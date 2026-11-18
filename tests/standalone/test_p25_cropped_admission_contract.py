"""Admission RED preparation executing the actual model prepare_attn method."""
import ast
import enum
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

SOURCE = Path(os.environ.get("P25_TEST_SOURCE", str(Path(__file__).resolve().parents[2])))
MAMBA = SOURCE / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"


class Mode(enum.Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


@pytest.fixture
def prepare(monkeypatch):
    prefix = "vllm_ascend.worker.v2.spec_decode.tree"
    for size in range(1, len(prefix.split(".")) + 1):
        name = ".".join(prefix.split(".")[:size])
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    name = prefix + ".state_rollback"
    path = SOURCE / "vllm_ascend/worker/v2/spec_decode/tree/state_rollback.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    # Configuration parsing is outside this test; execute the real admission.
    monkeypatch.setattr(module, "tree_state_slot_count", lambda config: 33)
    methods = [n for n in ast.walk(ast.parse(MAMBA.read_text(encoding="utf-8")))
               if isinstance(n, ast.FunctionDef) and n.name == "prepare_attn"]
    assert len(methods) == 1
    method = methods[0]
    cls = ast.ClassDef(name="ProductionPrepare", bases=[], keywords=[], body=[method], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(np=np, torch=torch, CUDAGraphMode=Mode,
                     AscendMambaHybridAttnMetadata=lambda **kw: types.SimpleNamespace(**kw),
                     build_attn_metadata=lambda **kw: kw)
    program = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(program, str(MAMBA), "exec"), namespace)

    def run(q, capacity, *, rows=1, capture=False, dummy=False, prefill=False,
            mode=Mode.FULL, draft=None, missing_depths=False):
        model = namespace["ProductionPrepare"]()
        model.num_accepted_tokens_gpu = torch.full((rows,), 3, dtype=torch.int32)
        model.vllm_config = types.SimpleNamespace(num_speculative_tokens=15, parallel_config=object())
        model.max_model_len = 33792
        paths = torch.zeros((rows, 32), dtype=torch.int32)
        sampled = torch.full((rows,), 4, dtype=torch.int32)
        model._tree_buffers = lambda device: (paths, sampled)
        starts = np.arange(rows + 1, dtype=np.int32) * q
        parents = torch.zeros((rows, 32), dtype=torch.int32)
        batch = types.SimpleNamespace(
            num_reqs=rows, num_reqs_after_padding=rows,
            num_tokens=rows*q, num_tokens_after_padding=rows*capacity,
            is_dummy=dummy, is_prefilling_np=np.full(rows, prefill, dtype=np.bool_),
            prefill_runs_as_decode_np=None, idx_mapping=torch.arange(rows, dtype=torch.int64),
            num_scheduled_tokens=np.full(rows, q, dtype=np.int32),
            num_draft_tokens_per_req=np.full(rows, q-1 if draft is None else draft, dtype=np.int32),
            query_start_loc_np=starts, query_start_loc=torch.from_numpy(starts),
            tree_parents=parents, tree_num_nodes=torch.full((rows,), q-1, dtype=torch.int32),
            tree_depths=None if missing_depths else torch.ones_like(parents),
            seq_lens=torch.full((rows,), 512, dtype=torch.int32),
            seq_lens_np=np.full(rows, 512, dtype=np.int32), dcp_local_seq_lens=None,
            positions=torch.arange(rows*capacity, dtype=torch.int64), attn_state=object())
        result = model.prepare_attn(batch, mode, (), torch.empty(0), [], object(), for_capture=capture)
        assert result["num_actual_tokens"] == rows*q
        assert result["num_tokens"] == rows*(capacity if mode == Mode.FULL else q)
        assert result["model_specific_attn_metadata"].tree_depths is batch.tree_depths
        return result["model_specific_attn_metadata"]

    return run


@pytest.mark.parametrize("q,capacity", [(2,9), (9,9), (16,17), (17,17), (24,25),
                                       (25,25), (26,33), (29,33), (32,33), (33,33)])
def test_real_bs1_decode_admits_actual_width_up_to_graph_capacity(prepare, q, capacity):
    metadata = prepare(q, capacity)
    assert metadata.share_tree_metadata, "Real cropped BS1 decode Q<=C<=S should share logical metadata"


@pytest.mark.parametrize("kwargs", [dict(q=1,capacity=1), dict(q=0,capacity=1),
    dict(q=29,capacity=25), dict(q=33,capacity=49), dict(q=29,capacity=33,draft=15),
    dict(q=33,capacity=33,capture=True), dict(q=33,capacity=33,dummy=True),
    dict(q=33,capacity=33,prefill=True), dict(q=33,capacity=33,rows=2),
    dict(q=33,capacity=33,mode=Mode.NONE), dict(q=33,capacity=33,mode=Mode.PIECEWISE)])
def test_noneligible_paths_keep_generic_metadata(prepare, kwargs):
    assert not prepare(**kwargs).share_tree_metadata


def test_missing_current_tree_depths_keeps_generic_fallback(prepare):
    assert not prepare(33, 33, missing_depths=True).share_tree_metadata
