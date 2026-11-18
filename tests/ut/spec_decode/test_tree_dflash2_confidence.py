# SPDX-License-Identifier: Apache-2.0
"""DFlash2 confidence contracts on CPU, without importing an NPU runtime.

Production modules are loaded into a private package, changing only their
imports. Model methods are extracted verbatim from their actual class; their
LM-head/TP boundaries are small CPU doubles. This is not model-load/NPU proof.
Run directly with --source-root, --reference-root and --json-report for RED/GREEN.
"""

import argparse
import ast
import copy
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


TREE_REL = Path("vllm_ascend/worker/v2/spec_decode/tree")
SOURCE_ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCE_ROOT = SOURCE_ROOT
REFERENCE_ROOT = None
JSON_REPORT = None


def load_tree(source_root, package_name):
    """Execute the complete CPU production modules with isolated imports."""
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root / TREE_REL)]
    sys.modules[package_name] = package

    class IsolateImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "vllm_ascend.ops.vocab_parallel_embedding":
                return None  # Used only by the unrelated Domino weight adapter.
            prefix = "vllm_ascend.worker.v2.spec_decode.tree"
            if node.module and node.module.startswith(prefix):
                node.module = package_name + node.module[len(prefix) :]
            return node

    modules = {}
    for name in ("budget_policy", "base_confidence", "layout", "scoring", "builder", "dflash2_builder"):
        path = source_root / TREE_REL / (name + ".py")
        module = types.ModuleType(package_name + "." + name)
        module.__file__ = str(path)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        code = IsolateImports().visit(ast.parse(path.read_text(encoding="utf-8")))
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)
        modules[name] = module
    return SimpleNamespace(**modules)


class UnquantizedHead:
    def apply(self, head, hidden, bias=None):
        assert bias is None
        self.calls = getattr(self, "calls", 0) + 1
        return hidden @ head.weight.T


def load_model_methods(source_root, tp=1, rank=0, gather=None):
    path = source_root / "vllm_ascend/models/qwen3_dflash2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlash2Qwen3ForCausalLM")
    methods = [
        copy.deepcopy(n) for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("compute_candidates")
    ]
    cls = ast.ClassDef(name="ModelMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    scope = {
        "torch": torch,
        "UnquantizedEmbeddingMethod": UnquantizedHead,
        "get_tensor_model_parallel_world_size": lambda: tp,
        "get_tensor_model_parallel_rank": lambda: rank,
        "tensor_model_parallel_all_gather": gather,
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope["ModelMethods"]


class RawAllGatherProcessor:
    """One local GEMM and one fake raw collective; no forward transforms."""

    def __init__(self, raw, vocab_size):
        self.raw = raw
        self.vocab_size = self.org_vocab_size = vocab_size
        self.use_all_gather = True
        self.logits_as_input = False
        self.calls = self.full_gathers = 0
        self.override_output = False
        self.output = None

    def _get_logits(self, hidden, head, bias):
        self.calls += 1
        local = head.quant_method.apply(head, hidden, bias=bias)
        first = head.tp_rank * head.num_embeddings_per_partition
        torch.testing.assert_close(local, self.raw[:, first : first + local.shape[-1]], rtol=0, atol=0)
        self.full_gathers += 1
        return self.output if self.override_output else self.raw[:, :self.org_vocab_size].clone()

    def __call__(self, *args, **kwargs):
        raise AssertionError("processor.forward would change transform order")


def layout(modules, rows, budget):
    return modules.layout.TreeLayout(
        torch.empty(rows, budget, dtype=torch.long),
        torch.empty(rows, budget, dtype=torch.long),
        torch.empty(rows, budget, dtype=torch.long),
        torch.empty(rows, dtype=torch.int32),
        torch.empty(rows, budget, budget, dtype=torch.bool),
        torch.empty(rows, budget + 1, dtype=torch.int32),
        torch.empty(rows, budget + 1, dtype=torch.int32),
    )


def summary(modules, rows, depth):
    return modules.budget_policy.TreeScoreSummary(
        torch.empty(rows, depth), torch.empty(rows, depth, dtype=torch.bool), torch.empty(rows, dtype=torch.bool)
    )


def fixture(modules, rows=2, depth=4, width=2, keep_origin=False):
    torch.manual_seed(731)
    vocab, candidates, rank = 11, 4, 3
    selector = SimpleNamespace(
        top_k=candidates,
        predecessor_codebook=torch.randn(vocab, rank),
        successor_codebook=torch.randn(vocab, rank),
        hidden_projection=torch.nn.Identity(),
    )
    scorer = modules.scoring.DFlash2Scorer(SimpleNamespace(candidate_selector=selector))
    builder = modules.dflash2_builder.DFlash2BeamTreeBuilder(
        budget=8, topk=width, scorer=scorer, keep_origin_chain=keep_origin
    )
    full_logits = torch.randn(rows, depth, vocab)
    ids = torch.stack([torch.randperm(vocab)[:candidates] for _ in range(rows * depth)])
    ids = ids.reshape(rows, depth, candidates)
    unary = full_logits.gather(-1, ids)
    hidden = torch.randn(rows, depth, rank)
    roots = torch.arange(rows, dtype=torch.long) + 5
    return builder, unary, ids, hidden, roots, full_logits


def build(modules, data, scores=None, source="scorer_logprob", base=None):
    builder, unary, ids, hidden, roots, full = data
    kwargs = {}
    if scores is not None:
        kwargs = dict(
            score_summary=scores,
            full_vocab_size=full.shape[-1],
            verification_confidence_source=source,
            base_confidence_logits=base,
        )
    return builder.build(
        unary,
        layout(modules, unary.shape[0], builder.budget),
        root_token_ids=roots,
        draft_candidate_ids=ids,
        draft_hidden=hidden,
        **kwargs,
    )


class TestDFlash2Confidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_tree(SOURCE_ROOT, "_echo_dflash2_candidate")
        cls.reference = None if REFERENCE_ROOT is None else load_tree(REFERENCE_ROOT, "_echo_dflash2_reference")

    def assert_layout_equal(self, first, second):
        for field in ("tokens", "depths", "parents", "num_nodes", "visibility", "first_child", "next_sibling"):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)

    def tp2_model(self, rank, *, dtype=torch.float32, softcap=None, ties=False, top_k=2):
        vocab, partition = 7, 4
        weights = torch.tensor(
            [[2., 1.], [-1., 3.], [4., 0.], [1., 2.],
             [2., 1.], [3., -1.], [-2., 0.], [99., 99.]], dtype=dtype,
        )
        if ties:
            weights[:vocab].zero_()
        hidden = torch.tensor([[1., 1.], [1., -1.], [-1., 2.]], dtype=dtype)
        raw = hidden @ weights.T
        rank_values, rank_ids = [], []
        for source_rank in range(2):
            first = source_rank * partition
            local = raw[:, first:first + partition].clone()
            real = min(partition, vocab - first)
            local[:, real:] = -torch.inf
            values, ids = torch.topk(local, top_k, -1)
            rank_values.append(values)
            rank_ids.append(ids + first)
        calls = []

        def gather(tensor, dim=-1):
            self.assertEqual(dim, -1)
            expected = rank_ids if tensor.dtype == torch.long else rank_values
            torch.testing.assert_close(tensor, expected[rank], rtol=0, atol=0)
            calls.append("ids" if tensor.dtype == torch.long else "values")
            return torch.cat(expected, dim=-1)

        model = load_model_methods(MODEL_SOURCE_ROOT, tp=2, rank=rank, gather=gather)()
        model.config = SimpleNamespace(vocab_size=vocab)
        model.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=top_k))
        first = rank * partition
        end = min(first + partition, vocab)
        model.lm_head = SimpleNamespace(
            weight=weights[first:first + partition].clone(),
            quant_method=UnquantizedHead(),
            tp_size=2, tp_rank=rank, parallel_group=None, disable_tp=False,
            num_embeddings_per_partition=partition, num_embeddings_padded=partition * 2,
            num_embeddings=vocab, org_vocab_size=vocab, num_added_embeddings=0,
            org_vocab_size_padded=partition * 2,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=first, org_vocab_end_index=end,
                padded_org_vocab_start_index=first, padded_org_vocab_end_index=first + partition,
                num_org_vocab_padding=partition - (end - first),
            ),
        )
        model.logits_processor = RawAllGatherProcessor(raw, vocab)
        model.draft_id_to_target_id = None
        model.output_multiplier = .7
        model.final_logit_softcapping = softcap
        return model, hidden, raw, calls

    def test_fixed_tree_matches_frozen_reference(self):
        if self.reference is None:
            self.skipTest("pass --reference-root for frozen fixed-path comparison")
        for keep_origin in (False, True):
            for width in (1, 2, 4):
                with self.subTest(origin=keep_origin, width=width):
                    actual_data = fixture(self.modules, width=width, keep_origin=keep_origin)
                    reference_data = fixture(self.reference, width=width, keep_origin=keep_origin)
                    actual = build(self.modules, actual_data)
                    expected = build(self.reference, reference_data)
                    self.assert_layout_equal(actual, expected)
                    for name in ("_tokens_buf", "_parents_buf", "_depths_buf", "_path_scores_buf", "_frontier_buf"):
                        torch.testing.assert_close(
                            getattr(actual_data[0], name), getattr(reference_data[0], name), rtol=0, atol=0
                        )

    def test_native_and_base_summaries_leave_fixed_tree_unchanged(self):
        for source in ("scorer_logprob", "base_full_vocab"):
            for keep_origin in (False, True):
                with self.subTest(source=source, origin=keep_origin):
                    data = fixture(self.modules, keep_origin=keep_origin)
                    expected = build(self.modules, data)
                    scores = summary(self.modules, 2, 4)
                    actual = build(self.modules, data, scores, source, data[-1])
                    self.assert_layout_equal(actual, expected)
                    self.assertTrue(scores.valid_depths.all())
                    self.assertFalse(scores.invalid_scores.any())

    def test_summary_tracks_every_stored_node_and_actual_parent(self):
        for source in ("scorer_logprob", "base_full_vocab"):
            for keep_origin in (False, True):
                with self.subTest(source=source, origin=keep_origin):
                    data = fixture(self.modules, keep_origin=keep_origin)
                    builder, unary, ids, hidden, roots, full = data
                    scores = summary(self.modules, 2, 4)
                    paths = [{0: 0.0}, {0: 0.0}]
                    maxima = []
                    original = builder._expand_candidates
                    base_logp = torch.log_softmax(full.double(), -1)

                    def recording(state, layer, child_depth, origin_frontier, **kwargs):
                        start = state.num_nodes + 1
                        live = 1 if child_depth == 1 else state.width
                        parents = state.frontier[:, :live].tolist()
                        edges = layer.scores[:, :live].double().tolist()
                        tokens = layer.token_ids[:, :live].tolist()
                        candidates = []
                        for row in range(2):
                            candidates.append([
                                paths[row][parent] + (
                                    edges[row][p][k] if source == "scorer_logprob"
                                    else float(base_logp[row, child_depth - 1, token])
                                )
                                for p, parent in enumerate(parents[row]) for k, token in enumerate(tokens[row][p])
                            ])
                        maxima.append([max(row) for row in candidates])
                        result = original(state, layer, child_depth, origin_frontier, **kwargs)
                        added = result[0]
                        for row in range(2):
                            for node in range(start, start + added):
                                parent = int(state.parents[row, node])
                                token = int(state.tokens[row, node - 1])
                                parent_slot = parents[row].index(parent)
                                candidate_slot = tokens[row][parent_slot].index(token)
                                index = parent_slot * len(tokens[row][parent_slot]) + candidate_slot
                                paths[row][node] = candidates[row][index]
                        return result

                    builder._expand_candidates = recording
                    build(self.modules, data, scores, source, full)
                    torch.testing.assert_close(
                        scores.max_path_logp.double(), torch.tensor(maxima, dtype=torch.double).T
                    )
                    expected_nodes = builder._pool_capacity(2, 4) + 1
                    self.assertEqual(len(paths[0]), expected_nodes)
                    for row, row_paths in enumerate(paths):
                        for node, value in row_paths.items():
                            self.assertAlmostEqual(float(builder._base_path_scores_buf[row, node]), value, places=5)

    def test_base_summary_includes_candidate_pruned_by_corrected_shortlist(self):
        data = fixture(self.modules, rows=1, depth=2, width=1)
        builder, unary, ids, hidden, roots, full = data
        ids[:] = torch.tensor([[[8, 3, 6, 2], [7, 4, 1, 9]]])
        unary.zero_()
        hidden.fill_(1)
        builder.scorer.predecessor_codebook.fill_(1)
        builder.scorer.successor_codebook.zero_()
        builder.scorer.successor_codebook[8].fill_(5)
        builder.scorer.successor_codebook[7].fill_(5)
        full.zero_()
        full[0, 0, 3] = 10  # Highest base confidence is not selected by correction.
        full[0, 1, 4] = 7
        scores = summary(self.modules, 1, 2)
        build(self.modules, data, scores, "base_full_vocab", full)
        logp = torch.log_softmax(full, -1)
        self.assertEqual(int(builder._tokens_buf[0, 0]), 8)
        torch.testing.assert_close(scores.max_path_logp[0, 0], logp[0, 0, 3])
        torch.testing.assert_close(scores.max_path_logp[0, 1], logp[0, 0, 8] + logp[0, 1, 4])
        torch.testing.assert_close(builder._base_path_scores_buf[0, 1], logp[0, 0, 8])

    def test_native_scorer_uses_global_token_ids_and_actual_frontier(self):
        data = fixture(self.modules)
        builder, unary, ids, hidden, roots, full = data
        state = SimpleNamespace(
            frontier=torch.tensor([[0, 2], [1, 0]]),
            tokens=torch.tensor([[9, 3], [7, 10]]),
        )
        ctx = builder.scorer.prepare(state, unary, roots, hidden, candidate_ids=ids)
        layer = builder.scorer.score(state, ctx, 2, False)
        parent_tokens = [[int(roots[0]), 3], [7, int(roots[1])]]
        expected = torch.empty(2, 2, 4)
        for row in range(2):
            for parent in range(2):
                for candidate in range(4):
                    token = int(ids[row, 2, candidate])
                    expected[row, parent, candidate] = unary[row, 2, candidate] + sum(
                        float(builder.scorer.predecessor_codebook[parent_tokens[row][parent], rank])
                        * float(hidden[row, 2, rank])
                        * float(builder.scorer.successor_codebook[token, rank]) for rank in range(3)
                    )
        torch.testing.assert_close(layer.scores, torch.log_softmax(expected, -1))
        torch.testing.assert_close(layer.token_ids, ids[:, 2, None, :].expand(2, 2, 4))

    def test_invalid_ids_do_not_wrap_to_codebook_tail(self):
        for root, invalid_id in ((True, -1), (False, -1), (False, 11)):
            with self.subTest(root=root, invalid_id=invalid_id):
                data = fixture(self.modules)
                if root:
                    data[4][0] = invalid_id
                else:
                    data[2][0, 0, 0] = invalid_id
                with self.assertRaises((IndexError, RuntimeError)):
                    build(self.modules, data)

    def test_wrong_lattice_metadata_is_rejected(self):
        for kind in ("float_ids", "float_roots", "root_shape", "candidate_shape", "width", "missing_ids"):
            with self.subTest(kind=kind):
                builder, unary, ids, hidden, roots, full = fixture(self.modules)
                if kind == "float_ids":
                    ids = ids.float()
                elif kind == "float_roots":
                    roots = roots.float()
                elif kind == "root_shape":
                    roots = roots[:, None]
                elif kind == "candidate_shape":
                    ids = ids[:, :-1]
                elif kind == "width":
                    ids, unary = ids[..., :-1], unary[..., :-1]
                else:
                    ids = None
                with self.assertRaises(ValueError):
                    builder.scorer.prepare(None, unary, roots, hidden, candidate_ids=ids)

    def test_base_requires_complete_logits_not_topk_or_implicit_native(self):
        for bad in (None, torch.zeros(2, 4, 4), torch.zeros(2, 3, 11), torch.zeros(2, 4, 11, dtype=torch.long)):
            with self.subTest(shape=None if bad is None else tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    build(self.modules, fixture(self.modules), summary(self.modules, 2, 4), "base_full_vocab", bad)

    def test_wrong_confidence_proof_or_source_is_rejected(self):
        for vocab, source in ((4, "scorer_logprob"), (True, "scorer_logprob"), (11, "other")):
            with self.subTest(vocab=vocab, source=source):
                data = fixture(self.modules)
                builder, unary, ids, hidden, roots, full = data
                with self.assertRaises(ValueError):
                    builder.build(unary, layout(self.modules, 2, 8), root_token_ids=roots, draft_hidden=hidden,
                                  draft_candidate_ids=ids, score_summary=summary(self.modules, 2, 4),
                                  full_vocab_size=vocab, verification_confidence_source=source)

    def test_full_vocab_nan_marks_only_affected_row(self):
        data = fixture(self.modules)
        data[-1][0, 3, 10] = torch.nan  # Also invalid when outside the retained lattice.
        scores = summary(self.modules, 2, 4)
        build(self.modules, data, scores, "base_full_vocab", data[-1])
        self.assertEqual(scores.invalid_scores.tolist(), [True, False])

    def test_native_invalid_edges_are_recorded_per_row(self):
        for value, invalid in ((torch.nan, True), (torch.inf, True), (.01, True), (-torch.inf, False)):
            with self.subTest(value=value):
                data = fixture(self.modules)
                scorer = data[0].scorer
                original = scorer.score

                def damaged(state, ctx, depth_slot, keep_origin_chain):
                    layer = original(state, ctx, depth_slot, keep_origin_chain)
                    if depth_slot == 1:
                        layer.scores[0, -1, -1] = value
                    return layer

                scorer.score = damaged
                scores = summary(self.modules, 2, 4)
                build(self.modules, data, scores)
                self.assertEqual(scores.invalid_scores.tolist(), [invalid, False])

    def test_inactive_first_layer_parents_do_not_poison_native_summary(self):
        data = fixture(self.modules)
        scorer = data[0].scorer
        original = scorer.score

        def damaged_padding(state, ctx, depth_slot, keep_origin_chain):
            layer = original(state, ctx, depth_slot, keep_origin_chain)
            if depth_slot == 0:
                layer.scores[:, 1:] = torch.nan
            return layer

        scorer.score = damaged_padding
        scores = summary(self.modules, 2, 4)
        build(self.modules, data, scores)
        self.assertFalse(scores.invalid_scores.any())
        self.assertTrue(torch.isfinite(scores.max_path_logp).all())

    def test_invalid_bootstrap_row_does_not_force_other_row_fallback(self):
        data = fixture(self.modules)
        data[-1][0, 2, 0] = torch.nan
        scores = summary(self.modules, 2, 4)
        tree = build(self.modules, data, scores, "base_full_vocab", data[-1])
        packet = torch.empty(2, 2, dtype=torch.int32)
        policy = self.modules.budget_policy
        policy.select_tree_prefix_device(tree, scores, packet, tau=.1, bootstrap_mask=torch.tensor([True, False]))
        self.assertEqual(int(packet[0, 1]), int(policy.PolicyStatus.BOOTSTRAP_FIXED))
        self.assertEqual(int(packet[1, 1]), int(policy.PolicyStatus.CONFIDENCE))

    def test_selector_width_and_metadata_proofs_reject_invalid_values(self):
        for top_k in (0, -1, True, 2.5, 12):
            with self.subTest(top_k=top_k):
                selector = SimpleNamespace(
                    top_k=top_k, predecessor_codebook=torch.zeros(11, 3),
                    successor_codebook=torch.zeros(11, 3), hidden_projection=torch.nn.Identity(),
                )
                with self.assertRaises(ValueError):
                    self.modules.scoring.DFlash2Scorer(SimpleNamespace(candidate_selector=selector))
        fn = self.modules.budget_policy.confidence_source_metadata
        for width, candidates, vocab in ((0, 4, 11), (5, 4, 11), (2, True, 11), (1, 1, True)):
            with self.subTest(width=width, candidates=candidates, vocab=vocab):
                with self.assertRaises(ValueError):
                    fn(source="scorer_logprob", scorer_name="dflash2", candidate_size=candidates,
                       prefix_len=0, width=width, vocab_size=vocab, threshold=.1)

    def test_capture_workspace_uses_lattice_width_and_stable_storage(self):
        data = fixture(self.modules, width=8)
        builder = data[0]
        builder.prepare_for_capture(2, 11, 4, torch.device("cpu"), reserve_base_confidence=True)
        self.assertEqual(builder._frontier_buf.shape[1], 4)
        pointers = tuple(t.data_ptr() for t in builder.workspace_tensors())
        for source in ("scorer_logprob", "base_full_vocab"):
            build(self.modules, data, summary(self.modules, 2, 4), source, data[-1])
            self.assertEqual(tuple(t.data_ptr() for t in builder.workspace_tensors()), pointers)

    def test_metadata_declares_topk_domain_and_preserves_domino(self):
        fn = self.modules.budget_policy.confidence_source_metadata
        for source in ("scorer_logprob", "base_full_vocab"):
            meta = fn(
                source=source, scorer_name="dflash2", candidate_size=4, prefix_len=0,
                width=2, vocab_size=11, threshold=.1,
            )
            self.assertEqual(
                meta["normalization_domain"], "selector_lattice" if source == "scorer_logprob" else "base_full_vocab"
            )
            self.assertEqual(meta["normalization_size"], 4 if source == "scorer_logprob" else 11)
            self.assertEqual(meta["token_mapping_requirement"], "target_vocab_ids")
            if self.reference is not None:
                args = dict(
                    source=source, scorer_name="domino", candidate_size=4, prefix_len=2,
                    width=2, vocab_size=11, threshold=.1,
                )
                self.assertEqual(fn(**args), self.reference.budget_policy.confidence_source_metadata(**args))
        scorer_meta = fixture(self.modules)[0].scorer.confidence_metadata()
        self.assertEqual(scorer_meta["normalization_domain"], "selector_lattice")
        self.assertEqual(scorer_meta["token_id_domain"], "target_vocab")

    def test_model_base_interface_retains_legacy_lattice_and_full_denominator(self):
        methods = load_model_methods(MODEL_SOURCE_ROOT)
        for softcap in (None, 1.7):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(softcap=softcap, dtype=dtype):
                    model = methods()
                    model.config = SimpleNamespace(vocab_size=7)
                    model.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=3))
                    model.lm_head = SimpleNamespace(
                        weight=torch.tensor(
                            [[3., -1.], [1., 2.], [0., -2.], [1., 0.],
                             [-3., 2.], [4., 1.], [0., 0.], [30., 30.]], dtype=dtype,
                        ),
                        quant_method=UnquantizedHead(),
                        shard_indices=SimpleNamespace(num_org_vocab_padding=1, org_vocab_start_index=0),
                    )
                    model.output_multiplier = 0.7
                    model.final_logit_softcapping = softcap
                    hidden = torch.tensor([[1., 2.], [2., -1.]], dtype=dtype)
                    legacy_ids, legacy_vals = model.compute_candidates(hidden)
                    ids, values, base = model.compute_candidates_with_base_logits(hidden)
                    torch.testing.assert_close(ids, legacy_ids, rtol=0, atol=0)
                    torch.testing.assert_close(values, legacy_vals, rtol=0, atol=0)
                    expected = (hidden @ model.lm_head.weight[:7].T).float() * 0.7
                    if softcap is not None:
                        expected = torch.tanh(expected / softcap) * softcap
                    torch.testing.assert_close(base, expected, rtol=0, atol=0)
                    self.assertEqual(tuple(base.shape), (2, 7))
                    self.assertTrue((ids < 7).all())
                    self.assertFalse(torch.allclose(torch.logsumexp(base, -1), torch.logsumexp(values, -1)))

    def test_model_full_base_interface_rejects_tp2_without_loaded_proof(self):
        model = load_model_methods(MODEL_SOURCE_ROOT, tp=2)()
        with self.assertRaises(ValueError):
            model.compute_candidates_with_base_logits(torch.zeros(1, 2))

    def test_tp2_base_matches_original_rankcat_ties_and_uses_one_raw_gather(self):
        for dtype in (torch.float32, torch.bfloat16):
            for softcap in (None, 1.7):
                for ties in (False, True):
                    results = []
                    for rank in (0, 1):
                        with self.subTest(dtype=dtype, softcap=softcap, ties=ties, rank=rank):
                            model, hidden, raw, calls = self.tp2_model(
                                rank, dtype=dtype, softcap=softcap, ties=ties
                            )
                            legacy_ids, legacy_values = model.compute_candidates(hidden)
                            self.assertEqual(calls, ["values", "ids"])
                            head_calls = model.lm_head.quant_method.calls
                            ids, values, base = model.compute_candidates_with_base_logits(hidden)
                            self.assertEqual(calls, ["values", "ids"])
                            self.assertEqual(model.lm_head.quant_method.calls, head_calls + 1)
                            self.assertEqual(model.logits_processor.calls, 1)
                            self.assertEqual(model.logits_processor.full_gathers, 1)
                            torch.testing.assert_close(ids, legacy_ids, rtol=0, atol=0)
                            torch.testing.assert_close(values, legacy_values, rtol=0, atol=0)
                            expected = raw[:, :7].float() * .7
                            if softcap is not None:
                                expected = torch.tanh(expected / softcap) * softcap
                            torch.testing.assert_close(base, expected, rtol=0, atol=0)
                            self.assertTrue((ids < 7).all())
                            results.append((ids, values, base))
                    if len(results) == 2:
                        for first, second in zip(*results):
                            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_tp2_rejects_incompatible_loaded_metadata_before_collective(self):
        for kind in (
            "head_group", "head_size", "head_rank", "partition", "added_vocab",
            "all_gather", "logits_as_input", "processor_vocab", "padding", "shard_start", "mapping",
            "k_zero", "k_exceeds_tail", "disable_tp",
        ):
            with self.subTest(kind=kind):
                model, hidden, raw, calls = self.tp2_model(1)
                head, processor = model.lm_head, model.logits_processor
                if kind == "head_group":
                    head.parallel_group = object()
                elif kind == "head_size":
                    head.tp_size = 1
                elif kind == "head_rank":
                    head.tp_rank = 0
                elif kind == "partition":
                    head.num_embeddings_per_partition += 1
                elif kind == "added_vocab":
                    head.num_added_embeddings = 1
                elif kind == "all_gather":
                    processor.use_all_gather = False
                elif kind == "logits_as_input":
                    processor.logits_as_input = True
                elif kind == "processor_vocab":
                    processor.org_vocab_size -= 1
                elif kind == "padding":
                    head.shard_indices.num_org_vocab_padding += 1
                elif kind == "shard_start":
                    head.shard_indices.org_vocab_start_index -= 1
                elif kind == "mapping":
                    model.draft_id_to_target_id = torch.arange(7)
                elif kind == "k_zero":
                    model.model.candidate_selector.top_k = 0
                elif kind == "k_exceeds_tail":
                    model.model.candidate_selector.top_k = 4
                else:
                    head.disable_tp = True
                with self.assertRaises(ValueError):
                    model.compute_candidates_with_base_logits(hidden)
                self.assertEqual(processor.calls, 0)
                self.assertEqual(calls, [])
                self.assertEqual(getattr(head.quant_method, "calls", 0), 0)

    def test_tp2_rejects_incomplete_or_transformed_raw_outputs(self):
        for output in (
            None, torch.zeros(3, 6, dtype=torch.bfloat16), torch.zeros(3, 8, dtype=torch.bfloat16),
            torch.zeros(3, 7), torch.zeros(3, 7, dtype=torch.long),
            torch.empty(3, 7, dtype=torch.bfloat16, device="meta"),
        ):
            with self.subTest(shape=None if output is None else (tuple(output.shape), output.dtype)):
                model, hidden, raw, calls = self.tp2_model(0, dtype=torch.bfloat16)
                model.logits_processor.override_output = True
                model.logits_processor.output = output
                with self.assertRaises(ValueError):
                    model.compute_candidates_with_base_logits(hidden)
                self.assertEqual(model.logits_processor.full_gathers, 1)

    def test_tp2_candidate_width_at_real_tail_boundary(self):
        for top_k in (1, 3):
            for rank in (0, 1):
                with self.subTest(top_k=top_k, rank=rank):
                    model, hidden, raw, calls = self.tp2_model(rank, top_k=top_k)
                    legacy_ids, legacy_values = model.compute_candidates(hidden)
                    ids, values, base = model.compute_candidates_with_base_logits(hidden)
                    torch.testing.assert_close(ids, legacy_ids, rtol=0, atol=0)
                    torch.testing.assert_close(values, legacy_values, rtol=0, atol=0)
                    self.assertEqual(tuple(ids.shape), (3, top_k))
                    self.assertEqual(tuple(base.shape), (3, 7))
                    self.assertEqual(calls, ["values", "ids"])
                    self.assertEqual(model.logits_processor.full_gathers, 1)

    def test_base_rejects_unsupported_or_noninteger_tp_group_sizes(self):
        for size in (0, 3, 4, True, None):
            with self.subTest(size=size):
                model = load_model_methods(MODEL_SOURCE_ROOT, tp=size)()
                with self.assertRaises(ValueError):
                    model.compute_candidates_with_base_logits(torch.zeros(1, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--json-report", type=Path)
    args, rest = parser.parse_known_args()
    SOURCE_ROOT, REFERENCE_ROOT, JSON_REPORT = args.source_root.resolve(), args.reference_root, args.json_report
    MODEL_SOURCE_ROOT = SOURCE_ROOT if args.model_source_root is None else args.model_source_root.resolve()
    if REFERENCE_ROOT is not None:
        REFERENCE_ROOT = REFERENCE_ROOT.resolve()
    program = unittest.main(argv=[sys.argv[0], *rest], exit=False, verbosity=2)
    result = program.result
    if JSON_REPORT is not None:
        paths = [
            TREE_REL / (name + ".py")
            for name in ("scoring", "builder", "dflash2_builder", "base_confidence", "budget_policy", "layout")
        ]
        paths.append(Path("vllm_ascend/models/qwen3_dflash2.py"))
        report = {
            "source_root": str(SOURCE_ROOT),
            "reference_root": None if REFERENCE_ROOT is None else str(REFERENCE_ROOT),
            "model_source_root": str(MODEL_SOURCE_ROOT),
            "tests_run": result.testsRun,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "success": result.wasSuccessful(),
            "failures": [{"test": test.id(), "detail": detail} for test, detail in result.failures],
            "errors": [{"test": test.id(), "detail": detail} for test, detail in result.errors],
            "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
            "source_sha256": {
                str(path): hashlib.sha256(
                    ((MODEL_SOURCE_ROOT if path.parts[-2] == "models" else SOURCE_ROOT) / path).read_bytes()
                ).hexdigest() for path in paths
            },
            "limitation": (
                "CPU production scorer/builder; isolated imports and model-boundary doubles; "
                "no model load or NPU graph execution"
            ),
        }
        with JSON_REPORT.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
