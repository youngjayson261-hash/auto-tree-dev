"""CPU contract of the actual finalize method, without importing NPU runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py"
if "--source-file" in sys.argv:
    position = sys.argv.index("--source-file")
    SOURCE = Path(sys.argv[position + 1])
    del sys.argv[position:position + 2]
module = ast.parse(SOURCE.read_text(encoding="utf-8"))
method = next(node for cls in module.body if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_run_tree_finalize")
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
FINALIZE = namespace["_run_tree_finalize"]


class Model:
    def __init__(self, base_width=11, full=False):
        self.model = SimpleNamespace(candidate_selector=object())
        self.base_width = base_width
        self.calls = []
        self.full = full

    def compute_candidates(self, hidden):
        self.calls.append("native")
        ids = torch.tensor([[8, 2, 5]]).expand(hidden.shape[0], -1)
        return ids, torch.zeros_like(ids, dtype=torch.float32)

    def compute_candidates_with_base_logits(self, hidden):
        ids, logits = self.compute_candidates(hidden)
        self.calls[-1] = "base"
        return ids, logits, torch.ones((hidden.shape[0], self.base_width))

    def compute_logits(self, hidden):
        return torch.ones((hidden.shape[0], self.base_width))


class Builder:
    def build(self, logits, layout, **kwargs):
        self.logits, self.kwargs = logits, kwargs
        return "built"


def runner(source="scorer_logprob", enabled=True, proven=True, base_width=11):
    builder = Builder()
    summary = object()
    return SimpleNamespace(
        _draft_hidden_buf=torch.arange(12).view(4, 3).float(),
        num_speculative_steps=2, sample_indices=torch.arange(4), draft_backend="dflash",
        model=Model(base_width), _tree_budget_enabled=enabled, _tree_confidence_source=source,
        _tree_full_vocab_size=11, _tree_dflash2_confidence_proven=proven,
        _load_layout_from_buffers=lambda _: "layout", num_query_per_req=3,
        input_buffers=SimpleNamespace(input_ids=torch.arange(6)),
        _tree_budget_runtime=SimpleNamespace(score_summary=lambda _: summary) if enabled else None,
        tree_builder=builder,
    )


class FinalizeContract(unittest.TestCase):
    def test_fixed_keeps_native_interface(self):
        obj = runner(enabled=False, proven=False)
        FINALIZE(obj, 2)
        self.assertEqual(obj.model.calls, ["native"])
        self.assertNotIn("base_confidence_logits", obj.tree_builder.kwargs)
        self.assertNotIn("score_summary", obj.tree_builder.kwargs)

    def test_native_ids_are_real_ids_with_batch_layout(self):
        obj = runner()
        FINALIZE(obj, 2)
        ids = obj.tree_builder.kwargs["draft_candidate_ids"]
        self.assertEqual(tuple(ids.shape), (2, 2, 3))
        self.assertEqual(ids[1, 1].tolist(), [8, 2, 5])
        self.assertEqual(obj.tree_builder.kwargs["root_token_ids"].tolist(), [0, 3])
        self.assertNotIn("base_confidence_logits", obj.tree_builder.kwargs)

    def test_base_receives_full_vocab_separately(self):
        obj = runner(source="base_full_vocab")
        FINALIZE(obj, 2)
        self.assertEqual(obj.model.calls, ["base"])
        self.assertEqual(tuple(obj.tree_builder.logits.shape), (2, 2, 3))
        self.assertEqual(tuple(obj.tree_builder.kwargs["base_confidence_logits"].shape), (2, 2, 11))

    def test_truncated_base_is_rejected(self):
        obj = runner(source="base_full_vocab", base_width=3)
        with self.assertRaisesRegex(ValueError, "full base-vocabulary"):
            FINALIZE(obj, 2)

    def test_unproven_native_or_base_is_rejected(self):
        for source in ("scorer_logprob", "base_full_vocab"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "verified scoring interface"):
                    FINALIZE(runner(source=source, proven=False), 2)

    def test_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported DFlash2 confidence source"):
            FINALIZE(runner(source="unknown"), 2)

    def test_non_candidate_full_vocab_path_is_preserved(self):
        obj = runner()
        del obj.model.model.candidate_selector
        FINALIZE(obj, 2)
        self.assertEqual(tuple(obj.tree_builder.logits.shape), (2, 2, 11))
        self.assertIsNone(obj.tree_builder.kwargs["draft_candidate_ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
