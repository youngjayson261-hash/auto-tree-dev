"""CPU wiring checks; this does not substitute for HCCL/stream validation."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
node = next(node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_publish_tree_budget")
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
PUBLISH = scope["_publish_tree_budget"]


class BroadcastContract(unittest.TestCase):
    def make(self, rank=0, size=2):
        events = []
        group = SimpleNamespace(world_size=size, rank_in_group=rank)
        def broadcast(packet, src):
            events.append(("broadcast", src, packet.clone()))
            packet.copy_(torch.tensor([[3, 1]], dtype=torch.int32))
            return packet
        group.broadcast = broadcast
        proof = SimpleNamespace(tp_size=size, tp_group=group)
        packet = torch.zeros((1, 2), dtype=torch.int32)
        def publish(tree, output):
            events.append(("publish",))
            output.fill_(7 + rank)
        runner = SimpleNamespace(_tree_budget_runtime=SimpleNamespace(publish=publish), tree=object(), tree_budget_packet=packet, input_batch=SimpleNamespace(num_reqs=1), _tree_dflash2_confidence_proof=proof)
        distributed = ModuleType("vllm.distributed")
        distributed.get_tp_group = lambda: group
        admission = ModuleType("vllm_ascend.worker.v2.spec_decode.tree.confidence_admission")
        admission.valid_tp2_confidence_proof = lambda candidate: candidate is proof
        modules = {"vllm.distributed": distributed, admission.__name__: admission}
        return runner, events, group, modules

    def test_publish_precedes_broadcast_on_both_ranks(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                obj, events, group, modules = self.make(rank=rank)
                with patch.dict(sys.modules, modules):
                    PUBLISH(obj)
                self.assertEqual([event[0] for event in events], ["publish", "broadcast"])
                self.assertEqual(events[1][1], 0)
                self.assertEqual(events[1][2].tolist(), [[7 + rank, 7 + rank]])
                self.assertEqual(obj.tree_budget_packet.tolist(), [[3, 1]])

    def test_tp1_keeps_original_publish_only(self):
        obj, events, _, modules = self.make(size=1)
        with patch.dict(sys.modules, modules):
            PUBLISH(obj)
        self.assertEqual([event[0] for event in events], ["publish"])

    def test_fixed_has_no_collective(self):
        obj, events, _, modules = self.make()
        obj._tree_budget_runtime = None
        obj._write_fixed_tree_budget = lambda: events.append(("fixed",))
        with patch.dict(sys.modules, modules):
            PUBLISH(obj)
        self.assertEqual(events, [("fixed",)])

    def test_stale_group_rejected_without_collective(self):
        obj, events, _, modules = self.make()
        modules["vllm.distributed"].get_tp_group = lambda: object()
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "proven live TP group"):
                PUBLISH(obj)
        self.assertEqual([event[0] for event in events], ["publish"])

    def test_invalid_proof_rejected_without_collective(self):
        obj, events, _, modules = self.make()
        modules["vllm_ascend.worker.v2.spec_decode.tree.confidence_admission"].valid_tp2_confidence_proof = lambda _: False
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "proven live TP group"):
                PUBLISH(obj)
        self.assertEqual([event[0] for event in events], ["publish"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
