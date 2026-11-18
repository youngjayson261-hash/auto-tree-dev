# SPDX-License-Identifier: Apache-2.0
"""Real two-rank HCCL budget transport; no model load or mocked collective.

Run only on the explicitly allocated physical cards 0,2. The parent uses
stdlib spawn; torch/NPU/vLLM import only in children after device visibility.
This checks production selector/publish/broadcast/seal/consumer methods, not
loaded-model admission, tree construction, KV/SSM commits, or FULL graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_record(obj):
    text = inspect.getsource(obj)
    return {
        "file": str(Path(inspect.getfile(obj)).resolve()),
        "line": inspect.getsourcelines(obj)[1],
        "signature": str(inspect.signature(obj)),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def case_data(generation, rank, rows):
    """Independent small oracle: 16 nodes per depth, four BFS depths."""
    scenario = ("normal", "bootstrap_nan", "score_fallback", "bad_layout", "cancel", "slot_change")[generation % 6]
    keep = [(generation + 2 * row + rank) % 5 for row in range(rows)]
    maxima = [[-0.5 if depth < kept else -6.0 for depth in range(4)] for kept in keep]
    parents = [[0 if node < 16 else node - 15 for node in range(64)] for _ in range(rows)]
    bootstrap, invalid = [False] * rows, [False] * rows
    expected = [[16 * kept, 1] for kept in keep]
    if rank == 0 and scenario == "bootstrap_nan":
        bootstrap[0], maxima[0][0] = True, float("nan")
        expected[0] = [64, 2]
    elif rank == 0 and scenario == "score_fallback":
        invalid[0] = True
        expected = [[64, 3] for _ in range(rows)]
    elif rank == 0 and scenario == "bad_layout":
        parents[0][0] = 1  # A root child cannot name itself as its parent.
        expected[0] = [0, 4]
    return scenario, maxima, parents, bootstrap, invalid, expected


def expect_error(kind, message, callback):
    try:
        callback()
    except kind as exc:
        if message not in str(exc):
            raise AssertionError(f"unexpected error: {exc}") from exc
        return str(exc)
    raise AssertionError(f"missing {kind.__name__}: {message}")


def make_handler(api, torch, np, device, spec, rows):
    """NPU buffers for production methods; no substitute method/collective.

    Upstream DraftTokensHandler.__init__ creates CUDA streams. This transport
    fixture bypasses that unrelated constructor, allocating the same fields
    used by the unmodified confidence-only production methods with NPU APIs.
    """
    handler = api.TreeDraftTokensHandler.__new__(api.TreeDraftTokensHandler)
    handler.device = device
    handler.copy_stream, handler.copy_event = torch.npu.Stream(), torch.npu.Event()
    handler.req_states = SimpleNamespace(
        max_num_reqs=rows,
        req_id_to_index={},
        tree_next_budget_cpu=np.full(rows, -919, dtype=np.int32),
    )
    handler.speculator = spec
    handler._budget_enabled, handler._budget_stats = True, None
    handler._pending_budget, handler._budget_sealed = None, False
    handler._budget_cpu = torch.empty((rows, 2), dtype=torch.int32, pin_memory=True)
    handler._budget_produced = torch.npu.Event()
    return handler


def exercise(args, rank, report):
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    import vllm
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from vllm.platforms import current_platform

    if not torch.distributed.is_backend_available("hccl"):
        raise RuntimeError("HCCL unavailable: no gloo/mock fallback is permitted")
    if current_platform.device_type != "npu":
        raise RuntimeError(f"expected native NPU platform, got {current_platform.device_type}")
    physical = current_platform.visible_device_id_to_physical_device_id(rank)
    if physical != (0, 2)[rank]:
        raise AssertionError(f"logical rank{rank} mapped to unexpected physical device {physical}")
    # Bind against the actual imported interfaces before any distributed call.
    init_kwargs = dict(
        world_size=2, rank=rank, local_rank=rank, backend="hccl",
        distributed_init_method=f"tcp://127.0.0.1:{args['port']}",
        timeout=timedelta(seconds=args["distributed_timeout"]),
    )
    inspect.signature(init_distributed_environment).bind(**init_kwargs)
    inspect.signature(initialize_model_parallel).bind(tensor_model_parallel_size=2, backend="hccl")
    report.update({
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "vllm": vllm.__version__,
        "device": str(device), "physical_device": (0, 2)[rank],
        "interfaces": {
            "init_distributed_environment": source_record(init_distributed_environment),
            "initialize_model_parallel": source_record(initialize_model_parallel),
        },
    })
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
    with set_current_vllm_config(config):
        try:
            init_distributed_environment(**init_kwargs)
            initialize_model_parallel(tensor_model_parallel_size=2, backend="hccl")
            group = get_tp_group()
            if group.world_size != 2 or group.rank_in_group != rank or group.ranks != [0, 1]:
                raise AssertionError("unexpected actual TP group")
            backend = str(torch.distributed.get_backend(group.device_group)).lower()
            if backend != "hccl":
                raise AssertionError(f"actual TP backend is {backend}, expected hccl")
            report["group"] = {"world_size": group.world_size, "rank_in_group": group.rank_in_group,
                               "ranks": group.ranks, "backend": backend}
            run_rounds(args, rank, report, torch, np, device, group)
            torch.npu.synchronize()
        finally:
            # Cleanup only groups owned by this fresh standalone process.
            had_error = sys.exc_info()[0] is not None
            cleanup_errors = []
            for cleanup in (destroy_model_parallel, destroy_distributed_environment):
                try:
                    cleanup()
                except BaseException:
                    cleanup_errors.append(traceback.format_exc())
            report["cleanup_errors"] = cleanup_errors
            if cleanup_errors and not had_error:
                raise RuntimeError("distributed cleanup failed; see rank cleanup_errors")


def run_rounds(args, rank, report, torch, np, device, group):
    from vllm_ascend.worker.v2.spec_decode.tree import draft_tokens as handler_api
    from vllm_ascend.worker.v2.spec_decode.tree.budget_policy import select_tree_prefix_device
    from vllm_ascend.worker.v2.spec_decode.tree.confidence_admission import DFlash2ConfidenceProof
    from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
    from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator, _TreeBudgetRuntime

    source_root = Path(args["source_root"]).resolve()
    for obj in (AscendTreeSpeculator, handler_api.TreeDraftTokensHandler, _TreeBudgetRuntime):
        if not Path(inspect.getfile(obj)).resolve().is_relative_to(source_root):
            raise AssertionError("production import escaped the requested isolated source")
    report["interfaces"].update({
        "publish": source_record(AscendTreeSpeculator._publish_tree_budget),
        "runtime_publish": source_record(_TreeBudgetRuntime.publish),
        "selector": source_record(select_tree_prefix_device),
        "set": source_record(handler_api.TreeDraftTokensHandler.set_draft_tokens),
        "seal": source_record(handler_api.TreeDraftTokensHandler.seal_budget_production),
        "consumer": source_record(handler_api.TreeDraftTokensHandler.get_draft_tokens),
        "consumer_result_type": source_record(handler_api.DraftTokenIds),
        "broadcast": source_record(type(group).broadcast),
    })
    rows = args["rows"]
    tree = TreeLayout(
        torch.zeros((rows, 64), dtype=torch.long, device=device),
        torch.arange(1, 5, device=device).repeat_interleave(16).expand(rows, -1).clone(),
        torch.empty((rows, 64), dtype=torch.long, device=device),
        torch.full((rows,), 64, dtype=torch.int32, device=device),
        torch.zeros((rows, 64, 64), dtype=torch.bool, device=device),
        torch.full((rows, 65), -1, dtype=torch.int32, device=device),
        torch.full((rows, 65), -1, dtype=torch.int32, device=device),
    )
    runtime = _TreeBudgetRuntime(rows, 4, device, tau=0.1)
    spec = SimpleNamespace(
        _tree_budget_runtime=runtime, tree=tree, budget=64,
        tree_budget_packet=torch.empty((rows, 2), dtype=torch.int32, device=device),
        input_batch=SimpleNamespace(num_reqs=rows),
        # Transport-only proof: actual group binding, no claim of model admission.
        _tree_dflash2_confidence_proof=DFlash2ConfidenceProof(248320, "selector_lattice", 16, 2, rank, group),
    )
    handler = make_handler(handler_api, torch, np, device, spec, rows)
    producer, delay_stream = torch.npu.Stream(), torch.npu.Stream()
    state_parents = torch.empty_like(tree.parents)
    state_generation = torch.zeros(1, dtype=torch.int32, device=device)
    busy_a = torch.full((1024, 1024), .01, dtype=torch.bfloat16, device=device)
    busy_b, busy_out = busy_a.clone(), torch.empty_like(busy_a)
    torch.npu.synchronize()
    report["rounds"] = []
    for generation in range(args["iterations"]):
        scenario, maxima, parents, bootstrap, invalid, local_expected = case_data(generation, rank, rows)
        expected = case_data(generation, 0, rows)[-1]
        if rank == 1 and local_expected == expected:
            raise AssertionError("fixture must differ from rank0 before broadcast")
        req_ids = [f"g{generation}-r{row}" for row in range(rows)]
        batch = SimpleNamespace(has_structured_output_reqs=False, req_ids=req_ids,
                                idx_mapping_np=np.arange(rows, dtype=np.int32))
        handler.req_states.req_id_to_index = dict(zip(req_ids, range(rows)))
        handler.req_states.tree_next_budget_cpu.fill(-919)
        host_parents = torch.tensor(parents, dtype=torch.long).pin_memory()
        host_maxima = torch.tensor(maxima, dtype=torch.float32).pin_memory()
        host_invalid = torch.tensor(invalid, dtype=torch.bool).pin_memory()
        host_bootstrap = np.asarray(bootstrap, dtype=np.bool_)
        spec.tree_budget_packet.fill_(-777)
        state_generation.fill_(-generation - 1)
        delay_stream.wait_stream(torch.npu.current_stream())
        producer.wait_stream(torch.npu.current_stream())
        ready = torch.npu.Event(enable_timing=True)
        published, state_done = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        with torch.npu.stream(delay_stream):
            for _ in range(args["delay_matmuls"]):
                torch.mm(busy_a, busy_b, out=busy_out)
            ready.record()
        pending_before_wait = not ready.query()
        with torch.npu.stream(producer):
            producer.wait_event(ready)
            tree.parents.copy_(host_parents, non_blocking=True)
            runtime.update_batch(host_bootstrap, rows)
            summary = runtime.score_summary(rows)
            summary.max_path_logp.copy_(host_maxima, non_blocking=True)
            summary.valid_depths.fill_(True)
            summary.invalid_scores.copy_(host_invalid, non_blocking=True)
            AscendTreeSpeculator._publish_tree_budget(spec)
            published.record()
            handler.set_draft_tokens(batch, None)
            expect_error(RuntimeError, "still pending", lambda: handler.set_draft_tokens(batch, None))
            expect_error(RuntimeError, "not sealed", handler.get_draft_tokens)
            # Same relative order as the runner: publish, set, tree-state IO, seal.
            state_parents.copy_(tree.parents)
            state_generation.fill_(generation + 1)
            state_done.record()
            handler.seal_budget_production()
            expect_error(RuntimeError, "already sealed", handler.seal_budget_production)
        pending_before_consumer = not handler._budget_produced.query()
        if scenario == "cancel":
            del handler.req_states.req_id_to_index[req_ids[-1]]
        elif scenario == "slot_change":
            handler.req_states.req_id_to_index[req_ids[-1]] = 0
        started = time.monotonic()
        if scenario == "bad_layout":
            error = expect_error(ValueError, "invalid confidence budget or tree layout", handler.get_draft_tokens)
            result = None
        elif scenario == "slot_change":
            error = expect_error(RuntimeError, "slot changed", handler.get_draft_tokens)
            result = None
        else:
            result, error = handler.get_draft_tokens(), None
        consumer_wall_ms = (time.monotonic() - started) * 1000
        packet = handler._budget_cpu[:rows].tolist()
        if packet != expected:
            raise AssertionError(f"rank{rank} g{generation} packet={packet}, rank0 oracle={expected}")
        if not handler._budget_produced.query():
            raise AssertionError("consumer returned before the original production seal event")
        if handler._pending_budget is not None or handler._budget_sealed:
            raise AssertionError("pending budget survived consumer success/error")
        if state_generation.cpu().tolist() != [generation + 1] or state_parents.cpu().tolist() != parents:
            raise AssertionError("consumer returned before post-publication tree state was complete")
        delivered = handler.req_states.tree_next_budget_cpu.tolist()
        if error is not None:
            if delivered != [-919] * rows:
                raise AssertionError("failed publication partially changed request budgets")
        else:
            accepted = rows - (scenario == "cancel")
            if result.req_ids != req_ids[:accepted] or result.draft_token_ids != [
                [-1] * p[0] for p in expected[:accepted]
            ]:
                raise AssertionError("wrong request/token budget result")
            if delivered != [p[0] for p in expected[:accepted]] + [-919] * (rows - accepted):
                raise AssertionError("wrong live request-state budget publication")
        report["rounds"].append({
            "generation": generation, "scenario": scenario,
            "local_expected_before_broadcast": local_expected, "rank0_expected": expected,
            "packet": packet, "request_budgets": delivered, "expected_consumer_error": error,
            "delay_event_pending_before_wait": pending_before_wait,
            "seal_event_pending_before_consumer": pending_before_consumer,
            "ready_to_published_ms": ready.elapsed_time(published),
            "published_to_state_done_ms": published.elapsed_time(state_done),
            "consumer_wall_ms": consumer_wall_ms, "passed": True,
        })
    if not any(row["delay_event_pending_before_wait"] for row in report["rounds"]):
        raise AssertionError("no asynchronous delay observed; increase --delay-matmuls and rerun under a new output")
    report["scope"] = {
        "real_hccl": True, "mock_collective": False, "production_publish_seal_consumer": True,
        "handler_constructor": "NPU field fixture; CUDA-only upstream constructor bypassed",
        "proof": "transport-only dataclass bound to actual TP group; not loaded-model admission",
        "tree_builder_model_full_graph_kv_ssm_tested": False,
    }


def child(rank, args):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    output = Path(args["output"])
    report = {"rank": rank, "pid": os.getpid(), "started_epoch": time.time(), "passed": False}
    code = 1
    with (output / f"rank{rank}.log").open("x", encoding="utf-8", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            exercise(args, rank, report)
            report["passed"], code = True, 0
        except BaseException:
            report["error"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            report["finished_epoch"] = time.time()
            write_new(output / f"rank{rank}.json", report)
    raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="new exclusive evidence directory")
    destination.add_argument("--report", type=Path, help="new parent aggregate JSON; sibling *-evidence directory")
    parser.add_argument("--tp-size", type=int, choices=(2,), default=2)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--devices", choices=("0,2",), default="0,2")
    parser.add_argument("--port", type=int, default=0, help="0 allocates an unused loopback TCP port")
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--rows", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--delay-matmuls", type=int, default=32)
    parser.add_argument("--distributed-timeout", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.iterations < 12 or args.delay_matmuls < 1 or args.distributed_timeout < 1 or args.timeout < 1:
        parser.error("require iterations>=12, positive delay/timeout values")
    if args.port != 0 and not 1024 <= args.port <= 65535:
        parser.error("port must be 0 or an unprivileged TCP port")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", args.devices) != args.devices:
        parser.error("conflicting ASCEND_RT_VISIBLE_DEVICES; refusing to expand/change an inherited mapping")
    source_root = args.source_root.resolve()
    if not (source_root / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py").is_file():
        parser.error("source-root must be the isolated production source")
    # Only the two explicitly allocated devices are ever selected by children.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(source_root))
    # A bind preflight rejects an occupied explicit port. Port 0 gives a unique
    # kernel allocation; a later bind race fails normally, without port cleanup.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
        args.port = reservation.getsockname()[1]
    aggregate = None if args.report is None else args.report.absolute()
    if aggregate is not None and os.path.lexists(aggregate):
        parser.error("report already exists; refusing to repeat/overwrite")
    output = (
        args.output.resolve() if aggregate is None else aggregate.parent / (aggregate.stem + "-evidence")
    )
    output.mkdir(parents=True, exist_ok=False)
    values = vars(args).copy()
    values.update(
        output=str(output), source_root=str(source_root), report=None if aggregate is None else str(aggregate)
    )
    write_new(output / "launch.json", {
        **values, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_devices": [0, 2], "spawn_ranks": 2, "started_epoch": time.time(),
    })
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=child, args=(rank, values), name=f"budget-tp2-rank{rank}") for rank in range(2)]
    timed_out, parent_error = False, None
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + args.timeout
        while any(process.is_alive() for process in processes):
            if time.monotonic() >= deadline or any(p.exitcode not in (None, 0) for p in processes):
                timed_out = time.monotonic() >= deadline
                break
            for process in processes:
                process.join(timeout=.2)
    except BaseException:
        parent_error = traceback.format_exc()
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()  # Only a child created by this invocation.
        for process in processes:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()  # Escalate only a still-live child of this invocation.
                    process.join(timeout=10)
    ranks = []
    for rank in range(2):
        path = output / f"rank{rank}.json"
        ranks.append(
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"rank": rank, "passed": False}
        )
    consistent = all(r.get("passed") for r in ranks) and (
        [row["packet"] for row in ranks[0]["rounds"]] == [row["packet"] for row in ranks[1]["rounds"]]
    )
    passed = consistent and not timed_out and parent_error is None and all(p.exitcode == 0 for p in processes)
    final_report = {
        "passed": passed, "actual_hccl_executed_successfully": passed,
        "rank0_consensus": consistent, "timed_out": timed_out,
        "parent_error": parent_error,
        "children": [{"rank": i, "pid": p.pid, "exitcode": p.exitcode} for i, p in enumerate(processes)],
        "iterations": args.iterations, "physical_devices": [0, 2], "ranks": ranks,
        "tp_size": 2, "evidence_directory": str(output),
        "finished_epoch": time.time(),
    }
    write_new(output / "report.json", final_report)
    if aggregate is not None:
        write_new(aggregate, final_report)
    print(json.dumps({"passed": passed, "rank0_consensus": consistent, "output": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
