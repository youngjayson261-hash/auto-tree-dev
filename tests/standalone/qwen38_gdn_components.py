# SPDX-License-Identifier: Apache-2.0
"""Real NPU GDN/conv acceptance entry; CPU mode checks only its references.

References derive from the existing test_gdn_tree_kernel.py and
test_gdn_tree_conv_state.py in the 20261031 tree research snapshot. This entry
uses the production NPU operator and page IO without mocks. TP=2 here means
one rank's local head dimensions, not a distributed/HCCL acceptance test.
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch

DIM, SLOTS, WIDTH = 128, 65, 4
GEARS = [1, 17, 33, 49, 65]
# Alternate shrink/grow across each capture and 16-token chunk boundary.
# The last step reuses a page for a new request with stale sibling snapshots.
QUERIES = [65, 13, 33, 1, 17, 49, 2, 15, 16, 18, 32, 34, 48, 50, 64, 65, 13]


def ancestry(node):
    path = [node]
    while path[-1]:
        path.append((path[-1] - 1) // 2)
    return list(reversed(path))


def recurrent_reference(inputs, initial, pages, root_page, q):
    """Independent parent-state recurrence, including BF16 page rounding."""
    query, key, value, decay, beta = [tensor.cpu() for tensor in inputs]
    repeat = value.shape[1] // query.shape[1]
    query, key = [tensor.repeat_interleave(repeat, dim=1).float() for tensor in (query, key)]
    states, outputs = initial.clone(), []
    for node in range(q):
        parent = root_page if node == 0 else pages[(node - 1) // 2]
        state = states[parent].float() * decay[node, :, None, None].float().exp()
        residual = value[node].float() - torch.einsum("hvk,hk->hv", state, key[node])
        state += (residual * beta[node].float()[:, None])[:, :, None] * key[node, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, query[node]) * DIM**-0.5)
        states[pages[node]] = state.to(states.dtype)
    return torch.stack(outputs).bfloat16(), states


def conv_reference(x, history, weight, bias, q, activation=True):
    outputs = []
    for node in range(q):
        taps = torch.cat((history, x[ancestry(node)]), dim=0)[-WIDTH:]
        value = (taps.float() * weight.T.float()).sum(dim=0)
        if bias is not None:
            value += bias.float()
        if activation:
            value = torch.nn.functional.silu(value)
        outputs.append(value.to(x.dtype))
    return torch.stack(outputs)


def cpu_self_check():
    q, nk, nv = 3, 1, 3
    query = torch.ones(q, nk, DIM).bfloat16() / math.sqrt(DIM)
    inputs = [query, query, torch.zeros(q, nv, DIM).bfloat16(),
              torch.full((q, nv), math.log(0.5)), torch.zeros(q, nv).bfloat16()]
    initial = torch.ones(5, nv, DIM, DIM)
    output, states = recurrent_reference(inputs, initial, [1, 2, 3], 4, q)
    torch.testing.assert_close(states[1], torch.full_like(states[1], 0.5))
    torch.testing.assert_close(states[2], torch.full_like(states[2], 0.25))
    torch.testing.assert_close(states[3], states[2])
    torch.testing.assert_close(states[0], initial[0])
    x = torch.arange(12).view(3, 4).bfloat16()
    history = torch.ones(3, 4).bfloat16()
    result = conv_reference(x, history, torch.ones(4, 4).bfloat16(), None, 3, False)
    torch.testing.assert_close(result[0], 3 + x[0])
    torch.testing.assert_close(result[1], 2 + x[0] + x[1])
    torch.testing.assert_close(result[2], 2 + x[0] + x[2])
    return {"cpu_reference_checked": True, "npu_executed": False, "reference_output_shape": list(output.shape)}


def make_inputs(generator, capacity, key_heads, value_heads):
    query, key = [torch.nn.functional.normalize(
        torch.randn(capacity, key_heads, DIM, generator=generator), dim=-1
    ).bfloat16() for _ in range(2)]
    return [query, key, torch.randn(capacity, value_heads, DIM, generator=generator).bfloat16(),
            -(torch.rand(capacity, value_heads, generator=generator) * 0.2 + 0.1),
            torch.rand(capacity, value_heads, generator=generator).bfloat16()]


def recurrent_case(args, dtype, graph_mode, report, api):
    generator = torch.Generator().manual_seed(20381116)
    nk, nv, page_count = 16 // args.tp_size, 48 // args.tp_size, 70
    pages = torch.randperm(page_count, generator=generator)[:SLOTS].tolist()
    elements, leading, trailing = nv * DIM * DIM, 64, 128
    initial = (torch.randn(page_count, nv, DIM, DIM, generator=generator) * 0.02).to(dtype)
    expected_state = initial.clone()
    backing = torch.full((page_count, leading + elements + trailing), -91.0, device="npu", dtype=dtype)
    state = backing[:, leading:leading + elements].view_as(initial)
    state.copy_(initial)
    indices = torch.tensor([pages], dtype=torch.int32, device="npu")
    parents = torch.tensor([[(node - 1) // 2 for node in range(1, SLOTS)]], dtype=torch.int32, device="npu")
    counts = torch.tensor([64], dtype=torch.int32, device="npu")
    accepted = torch.ones(1, dtype=torch.int32, device="npu")
    buffers = api.allocate_tree_verify_plan_buffers(SLOTS, 1, torch.device("npu"))
    captures = {}

    def plan_for(q):
        counts.fill_(q - 1)
        init = api.compute_tree_init_state_indices(indices, accepted, parents, counts)
        lengths = torch.tensor([q], device="npu")
        packed = api.pack(indices, lengths, SLOTS)
        init = api.pack(init, lengths, SLOTS)
        return api.compute_tree_verify_plan(lengths, packed, init, q).copy_into(buffers)

    def run(inputs, plan, capacity):
        return api.chunked(*[tensor.unsqueeze(0) for tensor in inputs], state, indices, accepted,
                           DIM**-0.5, plan, capacity)

    if graph_mode:
        for capacity in GEARS:
            inputs = [tensor.npu() for tensor in make_inputs(generator, capacity, nk, nv)]
            plan = plan_for(capacity)
            run(inputs, plan, capacity)
            torch.npu.synchronize()
            state.copy_(initial)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run(inputs, plan, capacity)
            captures[capacity] = (graph, inputs, output)
        state.copy_(initial)

    accepted_node = 0
    for step, q in enumerate(QUERIES):
        capacity = min(n for n in GEARS if n >= q)
        reused_request = step == len(QUERIES) - 1
        if reused_request:
            # A new request reuses the old page; stale sibling snapshots remain.
            new_root = (torch.randn(nv, DIM, DIM, generator=generator) * 0.02).to(dtype)
            state[pages[0]].copy_(new_root)
            expected_state[pages[0]] = new_root
            accepted_node = 0
        path = torch.full((1, 64), -1, dtype=torch.int32, device="npu")
        selected_path = ancestry(accepted_node)[1:]
        if selected_path:
            path[0, :len(selected_path)] = torch.tensor(selected_path, dtype=torch.int32, device="npu")
        sampled = torch.tensor([len(selected_path) + 1], dtype=torch.int32, device="npu")
        accepted.copy_(api.compute_tree_resume_column(path, sampled))
        inputs_cpu = make_inputs(generator, capacity, nk, nv)
        # Bitwise untouched-page checks compare against the actual device pool,
        # not the independently rounded reference from the preceding step.
        old_state = state.cpu()
        expected, expected_state = recurrent_reference(inputs_cpu, expected_state, pages, pages[accepted_node], q)
        plan = plan_for(q)
        if graph_mode:
            graph, inputs, output = captures[capacity]
            for target, source in zip(inputs, inputs_cpu):
                target.copy_(source)
            graph.replay()
        else:
            output = run([tensor.npu() for tensor in inputs_cpu], plan, capacity)
        torch.npu.synchronize()
        output_cpu, state_cpu = output[:q].cpu(), state.cpu()
        torch.testing.assert_close(output_cpu.float(), expected.float(), rtol=0.02, atol=0.001)
        state_rtol, state_atol = (0.02, 0.001) if dtype == torch.bfloat16 else (0.002, 0.0002)
        torch.testing.assert_close(state_cpu, expected_state, rtol=state_rtol, atol=state_atol)
        untouched = sorted(set(range(page_count)) - set(pages[:q]))
        torch.testing.assert_close(state_cpu[untouched], old_state[untouched], rtol=0, atol=0)
        assert torch.all(backing[:, :leading] == -91.0).cpu().item()
        assert torch.all(backing[:, leading + elements:] == -91.0).cpu().item()
        report["cases"].append(dict(component="recurrent", graph=graph_mode, state_dtype=str(dtype),
                                    q=q, physical=capacity, chunks=5, accepted_node=accepted_node,
                                    request_page_reused=reused_request,
                                    output_max_abs=float((output_cpu.float() - expected.float()).abs().max()),
                                    state_max_abs=float((state_cpu.float() - expected_state.float()).abs().max()),
                                    status="passed"))
        accepted_node = q - 1


def conv_case(args, graph_mode, report, api):
    generator = torch.Generator().manual_seed(20381117)
    channels = (2 * 16 + 48) * DIM // args.tp_size
    page_count, active_page, columns = 5, 3, SLOTS + WIDTH - 1
    elements, leading, trailing = channels * columns, 64, 37
    shape = (page_count, channels, columns) if api.dim_first() else (page_count, columns, channels)
    backing = torch.full((page_count, leading + elements + trailing), -91.0, dtype=torch.bfloat16, device="npu")
    state = backing[:, leading:leading + elements].view(shape)
    initial = torch.randn(shape, generator=generator).bfloat16()
    state.copy_(initial)
    history = (initial[active_page, :, :3].T if api.dim_first() else initial[active_page, :3]).clone()
    weight = torch.randn(channels, WIDTH, generator=generator).bfloat16()
    bias = torch.randn(channels, generator=generator).bfloat16()
    weight_device, bias_device = weight.npu(), bias.npu()
    parents = torch.tensor([[(node - 1) // 2 for node in range(1, SLOTS)]], dtype=torch.int32, device="npu")
    counts = torch.tensor([64], dtype=torch.int32, device="npu")
    cache = torch.full((1, SLOTS), active_page, dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 65], dtype=torch.int32, device="npu")
    prefix = torch.arange(3, dtype=torch.int32, device="npu")[None]
    windows = api.compute_tree_conv_rows(parents, counts, WIDTH, 3)
    captures = {}

    def update(q, prefix_cpu):
        counts.fill_(q - 1)
        starts.copy_(torch.tensor([0, q], dtype=torch.int32))
        prefix.copy_(prefix_cpu)
        windows.copy_(api.compute_tree_conv_rows(parents, counts, WIDTH, 3,
                                               row_widths=torch.tensor([q], device="npu")))

    def run(x):
        return api.tree_causal_conv1d(x, weight_device, state, cache, starts, windows[:x.shape[0]],
                                     prefix, bias_device, True)

    if graph_mode:
        for capacity in GEARS:
            x = torch.randn(capacity, channels, generator=generator).bfloat16().npu()
            update(capacity, torch.arange(3, dtype=torch.int32)[None])
            run(x)
            torch.npu.synchronize()
            state.copy_(initial)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run(x)
            captures[capacity] = (graph, x, output)
        state.copy_(initial)

    next_prefix = torch.arange(3, dtype=torch.int32)[None]
    for step, q in enumerate(QUERIES):
        capacity = min(n for n in GEARS if n >= q)
        reused_request = step == len(QUERIES) - 1
        if reused_request:
            history = torch.randn(3, channels, generator=generator).bfloat16()
            destination = state[active_page, :, :3] if api.dim_first() else state[active_page, :3]
            destination.copy_(history.T if api.dim_first() else history)
            next_prefix = torch.arange(3, dtype=torch.int32)[None]
        x_cpu = torch.randn(capacity, channels, generator=generator).bfloat16()
        expected = conv_reference(x_cpu, history, weight, bias, q)
        update(q, next_prefix)
        old_state = state.cpu()
        if graph_mode:
            graph, x, output = captures[capacity]
            x.copy_(x_cpu)
            graph.replay()
        else:
            output = run(x_cpu.npu())
        torch.npu.synchronize()
        actual, current = output[:q].cpu(), state.cpu()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
        actual_history = current[active_page, :, :3].T if api.dim_first() else current[active_page, :3]
        torch.testing.assert_close(actual_history, history, rtol=0, atol=0)
        others = [page for page in range(page_count) if page != active_page]
        torch.testing.assert_close(current[others], old_state[others], rtol=0, atol=0)
        assert torch.all(backing[:, :leading] == -91.0).cpu().item()
        assert torch.all(backing[:, leading + elements:] == -91.0).cpu().item()
        old_columns = old_state[active_page].T if api.dim_first() else old_state[active_page]
        new_columns = current[active_page].T if api.dim_first() else current[active_page]
        changed_padding = int((old_columns[3 + q:3 + capacity] != new_columns[3 + q:3 + capacity]).sum())
        report["cases"].append(dict(component="conv", graph=graph_mode, q=q, physical=capacity,
                                    channels=channels, request_page_reused=reused_request,
                                    output_max_abs=float((actual.float() - expected.float()).abs().max()),
                                    private_unused_padding_elements_changed=changed_padding, status="passed"))
        accepted_path = ancestry(q - 1)[1:]
        path = torch.full((1, 64), -1, dtype=torch.int32, device="npu")
        if accepted_path:
            path[0, :len(accepted_path)] = torch.tensor(accepted_path, dtype=torch.int32, device="npu")
        sampled = torch.tensor([len(accepted_path) + 1], dtype=torch.int32, device="npu")
        next_prefix = api.compute_tree_prefix_source(path, sampled, WIDTH).cpu()
        history = torch.cat((history, x_cpu[[0] + accepted_path]), dim=0)[-3:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--state-dtype", choices=["bfloat16", "float32", "both"], default="both")
    parser.add_argument("--execution", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--components", nargs="+", choices=["recurrent", "conv"], default=["recurrent", "conv"])
    parser.add_argument("--cpu-self-check", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(status="running", npu_executed=False, tp_local_head_shape_only=True,
                  tp_size=args.tp_size, key_heads=16 // args.tp_size, value_heads=48 // args.tp_size,
                  dimension=DIM, slots=SLOTS, query_sequence=QUERIES,
                  capture_gears=GEARS, cases=[])
    exit_code = 0
    try:
        report.update(cpu_self_check())
        if not args.cpu_self_check:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            import torch_npu  # noqa: F401
            from vllm_ascend import vllm_ascend_C  # noqa: F401
            from vllm_ascend.ops.gdn import _chunked_tree_verify, is_conv_state_dim_first, tree_causal_conv1d
            from vllm_ascend.ops.gdn_attn_builder import _pack_tree_rows_to_tokens
            from vllm_ascend.worker.v2.spec_decode.tree import state_rollback as rollback

            if not torch.npu.is_available():
                raise RuntimeError("A real NPU is required; this entry does not skip hardware failures")
            torch.npu.set_device(0)
            # Access the real registered op before accepting any component case.
            getattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule_tree")
            from types import SimpleNamespace
            api = SimpleNamespace(**{name: getattr(rollback, name) for name in (
                "compute_tree_init_state_indices", "compute_tree_verify_plan", "allocate_tree_verify_plan_buffers",
                "compute_tree_resume_column", "compute_tree_conv_rows", "compute_tree_prefix_source")},
                chunked=_chunked_tree_verify, pack=_pack_tree_rows_to_tokens, tree_causal_conv1d=tree_causal_conv1d,
                dim_first=is_conv_state_dim_first)
            modes = [False, True] if args.execution == "both" else [args.execution == "graph"]
            dtypes = [torch.bfloat16, torch.float32] if args.state_dtype == "both" else [getattr(torch, args.state_dtype)]
            report["npu_executed"] = True
            report["torch_version"], report["torch_npu_version"] = torch.__version__, torch_npu.__version__
            for graph_mode in modes:
                if "recurrent" in args.components:
                    for dtype in dtypes:
                        recurrent_case(args, dtype, graph_mode, report, api)
                if "conv" in args.components:
                    conv_case(args, graph_mode, report, api)
                torch.npu.empty_cache()
            report["status"] = "passed"
        else:
            report["status"] = "cpu_reference_only"
    except Exception:
        report["status"], report["error"] = "failed", traceback.format_exc()
        exit_code = 1
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(dict(status=report["status"], npu_executed=report["npu_executed"],
                              completed_cases=len(report["cases"]), report=str(args.report)), ensure_ascii=False))
        if report.get("error"):
            print(report["error"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
