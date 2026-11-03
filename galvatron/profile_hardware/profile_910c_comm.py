#!/usr/bin/env python3
"""Collect Galvatron hardware JSON on one Ascend node.

Run with one process per die (Atlas 800I A3: 16). Does not import Galvatron
or MindSpeed. Writes the files search_engine.py already knows how to read:

  hardware_configs/allreduce_bandwidth_<N>nodes_<G>gpus_per_node.json
  hardware_configs/p2p_bandwidth_<N>nodes_<G>gpus_per_node.json
  hardware_configs/overlap_coefficient.json
  hardware_configs/sp_time_<N>nodes_<G>gpus_per_node.json

Bandwidth values are bus GB/s, matching the NCCL "Avg bus bandwidth" unit
used by the cost model (time = message_MB / bandwidth_GBps, then scaled).
Group placement matches galvatron.core.profiler.generate_allreduce_groups
and generate_p2p_groups. On 910C with adjacent die ids, size=2 consec=1 is
the SIO pair and every other group crosses HCCS_SW.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def log(msg):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(msg, flush=True)


def allreduce_groups(world_size, group_size, consecutive):
    if group_size == world_size and not consecutive:
        return []
    num_groups = world_size // group_size
    groups = []
    if consecutive:
        for i in range(num_groups):
            groups.append(list(range(i * group_size, (i + 1) * group_size)))
    else:
        for i in range(num_groups):
            groups.append(list(range(i, world_size, num_groups)))
    return groups


def p2p_groups(world_size, pp_size):
    num_groups = world_size // pp_size
    return [list(range(i, world_size, num_groups)) for i in range(num_groups)]


def sync():
    torch.npu.synchronize()


def bus_gbps(nbytes, seconds, group_size):
    alg = (nbytes / seconds) / 1e9
    return alg * (2.0 * (group_size - 1) / group_size)


def bench_allreduce(numel, group, iters, warmup):
    tensor = torch.ones(numel, dtype=torch.float32, device="npu")
    for _ in range(warmup):
        dist.all_reduce(tensor, group=group)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor, group=group)
    sync()
    seconds = (time.perf_counter() - start) / iters
    n = dist.get_world_size(group)
    return bus_gbps(numel * 4, seconds, n), seconds


def bench_alltoall(numel, group, iters, warmup):
    n = dist.get_world_size(group)
    assert numel % n == 0
    inp = torch.ones(numel, dtype=torch.float32, device="npu")
    out = torch.empty_like(inp)
    for _ in range(warmup):
        dist.all_to_all_single(out, inp, group=group)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_to_all_single(out, inp, group=group)
    sync()
    return (time.perf_counter() - start) / iters


def bench_sendrecv(numel, peer_next, peer_prev, iters, warmup):
    send = torch.ones(numel, dtype=torch.float32, device="npu")
    recv = torch.empty_like(send)

    def once():
        ops = [
            dist.P2POp(dist.isend, send, peer_next),
            dist.P2POp(dist.irecv, recv, peer_prev),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    for _ in range(warmup):
        once()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        once()
    sync()
    seconds = (time.perf_counter() - start) / iters
    return (numel * 4 / seconds) / 1e9


def average_in_group(value, group):
    tensor = torch.tensor([value], dtype=torch.float64, device="npu")
    dist.all_reduce(tensor, group=group, op=dist.ReduceOp.SUM)
    return float(tensor.item()) / dist.get_world_size(group)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    log("wrote %s" % path)


def profile_allreduce(world_size, numel, iters, warmup):
    result = {}
    size = world_size
    while size >= 2:
        for consecutive in (1, 0):
            groups = allreduce_groups(world_size, size, consecutive)
            if not groups:
                continue
            pgs = [dist.new_group(ranks) for ranks in groups]
            dist.barrier()
            rank = dist.get_rank()
            local = None
            for ranks, pg in zip(groups, pgs):
                if rank in ranks:
                    bw, _ = bench_allreduce(numel, pg, iters, warmup)
                    local = average_in_group(bw, pg)
            dist.barrier()
            gathered = [torch.zeros(1, dtype=torch.float64, device="npu") for _ in range(world_size)]
            mine = torch.tensor([0.0 if local is None else local], dtype=torch.float64, device="npu")
            dist.all_gather(gathered, mine)
            if rank == 0:
                values = [t.item() for t in gathered if t.item() > 0]
                bw = sum(values) / len(values)
                key = "allreduce_size_%d_consec_%d" % (size, consecutive)
                result[key] = bw
                link = "SIO" if size == 2 and consecutive == 1 else "HCCS_SW_or_larger"
                log("  %-32s %8.2f GB/s   (%s)" % (key, bw, link))
        size //= 2
    return result


def profile_p2p(world_size, numel, iters, warmup):
    result = {}
    pp = 2
    rank = dist.get_rank()
    while pp <= world_size:
        groups = p2p_groups(world_size, pp)
        local = None
        for ranks in groups:
            if rank not in ranks:
                continue
            idx = ranks.index(rank)
            bw = bench_sendrecv(numel, ranks[(idx + 1) % len(ranks)], ranks[(idx - 1) % len(ranks)], iters, warmup)
            local = bw
        dist.barrier()
        gathered = [torch.zeros(1, dtype=torch.float64, device="npu") for _ in range(world_size)]
        mine = torch.tensor([0.0 if local is None else local], dtype=torch.float64, device="npu")
        dist.all_gather(gathered, mine)
        if rank == 0:
            values = [t.item() for t in gathered if t.item() > 0]
            bw = sum(values) / len(values)
            key = "pp_size_%d" % pp
            result[key] = bw
            log("  %-32s %8.2f GB/s" % (key, bw))
        pp *= 2
    return result


def profile_sp(world_size, size_mb_list, iters, warmup):
    result = {}
    rank = dist.get_rank()
    group_size = world_size
    while group_size >= 2:
        groups = allreduce_groups(world_size, group_size, 1)
        pgs = [dist.new_group(ranks) for ranks in groups]
        dist.barrier()
        for size_mb in size_mb_list:
            numel = (size_mb * 1024 * 1024) // 4
            numel = numel - (numel % group_size)
            local_ar = None
            local_a2a = None
            for ranks, pg in zip(groups, pgs):
                if rank not in ranks:
                    continue
                _, seconds = bench_allreduce(numel, pg, iters, warmup)
                local_ar = average_in_group(seconds, pg)
                seconds = bench_alltoall(numel, pg, iters, warmup)
                local_a2a = average_in_group(seconds, pg)
            dist.barrier()
            for name, local in (("allreduce", local_ar), ("all2all", local_a2a)):
                gathered = [torch.zeros(1, dtype=torch.float64, device="npu") for _ in range(world_size)]
                mine = torch.tensor([0.0 if local is None else local], dtype=torch.float64, device="npu")
                dist.all_gather(gathered, mine)
                if rank == 0:
                    values = [t.item() for t in gathered if t.item() > 0]
                    seconds = sum(values) / len(values)
                    key = "%s_size_%d_%dMB_time" % (name, group_size, size_mb)
                    result[key] = seconds
                    log("  %-40s %8.4f s" % (key, seconds))
        group_size //= 2
    return result


def profile_overlap(iters, warmup):
    device = "npu"
    comm = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device=device)
    left = torch.randn(4096, 4096, device=device)
    right = torch.randn(4096, 4096, device=device)
    comm_stream = torch.npu.Stream()
    compute_stream = torch.npu.Stream()

    def time_comm(stream, steps):
        for _ in range(warmup):
            with torch.npu.stream(stream):
                dist.all_reduce(comm)
        sync()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(stream)
        with torch.npu.stream(stream):
            for _ in range(steps):
                dist.all_reduce(comm)
        end.record(stream)
        sync()
        return start.elapsed_time(end) / steps

    def time_compute(stream, steps):
        for _ in range(warmup):
            with torch.npu.stream(stream):
                left.matmul(right)
        sync()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(stream)
        with torch.npu.stream(stream):
            for _ in range(steps):
                left.matmul(right)
        end.record(stream)
        sync()
        return start.elapsed_time(end) / steps

    comm_solo = time_comm(comm_stream, iters)
    compute_solo = time_compute(compute_stream, iters)

    for _ in range(warmup):
        with torch.npu.stream(comm_stream):
            dist.all_reduce(comm)
        with torch.npu.stream(compute_stream):
            left.matmul(right)
    sync()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record(comm_stream)
    with torch.npu.stream(comm_stream):
        for _ in range(iters):
            dist.all_reduce(comm)
    with torch.npu.stream(compute_stream):
        for _ in range(iters):
            left.matmul(right)
    end.record(comm_stream)
    sync()
    comm_overlap = start.elapsed_time(end) / iters

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record(compute_stream)
    with torch.npu.stream(comm_stream):
        for _ in range(iters):
            dist.all_reduce(comm)
    with torch.npu.stream(compute_stream):
        for _ in range(iters):
            left.matmul(right)
    end.record(compute_stream)
    sync()
    compute_overlap = start.elapsed_time(end) / iters

    coe = max(comm_overlap / comm_solo, compute_overlap / compute_solo, 1.0)
    packed = torch.tensor([coe], dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return float(packed.item()) / dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="hardware_configs")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--message-mb", type=int, default=256, help="Size used for the single bus-bandwidth number.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip-sp", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--sp-sizes-mb", default="1,2,4,8,16,32,64,128,256")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tag = "%dnodes_%dgpus_per_node" % (args.num_nodes, world_size)
    numel = (args.message_mb * 1024 * 1024) // 4

    log("world_size=%d local_rank=%d message=%dMB" % (world_size, local_rank, args.message_mb))
    if world_size != 16:
        log("warning: expected 16 dies on one 910C node, got %d" % world_size)

    log("allreduce bus bandwidth")
    allreduce = profile_allreduce(world_size, numel, args.iters, args.warmup)
    log("p2p bandwidth")
    p2p = profile_p2p(world_size, numel, args.iters, args.warmup)

    overlap = None
    if not args.skip_overlap:
        log("overlap coefficient")
        overlap = profile_overlap(args.iters, args.warmup)
        log("  overlap_coe %.4f" % overlap)

    sp = None
    if not args.skip_sp:
        log("sequence-parallel allreduce / alltoall time")
        sizes = [int(x) for x in args.sp_sizes_mb.split(",") if x]
        sp = profile_sp(world_size, sizes, args.iters, args.warmup)

    if rank == 0:
        out = os.path.abspath(args.output_dir)
        write_json(os.path.join(out, "allreduce_bandwidth_%s.json" % tag), allreduce)
        write_json(os.path.join(out, "p2p_bandwidth_%s.json" % tag), p2p)
        if overlap is not None:
            write_json(os.path.join(out, "overlap_coefficient.json"), {"overlap_coe": overlap})
        if sp is not None:
            write_json(os.path.join(out, "sp_time_%s.json" % tag), sp)
        sio = allreduce.get("allreduce_size_2_consec_1")
        cross = allreduce.get("allreduce_size_2_consec_0")
        if sio and cross:
            log("SIO pair %.1f GB/s, cross-chip pair %.1f GB/s, ratio %.2f" % (sio, cross, sio / cross))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
