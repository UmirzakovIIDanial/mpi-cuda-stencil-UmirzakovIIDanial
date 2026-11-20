import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64
import time

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2 # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
DX = 1.0 / 513.0
DY = 1.0 / 513.0
DT = 2.0e-6 

def get_gpu_uuid() -> str:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], encoding="utf-8")
    return out.strip().splitlines()[0].strip()

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}:{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

# ================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context
# ================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks, got {size}.")
    
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    row_rank, col_rank = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    return {
        "comm": comm, "cart_comm": cart_comm, "rank": rank,
        "row_rank": row_rank, "col_rank": col_rank,
        "nbr_up": nbr_up, "nbr_down": nbr_down,
        "nbr_left": nbr_left, "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid, "seed": instance_seed
    }

# ================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ================================================================
def initialize_subdomains(ctx):
    ny = nx = 256
    i = np.arange(258)
    j = np.arange(258)
    
    gy = ctx["row_rank"] * ny + i
    gx = ctx["col_rank"] * nx + j
    
    y = gy / 513.0
    x = gx / 513.0
    X, Y = np.meshgrid(x, y)

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    
    rng = np.random.default_rng(ctx["seed"])
    u = np.zeros((258, 258), dtype=np.float64)
    for _ in range(3):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        amp = rng.uniform(0.5, 1.5)
        sig = 0.05
        u += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sig**2))

    S = 0.5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
    
    if ctx["nbr_up"] == MPI.PROC_NULL: u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL: u[257, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL: u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL: u[:, 257] = 0.0
    
    return u, kappa, S

# ================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels
# ================================================================
@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    k = cuda.grid(1)
    if k >= ny:
        return
    p = k + 1
    send_up[k]    = u[1,  p]
    send_down[k]  = u[ny, p]
    send_left[k]  = u[p,  1]
    send_right[k] = u[p,  nx]

@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right, ny, nx):
    k = cuda.grid(1)
    if k >= ny:
        return
    p = k + 1
    u[0,   p] = recv_up[k]
    u[ny+1,p] = recv_down[k]
    u[p,   0] = recv_left[k]
    u[p,nx+1] = recv_right[k]

# ================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil Kernel
# ================================================================
@cuda.jit
def tiled_stencil_kernel(u_old, u_new, kappa, S, dt, inv_dx2, inv_dy2, lam, r_start, r_end, c_start, c_end):
    smem = cuda.shared.array((18, 18), dtype=float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y

    tid = ty * 16 + tx
    for e in range(tid, 18 * 18, 256):
        sy = e // 18
        sx = e % 18
        smem[sy, sx] = u_old[by * 16 + sy, bx * 16 + sx]

    cuda.syncthreads()

    i = by * 16 + ty + 1
    j = bx * 16 + tx + 1

    if r_start <= i <= r_end and c_start <= j <= c_end:
        c = smem[ty + 1, tx + 1]
        kip = 0.5 * (kappa[i + 1, j] + kappa[i, j])
        kim = 0.5 * (kappa[i, j] + kappa[i - 1, j])
        kjp = 0.5 * (kappa[i, j + 1] + kappa[i, j])
        kjm = 0.5 * (kappa[i, j] + kappa[i, j - 1])

        diff = (kip * (smem[ty + 2, tx + 1] - c) - kim * (c - smem[ty, tx + 1])) * inv_dx2 + \
               (kjp * (smem[ty + 1, tx + 2] - c) - kjm * (c - smem[ty + 1, tx])) * inv_dy2

        u_new[i, j] = c + dt * (diff - lam * c**3 + S[i, j])

# ================================================================
# TASK 5: Dual-Stream Orchestrator & Main Solver
# ================================================================
def run_solver(ctx):
    ny = nx = 256
    u0, kappa, S = initialize_subdomains(ctx)
    
    u_d = cuda.to_device(u0)
    un_d = cuda.device_array_like(u_d)
    kd = cuda.to_device(kappa)
    Sd = cuda.to_device(S)

    send_h = cuda.pinned_array((4, 256), dtype=np.float64)
    recv_h = cuda.pinned_array((4, 256), dtype=np.float64)
    send_d = cuda.device_array((4, 256), dtype=np.float64)
    recv_d = cuda.device_array((4, 256), dtype=np.float64)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    grid_pack = (1,)
    block_pack = (256,)
    grid_stencil = (16, 16)
    block_stencil = (16, 16)

    def step_overlapped(u, un):
        # Stage A: Pack & D2H
        pack_halos_kernel[grid_pack, block_pack, stream_boundary](u, send_d[0], send_d[1], send_d[2], send_d[3], ny, nx)
        for k in range(4):
            send_d[k].copy_to_host(send_h[k], stream=stream_boundary)
        stream_boundary.synchronize()

        # Stage B: MPI + Interior Compute
        reqs = []
        if ctx["nbr_down"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[1], source=ctx["nbr_down"], tag=0))
        if ctx["nbr_up"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[0], source=ctx["nbr_up"], tag=1))
        if ctx["nbr_right"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[3], source=ctx["nbr_right"], tag=2))
        if ctx["nbr_left"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[2], source=ctx["nbr_left"], tag=3))
        
        if ctx["nbr_up"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[0], dest=ctx["nbr_up"], tag=0))
        if ctx["nbr_down"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[1], dest=ctx["nbr_down"], tag=1))
        if ctx["nbr_left"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[2], dest=ctx["nbr_left"], tag=2))
        if ctx["nbr_right"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[3], dest=ctx["nbr_right"], tag=3))

        # Параллельно считаем внутреннюю часть
        tiled_stencil_kernel[grid_stencil, block_stencil, stream_interior](
            u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, 2, ny-1, 2, nx-1)

        # Stage C: Wait, H2D, Unpack, Boundary Compute
        MPI.Request.Waitall(reqs)
        for k in range(4):
            recv_d[k].copy_to_device(recv_h[k], stream=stream_boundary)
        
        unpack_halos_kernel[grid_pack, block_pack, stream_boundary](u, recv_d[0], recv_d[1], recv_d[2], recv_d[3], ny, nx)
        
        tiled_stencil_kernel[grid_stencil, block_stencil, stream_boundary](u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, 1, 1, 1, nx)
        tiled_stencil_kernel[grid_stencil, block_stencil, stream_boundary](u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, ny, ny, 1, nx)
        tiled_stencil_kernel[grid_stencil, block_stencil, stream_boundary](u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, 2, ny-1, 1, 1)
        tiled_stencil_kernel[grid_stencil, block_stencil, stream_boundary](u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, 2, ny-1, nx, nx)

        stream_interior.synchronize()
        stream_boundary.synchronize()

    def step_serialized(u, un):
        tiled_stencil_kernel[grid_stencil, block_stencil](u, un, kd, Sd, DT, 1/DX**2, 1/DY**2, LAMBDA_DAMP, 1, ny, 1, nx)
        pack_halos_kernel[grid_pack, block_pack](u, send_d[0], send_d[1], send_d[2], send_d[3], ny, nx)
        for k in range(4): send_d[k].copy_to_host(send_h[k])
        
        reqs = []
        if ctx["nbr_down"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[1], source=ctx["nbr_down"], tag=0))
        if ctx["nbr_up"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[0], source=ctx["nbr_up"], tag=1))
        if ctx["nbr_right"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[3], source=ctx["nbr_right"], tag=2))
        if ctx["nbr_left"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Irecv(recv_h[2], source=ctx["nbr_left"], tag=3))
        
        if ctx["nbr_up"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[0], dest=ctx["nbr_up"], tag=0))
        if ctx["nbr_down"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[1], dest=ctx["nbr_down"], tag=1))
        if ctx["nbr_left"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[2], dest=ctx["nbr_left"], tag=2))
        if ctx["nbr_right"] != MPI.PROC_NULL: reqs.append(ctx["cart_comm"].Isend(send_h[3], dest=ctx["nbr_right"], tag=3))
        
        MPI.Request.Waitall(reqs)
        
        for k in range(4): recv_d[k].copy_to_device(recv_h[k])
        unpack_halos_kernel[grid_pack, block_pack](u, recv_d[0], recv_d[1], recv_d[2], recv_d[3], ny, nx)

    # Бенчмарк: Serialized
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step_serialized(u_d, un_d)
        u_d, un_d = un_d, u_d
    serialized_ms = (time.perf_counter() - t0) * 1000

    # Сброс для overlapped
    u0_init, _, _ = initialize_subdomains(ctx)
    u_d = cuda.to_device(u0_init)
    un_d = cuda.device_array_like(u_d)
    
    energy_trace = []
    l2_trace = []
    
    def get_metrics(u_dev):
        u_h = u_dev.copy_to_host()
        # Считаем только внутренние ячейки
        local_E = np.sum(u_h[1:257, 1:257]) * DX * DY
        local_L2 = np.sum(u_h[1:257, 1:257]**2) * DX * DY
        
        # ИСПОЛЬЗУЕМ allreduce (с маленькой буквы)
        E = float(ctx["comm"].allreduce(local_E, op=MPI.SUM))
        L2 = float(ctx["comm"].allreduce(local_L2, op=MPI.SUM))
        return E, L2

    # Бенчмарк: Overlapped
    t0 = time.perf_counter()
    for step in range(STEPS + 1):
        if step % 100 == 0:
            E, L2 = get_metrics(u_d)
            energy_trace.append(E)
            l2_trace.append(L2)
        if step < STEPS:
            step_overlapped(u_d, un_d)
            u_d, un_d = un_d, u_d
    overlapped_ms = (time.perf_counter() - t0) * 1000

    # Проверка Parity (только 1 шаг)
    u_test = cuda.to_device(u0_init)
    un_test = cuda.device_array_like(u_test)
    step_serialized(u_test, un_test)
    u_ser_h = u_test.copy_to_host()
    
    u_ovl = cuda.to_device(u0_init)
    un_ovl = cuda.device_array_like(u_ovl)
    step_overlapped(u_ovl, un_ovl)
    u_ovl_h = u_ovl.copy_to_host()
    
    parity_max_diff = float(np.max(np.abs(u_ser_h - u_ovl_h)))

    # PTX Аудит
    ptx = tiled_stencil_kernel.inspect_asm()
    asm = "\n".join(ptx.values())
    ptx_counts = {
        "st_shared_f64": asm.count("st.shared.f64"),
        "ld_shared_f64": asm.count("ld.shared.f64"),
        "bar_sync": asm.count("bar.sync"),
    }
    ptx_audit_passed = all(v > 0 for v in ptx_counts.values())

    manifest = {
        "github_username": sys.argv[1],
        "gpu_uuid": ctx["gpu_uuid"],
        "instance_seed": ctx["seed"],
        "mpi_ranks": 4,
        "cart_dims": [2, 2],
        "serialized_ms": serialized_ms,
        "overlapped_ms": overlapped_ms,
        "parity_max_diff": parity_max_diff,
        "ptx_audit_passed": ptx_audit_passed,
        "ptx_counts": ptx_counts,
        "energy_trace": energy_trace,
        "l2_trace": l2_trace,
        "kappa_max": float(kappa.max()),
        "dt": DT,
        "dx": DX
    }

    if ctx["rank"] == 0:
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    ctx = init_distributed_context(sys.argv[1])
    run_solver(ctx)
