#!/usr/bin/env python3
import time
import argparse
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: silence the NumPy missing warning if you haven't installed numpy
warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy",
    category=UserWarning,
)

def fmt_bytes(n):
    return f"{n/(1024**3):.2f} GB"

def device_info():
    print("=== PyTorch / CUDA Environment ===")
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("cuDNN:", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        maj, minr = torch.cuda.get_device_capability(idx)
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU[{idx}]: {props.name}")
        print(f"  capability: {maj}.{minr}")
        print(f"  total VRAM: {fmt_bytes(total_b)}")
        print(f"  free  VRAM: {fmt_bytes(free_b)}")
        return idx, (maj, minr), free_b, total_b
    return None, (0, 0), 0, 0

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def pick_square_for_gemm(free_bytes, dtype=torch.float32, fraction=0.50):
    """Choose the largest N so that 3*N^2*bytes_per_elem < fraction * free_bytes."""
    bpe = torch.tensor([], dtype=dtype).element_size()
    target = free_bytes * fraction
    # A, B, C each N*N
    N = int(math.sqrt(max(target / (3 * bpe), 1)))
    # round down to multiple of 128 for nicer kernel tiling
    N = max((N // 128) * 128, 128)
    return N

def gemm_bench(iters=50, dtype=torch.float32, fraction=0.50, verbose=True):
    print("\n=== GEMM stress (A@B, square) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; skipping GEMM stress.")
        return

    free_b, total_b = torch.cuda.mem_get_info()
    N = pick_square_for_gemm(free_b, dtype=dtype, fraction=fraction)
    print(f"Targeting ~{fraction*100:.0f}% of free VRAM | dtype={dtype} | N={N}")

    # Allocate once
    A = torch.randn(N, N, device=device, dtype=dtype)
    B = torch.randn(N, N, device=device, dtype=dtype)
    C = torch.empty(N, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        C = A @ B
    sync()

    # Timed
    t0 = time.time()
    for _ in range(iters):
        C = A @ B
    sync()
    dt = time.time() - t0

    # FLOPs: 2*N^3 per matmul
    flops = 2.0 * (N**3) * iters
    tflops = flops / dt / 1e12
    print(f"Ran {iters} matmuls of {N}x{N} in {dt:.2f}s → {tflops:.2f} TFLOP/s")
    print(f"Max alloc during GEMM: {fmt_bytes(torch.cuda.max_memory_allocated())}")
    del A, B, C
    sync()
    torch.cuda.empty_cache()

def conv_bench(iters=50, H=512, W=512, Cin=64, Cout=64, K=3, B=32, dtype=torch.float32):
    print("\n=== Conv2d stress (NCHW) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; skipping Conv2d stress.")
        return

    torch.backends.cudnn.benchmark = True

    x = torch.randn(B, Cin, H, W, device=device, dtype=dtype)
    conv = nn.Conv2d(Cin, Cout, kernel_size=K, stride=1, padding=K//2, bias=False).to(device, dtype)
    # Warmup
    for _ in range(10):
        y = conv(x)
    sync()

    t0 = time.time()
    for _ in range(iters):
        y = conv(x)
    sync()
    dt = time.time() - t0

    # Rough FLOPs estimate (MACs*2): N * H * W * Cout * Cin * K * K * 2
    flops = B * H * W * Cout * Cin * (K * K) * 2.0 * iters
    tflops = flops / dt / 1e12
    print(f"x: {B}x{Cin}x{H}x{W}, conv: {Cin}->{Cout}, K={K} | {iters} iters in {dt:.2f}s")
    print(f"Throughput ≈ {tflops:.2f} TFLOP/s (rough)")
    print(f"Max alloc during Conv: {fmt_bytes(torch.cuda.max_memory_allocated())}")
    del x, y, conv
    sync()
    torch.cuda.empty_cache()

class TinyCNN(nn.Module):
    def __init__(self, Cin=3, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(Cin, 64, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

def train_bench(steps=300, B=64, Cin=3, H=224, W=224, num_classes=1000,
                lr=0.01, use_amp=False, dtype=torch.float32):
    print("\n=== Tiny CNN training stress ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available; skipping training stress.")
        return

    torch.backends.cudnn.benchmark = True
    model = TinyCNN(Cin=Cin, num_classes=num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    # Random data (synthetic)
    x = torch.randn(B, Cin, H, W, device=device, dtype=dtype)
    y = torch.randint(0, num_classes, (B,), device=device)

    # Warmup
    for _ in range(10):
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        opt.zero_grad(set_to_none=True)
        scaler.step(opt)
        scaler.update()
    sync()

    t0 = time.time()
    for i in range(steps):
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (i + 1) % max(steps // 5, 1) == 0:
            sync()
            print(f"step {i+1}/{steps} | loss={loss.item():.3f}")
    sync()
    dt = time.time() - t0

    # Very rough "images/sec"
    ips = (steps * B) / dt
    print(f"Trained {steps} steps, batch {B}, {Cin}x{H}x{W}, amp={use_amp} in {dt:.2f}s → {ips:.1f} img/s")
    print(f"Max alloc during Train: {fmt_bytes(torch.cuda.max_memory_allocated())}")

    del model, x, y, logits, loss
    sync()
    torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser(description="Heavier GPU stress tests for PyTorch")
    p.add_argument("--gemm-iters", type=int, default=60, help="iterations for GEMM")
    p.add_argument("--gemm-dtype", type=str, default="fp32", choices=["fp32","fp16","bf16"])
    p.add_argument("--gemm-mem-frac", type=float, default=0.50, help="fraction of free VRAM for GEMM matrices")
    p.add_argument("--conv-iters", type=int, default=80, help="iterations for conv")
    p.add_argument("--conv-HW", type=int, default=512, help="height/width for conv input")
    p.add_argument("--conv-Cin", type=int, default=64)
    p.add_argument("--conv-Cout", type=int, default=64)
    p.add_argument("--conv-K", type=int, default=3)
    p.add_argument("--conv-B", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--train-B", type=int, default=64)
    p.add_argument("--train-HW", type=int, default=224)
    p.add_argument("--amp", action="store_true", help="enable autocast+GradScaler for training")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; nothing to stress.")
        return

    idx, cc, free_b, total_b = device_info()
    maj, minr = cc
    # GTX 1070 Ti is sm_61 (no fast tensor cores) → fp16 may not be faster
    if maj < 7 and args.amp:
        print("Note: sm < 7.0; AMP may not speed up (Pascal/Maxwell).")

    # Dtype mapping
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.gemm_dtype]

    # Quick sanity op
    print("\nWarmup & sanity…")
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    sync()
    _ = a @ b
    sync()
    del a, b

    # Run benches
    gemm_bench(iters=args.gemm_iters, dtype=dtype, fraction=args.gemm_mem_frac)

    conv_bench(
        iters=args.conv_iters,
        H=args.conv_HW, W=args.conv_HW,
        Cin=args.conv_Cin, Cout=args.conv_Cout, K=args.conv_K,
        B=args.conv_B, dtype=torch.float32  # conv tends to like fp32 on older GPUs
    )

    train_bench(
        steps=args.train_steps,
        B=args.train_B,
        H=args.train_HW, W=args.train_HW,
        use_amp=args.amp,
        dtype=torch.float32
    )

    free_b2, _ = torch.cuda.mem_get_info()
    print("\n=== Done ===")
    print(f"Free VRAM before: {fmt_bytes(free_b)} | after: {fmt_bytes(free_b2)}")
    print("If you want *even* more load, increase --gemm-mem-frac and iters, or bump conv/train sizes.")

if __name__ == "__main__":
    main()

