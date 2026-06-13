"""
Benchmark a small nanoGPT training step on CPU and CUDA.

This script uses synthetic data so it does not require preparing a dataset.
"""
from contextlib import nullcontext
import argparse
import time

import torch

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark nanoGPT on CPU and CUDA.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cuda_dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_context(device, dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_batch(args, device):
    x = torch.randint(args.vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(args.vocab_size, (args.batch_size, args.block_size), device=device)
    return x, y


def run_bench(args, device, dtype):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    if args.compile:
        model = torch.compile(model)

    ctx = device_context(device, dtype)
    total_steps = args.warmup + args.steps
    measured_loss = None

    synchronize(device)
    start = None
    for step in range(total_steps):
        if step == args.warmup:
            synchronize(device)
            start = time.perf_counter()

        x, y = make_batch(args, device)
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        measured_loss = loss.item()

    synchronize(device)
    elapsed = time.perf_counter() - start
    ms_per_step = elapsed / args.steps * 1000
    tokens_per_second = args.batch_size * args.block_size * args.steps / elapsed
    return measured_loss, ms_per_step, tokens_per_second


def main():
    args = parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(
        "config: "
        f"batch_size={args.batch_size}, block_size={args.block_size}, "
        f"n_layer={args.n_layer}, n_head={args.n_head}, n_embd={args.n_embd}"
    )

    for device in devices:
        if device.type == "cuda":
            if args.cuda_dtype is None:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }[args.cuda_dtype]
        else:
            dtype = torch.float32

        loss, ms_per_step, tokens_per_second = run_bench(args, device, dtype)
        print(
            f"{device.type:>4} | dtype={str(dtype).removeprefix('torch.'):>8} | "
            f"loss={loss:.4f} | {ms_per_step:.2f} ms/step | "
            f"{tokens_per_second:.0f} tokens/sec"
        )

    if not torch.cuda.is_available():
        print("cuda | skipped: torch.cuda.is_available() is False")


if __name__ == "__main__":
    main()
