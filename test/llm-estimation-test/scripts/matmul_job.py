"""GPU job with a fixed amount of work (job package example).

Work = --iters (n x n) fp16 matrix multiplications. Running the same job at
different power/clock settings makes duration and energy comparable.

Example:
  python scripts/matmul_job.py --iters 20000 --tag default
"""

import argparse
from datetime import datetime, timezone
import json
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=4096, help="Matrix size")
p.add_argument("--iters", type=int, default=20000,
               help="Number of multiplications (amount of work)")
p.add_argument("--tag", default="", help="Run tag")
p.add_argument("--log", default="data/raw/job_log.jsonl")
args = p.parse_args()

assert torch.cuda.is_available(), "CUDA not available"
a = torch.randn(args.n, args.n, device="cuda", dtype=torch.float16)
b = torch.randn(args.n, args.n, device="cuda", dtype=torch.float16)
for _ in range(20):  # warm-up
  a @ b
torch.cuda.synchronize()

start_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
t0 = time.perf_counter()
for i in range(args.iters):
  c = a @ b  # queued on the GPU; Python does not wait
  if i % 500 == 0:
    torch.cuda.synchronize()
    print(f"\r{i}/{args.iters}", end="", flush=True)
torch.cuda.synchronize()  # wait for the GPU; without it duration is ~0
dur = time.perf_counter() - t0
end_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

tflops = 2 * args.n**3 * args.iters / dur / 1e12
rec = {"tag": args.tag, "n": args.n, "iters": args.iters,
       "start_utc": start_utc, "end_utc": end_utc,
       "duration_s": round(dur, 3), "tflops": round(tflops, 2)}
print("\n" + json.dumps(rec, ensure_ascii=False))
with open(args.log, "a", encoding="utf-8") as f:
  f.write(json.dumps(rec, ensure_ascii=False) + "\n")
