# Multi-head + exact linear Born attention in CNet (branch `born-mh-linear`, 2026-10-16)

## What
- `BornAttention(N, d, causal, eps, heads)`: `heads` heads of `dh = d/heads` consecutive complex dims,
  per-head scores and normalization. `RotaryEmbed(N, d, base, heads)`: the frequency ladder restarts per head.
- `setLinear(on, chunk)` (GPU only): the exact chunkwise density-matrix form of the same operator
  (`|<q,k>|^2 = <c(q), c(k)>` with `c(x)` the packed `dh^2` Hermitian features of `x x^H`). Forward: cuBLAS
  strided-batched GEMMs for the inter-chunk readout and the state update, one kernel for the intra-chunk
  `C x C` scores. Backward: Q gradients from the stored per-chunk forward prefix state, K/V gradients from a
  suffix state swept in reverse, lifts chained through a packed-Hermitian matvec. Constraints: causal,
  `N % chunk == 0`, `dh <= 64`. The CPU operator stays quadratic and is the oracle.
- Serialization: records with heads > 1 (or eps != 0) store `-dim` followed by `heads` (and `eps`); old files restore unchanged.
- Trainer flags: `-heads H -born_linear true -born_chunk C`.

## Fixed per-head decay (2026-10-18)
`BornAttention::setDecay(true)` / `-born_decay true`: `s[k,m] *= gamma_h^(k-m)`, `gamma_h = 1 - 2^-(5+h)`. CPU: score and
`b` weight scaled per pair; step mode: `R, M *= gamma_h` before adding the token. GPU quadratic: `born_gamma__(h)` in both
kernels. GPU linear (chunk `C`): keys lifted then row-scaled by `gamma^(C-1-j)` (`lin_cKw_`), queries by `gamma^(i+1)`
(`lin_cQd_`), state `M, Z` scaled by `gamma^C` per chunk, intra scores `gamma^(i-j)`; backward: `Bm` carries
`gamma^(k-j)`, the inter Q term `gamma^(k+1)` (abar uses `cQd`), later-chunk K/V terms use `cKw` and `Ug` rows scaled by
`gamma^(C-1-j)` (and `w_j Zs`), the suffix state `Ms, Zs` scaled by `gamma^C` with queries weighted by `gamma^(k+1)`.
Saver: heads stored negative when decay is on. All existing checks pass with decay (FD 3e-6, step-vs-full 4e-7, GPU parity,
linear-vs-quadratic on 6 configs incl. chunk 1).

## Tests
    make test_bornattn_mh && ./test_bornattn_mh        # 24 checks, no NCCL; -gpu 0 skips GPU parity
Finite differences (1-3 heads, causal/full), head-slicing oracle (exact 0), serialization round trip,
GPU-vs-CPU parity, linear-vs-quadratic forward+gradients on 8 configs (chunk 1 ... 256, 1-4 heads, eps 0 ... 1)
with O(1) inputs (abs tol 1e-3 on gradients of magnitude ~1). The dev main's `-ba_test -rope_test
-serial_born -ba_gpu -rope_gpu -tn_gpu -tw_gpu` also pass. Note: the framework's default `init_inputs` is
~1e-3 per entry, which makes Born scores vanish against eps; tests must set inputs explicitly.

## Measured
Attention layer only (`bench_bornattn -ba_dim 128 -ba_heads 4 -ba_batch 2 -ba_chunk 256`), fwd+bwd per step,
GTX 1080 Ti, eager. Linear peak memory includes ~1.07 GB of cuBLAS context that a real net pays anyway.

| N     | quadratic step | quadratic MB | linear step | linear MB |
|-------|---------------:|-------------:|------------:|----------:|
| 1024  | 121 ms         | 113          | 16 ms       | 1189      |
| 2048  | 824 ms         | 323          | 32 ms       | 1265      |
| 4096  | 5016 ms        | 1044         | 66 ms       | 1416      |
| 8192  | 25069 ms       | 3695         | 133 ms      | 1718      |
| 16384 | out of memory  | -            | 261 ms      | 2326      |

End to end (`dp_shakespeare -emb 256 -tokens 1024 -batch 8 -blocks 4 -born true -heads 4 -rope true
-token_norm true -tw_cublas true -fft_fast true -steps 300 -val_every 50`, A100, shared GPU): val nats/char

| step | quadratic | quadratic, same seed again | linear |
|-----:|----------:|---------------------------:|-------:|
| 50   | 3.333 | 3.415 | 3.337 |
| 100  | 3.103 | 2.963 | 3.033 |
| 150  | 2.941 | 2.770 | 2.945 |
| 200  | 2.810 | 2.648 | 2.756 |
| 250  | 2.685 | 2.583 | 2.608 |
| 300  | 2.638 | 2.541 | 2.567 |

The quadratic kernel's backward uses atomics, so two same-seed quadratic runs differ by up to 0.17 nats;
the linear run lies between them at every checkpoint. Solo A100 end-to-end step time (same model, batch 8, 100 steps):
1024 tokens 1346 ms quadratic vs 209 ms linear (6.4x); 4096 tokens: quadratic out of memory in the attention
scratch, linear 780 ms/step.

Attention layer only on the A100 (`bench_bornattn -ba_dim 128 -ba_heads 4 -ba_batch 4 -ba_chunk 256`, fwd+bwd):

| N     | quadratic step | quadratic MB | linear step | linear MB |
|-------|---------------:|-------------:|------------:|----------:|
| 1024  | 108 ms   | 252   | 19.6 ms | 4110 |
| 2048  | 222 ms   | 520   | 39 ms   | 4245 |
| 4096  | 692 ms   | 1359  | 78 ms   | 4513 |
| 8192  | 2147 ms  | 4245  | 155 ms  | 5050 |
| 16384 | 11325 ms | 14854 | 310 ms  | 6130 |
| 32768 | -        | -     | 620 ms  | 8294 |
| 65536 | -        | -     | 1242 ms | 12646 |

(linear MB includes the ~4 GB cuBLAS context on this CUDA version; logs: `~/cnet_mh/bench_a100.log`, `~/cnet_mh/e2e2_a100.log`, `~/cnet_mh/runs/e2e100_*.log`).

## Constant-state generation (done, CPU)
`impl/stepnet.h` builds the N=1 twin of a restored model (same layers and wiring, per-token sizes / N, weights
copied, learned positional table sliced per token, RoPE with a position offset) with `BornAttention::forwardStep`
keeping a running (R, M) density state per head. `generate -stream` (default) prefills the prompt through the
twin and does one state update per generated token; `-check K` compares the twin's next-token distribution with
the fixed-window model at the first K prompt positions.

    ./generate -model born_h4_e128_n512.mod -data t_sh.txt -prompt "..." -check 200   # STEPCHECK max|dp| 1e-5 PASS
    ./generate -model born_h4_e128_n512.mod -data t_sh.txt -prompt "..." -gen_len 300 # 2.55 s (vs 101 s -stream false)

Keep `-threads` small (default 8): a 120-thread OpenMP region per layer per token costs 250 s for 200 tokens.
Models with a learned positional table are capped at N positions; the generator re-prefills from the last N/2
tokens when it runs out (a sliding window, amortized). Fourier mixers and `-block_norm` have no constant-state
form and are rejected at build time. Not done: a GPU step kernel (the CPU path is 8 ms/token at E=128, 4 blocks).
