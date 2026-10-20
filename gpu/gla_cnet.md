# Gated linear attention over complex amplitudes in CNet (2026-10-17)

The degree-one counterpart of Born attention, built to compare the two inside CNet with everything else
held fixed. Layer `impl/gla.h` (`GatedLinearAttention`), GPU `gpu/gla_gpu.cpp`, tests `examples/test_gla.cpp`,
trainer flag `dp_shakespeare -gla true` (replaces the Born mixer in every attention block).

## Operator
Per head (dh complex dims), with the complex inner product `<q,k> = sum_m q_m conj(k_m)` (the convention of
`impl/bornattn.h` and Eq. 2 of the born_linear paper), a scalar per-head gate `alpha_t = sigmoid(Re g_t)` and
`s = 1/sqrt(dh)`:

    o_t = sum_{j<=t} ( prod_{j<u<=t} alpha_u ) * s * <q_t, k_j> * v_j            (complex amplitude weights)
    S_t = alpha_t S_{t-1} + conj(k_t) v_t^T,   o_t = s S_t^T q_t                 (dh x dh complex state per head)

The weights are amplitudes, not probabilities: contributions interfere by phase (with RoPE the phase rotates with
distance), and nothing sums to one, so the builder puts a `TokenNorm` right after the mixer (GLA's group norm).
Gate logits come from a `TokenwiseLinear(E -> heads)` on the pre-normed stream. The current token is never gated.
Not implemented: the real-weight form (`Re<q,k>`, what the PyTorch fla baseline computes) and per-dimension gates.

## Implementation
- CPU (`impl/gla.h`): quadratic oracle, Wirtinger backward for Q, K, V and the gate logits (real part), step mode
  (`forwardStep`, running `S` per head) for constant-state generation through `impl/stepnet.h`.
- GPU (`gpu/gla_gpu.cpp`, `GatedLinearAttentionGpu`): chunkwise. Per (clone, head) and chunk `C` (`-born_chunk`,
  `N % C == 0`, `dh <= 128`): log-gates and chunk-local cumulative sums; inter-chunk readout `s * exp(L_i) * S^T q_i`
  and the state update `S <- exp(L_C) S + Kt^H V` (decayed keys) as complex GEMMs = four real strided-batched cuBLAS
  GEMMs on the [re | im] planes; one kernel for the intra-chunk `C x C` amplitudes. Backward sweeps chunks in
  reverse: in-chunk Q/K/V gradients through `C x C` weighted amplitude planes, earlier chunks through the forward
  state snapshots, later chunks through a suffix state `(G, Gb)` built from queries and output gradients; gate
  gradients from the intra prefix/suffix sums, the inter-readout term, the later-chunk V term and one scalar per
  chunk for pairs straddling the chunk. `tw_blas_handle()` is exported from `reduce.cpp` for the GEMMs.
- Registry: type `isGLA = 27` (`impl/cnet.h` X-macro, `gpu/gpu_data.h`), saver record (N, dim, heads, scale).

## Tests (`make test_gla && ./test_gla`)
Finite differences on [Q;K;V;G] (1-3 heads, worst 4e-3 at eps 1e-3), step-vs-full recurrence 3.6e-7, no-gate limit
exact, serialization round trip, GPU-vs-CPU parity of the forward and all input gradients (abs tol 1e-3) on six
configurations (chunk 4..64, 1-4 heads, dh up to 32, 2 clones). ALL PASS on a GTX 1080 Ti (CUDA 11.3) and an
A100 (CUDA 12.8). `generate -check` on a trained 4-head ctx-4096 model: `max|p_stream - p_window| = 3e-6`.

## Measured: PG-19 characters, same model / data / seed / budget (one A100, 2026-10-18)
`-emb 256 -heads 4 -tokens 4096 -blocks 6 -batch 4 -steps 6000 -rope true -token_norm true -tw_cublas true
-fft_fast true -born_chunk 512 -ce_eps 0.001 -grad_clip 0.5 -seed 1000`, data = the first 1200 PG-19 training books:

| step | Born, linear (`-born true -born_linear true`) | Born, linear + fixed decay (`... -born_decay true`) | GLA (`-gla true`) |
|-----:|------:|------:|------:|
| 1000 | 2.494 | 2.231 | 1.964 |
| 2000 | 2.340 | 1.874 | 1.642 |
| 3000 | 2.195 | 1.712 | 1.529 |
| 4000 | 2.110 | 1.641 | 1.488 |
| 5000 | 2.017 | 1.595 | 1.465 |
| 6000 | 1.938 | 1.539 | **1.418** |
| ms/step | 815 | 835 | 942 |

Validation is the trainer's held-out split of the training stream (nats/char). The GLA kernel is a first version
(one thread per query for the intra-chunk amplitudes); the Born linear path's GEMM structure is tuned, hence the
step-time order. The decayed column is the fixed per-head decay of `gpu/born_linear_cnet.md` (`gamma_h = 1 - 2^-(5+h)`),
carried exactly in the chunkwise state; it closes about three quarters of the plain-Born gap at this budget and
costs 2% in step time. The PyTorch-mirror control of the same architecture (plain Born 1.370 / Born + decay 1.210 /
complex GLA 1.230) agrees on the ordering of the first two and puts the gated mixer level with decayed Born
there; in CNet the gated layer is still ahead at 6000 steps. All three twins pass `generate -check` (streaming
vs full-window next-token distributions agree to 5e-7 .. 3e-6). Logs: `SatComplex/results/a100_2026-10-18/pg19_twin_*.log`.
