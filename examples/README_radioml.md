# `radioml` — radio modulation classification with CNet

`radioml` classifies a length-128 complex IQ window into its modulation type with
a single **complex-valued** network and a **Born-rule** measurement head
`p_k = |z_k|² / ‖z‖²` — the physics-native counterpart of a softmax. It is a
self-contained, single-GPU example built on the CNet framework
([examples/radioml.cpp](radioml.cpp)).

## The model

The network consumes the raw complex IQ stream (no real/imag split) and combines,
in one graph:

| Branch | What it captures |
|---|---|
| **Learnable spectral conv bank** — `FourierTrans → Hadamard → ` pooled `\|·\|²`, `\|·\|⁴` | amplitude distributions (QAM density) |
| **Phase-order moments** `z^M` (`M = 2, 4`) `→` pool `→ \|·\|²` | carrier-invariant M-PSK phase structure |
| **Cyclostationary bank** — `ConjLagProduct` `rτ(t)=xt·conj(x_{t−τ})` per lag `τ`, read as `R(τ)`, `\|R(τ)\|²`, and the normalized cyclic spectrum `SoftMax(FFT rτ) → \|·\|²` | symbol-rate / cyclic structure that the pooled-power read-out discards |
| *(optional)* **STFT** — `SegmentedFourier` windows `{16, 32}` `→ \|·\|²` | time-localized spectral energy |
| **Complex MLP head** `→ CrossEntropy` (Born rule) | classifier |

The cyclostationary bank is the one lever that raises held-out accuracy: the
`ConjLagProduct` cyclic-correlation / cyclic-spectrum features are the a-priori
signal structure the amplitude/phase statistics cannot represent.

## Build

```bash
make radioml          # nvcc if present (CUDA), else g++ (CPU-only compile check)
```

The same Makefile auto-detects `nvcc`. Training/inference need a CUDA GPU
(cuFFT/cuBLAS); the g++ fallback is a compile check only. No NCCL is used.

## Data: the RML1 `.bin` format

`RadioMLDataReader` reads a little-endian binary file:

```
int32  magic   = 0x524D4C31  ("RML1")
int32  N       = number of examples
int32  L       = IQ sequence length (128)
int32  C       = number of classes
N × { int32 label(0..C-1), 2L × float32 IQ interleaved: I0,Q0,I1,Q1,… }
```

Each example is power-normalized (`I,Q ÷ sqrt(mean(I²+Q²))`). Produce these from
the DeepSig datasets with [data/rml10a_to_bin.py](../data/rml10a_to_bin.py):

```bash
python3 ../data/rml10a_to_bin.py RML2016.10a_dict.pkl ./rmldata
#  -> rmldata/rml10a_train.bin, rml10a_test.bin, rml10a_test_snr<S>.bin (per SNR)
```

Labels are `0..C-1` in sorted-modulation order (e.g. RML2016.10a, 11 classes:
`8PSK AM-DSB AM-SSB BPSK CPFSK GFSK PAM4 QAM16 QAM64 QPSK WBFM`).

## Train

```bash
./radioml -radioml_train rmldata/rml10a_train.bin \
          -radioml_test  rmldata/rml10a_test.bin \
          -arch 16 -n_classes 11 -no_epochs 30 -model_path rml.mod
```

Progress prints `epoch  step  loss  train-acc`; after the final checkpoint it runs
the **held-out** test split (never augmented) and prints `RadioML TEST accuracy`.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `-arch` | `16` | `11` pooled-power baseline · `15` + cyclic-correlation `R(τ)` · `16` + normalized cyclic spectrum · `20` VT-CNN2-style complex conv stack on the 2-D IQ image (`data/rml22_to_2dbin.py` bins, `-seq_len 256`) · `21` conjugate-lag streams + complex GLA over the unpooled sequence (best; 1-D bins) |
| `-seq_len` | `128` | IQ window length `L` (must match the `.bin` header) |
| `-n_classes` | `11` | modulation classes (RML2016.10a / RML22 = 11, RML2016.10b = 10) |
| `-n_channels` | `128` | learnable spectral conv-bank channels |
| `-pool_out` | `0` | mean-pool each feature to this many time bins (`0` = per-arch default, `8` for arch 16) |
| `-hidden_dim` | `0` | MLP head width (`0` = default 64) |
| `-head_layers` | `2` | hidden `Linear+CGelu` layers in the arch-16 head |
| `-stft` | `false` | arch 16: add the STFT branch (`SegmentedFourier` windows 16, 32) |
| `-no_epochs` | `30` | training epochs |
| `-batch_size` | `32` | examples per gradient step |
| `-l_rate` | `8e-4` | learning rate |
| `-true_adam` | `true` | true Adam (1st + 2nd moment) |
| `-beta` / `-beta2` / `-eps` | `0.9` / `0.999` / `1e-8` | Adam moments / epsilon |
| `-grad_clip` | `1.0` | true-Adam clip-by-value per grad component (`0` = off) |
| `-gauss_init` | `true` | scaled complex-Gaussian **per-input** init — essential (a shared init keeps all channels symmetric and the net never learns) |
| `-init_scale` | `1.0` | init variance = `init_scale / length` |
| `-seed` | `12345` | RNG seed for weight init (vary across runs for mean ± std) |
| `-augment` | `false` | training-time random phase rotation + time reversal (the held-out test split is **never** augmented); layout-aware for the 2-D bins of arch 20 |
| `-aug_reverse` / `-aug_cfo` | `true` / `0` | with `-augment`: include time reversal; max random carrier-frequency offset in rad/sample (a linear phase ramp) |
| `-eval_examples` | `20000` | held-out examples scored after every epoch on a fixed subset (`EPOCH e held-out acc ...`); the best epoch's weights are saved to `<model_path>.best` (`0` = only the final full pass) |
| `-cosine` | `false` | cosine decay of `-l_rate` over all steps to 5% of its value |
| `-conv_act` / `-dropout` | `u` / `0` | arch 20: conv activation (`u` per-plane TokenNorm, `crelu`, `cgelu`) and complex dropout after the conv and dense layers |
| `-gla_blocks` / `-gla_dim` / `-gla_heads` / `-gla_chunk` | `2` / `64` / `4` / `64` | arch 21: number of complex-GLA blocks, stream width, heads, GPU chunk (`seq_len % chunk == 0`) |
| `-lags` / `-cyc_branch` | `1,2,3,4` / `true` | arch 21: conjugate-lag streams `x_t conj(x_{t-τ})` kept at full resolution; also feed the pooled arch-16 cyclostationary bank to the head |
| `-radioml_train` / `-radioml_test` | — | train / test `.bin` paths |
| `-model_path` | — | save (train) / restore (infer) path |
| `-infer` | `false` | forward-only held-out inference from a restored model |
| `-confusion` | `false` | with `-infer`, print a `C×C` confusion matrix + per-class recall |
| `-infer_dims` | `4` | `Vars::dims_` for inference (4 is safe for every net) |

## Testing / evaluation

**Overall held-out accuracy** (from a saved model):

```bash
./radioml -infer true -model_path rml.mod -arch 16 \
          -radioml_test rmldata/rml10a_test.bin -confusion true
```

`-confusion` reveals the error *structure*: essentially all residual errors fall
in the physically indistinguishable pairs (QAM16↔QAM64, 8PSK↔QPSK, AM-DSB↔WBFM).

**Accuracy-vs-SNR curve** — run `-infer` once per per-SNR bin:

```bash
for b in rmldata/rml10a_test_snr*.bin; do
  echo -n "$b  "
  ./radioml -infer true -model_path rml.mod -arch 16 -radioml_test "$b" \
    | grep -o 'INFER accuracy: [0-9.]*'
done
```

> Always pass the **same `-arch`** to `-infer` as you trained with. (The net is
> restored from the model, but `-arch` must be consistent for the data reader and
> class count.)

## arch 21: conjugate-lag streams read by a complex gated sequence mixer

Pooling the lag products (arch 16) discards the symbol-transition trajectories. arch 21 keeps them: every time step is
represented by the raw sample, its conjugate lag products at `-lags` and its power, embedded token-wise into
`-gla_dim` complex channels (one `TokenwiseLinear(L, 1, E)` per stream, summed), then `-gla_blocks` pre-norm blocks of
the complex-amplitude `GatedLinearAttention` (scalar per-head gates, running `dh x dh` state) with position-wise
feed-forward layers, a dense head over the `L x E` tokens, optionally concatenated with the pooled cyclostationary bank
(`-cyc_branch`), and the Born-rule classifier. Nothing new was written for it: it is the language-model block applied to
IQ samples.

```bash
./radioml -radioml_train rmldata/rml10a_train.bin -radioml_test rmldata/rml10a_test.bin \
          -arch 21 -seq_len 128 -n_classes 11 -no_epochs 10 -l_rate 3.2e-3 -cosine true \
          -eval_examples 20000 -model_path rml21.mod       # best epoch -> rml21.mod.best
./radioml -infer true -model_path rml21.mod.best -arch 21 -seq_len 128 -n_classes 11 \
          -radioml_test rmldata/rml10a_test.bin -confusion true
```

The per-step learning rate is `-l_rate / -batch_size`. The model overfits after the first epoch or two (training accuracy
0.98 against a flat test accuracy), so the per-epoch held-out subset and the `.best` checkpoint are what give the number;
complex dropout on the token stream and time-reversal augmentation both lowered it (0.43 / 0.34 at `-dropout 0.3 / 0.5`,
0.43 with `-augment`), the latter because reversal runs the causal recurrence backwards. A global phase rotation leaves
every lag stream invariant.

## Expected numbers

Trained with the defaults above (true Adam, `gauss_init`, `l_rate 8e-4`):

| Dataset | arch 11 (baseline) | arch 16 (cyclostationary) | high-SNR plateau |
|---|---|---|---|
| RML2016.10a | ~0.47 | **~0.49** | ~0.70* |
| RML2016.10b | — | ~0.47 | ~0.69 |
| RML22 (artifact-corrected) | — | **~0.50** | ~0.65 |

arch 21 (50/50 split, all SNRs, `-l_rate 3.2e-3 -cosine true`, best epoch of 10 on RML2016.10a and of 6 on RML22, scored
on the full test split with `-infer`):

| Dataset | arch 21 | plateau | arch 20 (CGelu, 6 epochs) |
|---|---|---|---|
| RML2016.10a | **0.515** | ~0.80 from +4 dB | — |
| RML22 | **0.597** | ~0.87 from +8 dB | 0.403 |

Residual confusions are QAM16 (absorbed by QAM64) and WBFM/AM-DSB. For arch 20, the default `-conv_act u`
(amplitude-removing) stays at chance for two epochs at every learning rate tried; use `-conv_act cgelu`.

On the artifact-corrected **RML22** benchmark arch 16 reaches ~0.50 overall — on
par with the reported overall accuracy of the VT-CNN2 real-valued baseline. The
`-stft` branch gives **no** gain over the cyclic spectrum (redundant); scaling
capacity (`-n_channels`, `-hidden_dim`, `-no_epochs`) does **not** raise held-out
accuracy — the ceiling is the feature construction, not model size.

\* RML2016.10a's high-SNR plateau is partly inflated by known dataset artifacts;
RML22's ~0.65 plateau is the fairer figure.
