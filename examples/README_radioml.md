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
| `-arch` | `16` | `11` pooled-power baseline · `15` + cyclic-correlation `R(τ)` · `16` + normalized cyclic spectrum (best) |
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
| `-augment` | `false` | training-time random phase-rotation + time-reversal (the held-out test split is **never** augmented) |
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

## Expected numbers

Trained with the defaults above (true Adam, `gauss_init`, `l_rate 8e-4`):

| Dataset | arch 11 (baseline) | arch 16 (cyclostationary) | high-SNR plateau |
|---|---|---|---|
| RML2016.10a | ~0.47 | **~0.49** | ~0.70* |
| RML2016.10b | — | ~0.47 | ~0.69 |
| RML22 (artifact-corrected) | — | **~0.50** | ~0.65 |

On the artifact-corrected **RML22** benchmark arch 16 reaches ~0.50 overall — on
par with the reported overall accuracy of the VT-CNN2 real-valued baseline. The
`-stft` branch gives **no** gain over the cyclic spectrum (redundant); scaling
capacity (`-n_channels`, `-hidden_dim`, `-no_epochs`) does **not** raise held-out
accuracy — the ceiling is the feature construction, not model size.

\* RML2016.10a's high-SNR plateau is partly inflated by known dataset artifacts;
RML22's ~0.65 plateau is the fairer figure.
