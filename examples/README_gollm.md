# gollm — a Go move-prediction language model (CNet / Born attention)

`gollm` is an autoregressive language model over **Go move sequences**, built with
the **CNet** complex-valued C++/CUDA framework. It is the C++ counterpart of the
PyTorch `go_gen2.py` GPT, but uses CNet's native **Born-rule attention** (complex
`Q,K`; attention weights from normalized `|⟨Q,K⟩|²`) instead of softmax — CNet has
no softmax attention.

It can **train** on a corpus of games, **generate** new games (SGF output), and
**predict** the top-k next moves for a position.

Source: [`gollm.cpp`](gollm.cpp) · data prep: [`../data/prep_go.py`](../data/prep_go.py)

---

## Move encoding (vocabulary = 723)

Each move is a 3-character token `<color><col><row>`; a game starts with `222`:

```
id = color*361 + 19*(col-'a') + (row-'a')     # 0..360 black, 361..721 white
"222" (game start)  -> 722
```

So the vocabulary is **723** (19·19 black + 19·19 white + 1 start token). A board
point is `19*(col-'a') + (row-'a')` ∈ [0,361); black and white share one point
space (used by the legal-move filter). This matches `go_gen.py` / `go_gen2.py`.

---

## 1. Build

**Prerequisites:** `nvcc` (CUDA) and a host C++ compiler. NCCL is required *only*
for multi-GPU training. On a box where CUDA isn't on `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Build from `SatComplex/src`:

```bash
# multi-GPU (needs NCCL):
make gollm

# single-GPU / no NCCL (forces -world 1):
make gollm NCCL=0

# choose where the binary lands (default: the repo src dir):
TARGET_DIR=$HOME make gollm NCCL=0      # -> ~/gollm
```

The `NCCL=0` build compiles out all NCCL code (`#ifdef WITH_NCCL`) so it links on
machines without NCCL; it only supports `-world 1`.

---

## 2. Data preparation

`prep_go.py` turns a game corpus into the uint16 token bins `gollm` reads. Input
is a text file (or a `.zip` containing one) where **each line is one game**: the
`222` start token followed by 3-char moves, e.g. `2220qp1dq0cd...`.

```bash
python3 ../data/prep_go.py --in go_data.txt.zip --out ~/godata
#   --val_every_n 10   every Nth game -> val.bin (rest -> train.bin); default 10
#   --min_moves 2      skip games shorter than this
```

Outputs in `--out`:

| file | contents |
|------|----------|
| `train.bin` | training token stream (little-endian uint16) |
| `val.bin`   | validation token stream (held out, every 10th game) |
| `meta.txt`  | `vocab=723`, token counts, game count |

Example corpus: ~110k games → ~20.8M train / ~2.3M val tokens.

---

## 3. Training

`gollm` reads `<data>/train.bin` and `<data>/val.bin`, trains with the **Born-rule
sequence cross-entropy**, and checkpoints the best validation model.

### Working recipe (single GPU)

```bash
./gollm -world 1 -data ~/godata \
  -emb 256 -tokens 128 -blocks 6 -batch 16 \
  -born true -rope true -token_norm true -tw_cublas false -emb_gauss true \
  -ce_smooth 0 -ce_eps 1e-3 -grad_clip 1.0 \
  -lr 1e-3 -min_lr 1e-4 -warmup 100 -steps 4000 -val_every 200 \
  -save_path ~/go.mod
```

### Multi-GPU (data-parallel, needs the NCCL build)

One process per GPU; ranks rendezvous via a NCCL id file and all-reduce grads each
step. `batch` is the **global** batch (per-rank = `batch/world`). See
[`run_dp_shakespeare.sh`](run_dp_shakespeare.sh) for the launcher pattern:

```bash
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r ./gollm -world 4 -rank $r -data ~/godata \
    -emb 256 -tokens 128 -blocks 6 -batch 64 \
    -born true -rope true -token_norm true -tw_cublas false -emb_gauss true \
    -ce_smooth 0 -ce_eps 1e-3 -grad_clip 1.0 \
    -lr 1e-3 -min_lr 1e-4 -warmup 100 -steps 4000 -val_every 200 \
    -save_path ~/go.mod -nccl_id /tmp/gollm_id &
done; wait
```

### Metadata printed at startup (rank 0)

```
go-LM: tokens=<N> vocab=723  model E=256 N=128 L=6  global_batch=16 world=1
       born=1 rope=1 token_norm=1  ce_smooth=0  chance=ln(vocab)=6.58341
```

- `E` = embedding width, `N` = context length (moves), `L` = transformer blocks.
- `chance = ln(vocab) = 6.583` nats/move is the uniform baseline; the **unigram**
  (move-frequency) baseline is a bit lower.

### Loss

Reported in **nats/move** (natural-log perplexity per move). Training prints every
100 steps and validates every `val_every` steps:

```
step 400  train 5.25 nats/move  lr 0.000994  |g|=0  ...
  [val] step 400  val 5.25 nats/move
    [ckpt] new best val 5.25 -> saved ~/go.mod
```

- The loss is the **exact Born-rule** cross-entropy (`-ce_smooth 0`); `-ce_eps 1e-3`
  is a small target-prob floor for stability.
- **Best-val checkpointing:** `-save_path` saves the model each time validation
  improves (and the final model only if it's at least as good), so a late spike
  can't wipe out the best weights.
- A well-trained model reaches roughly **~4.7 nats/move** on the example corpus
  (from chance 6.58), with generated games showing real opening/middlegame shape.

### Critical training notes

- **`-tw_cublas false` is required** (see the saddle note below).
- **`-lr 1e-3` + `-ce_smooth 0`** (exact Born). Lower lr (e.g. 3e-4) or the
  Laplace-smoothed loss (`-ce_smooth 0.1`) also stall at the saddle.
- **`-emb_gauss true`** uses a symmetry-breaking complex embedding init.
- **Bool flags need an explicit value**: `-tw_cublas false`, `-born true`,
  `-generate true` — a bare `-generate` reads the *next* token as its value and
  silently stays false.

---

## The Born unigram saddle (why `-tw_cublas false`)

The Born-rule loss has a **unigram saddle**: a configuration that predicts the
move-frequency marginal (loss stalls just below chance and never uses context).
Escaping it is numerically **knife-edge** — the tiny (~1e-3) rounding difference
between the element-wise `TokenwiseLinear` kernel and cuBLAS's GEMM, which sum in
different orders, *deterministically* decides whether SGD rolls off the saddle:

- **cuBLAS off** (element-wise): escapes and learns (Go 6.58 → ~4.7).
- **cuBLAS on**: stays pinned at the saddle (~7.2 on WikiText, ~7.0 on Go).

This is **not a cuBLAS bug** — cuBLAS is numerically correct (verified by `-tw_gpu`
to ~1e-3) and deterministic. It is intrinsic to the loss landscape. We tried, and
**none** of these made cuBLAS-on escape:

| attempt | result |
|---|---|
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic algo) | still stuck |
| symmetric Gaussian weight init (`gauss_init`) | still stuck |
| `grad_noise` 0.01 / 0.1 (decaying Gaussian gradient noise) | still stuck |
| `dropout` 0.1 / 0.3 (complex multiplicative noise) | still stuck |

Intuition: the escape needs a *consistent directional* nudge (which the
element-wise summation order happens to provide); isotropic noise just jitters
around the saddle and averages out. So **use `-tw_cublas false`** (the default);
it is ~2–8× slower but the only path that learns.

`-dropout` and `-grad_noise` remain available as regularization / experimental
knobs (default off); they did not solve the saddle. A robust fix likely needs a
loss reformulation that de-sensitizes the saddle rather than relying on rounding.

## 4. Generating games

Generation and prediction run on the **CPU** (single-sequence autoregressive
sampling; no GPU/CUDA needed), so they work anywhere the binary + `.mod` exist.

```bash
./gollm -generate true -model ~/go.mod -gen_len 60 -temp 0.8 \
        -legal_moves true -sgf_path ~/game.sgf
```

- `-gen_len` — number of moves to play.
- `-temp` — sampling temperature; **lower (0.5–0.7) = stronger/greedier**, higher
  = more varied.
- `-legal_moves true` (default) — forbids the `222` token and any move onto an
  already-occupied point. **Occupancy only** — does not model captures or ko.
  Set `false` to see raw model output.
- `-moves 2220qp1dq` — optional seed to continue from a given opening (3-char
  tokens, start with `222`); omit to start from the empty board.
- `-sgf_path` — writes a 19×19 SGF (also printed to stdout). Open in any Go
  viewer (Sabaki, CGoban, online-go.com).

### Next-move prediction (top-k)

```bash
./gollm -predict true -model ~/go.mod -moves 2220qp1dq -topk 5 -legal_moves true
```

Prints the top-k candidate next moves with their Born probabilities, e.g.:

```
  1. B[pc]  (token 0pc, id 287)  p=0.302
  2. B[pd]  (token 0pd, id 288)  p=0.143
  ...
```

### Evaluating held-out loss

```bash
./gollm -eval_loss true -model ~/go.mod -data ~/godata -gen_len 500
```

Restores the model and reports the exact-Born validation loss (nats/move) over
`-gen_len` random held-out windows.

---

## 5. Flag reference

| flag | default | meaning |
|------|---------|---------|
| `-data` | `godata` | directory with `train.bin` / `val.bin` |
| `-vocab_size` | 723 | vocabulary (keep 723 for 19×19 Go) |
| `-emb` / `-tokens` / `-blocks` | 384 / 128 / 6 | model width E / context N / depth L |
| `-batch` | 64 | **global** batch (per-rank = batch/world) |
| `-world` / `-rank` | 1 / 0 | data-parallel process count / this rank |
| `-born` / `-rope` / `-token_norm` | true | Born attention / RoPE / per-token norm |
| `-tw_cublas` | false | **keep false** (see training notes) |
| `-emb_gauss` | false | symmetry-breaking embedding init (recommended) |
| `-ce_smooth` / `-ce_eps` | 0 / 0 | Born-loss smoothing / target-prob floor |
| `-dropout` | 0 | complex dropout prob on each sub-layer output (regularization; see saddle note) |
| `-lr` / `-min_lr` / `-warmup` | 3e-4 / 3e-5 / 200 | peak / cosine-floor lr / warmup steps |
| `-grad_clip` | 0 | clip-by-value per grad component (0 = off) |
| `-grad_norm_clip` | 0 | global grad-norm clip on the all-reduced gradient (0 = off) |
| `-grad_noise` | 0 | decaying Gaussian gradient noise σ=η/(1+t)^0.55 (experimental; see saddle note) |
| `-steps` / `-val_every` | 5000 / 200 | optimizer steps / validation period |
| `-save_path` | "" | best-val checkpoint path |
| `-generate` / `-predict` / `-eval_loss` | false | run mode (give `true`) |
| `-model` | "" | checkpoint to restore in the run modes |
| `-moves` | `222` | seed/context (3-char tokens) for generate/predict |
| `-gen_len` | 200 | moves to generate / eval windows |
| `-temp` | 0.8 | sampling temperature |
| `-topk` | 5 | candidates printed by `-predict` |
| `-legal_moves` | true | occupancy filter in generate/predict |

---

## Known limitations

- **Legality is occupancy-only** — no captures, ko, or suicide rules; generated
  games are self-consistent (no duplicate stones) but not fully rules-legal.
- **No color-alternation constraint** — color is part of the predicted token; a
  well-trained model learns to alternate, but it isn't enforced.
- Character of the model is research-grade: it reproduces opening/middlegame
  patterns from the training corpus, not strong play.
