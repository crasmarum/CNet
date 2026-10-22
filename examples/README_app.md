# Writing a new problem on `app::Trainer` and `app::Infer`

`impl/cnet_app.h` holds everything a CNet token-sequence model needs apart from the model itself and its
data: the training loop (single GPU or data-parallel over NCCL), Adam with warm-up and cosine decay, gradient
clipping, periodic validation with best-checkpoint saving, the final checkpoint, and inference on a saved model.
A new problem is one `.cpp` file with two small classes and a `main`.

## The two things you write

```cpp
#include "../impl/cnet_app.h"
// + the layer headers your graph uses (tokenwise.h, bornattn.h, ...)

struct MyData : app::DataSource {
    int vocab() const override;      // number of token ids
    int context() const override;    // window length N
    // one window: in[i] are the inputs, tgt[i] the token that follows in[i]
    void sample(bool heldout, std::mt19937_64 &rng, std::vector<int> &in, std::vector<int> &tgt) override;
};

struct MyModel : app::ModelDef {
    // add a CEmbedding(E, N, vocab) as the input and end in SequenceCrossEntropy(InSize(N * vocab), vocab)
    app::Heads build(ComplexNet &net, int vocab, int context) override {
        ...
        return app::Heads(embeddingId, lossId);
    }
};

int main(int argc, char **argv) {
    FLAGS::Parse(argc, argv);
    MyData data(...);
    MyModel model(...);
    app::Trainer trainer(model, data);   // options come from the standard flags
    return trainer.run();
}
```

`examples/char_lm.cpp` is a complete byte-level language model in about 100 lines.

## Build and run

```bash
make app SRC=examples/char_lm.cpp            # binary ./char_lm (multi-GPU build, needs NCCL)
make app SRC=examples/char_lm.cpp NCCL=0     # single-GPU build, no NCCL

./char_lm -data book.txt -steps 5000 -batch 8 -save_path lm.mod -tw_cublas true -tw_real_loss true
# 4 GPUs: one process per GPU
for r in 0 1 2 3; do ./char_lm -data book.txt -batch 24 -world 4 -rank $r -save_path lm.mod & done; wait
```

## Standard flags

| flag | default | meaning |
|---|---|---|
| `-steps` `-warmup` | 5000, 200 | optimizer steps, linear warm-up |
| `-lr` `-min_lr` | 1e-3, 1e-4 | peak rate and cosine floor |
| `-batch` | 8 | global windows per step, split across `-world` ranks |
| `-world` `-rank` `-nccl_id` | 1, 0, /tmp/cnet_app_id | data-parallel group |
| `-val_every` `-val_batches` `-log_every` | 500, 20, 100 | validation and logging |
| `-save_path` | | best-validation checkpoint, replaced by the final model if that is at least as good |
| `-init_model` | | start from saved weights (graph and data must match) |
| `-init_std` | 0.02 | fresh weights N(0, std²) for real and imaginary parts; 0 = CNet's old start (not recommended) |
| `-grad_clip` `-grad_norm_clip` | 0.5, 0 | per-component clip in Adam; global norm clip |
| `-ce_eps` | 0.001 | Born-loss probability floor in training (scoring is exact) |
| `-seed` | 1000 | data sampler seed (rank r uses seed + r) |
| `-tf32` | false | TF32 tensor cores for FP32 GEMMs |

CNet's kernel flags apply as well, notably `-tw_cublas true` and `-tw_real_loss true` (exact, halves the
linear layers' backward GEMMs; valid because the loss is real).

## The API

`app::Trainer`: `init()` (build or restore, initialize, allocate, join the group), `step()` (one optimizer step,
returns the training loss), `validate()`, `train()` (the full loop), `run()` (init + train), `saveModel(path)`,
`loadModel(path)`, `getModel()`, `globalStep()`, `bestValidation()`, `learningRate()`.

`app::Infer` (CPU, exact Born probabilities): `loadModel(path)`, `context()`, `vocab()`,
`nextProbs(ids)`, `sample(ids, temp, rng)`, `generate(prompt_ids, n, temp, seed, onToken)`, `score(ids)`
(nats/token in consecutive context-length windows, first token of each window unscored). Your program maps
text, moves or other symbols to ids and back.

## Scope

The graph must take one `CEmbedding` input and end in one `SequenceCrossEntropy`: language models over text,
bytes, Go moves or any other token stream. Classifiers with real-valued inputs (for example `radioml`) need a
second variant of the trainer. Inference runs on the CPU; for large models or long texts use the GPU
streaming scorer in `dp_shakespeare -eval_stream`.
