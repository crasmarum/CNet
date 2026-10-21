![logo](https://github.com/crasmarum/CNet/blob/main/docs/logo.png)
# CNet

CNet is a C++/CUDA framework for building and researching deep complex valued neural networks, as well as for optimization of complex valued functions using  gradient descent with [Wirtinger derivatives.](https://en.wikipedia.org/wiki/Wirtinger_derivatives) 
Every built-in layer ships both a CPU reference and a CUDA kernel. Adding a new operator means writing its CPU forward/backward class and a CUDA kernel, then wiring the kernel into the GPU dispatch via `GpuMapping` (`gpu/gpumapping.h`) and the `GPU_LAYER_TABLE` registration macro (`gpu/gpu_data.h`) — there is no automatic CPU→GPU codegen; the kernel is written once, by hand.

Why are the complex valued neural network (CVNN) interesting? For one, it is a research area that is currently neglected due to the huge commercial success of
the real valued ones.  This [Medium article](https://machine-learning-made-simple.medium.com/complex-valued-neural-networks-might-be-the-future-of-deep-learning-c51f71f4c835) summarizes well why CVNN might be the future of Deep Learning: _"complex numbers work very well for phasic data... The fields where these can be implemented most readily are: Signal Communications; Healthcare (both medical image and ECG); Deep Fake Detections; and Acoustic Analysis for Industrial Maintenance and expansion."_  

Despite the fact that mathematically the field of complex numbers $ℂ$ can be seen as a Clifford algebra and therefore one may achieve everything with only real numbers, 
the reality is a bit more complex and nuanced. It is worth quoting W. Rudin on this: _"in spite of the well-known and obvious identification of_ $ℂ$ _with_ $ℝ^2$, _these two are entirely different as far as their vector space structure is concerned."_

## Why CNet, and not PyTorch or JAX?

PyTorch and JAX both support complex tensors and complex-aware autodiff, and for mainstream work they are the right tools — far more mature, faster at scale, multi-GPU, and backed by large ecosystems. CNet does not compete on that axis. It exists because, _for research on complex-valued networks specifically_, a complex-first, transparent substrate has concrete advantages:

* **Complex is the default, not a bolted-on dtype.** In the mainstream frameworks complex coverage is uneven: many layers, normalizations, activations, initializers and optimizers assume real tensors, so a CVNN is assembled from real building blocks plus hand-written patches whose gradient conventions must be checked case by case. In CNet every variable, layer, loss and the optimizer are complex _by construction_.

* **Wirtinger gradients are explicit, first-class objects.** Each layer manipulates the pair $(\partial L/\partial z,\ \partial L/\partial \bar z)$ directly, and the non-holomorphic case (e.g. $|z|^2$) is handled in the open rather than hidden behind a conjugation convention. Every layer ships a CPU reference and a GPU kernel checked against each other and against finite differences, so the complex gradient is _auditable_, not trusted.

* **Spectral structure is built in.** The DFT, the Hadamard (spectral) product, the inverse DFT and the $\mathrm{conv} = \mathrm{IFFT}(\mathrm{FFT} \cdot \mathrm{FFT})$ equivalence are first-class layers, so problems whose natural domain is the frequency plane map onto the framework directly.

* **A physics-native model and loss — the Born rule.** This is the sharpest departure from the mainstream frameworks. CNet reads a network's output as a complex _amplitude vector_ $z$ (a _state_) and defines the class probabilities by the quantum-measurement (Born) rule $p_k = |z_k|^2 / \|z\|^2$, with loss $\mathcal{L}(z, y) = -\sum_k y_k \log\big(|z_k|^2 / \|z\|^2\big)$. Classification is a _measurement_ on an amplitude produced by a cascade of complex — often unitary — operations, rather than real logits passed through a softmax. This inductive bias is natural for physical and wave/field data: an IQ baseband sample _is_ the complex envelope of a real passband signal, an MRI $k$-space sample _is_ a complex Fourier coefficient, and a wavefunction _is_ a complex amplitude. PyTorch and JAX provide no such head or loss out of the box.

* **A small, auditable substrate.** The full forward/backward of every layer and its CUDA kernel is explicit and readable, and the GPU execution model — a depth-ordered graph cloned across the batch — is visible and controllable.

In short: **use PyTorch or JAX to ship; use CNet to _study_ complex-valued learning itself** — to model data in its native complex / amplitude form with a Born-rule measurement head, to prototype operators whose complex gradients you need to see and control, and to work natively in the spectral domain.

# Examples

## A Simple Complex Neural Net for MNIST

The following code shows how to build a minimal neural net for the MNIST/Fashion MNIST datasets, using the Fourier Transform and Hadamard layers with a Gelu activation function:
```c++
  CNet cnet;
  auto inp = cnet.add(new CInput(OutSize(28 * 28)));
  auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});

  auto h_data = cnet.add(new CInput(OutSize(28 * 28)));
  auto hdm = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft, h_data});

  auto gelu = cnet.add(new CGelu(InSize(28 * 28)), {hdm});

  auto l_data = cnet.add(new CInput(OutSize(28 * 28 * 10)));
  auto lin = cnet.add(new Linear(InSize(28 * 28), InSize(28 * 28 * 10)), {gelu, l_data});
  auto outp = cnet.add(new CrossEntropy(InSize(10)), {lin});
```

Please see the file [cnet.cpp](https://github.com/crasmarum/CNet/blob/main/cnet.cpp) for how to train on CPU/GPU the above complex valued neural net with the MNIST/Fashion MNIST datasets.

## Adding Custom Complex Valued Functions / NN Layers

In order to add a new complex valued function, say $Sigmoid : \mathbb C^n \to \mathbb C^n$ given by $Sigmoid(z)_i \mapsto 1 / (1 + e^{-z_i}),$
all you need to do in principle is to extend the `CFunc` class and provide implementations for the `forward()` and the `backward()` methods.

```c++
class CSigmoid: public CFunc {

public:
  CSigmoid(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
  }
};
```
While writing an implementation for the `forward()` method is usually straightforward:
```c++
virtual void forward() {
    for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
        complex<float> g = 1.0f / (1.0f + exp(-input().z(in_indx)));
        mutable_output()->real_[in_indx] = g.real();
        mutable_output()->imag_[in_indx] = g.imag();
    }
}
```
providing an implementation for the `backward()` method is in general more difficult. The CNet framework makes things easier whenever you can compute
the [Wirtinger derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives) for the corresponding multivariable functions as you can
write implementations for the `dz()` and the `dz_star()` methods instead.

For example for the Sigmoid function 
$Sigmoid\big(\textbf{z} = (z_0,\dots, z_i, \dots)\big)=\big(S_0(\textbf{z}),\dots,S_j(\textbf{z}),\dots\big)$
we have that the conjugate derivatives are all null because the sigmoid function is defined only in terms of $\textbf{z}=x+iy$ and not of the conjugate $z^\star=x-iy$:

$$\frac{d}{d z_i^\star} S_j = 0 \quad \text{for all } 0 \leq i,j < n$$

while the $z$ derivatives are easily computed as 

$$\frac{d}{d z_i} S_j = g(z_i)\,(1 - g(z_i)) \quad \text{for all } 0 \leq i = j < n, \quad g(z) = \frac{1}{1 + e^{-z}}$$

From the above observations we can easily implement the `dz()` and the `dz_star()` methods:

```c++
virtual complex<float> dz(int out_indx, int in_indx) {
    if (out_indx != in_indx) {
	return 0;
    }
    auto g = 1.0f / (1.0f + exp(-input().z(in_indx)));
    return g * (1.0f - g);
}

virtual complex<float> dz_star(int out_indx, int in_indx) {
    return 0;
}
```

We can test that the implementation is correct by minimizing the `Sigmoid` using a small neural net and Wirtinger gradient descent as you can see in the 
following snippet of code:

```c++
CNet net;
auto inp = net.add(new CInput(OutSize(128)));
auto sigm = net.add(new CSigmoid(InSize(128)), {inp});
auto l2 = net.add(new L2Out(InSize(128)), {sigm});

net.init_inputs();
net.init_exec_graph(true);
for (int var = 0; var < 1000; ++var) {
    net.forward();
    cout << var << "\tLoss: " << ((L2Out*)net[l2])->loss() << endl;
    net.backward(0);
    net.updateInputs(0.1);
}
```

Running the above code from the command line:

```
~/cnet -test_sigmoid true
```

one can observe that the loss is decreasing:

```
Depth 0: Input_1 3072, 
Depth 1: CSigmoid_2 3072, 
Depth 2: L2Out_3 3096, 
0	Loss: 32.008
1	Loss: 28.512
2	Loss: 25.425
3	Loss: 22.725
4	Loss: 20.377
5	Loss: 18.341
...
798	Loss: 0.100
799	Loss: 0.099
...
```

Please see the file [examples/sigmoid.h](https://github.com/crasmarum/CNet/blob/main/examples/sigmoid.h) for additional details.

# Computation Graph on CUDA

In order to execute the forward and the backward passes on a CNet complex valued neural net, the functional graph is ordered by depth, as shown in the
following diagram:
![Computation Graph](https://github.com/crasmarum/CNet/blob/main/docs/depth.png)

On an NVIDIA GPU the CNet framework will try to clone the CPU net and execute the forward and the backward passes in parallel following the graph computation depth. 
In the following image you can see how a batch of size 32 is deployed on a GPU:

![GPU clones](https://github.com/crasmarum/CNet/blob/main/docs/clones.png)

Because the clones execute concurrently across the batch, GPU memory grows linearly with the
batch size while the per-step wall-clock time stays essentially flat. The table below measures
this cost for a 6-block complex FNet (flattened token-major stream of `T * E = 16384` complex
values) on a single NVIDIA H100 (eager):

| batch | GPU mem (MB) | fwd (ms/step) | bwd (ms/step) |
|------:|-------------:|--------------:|--------------:|
|    16 |        167.8 |          0.33 |          0.32 |
|    32 |        335.5 |          0.33 |          0.32 |
|    64 |        671.1 |          0.33 |          0.32 |
|   128 |       1342.2 |          0.33 |          0.32 |
|   256 |       2684.4 |          0.33 |          0.32 |

Memory is linear in the batch size — each clone holds its own activations — while the forward and
backward times per step remain near-constant from batch 16 up to 256, since the clones are launched
level by level and run in parallel. Throughput therefore scales with the batch at roughly fixed
latency. A reduced-memory inference layout (`Vars::INFER_DIMS`) drops the per-example footprint
further by not allocating the gradient/optimizer buffers that only training needs.

## CUDA Graphs

Because the clone arena is static across steps — device pointers for activations, weights, and
Wirtinger gradients never move after `allocateOnGpu` — the entire level-by-level launch schedule
can be captured once into a CUDA graph and replayed with a single `cudaGraphLaunch`
(`CNet::GpuForwardGraphed` / `GpuBackwardGraphed`). This removes the per-kernel launch/sync
overhead: on an H100 the forward+backward step drops from **0.65 ms to 0.32 ms (~2×, flat across
batch)** while staying **bit-identical** to eager execution, and the captured step runs faster than
eager `torch.complex64`. The standalone `bench_clone` example reproduces this:

```bash
make bench_clone
./bench_clone -fft_fast true              # eager
./bench_clone -fft_fast true -bc_graph 1  # replay captured CUDA graphs
./bench_clone -fft_fast true -bc_check 1  # verify graph replay == eager (bit-identical)
```

# Complex Layers

## Input Layer

The Input Layer is used as the main input for the neural network as well as the input parameters for other layers, e.g., Linear or Hadamard layers.
For example, in the following code snippet `inp` is the main input and the `h_data` is the parameter for the Hadamard function.

```c++
#include "impl/cinput.h"

CNet cnet;
auto inp = cnet.add(new CInput(OutSize(28 * 28)));
auto h_data = cnet.add(new CInput(OutSize(28 * 28)));
auto hdm = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {inp, h_data});

```

The main variable for this layer is the output size, e.g.,  `CInput(OutSize(1024))`. As the main input of a neural net, on CPU you can set its complex values
from a batch using this method:

```c++
setInput(InputBatch& batch, int b_indx)
```
As the input parameter for other layers, its values are either randomly set at the beginning of training a network, e.g., `net.init_inputs()`, 
or are restored from a model via `net.restore(std::string file)`.

See  the file [cnet.cpp](https://github.com/crasmarum/CNet/blob/main/cnet.cpp)  for more details.

## Embedding Layer
The Embedding Layer is used only as the main input for the neural network. The main variables of an embedding layer are
the embedding dimension, the number of embeddings and the maximum number of input tokens. For example, in the following 
code snippet we create an embedding consisting of `no_embedings = 100` tokens of dimension `emb_dim = 300` that has an input
of maximum `max_in_tokens = 64` tokens and has an output of `emb_dim * max_no_tokens = 300 * 64` complex numbers. Note that the output is 
padded with zeros if necessary.

```c++
#include "impl/embed.h"

int emb_dim = 300;
int max_in_tokens = 64;
int no_embedings = 100;
CNet cnet;
auto emb = cnet.add(new CEmbedding(emb_dim, max_in_tokens, no_embedings));
auto fft = cnet.add(new FourierTrans(InSize(emb_dim * max_in_tokens)), {emb});
```
On a CPU you can set the output embedding values using a `vector<int>` data structure, e.g.,
```c++
embedding.setInput({2, 1, 1, 3});
```
while on GPU you can do it via the 
```c ++
net.batchToGpu(InputFunc *inp, OutputFunc *outp, Batch *batch);
```
method. The initial complex values for the embeddings are either randomly set at the beginning of training a network, e.g., `net.init_inputs()`, 
or are restored from a model via `net.restore(std::string file)`.

## Fourier Transform Layer

This layer implements the [Discrete Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) function

$$\mathrm{FFT}: \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{FFT}(z)_p = \frac{1}{\sqrt{N}} \sum_q z_q\, e^{\,i 2\pi p q / N}.$$

The main variable of this function is its input/output size. Example of its usage:
 ```c++
#include "impl/ft.h"

CNet cnet;
auto inp = cnet.add(new CInput(OutSize(28 * 28)));
auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});
```

## Inverse Fourier Transform Layer

This layer implements the [Inverse Discrete Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform), the exact (unitary) inverse of `FourierTrans`:

$$\mathrm{IFFT}: \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{IFFT}(z)_p = \frac{1}{\sqrt{N}} \sum_q z_q\, e^{-i 2\pi p q / N}.$$

It is typically used to bring a Hadamard product back to the time domain, i.e. to realise a circular convolution as $Conv(u, v) = IFFT\big(Hadamard(FFT(u), FFT(v))\big)$.

 ```c++
#include "impl/ft.h"

auto fft  = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});
auto hdm  = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft, h_data});
auto ifft = cnet.add(new InverseFourierTrans(InSize(28 * 28)), {hdm});
```

## Triangular (Causal) Fourier Layer

This layer implements a *causal* variant of the DFT: a lower-triangular-masked Fourier sum $\mathrm{TriangFourier}: \mathbb{C}^N \to \mathbb{C}^N$, so output position $p$ depends only on inputs $0 \leq q \leq p$:

$$\mathrm{TriangFourier}(z)_p = \frac{1}{\sqrt{N}} \sum_{q=0}^{p} z_q\, e^{\,i 2\pi p q / N}.$$

This makes it a *parameter-free causal token mixer* — the autoregressive analogue of self-attention used in [FNet](https://aclanthology.org/2022.naacl-main.319.pdf)-style language models: it mixes information across tokens while never letting a position see its own future. With a token-major `CEmbedding` layout (token $t$ occupies a contiguous block) the triangular mask over the flattened index is exactly token-level causality. The triangular mask breaks the usual FFT factorisation, so the default kernel is a direct $O(N^2)$ sum; an $O(N \log N)$ GPU fast path is also available (`-fft_fast`), which evaluates the causal transform via a Bluestein/chirp-z factorisation with batched cuFFT (forward and Wirtinger backward).

 ```c++
#include "impl/ft.h"

auto emb = cnet.add(new CEmbedding(emb_dim, max_in_tokens, no_embedings));
auto mix = cnet.add(new TriangFourier(InSize(emb_dim * max_in_tokens)), {emb});
```

## Born-Rule Attention Layer

This is a *content-based* causal token mixer: the complex analogue of single-head self-attention, but with the attention scores given by the **quantum-measurement (Born) rule** instead of a softmaxed dot product. The input is the concatenation of three complex blocks $[Q; K; V]$, each of shape $N \times d$ (token $k$, dimension $dd$ at index $k\,d + dd$), and the output $O$ has shape $N \times d$:

$$c_{k,m} = \sum_{dd} Q_{k,dd}\, \overline{K_{m,dd}}, \qquad s_{k,m} = |c_{k,m}|^2, \qquad A_{k,m} = \frac{s_{k,m}}{\sum_{m' \leq k} s_{k,m'}}, \qquad O_{k,dd} = \sum_{m \leq k} A_{k,m}\, V_{m,dd}.$$

The score $s_{k,m} = |\langle Q_k, K_m\rangle|^2$ is exactly the Born-rule probability of measuring query $k$ against key $m$, so routing is content-based like attention but requires no softmax — the squared magnitude of a complex inner product is already non-negative and is normalised causally over $m \leq k$. The Wirtinger backward (verified to $\sim 10^{-9}$ against finite differences on CPU, and exactly against the CUDA kernel) differentiates through $c$, $s$, the normalisation and the value mix. The constructor takes the number of tokens and the (complex) head dimension; $Q$, $K$, $V$ are produced by three `TokenwiseLinear` projections of the stream:

```c++
#include "impl/bornattn.h"

int N = 64, E = 128, d = E / 2;        // N tokens, width E, head dim d
auto Q = cnet.add(new TokenwiseLinear(N, E, d), {pre, wQ});
auto K = cnet.add(new TokenwiseLinear(N, E, d), {pre, wK});
auto V = cnet.add(new TokenwiseLinear(N, E, d), {pre, wV});
auto att = cnet.add(new BornAttention(N, d), {Q, K, V});   // input [Q;K;V] -> O
```

### Multiple heads

`BornAttention(N, d, causal, eps, heads)` splits the $d$ complex dims into `heads` heads of $d_h = d/\text{heads}$ consecutive dims; head $h$ owns dims $[h d_h, (h+1) d_h)$, computes its own scores and normalisation, and writes its own output dims. `heads = 1` is the single-head operator above. `RotaryEmbed(N, d, base, heads)` restarts the frequency ladder every $d_h$ dims so every head sees the full range of rotation frequencies. Checkpoints written with `heads > 1` (or a non-zero `eps`) use an extended record; old checkpoints restore unchanged. Smaller heads matter for the linear mode below: its state and per-token cost scale with $d_h^2$, and its compute crossover against the quadratic kernel sits near $N \approx d_h^2$ ($1$k tokens for $d_h = 32$, $16$k for a single $d_h = 128$ head).

### Exact linear (chunkwise) mode

Because the score is a *squared* inner product it has an exact finite feature map: lifting each vector to its outer product $\rho_x = x x^\dagger$ gives $|\langle Q_k, K_m\rangle|^2 = \mathrm{tr}(\rho_{Q_k}\rho_{K_m})$, a linear function of the $d_h^2$ (packed, real) features of $\rho_{K_m}$. The causal sums therefore collapse into a running state of $d_h^2$ numbers (the normaliser) plus $d_h^2 \times 2 d_h$ numbers (the value-weighted state) per head, read by a quadratic form in $Q_k$, and attention costs $O(N d_h^2)$ instead of $O(N^2 d_h)$ with no $N \times N$ scratch. `setLinear(true, chunk)` switches the **GPU** kernels to this form: tokens are processed in chunks of `chunk` (`N % chunk == 0`), each chunk mixes with itself through the quadratic kernel and with everything before it through the carried state, and the state advances by one cuBLAS batched GEMM per chunk. The backward is exact too (query gradients read the stored per-chunk forward state; key and value gradients a suffix state swept in reverse). The CPU operator stays quadratic and is the oracle: `make test_bornattn_mh && ./test_bornattn_mh` checks the linear forward and all input gradients against it on eight configurations, together with finite differences, a head-slicing oracle, serialization and GPU/CPU parity.

```c++
auto att = cnet.add(new BornAttention(N, d, /*causal=*/true, /*eps=*/0.f, /*heads=*/4), {Qr, Kr, V});
((BornAttention*) cnet[att])->setLinear(true, /*chunk=*/256);   // GPU: exact O(N d_h^2) form
```

The same operator, so the same model: a run switched between the two forms reaches the same loss (within the quadratic kernel's own run-to-run spread, which comes from its atomic backward). Attention layer alone, $d = 128$ as $4 \times 32$, batch 4, forward+backward per step on an A100-40GB:

| N | quadratic | linear (chunk 256) | speed-up |
|:--:|:--:|:--:|:--:|
| 1k | 108 ms, 0.25 GB | 20 ms | 5.5x |
| 4k | 692 ms, 1.4 GB | 78 ms | 8.9x |
| 16k | 11.3 s, 14.9 GB | 310 ms, 6.1 GB | 37x |
| 64k | out of memory | 1.24 s, 12.6 GB | --- |

Constraints: causal attention only, `N % chunk == 0`, $d_h \le 64$; the linear form has no attention-weight dropout (there are no weights). The design, the gradient derivation and the measurements are in [gpu/born_linear_cnet.md](gpu/born_linear_cnet.md).

### Fixed per-head decay (forgetting)

`setDecay(true)` (trainer flag `-born_decay true`) multiplies the score by $\gamma_h^{\,k-m}$ with the RetNet schedule $\gamma_h = 1 - 2^{-(5+h)}$ (effective windows of about 32, 64, 128, 256 tokens for four heads), so distant keys fade before normalisation. It costs no parameters and no wall-clock: the quadratic kernels multiply the score, the linear kernels carry it exactly in the state (keys weighted by $\gamma^{C-1-j}$ inside a chunk, the state scaled by $\gamma^{C}$ per chunk, queries read with $\gamma^{i+1}$), and the streaming decoder scales its running state by $\gamma_h$ per token. Checkpoints with decay use the extended record (negative head count). Finite differences, GPU parity, the linear-vs-quadratic oracle and the step-vs-full check all cover the decayed operator (`make test_bornattn_mh`). In the controlled PyTorch study of the companion paper the decay was the single most effective fix for long contexts (at 128k it recovered 0.66 of the 0.70 nats between plain Born and gated linear attention).

### Streaming (constant-state) decoding

The running state also gives constant-memory autoregressive decoding: one token in, one token out, the state updated in place, never touching past tokens. CNet graphs are static, so the decoder is a *second* graph with one token per step, built from a restored model by [impl/stepnet.h](impl/stepnet.h): the same layers and wiring, every per-token size divided by $N$, weights copied, the learned positional table sliced one row per token, `RotaryEmbed` given an absolute position (`setPos`), and `BornAttention` in step mode (`setStepMode(true)`, `forwardStep`). Fed the tokens one at a time it reproduces the fixed-window model's next-token distribution to $10^{-5}$ at every position. Supported layers: embedding, inputs, `TokenwiseLinear`, `TokenNorm`, `RotaryEmbed`, `BornAttention`, `GatedLinearAttention`, `Residual`, `CGelu`, `SequenceCrossEntropy`; Fourier mixers and the `SoftMax` block norm have no constant-state form and are rejected. See the `generate` example below.

## Gated Linear Attention (Complex-Amplitude) Layer

The degree-one counterpart of Born attention, added to compare the two inside CNet with everything else held fixed. Where Born attention routes with the *probability* $|\langle Q_k, K_m\rangle|^2$, this layer routes with the complex *amplitude* $\langle Q_k, K_m\rangle$ itself, damped by a data-dependent forget gate. The input is the concatenation $[Q; K; V; G]$: $Q, K, V$ as for `BornAttention` ($N \times d$ complex, `heads` heads of $d_h = d/\text{heads}$ dims), and $G$ an $N \times \text{heads}$ block of gate logits of which the real part is used. Per head, with $\alpha_t = \sigma(\mathrm{Re}\,G_{t,h})$ and $s = 1/\sqrt{d_h}$:

$$O_k = \sum_{m \leq k} \Big(\prod_{m < u \leq k} \alpha_u\Big)\, s\, \langle Q_k, K_m\rangle\, V_m, \qquad\text{equivalently}\qquad S_t = \alpha_t S_{t-1} + \overline{K_t}\, V_t^{\top},\quad O_t = s\, S_t^{\top} Q_t,$$

a $d_h \times d_h$ complex state per head that is constant in $N$. The weights are amplitudes, not probabilities: value contributions interfere by phase (with `RotaryEmbed` on $Q$ and $K$ the phase rotates with the distance $k - m$, so the state acts as a bank of positional filters), and nothing sums to one, so a `TokenNorm` normally follows the layer. The current token is never gated. The inner product uses the same convention as `BornAttention`, $\langle q, k\rangle = \sum_{dd} q_{dd}\,\overline{k_{dd}}$. The Wirtinger backward covers $Q$, $K$, $V$ and the gate logits and is verified against finite differences; `setStepMode(true)` / `forwardStep()` give the one-token recurrence for the streaming decoder.

```c++
#include "impl/gla.h"

int N = 4096, E = 256, d = E / 2, heads = 4;
auto Q = cnet.add(new TokenwiseLinear(N, E, d), {pre, wQ});     // then RotaryEmbed(N, d, 10000.0, heads) on Q and K
auto K = cnet.add(new TokenwiseLinear(N, E, d), {pre, wK});
auto V = cnet.add(new TokenwiseLinear(N, E, d), {pre, wV});
auto G = cnet.add(new TokenwiseLinear(N, E, heads), {pre, wG});  // per-head gate logits
auto gla = cnet.add(new GatedLinearAttention(N, d, heads), {Q, K, V, G});
((GatedLinearAttention*) cnet[gla])->setChunk(512);              // GPU chunk length (N % chunk == 0)
auto att = cnet.add(new TokenNorm(N, d), {gla});                 // output norm: the mix has no normaliser
```

On the GPU the layer runs chunkwise like the linear Born mode: the inter-chunk readout and the state update are complex GEMMs (four real cuBLAS GEMMs on the re/im planes), one kernel forms the intra-chunk amplitudes, and the backward sweeps the chunks in reverse with a suffix state. Constraints: causal only, `N % chunk == 0`, $d_h \le 128$. `make test_gla && ./test_gla` runs the finite-difference, recurrence, serialisation and GPU-parity checks. On the PG-19 character model (6 blocks, $E = 256$, 4 heads, context 4096, same seed and budget on one A100) it reaches 1.418 nats/char against 1.938 for linear Born attention and 1.539 for linear Born attention with the fixed decay (`-born_decay true`); the gated kernel is a first version (942 vs 815 and 835 ms per step). A controlled repeat of the same architecture in the PyTorch mirror puts the gated mixer level with Born plus a fixed per-head decay (1.230 vs 1.210 nats/char; plain Born 1.370), so its gain is forgetting; CNet's plain Born trailing its mirror is under investigation. Born attention remains CNet's recommended mixer; the gated layer is an option. Design, derivation and measurements: [gpu/gla_cnet.md](gpu/gla_cnet.md).

## Rotary Position Embedding (RoPE) Layer

`RotaryEmbed` applies a position-dependent complex phase rotation to each token's features — rotary position embeddings (RoPE) expressed as what they natively *are* in a complex network: a rotation. The input is $N$ tokens of $D$ complex features each (token $t$, dimension $d$ at index $t\,D + d$):

$$\mathrm{RotaryEmbed}(x)_{t,d} = x_{t,d}\, e^{\,i\,\theta_d\, t}, \qquad \theta_d = \mathrm{base}^{-d/D}.$$

Applied to the queries and keys of `BornAttention`, the per-feature inner product $Q_{k,d}\,\overline{K_{j,d}}$ picks up the **relative** phase $e^{\,i\,\theta_d(k-j)}$, so the Born score $|\langle Q_k,K_j\rangle|^2$ becomes position-aware. Because a single complex dimension is one rotation plane, there is one frequency per complex feature. The map $y = r\,x$ (with $r = e^{i\theta_d t}$ a unit-modulus constant per position/feature) is holomorphic, so the Wirtinger backward is exact and trivial — with $g = \partial L/\partial y$ and $\bar g = \partial L/\partial \bar y$, one has $\partial L/\partial x = g\,r$ and $\partial L/\partial \bar x = \bar g\,\bar r$. The layer is parameter-free.

```c++
#include "impl/rotary.h"

int N = 64, dh = 64;
auto Qr  = cnet.add(new RotaryEmbed(N, dh), {Q});   // rotate queries
auto Kr  = cnet.add(new RotaryEmbed(N, dh), {K});   // rotate keys
auto att = cnet.add(new BornAttention(N, dh), {Qr, Kr, V});
```

## Hadamard Layer

This layer implements the Hadamard function which is simply the element-wise multiplication:

$$\mathrm{Hadamard}: \mathbb{C}^N \times \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{Hadamard}(u, v)_p = u_p\, v_p.$$

 In the complex valued neural network world, the Hadamard Layer is equivalent with the Convolution layer, because
 the Fourier Transform famously commutes with the convolution: $FFT\big(Conv(u, v)\big)=Hadamard\big(FFT(u), FFT(v)\big).$ 
 See also [On the Equivalence of Convolutional and Hadamard Networks using DFT](https://arxiv.org/abs/1810.11650).

 You can see an example of using the Hadamard layer `hdm` with some parameters provided by an Input Layer `h_data` in the following code snippet:
 ```c++
#include "impl/hadamard.h"

CNet cnet;
auto inp = cnet.add(new CInput(OutSize(28 * 28)));
auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});

auto h_data = cnet.add(new CInput(OutSize(28 * 28)));
auto hdm = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft, h_data});
```

## Residual Layer

This layer implements the Residual function which computes the element-wise addition:

$$\mathrm{Residual}: \mathbb{C}^N \times \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{Residual}(u, v)_p = u_p + v_p.$$

You can see an example of using the Residual layer `res` in the following code snippet:
 
 ```c++
#include "impl/residual.h"

CNet cnet;
//...
auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});
//...
auto hdm = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft, h_data});

auto res = cnet.add(new Residual(InSize(28 * 28), InSize(28 * 28)), {fft, hdm});
```

## Linear Layer
This layer is the equivalent of the fully connected / dense layer and is performing a matrix multiplication:

$$\mathrm{Linear}: \mathbb{C}^N \times \mathbb{C}^{N \cdot M} \to \mathbb{C}^M, \qquad \mathrm{Linear}(u, W) = u\,W.$$

You can see an example of using the Linear layer `lin` in the following code snippet:

```c++
#include "impl/linear.h"

CNet cnet;
// ...
auto l_data = cnet.add(new CInput(OutSize(512 * 10)));
auto lin = cnet.add(new Linear(InSize(512), InSize(512 * 10)), {gelu, l_data});
auto outp = cnet.add(new CrossEntropy(InSize(10)), {lin});
```

## Tokenwise Linear Layer

This is a *position-wise* linear layer: the input is seen as $N$ token blocks of width $E_{in}$, and the **same** $E_{in} \times E_{out}$ weight matrix is applied to every token independently (a shared per-token dense/FFN projection).

$TokenwiseLinear : \mathbb{C}^{N \cdot E_{in}} \times \mathbb{C}^{E_{in} \cdot E_{out}} \to \mathbb{C}^{N \cdot E_{out}}$.

Sharing the weight across positions costs $O(N \cdot E_{in} \cdot E_{out})$ parameters and compute, versus $O\big((N \cdot E)^2\big)$ for a full dense `Linear` over the flattened sequence — this is what makes a per-token feed-forward block affordable in a sequence model. The constructor takes the number of tokens and the input/output embedding widths:

```c++
#include "impl/tokenwise.h"

int N = 64, E = 128, H = 4 * E;        // N tokens, width E, hidden H
auto w1 = cnet.add(new CInput(OutSize(E * H)));
auto ff = cnet.add(new TokenwiseLinear(N, E, H), {hidden, w1});
```

On the GPU, `TokenwiseLinear` has two kernel paths: a naive per-element complex multiply-accumulate, and a batched cuBLAS GEMM path (the complex product is four real `cublasSgemmBatched` calls). The batched path is a large win once the projection is wide — most visibly a language-model head at vocabulary width — where it measured **≈3.4× faster** (607 → 177 ms/step) on a vocab-8192, `N=256`, `E=256` model on an H100; at small output widths (e.g. a character vocabulary) the two paths are indistinguishable. The `dp_shakespeare` example selects it with `-tw_cublas true`.

## CRelu Layer

This activation function is the equivalent of Relu:

$$\mathrm{CRelu}: \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{CRelu}(x + iy)_k = x_k + i y_k \ \text{ if } x_k, y_k > 0, \ \text{ else } 0.$$

You can see an example of using the CRelu layer `rel` in the following code snippet:

```c++
#include "impl/relu.h"

CNet cnet;
// ...
auto inp = cnet.add(new CInput(OutSize(512)));
auto rel = cnet.add(new Crelu(InSize(512)), {inp});
```

## Gelu Layer

This activation function is the equivalent of Gelu:

$$\mathrm{CGelu}: \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{CGelu}(x + iy)_k = \mathrm{Gelu}(x_k) + i\,\mathrm{Gelu}(y_k).$$

You can read more about $Gelu$ in the paper [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
You can see an example of using the CGelu layer `rel` in the following code snippet:

```c++
#include "impl/relu.h"

CNet cnet;
// ...
auto inp = cnet.add(new CInput(OutSize(512)));
auto rel = cnet.add(new CGelu(InSize(512)), {inp});
```

## SoftMax (L2 Normalization) Layer

Despite its name, this layer is the complex $L2$ normalization — it divides by the Euclidean norm:

$$\mathrm{SoftMax}: \mathbb{C}^N \to \mathbb{C}^N, \qquad \mathrm{SoftMax}(z)_k = \frac{z_k}{\|z\|}, \quad \|z\| = \sqrt{\sum_j |z_j|^2}.$$

Unlike `CGelu`, this activation is scale-covariant and **commutes with the (unitary) Fourier Transform**, so it can be used inside a spectral network without breaking the FFT/convolution equivalence — useful as the nonlinearity between spectral mixing layers.

```c++
#include "impl/softmax.h"

auto inp = cnet.add(new CInput(OutSize(512)));
auto nrm = cnet.add(new SoftMax(InSize(512)), {inp});
```

## TokenNorm (Per-Token RMS Normalization) Layer

Where `SoftMax` normalizes a whole block by its *global* Euclidean norm, `TokenNorm` normalizes **each token independently** by its own root-mean-power — the complex analogue of (gain-free) RMSNorm/LayerNorm, and the per-token counterpart of `SoftMax`. The input is $N$ tokens of $d$ complex features each (token $t$, dimension $dd$ at index $t\,d + dd$):

$$r_t = \sqrt{\frac{1}{d}\sum_{dd} |x_{t,dd}|^2 + \varepsilon}, \qquad \mathrm{TokenNorm}(x)_{t,dd} = \frac{x_{t,dd}}{r_t}.$$

A learnable affine is intentionally omitted (a following `TokenwiseLinear` supplies gain and bias), which keeps the layer **parameter-free** and its Wirtinger backward exact. With $T = \sum_{dd}\big(\tfrac{\partial L}{\partial y}\,x + \tfrac{\partial L}{\partial \bar y}\,\bar x\big)$ the input gradients are $\partial L/\partial x_{dd} = (\partial L/\partial y_{dd})/r - \bar{x}_{dd}\,T/(2 d r^3)$ and $\partial L/\partial \bar{x}_{dd} = (\partial L/\partial \bar{y}_{dd})/r - x_{dd}\,T/(2 d r^3)$ (verified: CPU finite-difference and GPU-vs-CPU to machine precision). Used as a *pre-norm* (normalize the residual-stream input to each sub-block), it lets deep complex transformers train stably — on tiny-shakespeare it breaks the plateau that global block-normalization hits. The constructor takes the number of tokens, the per-token feature width, and an optional $\varepsilon$:

```c++
#include "impl/tokennorm.h"

int N = 64, E = 128;
auto normed = cnet.add(new TokenNorm(N, E), {last});       // per-token RMS pre-norm
auto Q      = cnet.add(new TokenwiseLinear(N, E, E / 2), {normed, wQ});
```

## Complex Dropout Layer

`ComplexDropout` is inverted dropout for complex activations: at **training** time each element is kept with probability $1-p$ and rescaled by $1/(1-p)$, so the expected activation is unchanged; at **inference** it is the identity.

$$\mathrm{Dropout}(z)_k = \frac{m_k}{1-p}\, z_k, \quad m_k \sim \mathrm{Bernoulli}(1-p) \ \text{(train)}, \qquad \mathrm{Dropout}(z)_k = z_k \ \text{(inference)}.$$

Since the map is multiplication by a real per-element constant $c_k = m_k/(1-p)$, the Wirtinger backward is trivial: $\partial L/\partial z_k = c_k\,\partial L/\partial y_k$ and $\partial L/\partial \bar z_k = c_k\,\partial L/\partial \bar y_k$. The mask is **not stored** — it is a deterministic hash of `(seed, step, element)` recomputed identically in the forward and backward kernels, and the GPU mapping bumps `step` each forward for a fresh mask (the `seed` folds in the layer depth so stacked dropout layers are independent). The CPU path is the identity, so generation/evaluation are unaffected. The constructor takes the input size and the drop probability:

```c++
#include "impl/dropout.h"

int N = 64, E = 128;
auto sub  = cnet.add(new TokenwiseLinear(N, E, E), {g, w});   // a sub-layer output
auto drop = cnet.add(new ComplexDropout(InSize(N * E), 0.1f), {sub});
```

**Caveat for the Born-rule loss.** Dropout is a conventional regularizer, but the Born-rule `SequenceCrossEntropy` has a *unigram saddle* whose escape is numerically knife-edge (it is tipped by the summation order of the `TokenwiseLinear` GEMM — see the language-model example). Empirically, **any stochastic perturbation — dropout or gradient noise — suppresses that escape**: on tiny-shakespeare / sub-word data a model that learns fine with dropout off gets pinned at the unigram plateau with `-dropout 0.1`. So for Born-rule language-model training, leave dropout off; it is provided for conventional (real-logit `CrossEntropy`) networks and as a documented research knob.

## Conjugate Lag Product (Cyclic-Correlation) Layer

`ConjLagProduct` computes the instantaneous autocorrelation at a fixed lag `tau`:

$$\mathrm{ConjLagProduct}_\tau(x)_t = x_t\,\overline{x_{t-\tau}} \quad (t \ge \tau;\ 0 \text{ otherwise}).$$

It is the nonlinear, non-holomorphic primitive behind **cyclostationary** and higher-order modulation features: the time-mean of the output estimates the lag-`tau` autocorrelation `R(tau)` (bounded `|R| <= 1` for power-normalised input, so well-scaled), while an FFT of the output over time gives the **cyclic spectrum** (spectral lines at the symbol-rate / carrier cycle frequencies). It is the complex, two-input generalisation of `CModulus2` (`|z|^2 = x*conj(x)` is the `tau = 0` case). The Wirtinger backward (verified to ~1e-4 against central finite differences on CPU, and exactly against the CUDA kernel) differentiates `y = a*conj(b)` through both inputs `a = x_t`, `b = x_{t-tau}`. The constructor takes the input length and the lag:

```c++
#include "impl/conjlag.h"

int N = 128, tau = 8;
auto r   = cnet.add(new ConjLagProduct(InSize(N), tau), {x});     // x_t * conj(x_{t-tau})
auto Rt  = cnet.add(new MeanPool(InSize(N), OutSize(4)), {r});    // pooled autocorrelation R(tau)
```

This is the a-priori signal structure the pooled-power features miss. On held-out RML2016.10a (11-way modulation), adding a small bank of `ConjLagProduct` lags (`R(tau)` and `|R(tau)|^2`) to the arch-11 spectral-conv classifier raised accuracy from **0.340 to 0.363** under an identical recipe — a modest but real gain in the predicted direction (separating the same-band QAM16/QAM64 and 8PSK/QPSK confusions that linear/power features cannot). A note on scaling: read out the time-mean `R(tau)` (O(1)), not the raw `|FFT(r_tau)|^2` (energy ~N), which otherwise saturates the Born-rule logits.

The full cyclostationary front end adds a **SoftMax-normalised cyclic spectrum** on top of this: `SoftMax(FFT(r_tau))`, an O(1) read-out of the spectral-line pattern that keeps the Born logits well-scaled. On the artifact-corrected **RML22** benchmark this cyclostationary classifier reaches **~0.50 overall — on par with the VT-CNN2 real-valued baseline** (held-out RML2016.10a ~0.49), closing most of the pooled-power gap.

## Segmented (Windowed) Fourier / STFT Layer

`SegmentedFourier` is a short-time Fourier transform expressed as a block-diagonal DFT: the length-`N` input is split into `N/window` contiguous windows of length `window`, and each window is DFT'd independently (unitary, `1/sqrt(window)` scaling). Output index `b*w+p` (window `b`, frequency bin `p`):

$$\mathrm{SegmentedFourier}(x)_{b,p} = \frac{1}{\sqrt{w}} \sum_{q=0}^{w-1} x_{b,q}\, e^{\,i 2\pi p q / w}.$$

With a following `CModulus2` and `MeanPool` it yields a spectrogram (`N/window` time steps × `window` frequency bins) — a time–frequency feature that keeps local temporal structure the global `FourierTrans` averages away. It is the exact (unitary) `FourierTrans` when `window = N`; its Wirtinger backward mirrors `FourierTrans` per window and is verified to ~2e-4 against central finite differences on CPU, and exactly against the CUDA kernel (roots are computed on the fly in the kernel). The constructor takes the input length and the window length (which must divide it):

```c++
#include "impl/segfourier.h"

int N = 128, window = 32;
auto stft = cnet.add(new SegmentedFourier(InSize(N), window), {x});   // 4 windows x 32 bins
auto mag  = cnet.add(new CModulus2(InSize(N)), {stft});               // |.|^2 spectrogram
```

Empirically, on RML a segmented-Fourier branch (windows `{16, 32}`) gave **no gain over the cyclic spectrum** above — the two features are redundant — which we record as an honest negative. It is provided as a general-purpose spectral/time–frequency primitive.

## L2Out Loss function

This loss function is simply the square of the $L2$ norm:

$$\mathrm{L2Out}: \mathbb{C}^N \to \mathbb{R}, \qquad \mathrm{L2Out}(z) = \sum_k z_k\, z_k^{\star} = \sum_k |z_k|^2.$$

You can see an example of using the `L2Out` loss function `l2` in the following code snippet:

```c++
#include "impl/l2out.h"

CNet net;
auto inp = net.add(new CInput(OutSize(128)));
auto sigm = net.add(new CSigmoid(InSize(128)), {inp});
auto l2 = net.add(new L2Out(InSize(128)), {sigm});
```

## Cross Entropy Loss function

The CrossEntropy loss implements the multivariable function

$$\mathrm{CrossEntropy}: \mathbb{C}^N \times \mathbb{R}^N \to \mathbb{R}, \qquad \mathrm{CrossEntropy}(z, y) = \sum_k -y_k \log\!\big(z_k\, z_k^{\star} / \|z\|^2\big).$$

You can read more on it in the [On the Equivalence of Convolutional and Hadamard Networks using DFT](https://arxiv.org/abs/1810.11650) research paper.
You can see an example of using the `CrossEntropy` loss function `ce` in the following code snippet, as well as in the 
[cnet.cpp](https://github.com/crasmarum/CNet/blob/main/cnet.cpp) file:

```c++
#include "impl/crossent.h"

CNet cnet;
// ...
auto l_data = cnet.add(new CInput(OutSize(28 * 28 * 10)));
auto lin = cnet.add(new Linear(InSize(28 * 28), InSize(28 * 28 * 10)), {gelu, l_data});
auto ce = cnet.add(new CrossEntropy(InSize(10)), {lin});
```

## Sequence Cross Entropy Loss function

This is the autoregressive analogue of `CrossEntropy`. The input is $N$ contiguous vocabulary-sized blocks (one per sequence position); each block is treated as an independent Born-rule measurement, and the loss is the mean over positions of the negative log-probability of that position's target token:

$$\mathrm{SeqCE}: \mathbb{C}^{N \cdot V} \times \mathbb{R}^{N} \to \mathbb{R}, \qquad \mathrm{SeqCE}(z, y) = \frac{1}{N}\sum_{p=0}^{N-1} -\log\!\big(|z_{p,y_p}|^2 / \|z_p\|^2\big).$$

The per-position targets (the input tokens shifted left by one) are set with `setTargets(std::vector<int>)` before each forward pass. Combined with a token-major `CEmbedding`, a causal `TriangFourier` mixer and `TokenwiseLinear` feed-forward blocks, this trains a character-level language model where every position predicts its next token in a single pass.

```c++
#include "impl/seqcrossent.h"

int N = 64, vocab = 65;
auto logits = cnet.add(new TokenwiseLinear(N, emb_dim, vocab), {last, w_out});
auto seq    = cnet.add(new SequenceCrossEntropy(InSize(N * vocab), vocab), {logits});
// ... each step:
seq->setTargets(targets);   // per-position next-token targets
```

# Optimizers

The learnable `CInput` / `CEmbedding` parameters are trained with the Wirtinger gradients accumulated during the backward pass. Three update rules are available, each on both CPU and GPU:

* **Plain gradient descent** — `updateInputs(l_rate)` (CPU) / `gpuUpdateInputs(l_rate)` (GPU).
* **Momentum Adam** (first moment only) — `adamUpdate(...)`.
* **Full Adam** (first + second moment with bias correction, and optional gradient clip-by-value) — `trueAdamUpdate(...)`.

On GPU a batch is trained as a set of net clones, so the update is called once per step with the learning rate scaled by the batch size (the per-clone gradients are summed onto the ancestor parameters first):

```c++
// plain SGD
cnet.gpuUpdateInputs(l_rate / batch.size());
// momentum Adam
cnet.adamUpdate(l_rate / batch.size(), beta, step);
// full Adam with gradient clipping (grad_clip = 0 disables the clip)
cnet.trueAdamUpdate(l_rate / batch.size(), beta1, beta2, eps, step, grad_clip);
```

On CPU, where gradients are accumulated over several samples before an update, the Adam variants additionally take that accumulation batch size:

```c++
net.adamUpdate(l_rate, batch_size, beta, step);
net.trueAdamUpdate(l_rate, batch_size, beta1, beta2, eps, step);
```

Because the Born-rule `CrossEntropy` / `SequenceCrossEntropy` losses are ill-conditioned, full Adam (`trueAdamUpdate`) with gradient clipping is recommended over plain SGD for classification and language-model training.

The ill-conditioning is concrete: the exact gradient of $-\log(|z_t|^2/\|z\|^2)$ grows like $1/|z_t|$ when the model assigns a near-zero amplitude to the *correct* token, so a single hard example can produce an explosive update that parameter-level gradient clipping cannot catch (the bad gradient is already summed into the minibatch). `SequenceCrossEntropy` therefore offers an optional **probability floor** `setEps(eps)` (the `-ce_eps` flag in the language-model example): the target modulus is clamped from below, $|z_t|^2 \leftarrow \max(|z_t|^2,\ \varepsilon\,\|z\|^2)$, **in the gradient only** — a uniform measurement-noise floor, i.e. label-smoothing for the Born rule. The forward loss stays exact, so the reported metric is unchanged; only the optimization is regularised, and `eps = 0` is bit-identical to the exact gradient. A value around $10^{-3}$ removes the training-loss spikes that otherwise appear with deep models and higher learning rates.

# Building the Software

CNet is currently supported only on Linux/MacOS and you can build it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd CNet
make
```

With nvcc the build emits native code for Ampere (`sm_80`, A100) and Hopper (`sm_90`, H100/H200/GH200) plus `compute_90` PTX that newer GPUs JIT-compile. Override the targets with `GPU_ARCH`, for example `make GPU_ARCH="-gencode arch=compute_100,code=sm_100"` for Blackwell. Example programs (`make dp_shakespeare`, `make generate`, `make radioml`, the tests) are rebuilt when any header they include changes; if a binary misbehaves after a large edit, `rm -rf build/obj` and rebuild.
# Training on GPU

If the  NVIDIA CUDA Compiler is found, the make file will automatically compile CNet with CUDA support. For example, once you
download locally the MNIST/Fashion MNIST datasets, you can train the example neural network from the command line like follows.

```
~/cnet -mnist_images ~/train-images.idx3-ubyte -mnist_labels ~/train-labels.idx1-ubyte \
       -model_path ~/test.mod  -mnist_gpu_train true
```

You can change the default learning rate and the GPU batch size via the `l_rate` and `batch_size` command line flags:
```
~/cnet -mnist_images ~/train-images.idx3-ubyte -mnist_labels ~/train-labels.idx1-ubyte \
       -model_path ~/test.mod  -mnist_gpu_train true -l_rate 0.001 -batch_size 40
```
You can find the example training code in the [cnet.cpp](https://github.com/crasmarum/CNet/blob/main/cnet.cpp) file:
```c++
MnistDataReader reader;
assert(reader.Open(mnist_images, mnist_labels, 60000));
assert(reader.readData());

CNet cnet;
assert(cnet.hasGPU());
int cinp, coutp;
createMnistNet(cnet, &cinp, &coutp);
std::cout << cnet.cpuNet().toString() << std::endl;

CInput *inp = (CInput*)cnet[cinp];
CrossEntropy *cen = (CrossEntropy*)cnet[coutp];
cnet.init_inputs();

assert(cnet.allocateOnGpu(batch_size));
std::cout << std::setprecision(4) << "L_rate: " << l_rate
          << "\t batch_size: " << batch_size << std::endl;

float avg_loss = 0;
for (int epoch = 0; epoch < no_epochs; ++epoch) {
	reader.shuffle();
	for (int time = 0; time < reader.size() / batch_size; ++time) {
		auto batch = reader.nextBatch(batch_size);

		cnet.gpuForward(inp, cen, batch);
		avg_loss += cnet.getLoss(cen, &batch);

		if (time % 100 == 99) {
			std::cout << epoch << "\t" << time << "\t loss: " << (avg_loss / 100) << std::endl;
			avg_loss = 0;
		}

		if (time % 1000 == 999) {
			if (!cnet.getInputsFromGpu()) {
				std::cerr << "Warning: could get data from GPU." << std::endl;
			} else {
				cnet.save(model_path);
				std::cout << "Model saved at: " << model_path << std::endl;
			}
		}

		cnet.gpuBackward();
		cnet.gpuUpdateInputs(l_rate / batch.size());
	}
}
```

# Multi-GPU Training (Data Parallel)

CNet trains across multiple GPUs with **data parallelism**, using the same
clone mechanism it uses to batch within one GPU — only one level up. Each GPU
holds a full copy of the network; the global batch is split across GPUs; after
each backward the per-GPU gradients are summed with an **NCCL all-reduce**; and
every replica applies the identical optimizer step, so the replicas stay in sync
and the result is numerically **identical to single-GPU training** with the full
batch.

Three small pieces make this work (see `gpu/gpu_func.h`): `ancestorGradBuffers()`
exposes each parameter's gradient buffer for an in-place all-reduce, and
`collectAncestorParams()` / `loadAncestorParams()` broadcast the initial weights.

The standard DDP architecture — **one process per GPU** — avoids serializing
CNet's many kernel launches through a single CUDA context, and scales
near-linearly. A complete, self-contained example trains a complex FNet
character-level language model on tiny-shakespeare:

```bash
make dp_shakespeare

# 1 GPU
./dp_shakespeare -world 1 -batch 48 -steps 3000 -data tiny_shakespear2.txt

# N GPUs (one process per GPU); the helper script launches and pins them
./examples/run_dp_shakespeare.sh 4 -batch 48 -steps 3000
```

By default the example uses the parameter-free `TriangFourier` causal mixer. Flags let it assemble a fully complex transformer instead: `-born true` swaps in the `BornAttention` content-based mixer (with `TokenwiseLinear` Q/K/V/O projections), `-rope true` adds `RotaryEmbed` rotary position embeddings on the queries/keys, `-token_norm true` adds per-token `TokenNorm` pre-normalization (pre-LN), and `-ce_eps 0.001` enables the Born-loss probability floor for stable deep training:

```bash
./examples/run_dp_shakespeare.sh 4 -born true -rope true -token_norm true -ce_eps 0.001 \
    -emb 160 -tokens 64 -blocks 8 -batch 64 -steps 8000 -grad_clip 0.5
```

`-val_data <file>` takes a separate held-out text file, and then all of `-data` is training data; without it the last 10% of `-data` is held out. The character corpus is memory-mapped (one byte per character, 64-bit positions, one page-cache copy shared by all ranks on a node), so multi-gigabyte corpora such as the full PG-19 training set (11.7 GB) load in seconds with a few GB of host RAM. `-born_decay true` adds the fixed per-head decay described above (recommended for contexts beyond a few thousand tokens); `-pos_table false` drops the learned positional table (RoPE only), which is what lets the trained model stream past its context (see below). Two more flags select the multi-head and linear forms of the mixer: `-heads 4` splits the $E/2$-wide complex head into four, and `-born_linear true -born_chunk 256` runs the exact chunkwise linear kernels (GPU only; `-tokens` must be a multiple of `-born_chunk`). With them the same model trains at contexts the quadratic kernel cannot reach: at `-tokens 4096 -batch 8` the quadratic kernel runs out of memory on a 40 GB A100 while the linear form steps in 0.78 s, and at `-tokens 1024` the whole training step is 6.4x faster (0.21 s vs 1.35 s). Everything else — RoPE, TokenNorm, the Born read-out, checkpoints — is unchanged, and a checkpoint trained one way can be used the other way.

```bash
./examples/run_dp_shakespeare.sh 1 -born true -heads 4 -born_linear true -born_chunk 256 \
    -rope true -token_norm true -tw_cublas true -ce_eps 0.001 \
    -emb 256 -tokens 4096 -blocks 4 -batch 8 -steps 3000 -save_path born4k.mod
```

`-gla true` replaces the Born mixer in every attention block by the gated linear attention over complex amplitudes (`GatedLinearAttention` with a `TokenwiseLinear` gate projection and a `TokenNorm` after the mix; `-heads`, `-rope`, `-token_norm` and `-born_chunk` apply as before; GPU chunkwise kernels, so `-tokens` must be a multiple of `-born_chunk`). Checkpoints trained either way serialise, restore and stream through `generate` unchanged:

```bash
./examples/run_dp_shakespeare.sh 1 -gla true -heads 4 -born_chunk 512 \
    -rope true -token_norm true -tw_cublas true -ce_eps 0.001 \
    -emb 256 -tokens 4096 -blocks 6 -batch 4 -steps 6000 -save_path gla4k.mod
```

The trained model can be checkpointed (`-save_path model.mod`) and then sampled without retraining. Training also writes a `model.mod.vocab` sidecar holding the character⟷id table, so generation is **self-contained** and no longer needs the training corpus to reconstruct the (insertion-order) vocabulary; older checkpoints without a sidecar fall back to `-data`. Add `-gen_gpu true` to sample on the device through the fast Bluestein/Born forward instead of the CPU reference — seconds rather than minutes for a hybrid/FNet model:

```bash
./dp_shakespeare -generate true -gen_gpu true -model model.mod -prompt "ROMEO:" -gen_len 600 -temp 0.8
```

A pretrained fully-complex Born-rule model (Born attention + RoPE + TokenNorm, ~2.1M complex parameters, val 1.52 nats/char on tiny-shakespeare) is available as a [GitHub release asset](https://github.com/crasmarum/CNet/releases/download/v0.1/rope20k.mod) and on the [Hugging Face Hub](https://huggingface.co/crasmarum/born-attention-lm) — download it and generate directly:

```bash
curl -L -o rope20k.mod https://github.com/crasmarum/CNet/releases/download/v0.1/rope20k.mod
./dp_shakespeare -generate true -model rope20k.mod -prompt "ROMEO:" -gen_len 600 -temp 0.8
```

### Constant-state generation with `generate`

The standalone `generate` example (`make generate`, CPU only, no NCCL) samples from any all-Born or all-GLA checkpoint through the streaming decoder of `impl/stepnet.h`: the prompt is prefilled one token at a time and every generated character costs one state update per layer, independent of how much has been generated. Characters are flushed as they are sampled, so the text appears live. The vocabulary is rebuilt from the training text in insertion order, so `-data` must be the file the model was trained on (for the published checkpoint, the standard 1,115,394-byte tiny-shakespeare with 65 distinct characters):

```bash
make generate
./generate -model rope20k.mod -data tiny_shakespear2.txt -prompt "ROMEO:" -gen_len 400 -temp 0.8
./generate -model rope20k.mod -data tiny_shakespear2.txt -prompt "ROMEO:" -gen_len 400 -stream false   # recompute the window per token
./generate -model rope20k.mod -data tiny_shakespear2.txt -check 64 \
    -prompt "First Citizen: Before we proceed any further, hear me speak."    # streaming vs window: max |dp| ~ 5e-6
```

`-check K` feeds the first `K` prompt tokens (`K` at most the model's context) through both paths and reports the largest difference between their next-token distributions. On a 4-block, 4-head model with a 512-token context, 300 characters take 2.6 s streaming against 101 s through the full-window path. Keep `-threads` (default 8) at or below the physical cores: the per-token layers are tiny and a parallel region per layer per token over a hundred threads is pure overhead. Models with a learned positional table are limited to their training context; when it runs out the generator re-prefills from the last $N/2$ characters, a sliding window amortised over $N/2$ tokens rather than paid per token.

### Speed: precision, profiling, and what a step costs

Three trainer flags matter for speed:

| flag | effect |
|---|---|
| `-tw_cublas true` | token-wise layers through batched cuBLAS GEMMs instead of the reference kernel; always use it |
| `-tf32 true` | TF32 tensor cores for the cuBLAS GEMMs (default strict FP32). About 1.6x faster per step, but on the PG-19 byte model it trails FP32 by 0.04 validation nats at equal steps over 1,500 steps (it is ahead at equal wall-clock time); opt in deliberately |
| `-profile_steps N` | after 20 warm-up steps, time N steps per layer type (forward, backward, share) and per phase (batch, forward, loss, backward, gradient reduce, Adam), print the table and exit |

`-ksync false` skips the host synchronization after every kernel; results are identical, and on the models measured it does not change the step time (the GPU is the bottleneck, not launches).

```bash
./dp_shakespeare -data pg19_train.txt -emb 256 -heads 4 -tokens 4096 -blocks 6 -batch 8 -steps 100 \
    -rope true -pos_table false -token_norm true -tw_cublas true -fft_fast true -born_chunk 512 -gla true -profile_steps 10
# PHASE / PROFILE / SUBSECTION lines
```

The attention layers compute their within-chunk part with cuBLAS GEMMs (scores, decay and causal mask applied elementwise, then weights times values), so their cost no longer depends on per-thread register arrays. Measured on one H100 SXM, PG-19 byte model (E=256, 6 blocks, 4 heads, context 4096, batch 8), ms per training step:

| mixer | FP32 | TF32 |
|---|:--:|:--:|
| gated complex mixer (`-gla true`) | 329 | 203 |
| linear Born + decay (`-born true -born_linear true -born_decay true`) | 383 | 246 |

In FP32 the token-wise layers are about 40% of the step: their backward runs 16 real GEMMs because CNet keeps both Wirtinger gradients.

### Large-window inference: scoring millions of bytes with a constant state

Streaming is bounded only by what the model itself remembers of position. A model trained with the default learned positional table (`-pos_table true`, one row per context position) cannot be fed past its context: the twin has no row to read. With `-pos_table false` the model is RoPE-only (the embedding feeds the first block through a `TokenNorm`), RoPE takes the absolute position in the streaming twin, and the state runs for as long as the input lasts. `generate -eval` scores a text file through the twin in byte-counted windows:

```bash
./generate -model pg19.mod -eval pg19_test.txt -eval_offset 0 -eval_chars 5242880 -eval_window 5242880 -threads 8
# EVAL ... scored=5242879 skipped=0 nats/char=... bits/char=... bpb_all=...
```

`-eval_window` is the number of bytes between state resets (`0` = the training context, the old behaviour). Bytes outside the training vocabulary are not scored and are charged 8 bits each in `bpb_all`. A long file is sharded by whole windows (`-eval_offset` and `-eval_chars` as multiples of the window); the CPU twin runs at a few hundred bytes per second per process.

The same recurrence runs on the GPU, `N` tokens at a time. `dp_shakespeare -eval_stream` restores a model, allocates `-eval_batch` windows in parallel and feeds each window as consecutive `N`-token forwards with the mixer state carried between them (`GatedLinearAttentionGpu::setCarry`; `BornAttentionGpu::setCarry` on the density state and the normalizer, restored Born layers being switched to the linear form with chunk `-born_chunk`) and RoPE at the absolute byte position; per-position losses are read back, and the state is reset at window boundaries only.

```bash
./dp_shakespeare -tw_cublas true -fft_fast true -born_chunk 512 -eval_stream pg19_test.txt \
    -model pg19.mod -eval_window 5242880 -eval_batch 8
# WINDOW k ... nats/char ... bits/char ...   then   EVAL_STREAM ... bits/char=... bpb_all=...
```

Pass `-tw_cublas true` (without it the token-wise layers run through the slow reference kernel), and `-eval_batch` must be at least 2. On a GH200 a 41 MB test set in eight 5 MB windows takes about three minutes for a 6-block model with the gated mixer and six with linear Born. The GPU pass reproduces the CPU twin exactly for the gated mixer and to within 0.1% of the summed loss for linear Born (fp32 chunkwise accumulation order).

**PG-19 byte models.** [scripts/pg19_5m/](scripts/pg19_5m/) fetches PG-19 (`fetch_pg19.sh 1200`: the test and validation partitions and the first 1,200 training books from the public bucket), trains two RoPE-only twins (`train_5m.sh 8 15000`: E=256, 6 blocks, 4 heads, context 4096, batch 8, 15,000 steps on one GH200, about two hours each; linear Born with the fixed decay, then the gated mixer) and scores them (`score_5m_gpu.sh` on the GPU, `eval5m.sh` on the CPU). Scored on the official test partition in 5,242,880-byte windows, the protocol of the MBLM hierarchy (Egli et al., 2025):

| model (4.03M complex parameters, 0.49 GB of training bytes) | best val (nats/char) | test, 5M-byte windows (bits/byte) | validation, 5M windows / 4k resets |
|---|:--:|:--:|:--:|
| linear Born + fixed decay (`-born true -born_linear true -born_decay true -pos_table false`) | 1.121 | **1.656** | 1.690 / 1.696 |
| gated complex mixer (`-gla true -pos_table false`) | 1.148 | 1.700 | 1.732 / 1.736 |
| MBLM (Mamba-2 + two Transformer stages, 350M parameters, 100 GB of bytes, 5M-byte context) | — | 2.448 | — |

The carried state neither drifts over five million bytes nor adds much beyond the training context (the 5M-window and 4k-reset scores are within 0.006 of each other), and without the positional table the decayed Born twin leads the gated one at every checkpoint. Samples from both twins are in the papers' results.

The per-step work is the forward/backward plus a single coalesced NCCL
all-reduce of the gradients; the validation loss is identical across GPU counts,
confirming correctness. Measured on RTX A6000 (CUDA 12.8, NCCL 2.26), same
global batch:

| GPUs | ms / step | speed-up |
|:----:|:---------:|:--------:|
| 1    | 157       | 1.00x    |
| 2    | 81        | 1.94x    |
| 3    | 56        | 2.79x    |

# Caveat

The current version of the CNet framework has some limitations:
* multi-GPU training is data-parallel only (no model/tensor parallelism yet); per-layer kernel-launch overhead on the single-GPU path is eliminated by optional CUDA Graph capture (see above);
* adding a layer means hand-writing both its CPU reference and its CUDA kernel (wired through `GpuMapping` / `GPU_LAYER_TABLE`); there is no higher-level CPU→GPU codegen yet.
* supported only on Linux/MacOS. 

