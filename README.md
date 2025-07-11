![logo](https://github.com/crasmarum/CNet/blob/main/docs/logo.png)
# CNet

CNet is a C++/CUDA framework for building and researching deep complex valued neural networks, as well as for optimization of complex valued functions using  gradient descent with [Wirtinger derivatives.](https://en.wikipedia.org/wiki/Wirtinger_derivatives) 
In the current version it should be relatively straighforward to implement new CPU-only functions / layers. In subsequent releases support will be added to easily implement CUDA layers, too.

Why are the complex valued neural network (CVNN) interesting? For one, it is a research area that is currently neglected due to the huge commercial success of
the real valued ones.  This [Medium article](https://machine-learning-made-simple.medium.com/complex-valued-neural-networks-might-be-the-future-of-deep-learning-c51f71f4c835) summarizes well why CVNN might be the future of Deep Learning: _"complex numbers work very well for phasic data... The fields where these can be implemented most readily are: Signal Communications; Healthcare (both medical image and ECG); Deep Fake Detections; and Acoustic Analysis for Industrial Maintenance and expansion."_  

The current mainstream ML frameworks PyTorch and JAX offer support for complex differentiation, each framework having its own limitations. However, despite the fact that mathematically the field of complex numbers $ℂ$ can be seen as a Clifford algebra and therefore one may achieve everything with only real numbers, 
the reality is a bit more complex and nuanced. It is worth quoting W. Rudin on this: _"in spite of the well-known and obvious identification of_ $ℂ$ _with_ $ℝ^2$, _these two are entirely different as far as their vector space structure is concerned."_

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

Please see the file [cnet.cpp](https://github.com/crasmarum/CNet/blob/a31b701317c5aa98da7dc59ca7bf928fcc53af8c/cnet.cpp) for how to train on CPU/GPU the above complex valued neural net with the MNIST/Fashion MNIST datasets.

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
<p align="center">
$\frac{d}{d z_i^\star}S_j=0 \text{ for all } 0\leq i,j \lt n$ 
</p>

while the $z$ derivatives are easily computed as 

<p align="center">
$\frac{d}{d z_i}S_j=g(z_i)(1−g(z_i)) \text{ for all } 0\leq i = j \lt n \text{ where } g(z) \mapsto 1 / (1 + e^{-z}).$ 
</p>

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

Please see the file [examples/sigmoid.h](https://github.com/crasmarum/CNet/blob/84ff9f20457c6fab2306f39b840b7f6aa1da813f/examples/sigmoid.h) for additional details.

# Computation Graph on CUDA

In order to execute the forward and the backward passes on a CNet complex valued neural net, the functional graph is ordered by depth, as shown in the
following diagram:
![Computation Graph](https://github.com/crasmarum/CNet/blob/main/docs/depth.png)

On an NVIDIA GPU the CNet framework will try to clone the CPU net and execute the forward and the backward passes in parallel following the graph computation depth. 
In the following image you can see how a batch of size 32 is deployed on a GPU:

![GPU clones](https://github.com/crasmarum/CNet/blob/main/docs/clones.png)

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

See  the file [cnet.cpp](https://github.com/crasmarum/CNet/blob/a31b701317c5aa98da7dc59ca7bf928fcc53af8c/cnet.cpp)  for more details.

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

 $FFT : \mathbb{C}^N \to \mathbb{C}^N \text{ given by } FFT(z)_p \mapsto \sum_q z_q e^{i2{\pi}pq / N} / \sqrt{N}$.

The main variable of this function is its input/output size. Example of its usage:
 ```c++
#include "impl/ft.h"

CNet cnet;
auto inp = cnet.add(new CInput(OutSize(28 * 28)));
auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {inp});
```

## Hadamard Layer

This layer implements the Hadamard function which is simply the element-wise multiplication:

 $Hadamard : \mathbb{C}^N \times \mathbb{C}^N \to \mathbb{C}^N \text{ given by } Hadamard(u, v)_p \mapsto u_p * v_p$.

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

$Residual : \mathbb{C}^N \times \mathbb{C}^N \to \mathbb{C}^N \text{ given by } Residual(u, v)_p \mapsto u_p + v_p$.

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

$Linear : \mathbb{C}^N \times \mathbb{C}^{N*M} \to \mathbb{C}^M \text{ given by } Linear(u, W) \mapsto u * M$.

You can see an example of using the Linear layer `lin` in the following code snippet:

```c++
#include "impl/linear.h"

CNet cnet;
// ...
auto l_data = cnet.add(new CInput(OutSize(512 * 10)));
auto lin = cnet.add(new Linear(InSize(512), InSize(512 * 10)), {gelu, l_data});
auto outp = cnet.add(new CrossEntropy(InSize(10)), {lin});
```

## CRelu Layer

This activation function is the equivalent of Relu:

$CRelu : \mathbb{C}^N \to \mathbb{C}^N \text{ given by } 
CRelu(x + iy)_k \mapsto x_k + iy_k \text{ if } x_k,y_k >0, \space 0 \text{ otherwise}$.

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

$CGelu : \mathbb{C}^N \to \mathbb{C}^N \text{ given by } CGelu(x + iy)_k \mapsto Gelu(x_k) + iGelu(y_k)$.

You can read more about $Gelu$ in the paper [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
You can see an example of using the CGelu layer `rel` in the following code snippet:

```c++
#include "impl/relu.h"

CNet cnet;
// ...
auto inp = cnet.add(new CInput(OutSize(512)));
auto rel = cnet.add(new CGelu(InSize(512)), {inp});
```

## L2Out Loss function

This loss function is simply the square of the $L2$ norm:

$L2Out : \mathbb{C}^N \to \mathbb{R} \text{ given by } L2Out(z) \mapsto \sum_k {z_k * z_k^{\star}} = \sum_k |z_k|^2$.

You can see an example of using the `L2Out` loss function `l2` in the following code snippet:

```c++
#include "impl/l2out.h"

CNet net;
auto inp = net.add(new CInput(OutSize(128)));
auto sigm = net.add(new CSigmoid(InSize(128)), {inp});
auto l2 = net.add(new L2Out(InSize(128)), {sigm});
```

## Cross Entropy Loss function

The CroossEntropy loss implements the multivariable function

$CroossEntropy : \mathbb{C}^N \times \mathbb{R}^N \to \mathbb{R} \text{ given by } CroossEntropy(z, y) \mapsto 
\sum_k -y_k \log({z_k * z_k^{\star}}/ \|z\|^2)$.

You can read more on it in the [On the Equivalence of Convolutional and Hadamard Networks using DFT](https://arxiv.org/abs/1810.11650) research paper.
You can see an example of using the `CroossEntropy` loss function `ce` in the following code snippet, as well as in the 
[cnet.cpp](https://github.com/crasmarum/CNet/blob/a31b701317c5aa98da7dc59ca7bf928fcc53af8c/cnet.cpp) file:

```c++
#include "impl/crossent.h"

CNet cnet;
// ...
auto l_data = cnet.add(new CInput(OutSize(28 * 28 * 10)));
auto lin = cnet.add(new Linear(InSize(28 * 28), InSize(28 * 28 * 10)), {gelu, l_data});
auto ce = cnet.add(new CrossEntropy(InSize(10)), {lin});
```


# Building the Software

CNet is currently supported only on Linux/MacOS and you can build it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd CNet
make
```
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
You can find the example training code in the [cnet.cpp](https://github.com/crasmarum/CNet/blob/a31b701317c5aa98da7dc59ca7bf928fcc53af8c/cnet.cpp) file:
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
# Caveat

The current version of the CNet framework has certain limitations:
* certain CUDA layer implementations are not optimized, e.g., the Linear layer;
* there is no support for multiple GPUs;
* the ADAM optimizer is a work in progress;
* APIs to create new functions / layers are for CPU-only. Support for APIs for implementing CUDA layers, to follow.
* supported only on Linux/MacOS. 

