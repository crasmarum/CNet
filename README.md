# CNet

CNet is a C++/CUDA framework for building and researching deep complex valued networks, as well as for optimization of complex valued functions using  gradient descent with Wirtinger derivatives. In the current version it should be relatively straighforward to implement CPU-only functions / layers.

# Examples

## A Simple Hadamard Neural Net for MNIST

The following code shows how to build a minimal neural net for the MNIST/Fashion MNIST datasets, using a Fourier Transform and Hadamard layers with a Gelu activation function:
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

Please see the file cnet.cpp for how to train on CPU/GPU the above complex valued neural net with the MNIST/Fashion MNIST datasets.

## Adding Custom Complex Valued Functions / NN Layers

Let's add a new complex valued function, say $Sigmoid : \mathbb C^n \to \mathbb C^n$ given by $Sigmoid(z)_i \mapsto 1 / (1 + e^{-z_i}).$

All you need to do in principle is to extend the `CFunc` class and provide implementations for the `forward()` and the `backward()` methods.

```c++
class CSigmoid: public CFunc {

public:
  CSigmoid(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
  }
};
```
While providing an implementation for the forward method should be straightforward:
```c++
virtual void forward() {
  for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
    auto g = 1.0f / (1.0f + std::exp(input().z(in_indx)));
    output().real_[in_indx] = g.real();
    output().imag_[in_indx] = g.imag();
  }
}
```
providing an implementation for the `backward()` method is usually more difficult. The CNet framework makes things easier 

# Building

CNet is currently supported only on Linux/MacOS and you can build it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd src
make
```


