# CNet

CNet is a C++/CUDA framework for building and researching deep complex valued networks, as well as for optimization of complex valued functions using  gradient descent with Wirtinger derivatives. In the current version it should be relatively straighforward to implement CPU-only functions / layers.

# Examples

## A Simple Complex Neural Net for MNIST

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

In order to add a new complex valued function, say $Sigmoid : \mathbb C^n \to \mathbb C^n$ given by $Sigmoid(z)_i \mapsto 1 / (1 + e^{-z_i}),$
all you need to do in principle is to extend the `CFunc` class and provide implementations for the `forward()` and the `backward()` methods.

```c++
class CSigmoid: public CFunc {

public:
  CSigmoid(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
  }
};
```
While writing an implementation for the `forward()` method is usaually straightforward:
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
the [Wirtinger derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives) for the corresponding multivariable functions: you can
write implementations for the `dz()` and the `dz_star()` methods instead.

For example for the Sigmoid function 
$Sigmoid\big(\textbf{z} = (z_0,\dots, z_i, \dots)\big)=\big(S_0(\textbf{z}),\dots,S_j(\textbf{z}),\dots\big)$
we have that the conjugate derivatives are all null because the sigmoid function is defined only in terms of $\textbf{z}$ and not of the conjugate $z^\star$:
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

We can easily test that the implementation is correct by minimizing the `Sigmoid` using a small neural net and Wirtinger gradient descent like in the 
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

Running the above code you can see that the loss is decreasing:

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

Please see the file examples/sigmoid.h for additional details.

# Building

CNet is currently supported only on Linux/MacOS and you can build it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd src
make
```


