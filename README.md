# CNet

CNet is a C++/CUDA framework for building and researching deep complex valued networks, as well as for optimization of complex valued functions using  gradient descent with Wirtinger derivatives. In the current version it should be relatively straighforward to implement CPU-only functions / layers.

# 

# Building

CNet is currently supported only on Linux/MacOS and you can build it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd src
make
```


