# CNet

CNet is a C++/CUDA framework for researching deep complex valued networks. 

# Building

CNet is currently supported only on Linux/MacOS and you can buld it using make. 
The Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc) and if nvcc is found, it compiles the project with CUDA support.
Otherwise, it falls back to using g++.

```
git clone https://github.com/crasmarum/CNet.git
cd src
make
```


