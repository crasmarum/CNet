# Phase-generalisation benchmark

K=8 constant-modulus waveforms of n=32 samples, per-sample SNR -6 dB, random amplitude 0.5-1.5; training phases in [0, 90) deg (real_aug: full circle); test over the full circle in 24 bins x 500; 40 epochs, Adam 0.003, 3 seeds; chance 0.125.

Parameters (real degrees of freedom): complex 8,544, tied_real 8,544, real_mlp 8,639, real_aug 8,639, real_modulus 8,592.


## Training set 256

| model | in training range | outside it | worst bin | max change of p under a phase rotation |
|---|---|---|---|---|
| complex | 0.722 [0.666-0.753] | 0.728 [0.674-0.759] | 0.709 | 3.0e-07 |
| tied_real | 0.687 [0.645-0.734] | 0.684 [0.630-0.735] | 0.663 | 3.0e-07 |
| real_mlp | 0.848 [0.830-0.869] | 0.161 [0.155-0.166] | 0.000 | 1.0e+00 |
| real_aug | 0.316 [0.272-0.357] | 0.354 [0.310-0.390] | 0.292 | 9.8e-01 |
| real_modulus | 0.128 [0.123-0.131] | 0.126 [0.125-0.128] | 0.113 | 1.8e-07 |
| matched | 0.878 [0.869-0.884] | 0.879 [0.874-0.886] | 0.864 | - |

## Training set 1,024

| model | in training range | outside it | worst bin | max change of p under a phase rotation |
|---|---|---|---|---|
| complex | 0.850 [0.838-0.860] | 0.852 [0.848-0.861] | 0.837 | 3.3e-07 |
| tied_real | 0.849 [0.837-0.860] | 0.852 [0.848-0.859] | 0.828 | 3.0e-07 |
| real_mlp | 0.881 [0.868-0.896] | 0.170 [0.165-0.173] | 0.000 | 1.0e+00 |
| real_aug | 0.689 [0.674-0.703] | 0.714 [0.712-0.716] | 0.665 | 1.0e+00 |
| real_modulus | 0.125 [0.115-0.130] | 0.125 [0.124-0.126] | 0.107 | 2.1e-07 |
| matched | 0.878 [0.869-0.884] | 0.879 [0.874-0.886] | 0.864 | - |

## Training set 4,096

| model | in training range | outside it | worst bin | max change of p under a phase rotation |
|---|---|---|---|---|
| complex | 0.866 [0.858-0.872] | 0.866 [0.859-0.873] | 0.854 | 3.6e-07 |
| tied_real | 0.865 [0.858-0.870] | 0.866 [0.860-0.874] | 0.856 | 3.9e-07 |
| real_mlp | 0.892 [0.889-0.897] | 0.191 [0.191-0.192] | 0.000 | 1.0e+00 |
| real_aug | 0.804 [0.797-0.809] | 0.812 [0.807-0.821] | 0.789 | 1.0e+00 |
| real_modulus | 0.125 [0.121-0.128] | 0.123 [0.119-0.126] | 0.100 | 3.0e-07 |
| matched | 0.878 [0.869-0.884] | 0.879 [0.874-0.886] | 0.864 | - |

## Training set 16,384

| model | in training range | outside it | worst bin | max change of p under a phase rotation |
|---|---|---|---|---|
| complex | 0.855 [0.844-0.863] | 0.857 [0.853-0.861] | 0.839 | 3.4e-06 |
| tied_real | 0.859 [0.858-0.861] | 0.860 [0.858-0.863] | 0.843 | 1.5e-06 |
| real_mlp | 0.897 [0.891-0.905] | 0.266 [0.262-0.272] | 0.000 | 1.0e+00 |
| real_aug | 0.831 [0.821-0.837] | 0.835 [0.831-0.841] | 0.806 | 1.0e+00 |
| real_modulus | 0.129 [0.125-0.133] | 0.125 [0.123-0.129] | 0.113 | 1.8e-07 |
| matched | 0.878 [0.869-0.884] | 0.879 [0.874-0.886] | 0.864 | - |

Mean over seeds [min-max] of the accuracy averaged over the test-phase bins in / outside the training range; worst bin: lowest seed-mean bin accuracy. Last column: max |p_k(z) - p_k(e^{it} z)| over test signals and random t; below about 1e-4 means invariant up to float32 rounding.
