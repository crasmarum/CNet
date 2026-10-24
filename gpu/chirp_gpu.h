#ifndef GPU_CHIRP_GPU_H_
#define GPU_CHIRP_GPU_H_

// GPU forms of ChirpFilter and BoxcarMax (CPU references and derivations: impl/chirp.h).

#include "gpumapping.h"

// Learnable coherent dedispersion, batched over clones with cuFFT:
//   forward : X = FFT(x) (batch M), Y_k = X . D_k, y_k = IFFT(Y_k)/L (batch M*K)
//   backward: Ig = IFFT(g)/L, Ih = IFFT(conj h)/L (batch M*K); per (clone, k) a block reduction gives
//             S1 = sum_f Ig V, S2 = sum_f Ih V (V = i phi dm_scale D X)  ->  dL/dd_k = (S1 + conj S2)/2;
//             dL/dx += sum_k FFT(D . Ig),   dL/dx* += sum_k IFFT(L conj(D Ih))/L  (FFT(h) = L conj(IFFT(conj h))).
class ChirpFilterGpu : public GpuMapping {
	int planM_ = -1, planMK_ = -1;       // cufftHandle for size L with batch M, and with batch M*K
	void *X_ = NULL;                       // M x L     spectrum of the input (kept for backward)
	void *A_ = NULL, *B_ = NULL;          // M*K x L   work spectra
	float *fcdm_ = NULL;                   // M x (1 + K): fc (MHz) and DM_k per clone, copied from the input
	int L_ = 0, K_ = 0, M_ = 0;
	void ensure(int L, int K, int M);
	void release();
public:
	ChirpFilterGpu(int depth) : GpuMapping(depth) {}
	virtual ~ChirpFilterGpu() { release(); }
	void gpu_forward();
	void gpu_backward();
	virtual void forward() override { gpu_forward(); }
	virtual void backward(int label) override { gpu_backward(); }
};

// Boxcar search: per (clone, row k, width w) the max over strided start positions of (window sum - w)/sqrt(w);
// backward sends the output gradient / sqrt(w) to every sample of the winning window.
class BoxcarMaxGpu : public GpuMapping {
	int *argmax_ = NULL;                   // M x K x W
	int *widths_ = NULL;                   // W, on the device
	int M_ = 0, K_ = 0, W_ = 0, maxw_ = 0;
	void ensure(int M, int K, const std::vector<int> &widths);
	void release();
public:
	BoxcarMaxGpu(int depth) : GpuMapping(depth) {}
	virtual ~BoxcarMaxGpu() { release(); }
	void gpu_forward();
	void gpu_backward();
	virtual void forward() override { gpu_forward(); }
	virtual void backward(int label) override { gpu_backward(); }
};

#endif /* GPU_CHIRP_GPU_H_ */
