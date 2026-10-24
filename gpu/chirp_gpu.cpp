// GPU forms of ChirpFilter and BoxcarMax. CPU references and the Wirtinger derivations: impl/chirp.h.
// Layout per clone (CNet planar vars): input [x (L) | fc (1) | d (K)], output [y_0 (L) | ... | y_{K-1} (L)].
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "chirp_gpu.h"
#include "kernels.h"
#include "../impl/chirp.h"
#ifdef __CUDACC__
#include <cufft.h>
#endif

#ifdef __CUDACC__

#define CHIRP_KDM 4.148808e15      // = ChirpFilter::KDM
#define CHIRP_DT  2.56e-6          // = ChirpFilter::DT

__device__ inline double chirp_phi__(int j, int L, double fc_mhz) {
	double f = (double) (j < (L + 1) / 2 ? j : j - L) / (L * CHIRP_DT);
	double fc = fc_mhz * 1e6;
	return 2.0 * 3.14159265358979323846 * CHIRP_KDM * f * f / (fc * fc * (fc + f));
}

// fc and DM_k = dm_scale * Re(d_k) per clone
__global__ void chirp_params__(GpuInVar *in, float *fcdm, int L, int K, int M, float dm_scale) {
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	if (m >= M) return;
	fcdm[m * (1 + K)] = *Z_real_(in[m], L);
	for (int k = 0; k < K; ++k) fcdm[m * (1 + K) + 1 + k] = dm_scale * *Z_real_(in[m], L + 1 + k);
}

__global__ void chirp_gather__(GpuInVar *in, cufftComplex *X, int L, int M) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M * L) return;
	int m = tid / L, t = tid % L;
	X[tid] = make_cuFloatComplex(*Z_real_(in[m], t), *Z_imag_(in[m], t));
}

// A[(m*K + k)*L + j] = X[m*L + j] * exp(i phi_j DM_mk)
__global__ void chirp_apply__(const cufftComplex *X, cufftComplex *A, const float *fcdm, int L, int K, int M) {
	long tid = (long) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (long) M * K * L) return;
	int j = tid % L; long mk = tid / L; int k = mk % K, m = mk / K;
	double ph = chirp_phi__(j, L, fcdm[m * (1 + K)]) * fcdm[m * (1 + K) + 1 + k];
	double s, c; sincos(ph, &s, &c);
	cufftComplex x = X[(long) m * L + j];
	A[tid] = make_cuFloatComplex((float) (x.x * c - x.y * s), (float) (x.x * s + x.y * c));
}

__global__ void chirp_scatter__(const cufftComplex *A, GpuOutVar *out, int L, int K, int M) {
	long tid = (long) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (long) M * K * L) return;
	int t = tid % L; long mk = tid / L; int k = mk % K, m = mk / K;
	float inv = 1.f / L;
	*Z_real_(out[m], k * L + t) = A[tid].x * inv;
	*Z_imag_(out[m], k * L + t) = A[tid].y * inv;
}

// A <- dL/dy (= g), B <- conj(dL/dy*) (= conj h)
__global__ void chirp_gather_grad__(GpuOutVar *out, cufftComplex *A, cufftComplex *B, int L, int K, int M) {
	long tid = (long) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (long) M * K * L) return;
	int t = tid % L; long mk = tid / L; int k = mk % K, m = mk / K;
	int p = k * L + t;
	A[tid] = make_cuFloatComplex(*dZ_real_(out[m], p), *dZ_imag_(out[m], p));
	B[tid] = make_cuFloatComplex(*dZ_star_real_(out[m], p), -*dZ_star_imag_(out[m], p));
}

// One block per (clone, k): S1 = sum Ig V, S2 = sum Ih V; rewrite A <- D Ig, B <- L conj(D Ih).
// A, B hold the unnormalised inverse FFTs on entry (Ig = A / L, Ih = B / L).
__global__ void chirp_reduce__(cufftComplex *A, cufftComplex *B, const cufftComplex *X, const float *fcdm,
		GpuInVar *in, int L, int K, int M, float dm_scale) {
	int mk = blockIdx.x; int k = mk % K, m = mk / K;
	double fc = fcdm[m * (1 + K)], DM = fcdm[m * (1 + K) + 1 + k];
	double s1r = 0, s1i = 0, s2r = 0, s2i = 0;
	long base = (long) mk * L;
	for (int j = threadIdx.x; j < L; j += blockDim.x) {
		double phi = chirp_phi__(j, L, fc);
		double s, c; sincos(phi * DM, &s, &c);
		cufftComplex x = X[(long) m * L + j];
		// V = i phi dm_scale D X
		double dr = x.x * c - x.y * s, di = x.x * s + x.y * c;
		double vr = -phi * dm_scale * di, vi = phi * dm_scale * dr;
		double gr = A[base + j].x / L, gi = A[base + j].y / L;
		double hr = B[base + j].x / L, hi = B[base + j].y / L;
		s1r += gr * vr - gi * vi; s1i += gr * vi + gi * vr;
		s2r += hr * vr - hi * vi; s2i += hr * vi + hi * vr;
		// D Ig  and  L conj(D Ih)
		A[base + j] = make_cuFloatComplex((float) (gr * c - gi * s), (float) (gr * s + gi * c));
		double ar = hr * c - hi * s, ai = hr * s + hi * c;
		B[base + j] = make_cuFloatComplex((float) (L * ar), (float) (-L * ai));
	}
	__shared__ double sh[4][256];
	sh[0][threadIdx.x] = s1r; sh[1][threadIdx.x] = s1i; sh[2][threadIdx.x] = s2r; sh[3][threadIdx.x] = s2i;
	__syncthreads();
	for (int st = blockDim.x / 2; st > 0; st >>= 1) {
		if (threadIdx.x < st)
			for (int q = 0; q < 4; ++q) sh[q][threadIdx.x] += sh[q][threadIdx.x + st];
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		// dL/dd = dL/dd* = (S1 + conj S2) / 2
		float re = (float) (0.5 * (sh[0][0] + sh[2][0])), im = (float) (0.5 * (sh[1][0] - sh[3][0]));
		int p = L + 1 + k;
		atomicAdd(dZ_real_(in[m], p), re);      atomicAdd(dZ_imag_(in[m], p), im);
		atomicAdd(dZ_star_real_(in[m], p), re); atomicAdd(dZ_star_imag_(in[m], p), im);
	}
}

// dL/dx += sum_k FFT(D Ig) (in A, after a forward FFT);  dL/dx* += sum_k IFFT(...)/L (in B, after an inverse FFT)
__global__ void chirp_accum__(const cufftComplex *A, const cufftComplex *B, GpuInVar *in, int L, int K, int M) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M * L) return;
	int m = tid / L, t = tid % L;
	float ar = 0, ai = 0, br = 0, bi = 0;
	for (int k = 0; k < K; ++k) {
		long i = ((long) m * K + k) * L + t;
		ar += A[i].x; ai += A[i].y; br += B[i].x; bi += B[i].y;
	}
	float inv = 1.f / L;
	atomicAdd(dZ_real_(in[m], t), ar);              atomicAdd(dZ_imag_(in[m], t), ai);
	atomicAdd(dZ_star_real_(in[m], t), br * inv);   atomicAdd(dZ_star_imag_(in[m], t), bi * inv);
}

static inline unsigned nblk(long n, int tb) { return (unsigned) ((n + tb - 1) / tb); }

void ChirpFilterGpu::ensure(int L, int K, int M) {
	if (L == L_ && K == K_ && M == M_ && X_) return;
	release();
	cufftHandle p1, p2;
	cufftResult r1 = cufftPlan1d(&p1, L, CUFFT_C2C, M);
	cufftResult r2 = cufftPlan1d(&p2, L, CUFFT_C2C, M * K);
	if (r1 != CUFFT_SUCCESS || r2 != CUFFT_SUCCESS) { fprintf(stderr, "ChirpFilterGpu: cufftPlan1d failed\n"); abort(); }
	planM_ = (int) p1; planMK_ = (int) p2;
	gpuErrchk(cudaMalloc(&X_, sizeof(cufftComplex) * (size_t) M * L));
	gpuErrchk(cudaMalloc(&A_, sizeof(cufftComplex) * (size_t) M * K * L));
	gpuErrchk(cudaMalloc(&B_, sizeof(cufftComplex) * (size_t) M * K * L));
	gpuErrchk(cudaMalloc(&fcdm_, sizeof(float) * (size_t) M * (1 + K)));
	L_ = L; K_ = K; M_ = M;
}

void ChirpFilterGpu::release() {
	if (planM_ != -1) { cufftDestroy((cufftHandle) planM_); planM_ = -1; }
	if (planMK_ != -1) { cufftDestroy((cufftHandle) planMK_); planMK_ = -1; }
	if (X_) { cudaFree(X_); X_ = NULL; }
	if (A_) { cudaFree(A_); A_ = NULL; }
	if (B_) { cudaFree(B_); B_ = NULL; }
	if (fcdm_) { cudaFree(fcdm_); fcdm_ = NULL; }
	L_ = K_ = M_ = 0;
}

void ChirpFilterGpu::gpu_forward() {
	ChirpFilter *cf = (ChirpFilter*) cpu_func_.front();
	int L = cf->L(), K = cf->K(), M = getNoMappings();
	ensure(L, K, M);
	cufftComplex *X = (cufftComplex*) X_, *A = (cufftComplex*) A_;
	chirp_params__ CUDA2(nblk(M, 128), 128) (gpu_in_ptr_, fcdm_, L, K, M, cf->dmScale());
	chirp_gather__ CUDA2(nblk((long) M * L, 256), 256) (gpu_in_ptr_, X, L, M);
	cufftSetStream((cufftHandle) planM_, g_cnet_stream);
	cufftExecC2C((cufftHandle) planM_, X, X, CUFFT_FORWARD);
	chirp_apply__ CUDA2(nblk((long) M * K * L, 256), 256) (X, A, fcdm_, L, K, M);
	cufftSetStream((cufftHandle) planMK_, g_cnet_stream);
	cufftExecC2C((cufftHandle) planMK_, A, A, CUFFT_INVERSE);
	chirp_scatter__ CUDA2(nblk((long) M * K * L, 256), 256) (A, gpu_out_ptr_, L, K, M);
	CNET_KSYNC();
}

void ChirpFilterGpu::gpu_backward() {
	ChirpFilter *cf = (ChirpFilter*) cpu_func_.front();
	int L = cf->L(), K = cf->K(), M = getNoMappings();
	ensure(L, K, M);   // X_ and fcdm_ hold the forward's spectrum and parameters
	cufftComplex *X = (cufftComplex*) X_, *A = (cufftComplex*) A_, *B = (cufftComplex*) B_;
	chirp_gather_grad__ CUDA2(nblk((long) M * K * L, 256), 256) (gpu_out_ptr_, A, B, L, K, M);
	cufftSetStream((cufftHandle) planMK_, g_cnet_stream);
	cufftExecC2C((cufftHandle) planMK_, A, A, CUFFT_INVERSE);
	cufftExecC2C((cufftHandle) planMK_, B, B, CUFFT_INVERSE);
	chirp_reduce__ CUDA2(M * K, 256) (A, B, X, fcdm_, gpu_in_ptr_, L, K, M, cf->dmScale());
	cufftExecC2C((cufftHandle) planMK_, A, A, CUFFT_FORWARD);
	cufftExecC2C((cufftHandle) planMK_, B, B, CUFFT_INVERSE);
	chirp_accum__ CUDA2(nblk((long) M * L, 256), 256) (A, B, gpu_in_ptr_, L, K, M);
	CNET_KSYNC();
}

// ---------------- BoxcarMax ----------------
__global__ void boxcar_forward__(GpuInVar *in, GpuOutVar *out, int *argmax, const int *widths, int L, int K, int W, int M) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M * K * W) return;
	int iw = tid % W, k = (tid / W) % K, m = tid / (W * K);
	int w = widths[iw], st = w / 4 > 1 ? w / 4 : 1;
	const float *p = Z_real_(in[m], k * L);
	double s = 0;
	for (int t = 0; t < w; ++t) s += p[t];
	double best = s; int arg = 0;
	for (int pos = st; pos + w <= L; pos += st) {
		for (int t = 0; t < st; ++t) s += p[pos + w - st + t] - p[pos - st + t];
		if (s > best) { best = s; arg = pos; }
	}
	argmax[tid] = arg;
	*Z_real_(out[m], k * W + iw) = (float) ((best - w) / sqrt((double) w));
	*Z_imag_(out[m], k * W + iw) = 0.f;
}

__global__ void boxcar_backward__(GpuInVar *in, GpuOutVar *out, const int *argmax, const int *widths, int L, int K, int W,
		int M, int maxw) {
	long tid = (long) blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (long) M * K * W * maxw) return;
	int t = tid % maxw; long r = tid / maxw;
	int iw = r % W, k = (r / W) % K, m = r / (W * K);
	int w = widths[iw];
	if (t >= w) return;
	float sc = rsqrtf((float) w);
	int o = k * W + iw, i = k * L + argmax[r] + t;
	atomicAdd(dZ_real_(in[m], i), *dZ_real_(out[m], o) * sc);
	atomicAdd(dZ_imag_(in[m], i), *dZ_imag_(out[m], o) * sc);
	atomicAdd(dZ_star_real_(in[m], i), *dZ_star_real_(out[m], o) * sc);
	atomicAdd(dZ_star_imag_(in[m], i), *dZ_star_imag_(out[m], o) * sc);
}

void BoxcarMaxGpu::ensure(int M, int K, const std::vector<int> &widths) {
	int W = (int) widths.size();
	if (M == M_ && K == K_ && W == W_ && argmax_) return;
	release();
	gpuErrchk(cudaMalloc(&argmax_, sizeof(int) * (size_t) M * K * W));
	gpuErrchk(cudaMalloc(&widths_, sizeof(int) * W));
	gpuErrchk(cudaMemcpy(widths_, widths.data(), sizeof(int) * W, cudaMemcpyHostToDevice));
	maxw_ = 0;
	for (int w : widths) maxw_ = w > maxw_ ? w : maxw_;
	M_ = M; K_ = K; W_ = W;
}

void BoxcarMaxGpu::release() {
	if (argmax_) { cudaFree(argmax_); argmax_ = NULL; }
	if (widths_) { cudaFree(widths_); widths_ = NULL; }
	M_ = K_ = W_ = maxw_ = 0;
}

void BoxcarMaxGpu::gpu_forward() {
	BoxcarMax *bm = (BoxcarMax*) cpu_func_.front();
	int M = getNoMappings(), K = bm->K(), L = bm->L(), W = (int) bm->widths().size();
	ensure(M, K, bm->widths());
	boxcar_forward__ CUDA2(nblk((long) M * K * W, 128), 128) (gpu_in_ptr_, gpu_out_ptr_, argmax_, widths_, L, K, W, M);
	CNET_KSYNC();
}

void BoxcarMaxGpu::gpu_backward() {
	BoxcarMax *bm = (BoxcarMax*) cpu_func_.front();
	int M = getNoMappings(), K = bm->K(), L = bm->L(), W = (int) bm->widths().size();
	boxcar_backward__ CUDA2(nblk((long) M * K * W * maxw_, 256), 256) (gpu_in_ptr_, gpu_out_ptr_, argmax_, widths_,
			L, K, W, M, maxw_);
	CNET_KSYNC();
}

#else   // CPU-only build: the GPU mappings are never instantiated

void ChirpFilterGpu::ensure(int, int, int) {}
void ChirpFilterGpu::release() {}
void ChirpFilterGpu::gpu_forward() {}
void ChirpFilterGpu::gpu_backward() {}
void BoxcarMaxGpu::ensure(int, int, const std::vector<int>&) {}
void BoxcarMaxGpu::release() {}
void BoxcarMaxGpu::gpu_forward() {}
void BoxcarMaxGpu::gpu_backward() {}

#endif
