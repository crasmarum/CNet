// Gated linear attention over complex amplitudes, chunkwise GPU form (layer: impl/gla.h).
//
// Per (clone, head), chunk C, scalar per-head gate alpha_t = sigmoid(Re g_t), lambda_t = log alpha_t,
// L_t = chunk-local inclusive cumulative sum of lambda, LC = L at the chunk's last token:
//   state S (dh x dh complex) = sum_{j before the chunk} (decay j+1..c0-1) conj(K_j) V_j^T
//   o_i = s exp(L_i) S^T q_i  +  sum_{j<=i in chunk} s exp(L_i - L_j) <q_i,k_j> v_j
//   S <- exp(LC) S + sum_j exp(LC - L_j) conj(K_j) V_j^T
// Backward sweeps chunks in reverse with the suffix state G = sum_{k later} (decay) q_k g_k^T and
// Gb the same with conj(q_k) gb_k; see the derivation in impl/gla.h and the comments below.
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "reduce.h"
#include "allocator.h"
#include "kernels.h"
#ifdef __CUDACC__
#include <cublas_v2.h>
cublasHandle_t tw_blas_handle();
#endif

#define GLA_MAX_DH 128

#ifdef __CUDACC__

// ---------- small device helpers ----------
__device__ inline float gla_logsigmoid(float x) { return x >= 0.f ? -log1pf(expf(-x)) : x - log1pf(expf(x)); }

// one thread per (clone, head, chunk): lambda and the chunk-local inclusive cumulative sums
__global__ void gla_gates__(GpuInVar *in, float *lam, float *L, int N, int d, int H, int C, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int NC = N / C;
	if (tid >= B * H * NC) return;
	const int c = tid % NC, r = tid / NC, h = r % H, clone = r / H;
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	const int gbase = 3 * N * d;
	float acc = 0.f;
	for (int t = c * C; t < (c + 1) * C; ++t) {
		float x = inv.input_ptr_[gbase + t * H + h];               // Re G[t,h]
		float l = gla_logsigmoid(x);
		acc += l;
		lam[bh * N + t] = l; L[bh * N + t] = acc;
	}
}

// planes[((bh*2 + pl)*C + t)*dh + j]: Qs = exp(L_t) q, Kt = exp(LC - L_t) k, Qr = q, Kr = k, V = v
__global__ void gla_pack__(GpuInVar *in, const float *L, float *Qs, float *Kt, float *Qr, float *Kr, float *V,
		int c0, int C, int N, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	const int tok = c0 + t, Mv = N * d, p = tok * d + h * dh + j;
	const float Lt = L[bh * N + tok], LC = L[bh * N + c0 + C - 1];
	const float eq = expf(Lt), ek = expf(LC - Lt);
	cmplx_ q = Z_(inv, p), k = Z_(inv, Mv + p), v = Z_(inv, 2 * Mv + p);
	const size_t o0 = ((bh * 2 + 0) * C + t) * dh + j, o1 = ((bh * 2 + 1) * C + t) * dh + j;
	Qs[o0] = eq * q.real; Qs[o1] = eq * q.imag;
	Kt[o0] = ek * k.real; Kt[o1] = ek * k.imag;
	Qr[o0] = q.real; Qr[o1] = q.imag;
	Kr[o0] = k.real; Kr[o1] = k.imag;
	V[o0]  = v.real; V[o1]  = v.imag;
}

// one thread per (clone, head, i in chunk): intra-chunk amplitudes a_ij = s exp(L_i - L_j) <q_i,k_j>,
// o_i = Oint_i + sum_j a_ij v_j
__global__ void gla_intra_fwd__(GpuInVar *in, GpuOutVar *out, const float *L, const float *Oint,
		int c0, int C, int N, int d, int H, int B, float s) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	const int ic = tid % C, r = tid / C, h = r % H, clone = r / H;
	const int i = c0 + ic, dh = d / H, d0 = h * dh, Mv = N * d;
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone]; GpuOutVar ov = out[clone];
	cmplx_ q[GLA_MAX_DH], o[GLA_MAX_DH];
	for (int m = 0; m < dh; ++m) { q[m] = Z_(inv, i * d + d0 + m); o[m] = cmplx(0.f, 0.f); }
	const float Li = L[bh * N + i];
	for (int jc = 0; jc <= ic; ++jc) {
		const int j = c0 + jc;
		cmplx_ c = cmplx(0.f, 0.f);
		for (int m = 0; m < dh; ++m) c = c + q[m] * conj_(Z_(inv, Mv + j * d + d0 + m));
		const float w = s * expf(Li - L[bh * N + j]);
		cmplx_ a = w * c;
		for (int v = 0; v < dh; ++v) o[v] = o[v] + a * Z_(inv, 2 * Mv + j * d + d0 + v);
	}
	const float *oir = Oint + ((bh * 2 + 0) * C + ic) * dh, *oii = Oint + ((bh * 2 + 1) * C + ic) * dh;
	for (int v = 0; v < dh; ++v) {
		ov.out_ptr_[i * d + d0 + v] = o[v].real + oir[v];
		ov.out_ptr_[ov.out_length_ + i * d + d0 + v] = o[v].imag + oii[v];
	}
}

// S[bh] *= exp(LC) (both planes); one thread per element
__global__ void gla_scale_state__(float *S, const float *L, int c0, int C, int N, int dh, int BH) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const size_t per = (size_t) 2 * dh * dh;
	if (tid >= (size_t) BH * per) return;
	const int bh = (int) (tid / per);
	S[tid] *= expf(L[(size_t) bh * N + c0 + C - 1]);
}

// backward packing: Gg = gO, Gb = gOb planes [2][C][dh]
__global__ void gla_pack_grad__(GpuOutVar *out, float *Gg, float *Gb, int c0, int C, int d, int H, int B) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuOutVar ov = out[clone];
	const int p = (c0 + t) * d + h * dh + j;
	cmplx_ g = dZ_(ov, p), gb = dZ_star_(ov, p);
	const size_t o0 = ((bh * 2 + 0) * C + t) * dh + j, o1 = ((bh * 2 + 1) * C + t) * dh + j;
	Gg[o0] = g.real; Gg[o1] = g.imag; Gb[o0] = gb.real; Gb[o1] = gb.imag;
}

// one thread per (clone, head, i in chunk): amplitudes A[i][j], weighted PW = p_ij s w_ij, PbW,
// E[i][j] = Re(p a + pb conj a), eps_i = Re(g_i.Oint_i + gb_i.conj(Oint_i)); zeros above the diagonal
__global__ void gla_intra_bwd__(GpuInVar *in, const float *L, const float *Oint, const float *Gg, const float *Gb,
		float *A, float *PW, float *PbW, float *E, float *eps_out,
		int c0, int C, int N, int d, int H, int B, float s) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	const int ic = tid % C, r = tid / C, h = r % H, clone = r / H;
	const int i = c0 + ic, dh = d / H, d0 = h * dh, Mv = N * d;
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	cmplx_ q[GLA_MAX_DH], g[GLA_MAX_DH], gb[GLA_MAX_DH];
	const float *ggr = Gg + ((bh * 2 + 0) * C + ic) * dh, *ggi = Gg + ((bh * 2 + 1) * C + ic) * dh;
	const float *gbr = Gb + ((bh * 2 + 0) * C + ic) * dh, *gbi = Gb + ((bh * 2 + 1) * C + ic) * dh;
	for (int m = 0; m < dh; ++m) { q[m] = Z_(inv, i * d + d0 + m); g[m] = cmplx(ggr[m], ggi[m]); gb[m] = cmplx(gbr[m], gbi[m]); }
	const float Li = L[bh * N + i];
	const size_t row = (bh * C + ic) * C, pr = (bh * 2 + 0) * C * C + ic * C, pi = (bh * 2 + 1) * C * C + ic * C;
	for (int jc = 0; jc < C; ++jc) {
		if (jc > ic) { A[pr + jc] = A[pi + jc] = 0.f; PW[pr + jc] = PW[pi + jc] = 0.f; PbW[pr + jc] = PbW[pi + jc] = 0.f; E[row + jc] = 0.f; continue; }
		const int j = c0 + jc;
		cmplx_ c = cmplx(0.f, 0.f), p = cmplx(0.f, 0.f), pb = cmplx(0.f, 0.f);
		for (int m = 0; m < dh; ++m) {
			cmplx_ v = Z_(inv, 2 * Mv + j * d + d0 + m);
			c = c + q[m] * conj_(Z_(inv, Mv + j * d + d0 + m));
			p = p + g[m] * v;
			pb = pb + gb[m] * conj_(v);
		}
		const float w = s * expf(Li - L[bh * N + j]);
		cmplx_ a = w * c;
		A[pr + jc] = a.real; A[pi + jc] = a.imag;
		cmplx_ pw = w * p, pbw = w * pb;
		PW[pr + jc] = pw.real; PW[pi + jc] = pw.imag;
		PbW[pr + jc] = pbw.real; PbW[pi + jc] = pbw.imag;
		E[row + jc] = (p * a).real + (pb * conj_(a)).real;
	}
	const float *oir = Oint + ((bh * 2 + 0) * C + ic) * dh, *oii = Oint + ((bh * 2 + 1) * C + ic) * dh;
	float e = 0.f;
	for (int v = 0; v < dh; ++v) { cmplx_ o = cmplx(oir[v], oii[v]); e += (g[v] * o).real + (gb[v] * conj_(o)).real; }
	eps_out[bh * C + ic] = e;
}

// Ecum[k][u] = sum_{m<u} E[k][m]; one thread per (bh, k)
__global__ void gla_e_prefix__(const float *E, float *Ecum, int C, int BH) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= BH * C) return;
	const size_t row = (size_t) tid * C;
	float acc = 0.f;
	for (int u = 0; u < C; ++u) { Ecum[row + u] = acc; acc += E[row + u]; }
}

// eta_m = Re sum_v (v_mv dVlat_mv + conj(v_mv) dVblat_mv): one thread per (bh, m)
__global__ void gla_eta__(const float *V, const float *dV, const float *dVb, float *eta, int C, int dh, int BH) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= BH * C) return;
	const int m = tid % C, bh = tid / C;
	const size_t r0 = ((size_t) bh * 2 + 0) * C * dh + (size_t) m * dh, r1 = ((size_t) bh * 2 + 1) * C * dh + (size_t) m * dh;
	float e = 0.f;
	for (int v = 0; v < dh; ++v) {
		cmplx_ vv = cmplx(V[r0 + v], V[r1 + v]), a = cmplx(dV[r0 + v], dV[r1 + v]), b = cmplx(dVb[r0 + v], dVb[r1 + v]);
		e += (vv * a).real + (conj_(vv) * b).real;
	}
	eta[tid] = e;
}

// D_c[bh] = s exp(LC) Re sum_{m,v} (G[m,v] S0[m,v] + Gb[m,v] conj(S0[m,v])): one thread per bh
__global__ void gla_dc__(const float *G, const float *Gb, const float *S0, const float *L, float *Dc,
		int c0, int C, int N, int dh, int BH, float s) {
	int bh = blockIdx.x * blockDim.x + threadIdx.x;
	if (bh >= BH) return;
	const size_t base = (size_t) bh * 2 * dh * dh, n = (size_t) dh * dh;
	float acc = 0.f;
	for (size_t e = 0; e < n; ++e) {
		cmplx_ g = cmplx(G[base + e], G[base + n + e]), gb = cmplx(Gb[base + e], Gb[base + n + e]), s0 = cmplx(S0[base + e], S0[base + n + e]);
		acc += (g * s0).real + (gb * conj_(s0)).real;
	}
	Dc[bh] = s * expf(L[(size_t) bh * N + c0 + C - 1]) * acc;
}

// gate gradient per (bh, u in chunk): dlambda_u = sum_{k>=u} Ecum[k][u] + sum_{k>=u} eps_k + sum_{m<u} eta_m + D_c
__global__ void gla_gate_grad__(GpuInVar *in, const float *Ecum, const float *epsk, const float *eta, const float *Dc,
		int c0, int C, int N, int d, int H, int B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B * H * C) return;
	const int uc = tid % C, r = tid / C, h = r % H, clone = r / H;
	const size_t bh = (size_t) clone * H + h;
	float acc = Dc[bh];
	for (int k = uc; k < C; ++k) acc += Ecum[(bh * C + k) * C + uc] + epsk[bh * C + k];
	for (int m = 0; m < uc; ++m) acc += eta[bh * C + m];
	GpuInVar inv = in[clone];
	const int u = c0 + uc, p = 3 * N * d + u * H + h;
	const float x = inv.input_ptr_[p];
	const float dx = acc / (1.f + expf(x)) * 0.5f;         // dlambda/dx = sigmoid(-x); Wirtinger halves
	atomicAdd(dZ_real_(inv, p), dx);
	atomicAdd(dZ_star_real_(inv, p), dx);
}

// accumulate the Q, K, V gradients of the chunk into the input gradient planes
__global__ void gla_accum__(GpuInVar *in, const float *L, const float *dQ, const float *dQb, const float *dQi, const float *dQbi,
		const float *dK, const float *dKb, const float *dKl, const float *dKlb, const float *dV, const float *dVb,
		int c0, int C, int N, int d, int H, int B, float s) {
	size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const int dh = d / H;
	if (tid >= (size_t) B * H * C * dh) return;
	int j = (int) (tid % dh); size_t r = tid / dh;
	int t = (int) (r % C); r /= C;
	int h = (int) (r % H); int clone = (int) (r / H);
	const size_t bh = (size_t) clone * H + h;
	GpuInVar inv = in[clone];
	const int tok = c0 + t, Mv = N * d, p = tok * d + h * dh + j;
	const float Lt = L[bh * N + tok], LC = L[bh * N + c0 + C - 1];
	const float eq = s * expf(Lt), wk = expf(LC - Lt);
	const size_t o0 = ((bh * 2 + 0) * C + t) * dh + j, o1 = ((bh * 2 + 1) * C + t) * dh + j;
	atomicAdd(dZ_real_(inv, p),      dQ[o0]  + eq * dQi[o0]);
	atomicAdd(dZ_imag_(inv, p),      dQ[o1]  + eq * dQi[o1]);
	atomicAdd(dZ_star_real_(inv, p), dQb[o0] + eq * dQbi[o0]);
	atomicAdd(dZ_star_imag_(inv, p), dQb[o1] + eq * dQbi[o1]);
	atomicAdd(dZ_real_(inv, Mv + p),      dK[o0]  + wk * dKlb[o0]);
	atomicAdd(dZ_imag_(inv, Mv + p),      dK[o1]  + wk * dKlb[o1]);
	atomicAdd(dZ_star_real_(inv, Mv + p), dKb[o0] + wk * dKl[o0]);
	atomicAdd(dZ_star_imag_(inv, Mv + p), dKb[o1] + wk * dKl[o1]);
	atomicAdd(dZ_real_(inv, 2 * Mv + p),      dV[o0]);
	atomicAdd(dZ_imag_(inv, 2 * Mv + p),      dV[o1]);
	atomicAdd(dZ_star_real_(inv, 2 * Mv + p), dVb[o0]);
	atomicAdd(dZ_star_imag_(inv, 2 * Mv + p), dVb[o1]);
}

// ---------- complex GEMM on planes ----------
// Row-major Z[n x m] = alpha * op(X)[n x k] * op(Y)[k x m] + beta * Z, batched over BH with the given
// plane strides; X stored row-major [n x k] (ld k) or, if tX, [k x n] (ld n); same for Y. cX / cY
// conjugate the operand. Four real strided-batched GEMMs (cuBLAS column-major via the transpose trick).
struct Planes { const float *r, *i; int ld; long long stride; };
struct PlanesOut { float *r, *i; int ld; long long stride; };
static void rm_gemm(cublasHandle_t hb, bool tX, bool tY, int n, int m, int k, float alpha,
		const float *X, int ldX, long long sX, const float *Y, int ldY, long long sY, float beta, float *Z, int ldZ, long long sZ, int batch) {
	cublasSgemmStridedBatched(hb, tY ? CUBLAS_OP_T : CUBLAS_OP_N, tX ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
		Y, ldY, sY, X, ldX, sX, &beta, Z, ldZ, sZ, batch);
}
static void cgemm(cublasHandle_t hb, bool tX, bool cX, bool tY, bool cY, int n, int m, int k, float alpha,
		Planes X, Planes Y, float beta, PlanesOut Z, int batch) {
	const float sx = cX ? -1.f : 1.f, sy = cY ? -1.f : 1.f;
	rm_gemm(hb, tX, tY, n, m, k, alpha,            X.r, X.ld, X.stride, Y.r, Y.ld, Y.stride, beta, Z.r, Z.ld, Z.stride, batch);
	rm_gemm(hb, tX, tY, n, m, k, -sx * sy * alpha, X.i, X.ld, X.stride, Y.i, Y.ld, Y.stride, 1.f,  Z.r, Z.ld, Z.stride, batch);
	rm_gemm(hb, tX, tY, n, m, k, sy * alpha,       X.r, X.ld, X.stride, Y.i, Y.ld, Y.stride, beta, Z.i, Z.ld, Z.stride, batch);
	rm_gemm(hb, tX, tY, n, m, k, sx * alpha,       X.i, X.ld, X.stride, Y.r, Y.ld, Y.stride, 1.f,  Z.i, Z.ld, Z.stride, batch);
}
// chunk planes [bh][2][C][dh]; state planes [bh][2][dh][dh]; C x C planes [bh][2][C][C]
static Planes chunkP(const float *p, int C, int dh) { return Planes{p, p + (size_t) C * dh, dh, (long long) 2 * C * dh}; }
static PlanesOut chunkO(float *p, int C, int dh) { return PlanesOut{p, p + (size_t) C * dh, dh, (long long) 2 * C * dh}; }
static Planes stateP(const float *p, int dh) { return Planes{p, p + (size_t) dh * dh, dh, (long long) 2 * dh * dh}; }
static PlanesOut stateO(float *p, int dh) { return PlanesOut{p, p + (size_t) dh * dh, dh, (long long) 2 * dh * dh}; }
static Planes sqP(const float *p, int C) { return Planes{p, p + (size_t) C * C, C, (long long) 2 * C * C}; }

#endif  // __CUDACC__

void GatedLinearAttentionGpu::ensure_scratch(int N, int d, int H, int C, int B) {
#ifdef __CUDACC__
	const int dh = d / H, NC = N / C;
	if (gl_Qs_ && gl_N_ == N && gl_C_ == C && gl_B_ == B && gl_H_ == H && gl_dh_ == dh) return;
	free_scratch();
	gl_N_ = N; gl_C_ = C; gl_B_ = B; gl_H_ = H; gl_dh_ = dh;
	const size_t BH = (size_t) B * H, chunk = BH * 2 * C * dh, state = BH * 2 * dh * dh, sq = BH * 2 * C * C;
	gpuErrchk(cudaMalloc((void**) &gl_lam_,   sizeof(float) * BH * N));
	gpuErrchk(cudaMalloc((void**) &gl_L_,     sizeof(float) * BH * N));
	float **cp[] = {&gl_Qs_, &gl_Kt_, &gl_Qr_, &gl_Kr_, &gl_V_, &gl_Oint_, &bw_Gg_, &bw_Gb_,
		&bw_dQ_, &bw_dQb_, &bw_dQi_, &bw_dQbi_, &bw_dK_, &bw_dKb_, &bw_dKl_, &bw_dKlb_, &bw_dV_, &bw_dVb_};
	for (float **p : cp) gpuErrchk(cudaMalloc((void**) p, sizeof(float) * chunk));
	gpuErrchk(cudaMalloc((void**) &gl_S_,     sizeof(float) * state));
	gpuErrchk(cudaMalloc((void**) &carry_S_,  sizeof(float) * state));
	gpuErrchk(cudaMemset(carry_S_, 0, sizeof(float) * state));
	gpuErrchk(cudaMalloc((void**) &gl_Shist_, sizeof(float) * state * NC));
	gpuErrchk(cudaMalloc((void**) &bw_G_,     sizeof(float) * state));
	gpuErrchk(cudaMalloc((void**) &bw_Gbs_,   sizeof(float) * state));
	gpuErrchk(cudaMalloc((void**) &bw_A_,     sizeof(float) * sq));
	gpuErrchk(cudaMalloc((void**) &bw_PW_,    sizeof(float) * sq));
	gpuErrchk(cudaMalloc((void**) &bw_PbW_,   sizeof(float) * sq));
	gpuErrchk(cudaMalloc((void**) &bw_E_,     sizeof(float) * BH * C * C));
	gpuErrchk(cudaMalloc((void**) &bw_Ecum_,  sizeof(float) * BH * C * C));
	gpuErrchk(cudaMalloc((void**) &bw_eps_,   sizeof(float) * BH * C));
	gpuErrchk(cudaMalloc((void**) &bw_eta_,   sizeof(float) * BH * C));
	gpuErrchk(cudaMalloc((void**) &bw_Dc_,    sizeof(float) * BH));
#endif
}

void GatedLinearAttentionGpu::free_scratch() {
#ifdef __CUDACC__
	float **ps[] = {&gl_lam_, &gl_L_, &gl_Qs_, &gl_Kt_, &gl_Qr_, &gl_Kr_, &gl_V_, &gl_Oint_, &gl_S_, &gl_Shist_, &carry_S_,
		&bw_Gg_, &bw_Gb_, &bw_A_, &bw_PW_, &bw_PbW_, &bw_E_, &bw_Ecum_, &bw_eps_, &bw_eta_, &bw_Dc_,
		&bw_dQ_, &bw_dQb_, &bw_dQi_, &bw_dQbi_, &bw_dK_, &bw_dKb_, &bw_dKl_, &bw_dKlb_, &bw_dV_, &bw_dVb_, &bw_G_, &bw_Gbs_};
	for (float **p : ps) if (*p) { cudaFree(*p); *p = NULL; }
	gl_N_ = gl_C_ = gl_B_ = gl_H_ = gl_dh_ = 0;
#endif
}

void GatedLinearAttentionGpu::gpu_gla_forward() {
#ifdef __CUDACC__
	GatedLinearAttention *f = (GatedLinearAttention*) cpu_func_.front();
	const int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings();
	int C = f->chunk(); if (C > N) C = N;
	const int dh = d / H, NC = N / C;
	if (N % C != 0 || dh > GLA_MAX_DH) {
		fprintf(stderr, "GatedLinearAttentionGpu: needs N %% chunk == 0 and head dim <= %d (N=%d C=%d dh=%d)\n", GLA_MAX_DH, N, C, dh);
		exit(1);
	}
	const size_t BH = (size_t) B * H, state = BH * 2 * dh * dh;
	const float s = f->scale();
	ensure_scratch(N, d, H, C, B);
	cudaStream_t st = g_cnet_stream ? g_cnet_stream : cudaStreamPerThread;
	cublasHandle_t hb = tw_blas_handle();
	cublasSetStream(hb, st);
	const int tb = 128;
	const size_t nGate = BH * NC, nPack = BH * C * dh, nTok = BH * C;
	if (carry_) { gpuErrchk(cudaMemcpyAsync(gl_S_, carry_S_, sizeof(float) * state, cudaMemcpyDeviceToDevice, st)); }
	else        { gpuErrchk(cudaMemsetAsync(gl_S_, 0, sizeof(float) * state, st)); }
	gla_gates__ CUDA2((unsigned) ((nGate + tb - 1) / tb), tb) (gpu_in_ptr_, gl_lam_, gl_L_, N, d, H, C, B);
	for (int c = 0; c < NC; ++c) {
		const int c0 = c * C;
		gla_pack__ CUDA2((unsigned) ((nPack + tb - 1) / tb), tb) (gpu_in_ptr_, gl_L_, gl_Qs_, gl_Kt_, gl_Qr_, gl_Kr_, gl_V_, c0, C, N, d, H, B);
		gpuErrchk(cudaMemcpyAsync(gl_Shist_ + (size_t) c * state, gl_S_, sizeof(float) * state, cudaMemcpyDeviceToDevice, st));
		// inter-chunk readout: Oint[C x dh] = s * Qs[C x dh] * S[dh x dh]   (o_i,v = s exp(L_i) sum_m q_im S[m,v])
		cgemm(hb, false, false, false, false, C, dh, dh, s, chunkP(gl_Qs_, C, dh), stateP(gl_S_, dh), 0.f, chunkO(gl_Oint_, C, dh), (int) BH);
		gla_intra_fwd__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (gpu_in_ptr_, gpu_out_ptr_, gl_L_, gl_Oint_, c0, C, N, d, H, B, s);
		// state: S <- exp(LC) S + Kt^H V
		gla_scale_state__ CUDA2((unsigned) ((state + tb - 1) / tb), tb) (gl_S_, gl_L_, c0, C, N, dh, (int) BH);
		cgemm(hb, true, true, false, false, dh, dh, C, 1.f, chunkP(gl_Kt_, C, dh), chunkP(gl_V_, C, dh), 1.f, stateO(gl_S_, dh), (int) BH);
	}
	if (carry_) { gpuErrchk(cudaMemcpyAsync(carry_S_, gl_S_, sizeof(float) * state, cudaMemcpyDeviceToDevice, st)); }
	cublasSetStream(hb, cudaStreamPerThread);
	CNET_KSYNC();
#endif
}

void GatedLinearAttentionGpu::resetCarry() {
#ifdef __CUDACC__
	if (carry_S_) {
		const size_t state = (size_t) gl_B_ * gl_H_ * 2 * gl_dh_ * gl_dh_;
		gpuErrchk(cudaMemset(carry_S_, 0, sizeof(float) * state));
	}
#endif
}

void GatedLinearAttentionGpu::gpu_gla_backward() {
#ifdef __CUDACC__
	GatedLinearAttention *f = (GatedLinearAttention*) cpu_func_.front();
	const int N = f->nTokens(), d = f->dim(), H = f->heads(), B = getNoMappings();
	int C = f->chunk(); if (C > N) C = N;
	const int dh = d / H, NC = N / C;
	const size_t BH = (size_t) B * H, state = BH * 2 * dh * dh;
	const float s = f->scale();
	cudaStream_t st = g_cnet_stream ? g_cnet_stream : cudaStreamPerThread;
	cublasHandle_t hb = tw_blas_handle();
	cublasSetStream(hb, st);
	const int tb = 128;
	const size_t nPack = BH * C * dh, nTok = BH * C;
	gpuErrchk(cudaMemsetAsync(bw_G_, 0, sizeof(float) * state, st));
	gpuErrchk(cudaMemsetAsync(bw_Gbs_, 0, sizeof(float) * state, st));
	for (int c = NC - 1; c >= 0; --c) {
		const int c0 = c * C;
		const float *S0 = gl_Shist_ + (size_t) c * state;
		gla_pack__ CUDA2((unsigned) ((nPack + tb - 1) / tb), tb) (gpu_in_ptr_, gl_L_, gl_Qs_, gl_Kt_, gl_Qr_, gl_Kr_, gl_V_, c0, C, N, d, H, B);
		gla_pack_grad__ CUDA2((unsigned) ((nPack + tb - 1) / tb), tb) (gpu_out_ptr_, bw_Gg_, bw_Gb_, c0, C, d, H, B);
		// Oint (forward's inter readout, for the gate term eps_k)
		cgemm(hb, false, false, false, false, C, dh, dh, s, chunkP(gl_Qs_, C, dh), stateP(S0, dh), 0.f, chunkO(gl_Oint_, C, dh), (int) BH);
		gla_intra_bwd__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (gpu_in_ptr_, gl_L_, gl_Oint_, bw_Gg_, bw_Gb_,
			bw_A_, bw_PW_, bw_PbW_, bw_E_, bw_eps_, c0, C, N, d, H, B, s);
		// V from later chunks: dV = s conj(Kt) G ; dVb = s Kt Gb   (rows scaled by w_j through Kt)
		cgemm(hb, false, true,  false, false, C, dh, dh, s, chunkP(gl_Kt_, C, dh), stateP(bw_G_, dh),   0.f, chunkO(bw_dV_, C, dh),  (int) BH);
		cgemm(hb, false, false, false, false, C, dh, dh, s, chunkP(gl_Kt_, C, dh), stateP(bw_Gbs_, dh), 0.f, chunkO(bw_dVb_, C, dh), (int) BH);
		gla_eta__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (gl_V_, bw_dV_, bw_dVb_, bw_eta_, C, dh, (int) BH);
		// V in-chunk: dV += A^T Gg ; dVb += conj(A)^T Gb
		cgemm(hb, true, false, false, false, C, dh, C, 1.f, sqP(bw_A_, C), chunkP(bw_Gg_, C, dh), 1.f, chunkO(bw_dV_, C, dh),  (int) BH);
		cgemm(hb, true, true,  false, false, C, dh, C, 1.f, sqP(bw_A_, C), chunkP(bw_Gb_, C, dh), 1.f, chunkO(bw_dVb_, C, dh), (int) BH);
		// Q: dQ = PW conj(Kr) ; dQb = PbW Kr ; inter: dQi = Gg S0^T ; dQbi = Gb conj(S0)^T  (scaled by s exp(L_i) at accumulate)
		cgemm(hb, false, false, false, true,  C, dh, C,  1.f, sqP(bw_PW_, C),  chunkP(gl_Kr_, C, dh), 0.f, chunkO(bw_dQ_, C, dh),  (int) BH);
		cgemm(hb, false, false, false, false, C, dh, C,  1.f, sqP(bw_PbW_, C), chunkP(gl_Kr_, C, dh), 0.f, chunkO(bw_dQb_, C, dh), (int) BH);
		cgemm(hb, false, false, true,  false, C, dh, dh, 1.f, chunkP(bw_Gg_, C, dh), stateP(S0, dh), 0.f, chunkO(bw_dQi_, C, dh),  (int) BH);
		cgemm(hb, false, false, true,  true,  C, dh, dh, 1.f, chunkP(bw_Gb_, C, dh), stateP(S0, dh), 0.f, chunkO(bw_dQbi_, C, dh), (int) BH);
		// K: dKb (dz*) = PW^T Qr ; dK (dz) = PbW^T conj(Qr) ; later: dKl = s V G^T ; dKlb = s conj(V) Gb^T (scaled by w_j at accumulate)
		cgemm(hb, true, false, false, false, C, dh, C,  1.f, sqP(bw_PW_, C),  chunkP(gl_Qr_, C, dh), 0.f, chunkO(bw_dKb_, C, dh), (int) BH);
		cgemm(hb, true, false, false, true,  C, dh, C,  1.f, sqP(bw_PbW_, C), chunkP(gl_Qr_, C, dh), 0.f, chunkO(bw_dK_, C, dh),  (int) BH);
		cgemm(hb, false, false, true, false, C, dh, dh, s,   chunkP(gl_V_, C, dh), stateP(bw_G_, dh),   0.f, chunkO(bw_dKl_, C, dh),  (int) BH);
		cgemm(hb, false, true,  true, false, C, dh, dh, s,   chunkP(gl_V_, C, dh), stateP(bw_Gbs_, dh), 0.f, chunkO(bw_dKlb_, C, dh), (int) BH);
		// gates
		gla_e_prefix__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (bw_E_, bw_Ecum_, C, (int) BH);
		gla_dc__ CUDA2((unsigned) ((BH + tb - 1) / tb), tb) (bw_G_, bw_Gbs_, S0, gl_L_, bw_Dc_, c0, C, N, dh, (int) BH, s);
		gla_gate_grad__ CUDA2((unsigned) ((nTok + tb - 1) / tb), tb) (gpu_in_ptr_, bw_Ecum_, bw_eps_, bw_eta_, bw_Dc_, c0, C, N, d, H, B);
		gla_accum__ CUDA2((unsigned) ((nPack + tb - 1) / tb), tb) (gpu_in_ptr_, gl_L_, bw_dQ_, bw_dQb_, bw_dQi_, bw_dQbi_,
			bw_dK_, bw_dKb_, bw_dKl_, bw_dKlb_, bw_dV_, bw_dVb_, c0, C, N, d, H, B, s);
		// suffix state: G <- exp(LC) G + Qs^T Gg ; Gb <- exp(LC) Gb + conj(Qs)^T Gb_chunk
		gla_scale_state__ CUDA2((unsigned) ((state + tb - 1) / tb), tb) (bw_G_, gl_L_, c0, C, N, dh, (int) BH);
		gla_scale_state__ CUDA2((unsigned) ((state + tb - 1) / tb), tb) (bw_Gbs_, gl_L_, c0, C, N, dh, (int) BH);
		cgemm(hb, true, false, false, false, dh, dh, C, 1.f, chunkP(gl_Qs_, C, dh), chunkP(bw_Gg_, C, dh), 1.f, stateO(bw_G_, dh),   (int) BH);
		cgemm(hb, true, true,  false, false, dh, dh, C, 1.f, chunkP(gl_Qs_, C, dh), chunkP(bw_Gb_, C, dh), 1.f, stateO(bw_Gbs_, dh), (int) BH);
	}
	cublasSetStream(hb, cudaStreamPerThread);
	CNET_KSYNC();
#endif
}
