#ifndef IMPL_HADAMARDCONV_H_
#define IMPL_HADAMARDCONV_H_

#include <complex>
#include <vector>
#include "cfunc.h"
#include "vars.h"
#include "utils.h"

// Multi-plane complex spectral convolution (the Hadamard-conv layer of
// arXiv:1810.11650), done in one layer so a C_in -> F conv does not explode the
// graph into C_in*F FourierTrans/Hadamard sub-nodes.
//
// The input is already in the frequency domain (the net applies one FourierTrans
// up front, and the L2 activation A(z)=z/||z|| == SoftMax commutes with the DFT,
// so every layer stays spectral). Each output filter f is a localized kernel of
// K free complex taps per input plane, scattered into the length-D plane at the
// fixed positions kernel_index[.] and DFT'd -- equivalently, summed directly:
//
//   W_{f,p}[k] = sum_{j=0}^{K-1} taps[f,p,j] * root(index[j]*k),   root(m)=e^{i2pi m/D}
//   out[f,k]   = sum_{p=0}^{C-1} X[p,k] * W_{f,p}[k]  (+ D*bias[f] at k==0 only)
//
// so out[f,k] = sum_{p,j} taps[f,p,j] * X[p,k] * root(index[j]*k) (+ DC bias).
// The map is holomorphic in BOTH X and the parameters (taps, bias) -- the roots
// are constants -- so dOut/d(.)* = 0 and the Wirtinger backward is exact:
//   dL/dX[p,k]      : dz += sum_f dLo_z[f,k]  * W_{f,p}[k]
//                     dz~ += sum_f dLo_z~[f,k] * conj(W_{f,p}[k])
//   dL/dtaps[f,p,j] : dz += sum_k dLo_z[f,k]  * X[p,k]*root(index[j]*k)
//                     dz~ += sum_k dLo_z~[f,k] * conj(X[p,k]*root(index[j]*k))
//   dL/dbias[f]     : dz += D*dLo_z[f,0] ;  dz~ += D*dLo_z~[f,0]
// (verified against central finite differences, see test_hadamardconv_fd.cpp).
//
// Input layout (concatenated, as CNet passes a layer its parents in order):
//   [ X : C*D ] [ taps : F*C*K ] [ bias : F ]      Output: [ F*D ].
// Pass X as the data parent and a single CInput(F*C*K + F) as the learnable
// parameter parent. kernel_index (K positions in [0,D)) is a fixed structural
// parameter of the layer (K==D with index 0..D-1 gives a full/global filter).
class HadamardConv : public CFunc {
	int in_planes_;   // C
	int plane_size_;  // D
	int filters_;     // F
	std::vector<int> kernel_index_;  // K tap positions in [0, D)
	UnityRoots u_roots_;

	static int inLen(int C, int D, int F, int K) { return C * D + F * C * K + F; }

public:
	HadamardConv(Uid uid, int in_planes, int plane_size, int filters, std::vector<int> kernel_index)
		: CFunc(uid, InSize(inLen(in_planes, plane_size, filters, (int) kernel_index.size())),
				OutSize(filters * plane_size)),
		  in_planes_(in_planes), plane_size_(plane_size), filters_(filters),
		  kernel_index_(kernel_index), u_roots_(plane_size) {}

	HadamardConv(int in_planes, int plane_size, int filters, std::vector<int> kernel_index)
		: CFunc(InSize(inLen(in_planes, plane_size, filters, (int) kernel_index.size())),
				OutSize(filters * plane_size)),
		  in_planes_(in_planes), plane_size_(plane_size), filters_(filters),
		  kernel_index_(kernel_index), u_roots_(plane_size) {}

	virtual ~HadamardConv() {}

	int inPlanes()  const { return in_planes_; }
	int planeSize() const { return plane_size_; }
	int filters()   const { return filters_; }
	int taps()      const { return (int) kernel_index_.size(); }
	const std::vector<int>& kernelIndex() const { return kernel_index_; }

	virtual CFunc* clone(Uid uid) {
		return new HadamardConv(uid, in_planes_, plane_size_, filters_, kernel_index_);
	}

	virtual std::string getName() {
		return "HadamardConv_" + std::to_string(uid_);
	}

	// base offsets into the concatenated input
	int tapsBase() const { return in_planes_ * plane_size_; }
	int biasBase() const { return in_planes_ * plane_size_ + filters_ * in_planes_ * (int) kernel_index_.size(); }

	virtual void forward() {
		const int C = in_planes_, D = plane_size_, F = filters_, K = (int) kernel_index_.size();
		const int tb = tapsBase(), bb = biasBase();
		for (int oi = 0; oi < no_outputs(); ++oi) {
			#pragma omp parallel for
			for (int f = 0; f < F; ++f) {
				for (int k = 0; k < D; ++k) {
					std::complex<float> acc(0, 0);
					for (int p = 0; p < C; ++p) {
						std::complex<float> W(0, 0);
						for (int j = 0; j < K; ++j) {
							W += input().z(tb + f * C * K + p * K + j)
							     * u_roots_.root((kernel_index_[j] * k) % D);
						}
						acc += input().z(p * D + k) * W;
					}
					if (k == 0) {
						acc += (float) D * input().z(bb + f);
					}
					output(oi).real_[offset(oi) + f * D + k] = acc.real();
					output(oi).imag_[offset(oi) + f * D + k] = acc.imag();
				}
			}
		}
	}

	virtual void backward() {
		const int C = in_planes_, D = plane_size_, F = filters_, K = (int) kernel_index_.size();
		const int tb = tapsBase(), bb = biasBase();
		for (int oi = 0; oi < no_outputs(); ++oi) {
			// dL/dX[p,k]
			#pragma omp parallel for
			for (int p = 0; p < C; ++p) {
				for (int k = 0; k < D; ++k) {
					std::complex<float> sdz(0, 0), sdzs(0, 0);
					for (int f = 0; f < F; ++f) {
						std::complex<float> W(0, 0);
						for (int j = 0; j < K; ++j) {
							W += input().z(tb + f * C * K + p * K + j)
							     * u_roots_.root((kernel_index_[j] * k) % D);
						}
						sdz  += output(oi).dz(offset(oi) + f * D + k) * W;
						sdzs += output(oi).dz_star(offset(oi) + f * D + k) * std::conj(W);
					}
					input().dz_real_[p * D + k]      += sdz.real();
					input().dz_imag_[p * D + k]      += sdz.imag();
					input().dz_star_real_[p * D + k] += sdzs.real();
					input().dz_star_imag_[p * D + k] += sdzs.imag();
				}
			}
			// dL/dtaps[f,p,j]
			#pragma omp parallel for
			for (int f = 0; f < F; ++f) {
				for (int p = 0; p < C; ++p) {
					for (int j = 0; j < K; ++j) {
						std::complex<float> sdz(0, 0), sdzs(0, 0);
						for (int k = 0; k < D; ++k) {
							std::complex<float> xr = input().z(p * D + k)
							                         * u_roots_.root((kernel_index_[j] * k) % D);
							sdz  += output(oi).dz(offset(oi) + f * D + k) * xr;
							sdzs += output(oi).dz_star(offset(oi) + f * D + k) * std::conj(xr);
						}
						int t = tb + f * C * K + p * K + j;
						input().dz_real_[t]      += sdz.real();
						input().dz_imag_[t]      += sdz.imag();
						input().dz_star_real_[t] += sdzs.real();
						input().dz_star_imag_[t] += sdzs.imag();
					}
				}
			}
			// dL/dbias[f]  (DC term only)
			#pragma omp parallel for
			for (int f = 0; f < F; ++f) {
				auto dz  = output(oi).dz(offset(oi) + f * D + 0);
				auto dzs = output(oi).dz_star(offset(oi) + f * D + 0);
				input().dz_real_[bb + f]      += (float) D * dz.real();
				input().dz_imag_[bb + f]      += (float) D * dz.imag();
				input().dz_star_real_[bb + f] += (float) D * dzs.real();
				input().dz_star_imag_[bb + f] += (float) D * dzs.imag();
			}
		}
	}

	virtual void backward(int label) {}

	virtual std::complex<float>      dz(int out_indx, int in_index) { return 0; }
	virtual std::complex<float> dz_star(int out_indx, int in_index) { return 0; }
};

#endif /* IMPL_HADAMARDCONV_H_ */
