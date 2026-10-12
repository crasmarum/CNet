#ifndef IMPL_SEGFOURIER_H_
#define IMPL_SEGFOURIER_H_

#include <complex>
#include "cfunc.h"
#include "vars.h"
#include "utils.h"

// Segmented (windowed) Fourier transform = a short-time Fourier transform (STFT)
// expressed as a block-diagonal DFT. The length-N input is split into N/window
// contiguous windows of length `window`; each window is DFT'd independently
// (unitary, 1/sqrt(window) scaling). With a following |.|^2 and mean-pool this
// gives a spectrogram / time-frequency feature -- the time-localized spectral
// structure that global FFT and the cyclic features miss (cf. IQFormer's STFT
// stream, F=window frequency bins).
//
// Output index b*w+p (window b, frequency p):
//   y_{b,p} = (1/sqrt(w)) sum_{q=0}^{w-1} x_{b,q} * root(p*q),  root(k)=e^{i2pi k/w}
// This is a holomorphic linear map y = M x (block-diagonal), so the Wirtinger
// backward mirrors FourierTrans, per window: dL/dx += sum_p dz_p * root(p*q),
// dL/dx~ += sum_p dz~_p * conj(root(p*q)), scaled by 1/sqrt(w).
class SegmentedFourier : public CFunc {
	int window_;
	UnityRoots u_roots_;

public:
	SegmentedFourier(Uid uid, InSize in_size, int window)
		: CFunc(uid, in_size, OutSize(in_size.value())), window_(window), u_roots_(window) {}

	SegmentedFourier(InSize in_size, int window)
		: CFunc(in_size, OutSize(in_size.value())), window_(window), u_roots_(window) {}

	virtual ~SegmentedFourier() {}

	int window() const { return window_; }

	virtual CFunc* clone(Uid uid) {
		return new SegmentedFourier(uid, InSize(input().length_), window_);
	}

	virtual std::string getName() {
		return "SegmentedFourier_" + std::to_string(uid_);
	}

	virtual void forward() {
		const int w = window_, L = input().length_, W = L / w;
		const float scale = std::sqrt((float) w);
		for (int oi = 0; oi < no_outputs(); ++oi) {
			#pragma omp parallel for
			for (int b = 0; b < W; ++b) {
				for (int p = 0; p < w; ++p) {
					std::complex<float> acc(0, 0);
					for (int q = 0; q < w; ++q) {
						std::complex<float> x(input().real_[b * w + q], input().imag_[b * w + q]);
						acc += x * u_roots_.root((p * q) % w);
					}
					acc /= scale;
					output(oi).real_[offset(oi) + b * w + p] = acc.real();
					output(oi).imag_[offset(oi) + b * w + p] = acc.imag();
				}
			}
		}
	}

	virtual void backward() {
		const int w = window_, L = input().length_, W = L / w;
		const float scale = std::sqrt((float) w);
		for (int oi = 0; oi < no_outputs(); ++oi) {
			#pragma omp parallel for
			for (int b = 0; b < W; ++b) {
				for (int q = 0; q < w; ++q) {
					std::complex<float> sum_dz(0, 0), sum_star_dz(0, 0);
					for (int p = 0; p < w; ++p) {
						auto dLdz      = output(oi).dz(offset(oi) + b * w + p);
						auto dLdz_star = output(oi).dz_star(offset(oi) + b * w + p);
						auto r = u_roots_.root((p * q) % w);
						sum_dz      += dLdz * r;
						sum_star_dz += dLdz_star * std::conj(r);
					}
					input().dz_real_[b * w + q]      += sum_dz.real() / scale;
					input().dz_imag_[b * w + q]      += sum_dz.imag() / scale;
					input().dz_star_real_[b * w + q] += sum_star_dz.real() / scale;
					input().dz_star_imag_[b * w + q] += sum_star_dz.imag() / scale;
				}
			}
		}
	}

	virtual void backward(int label) {}

	virtual std::complex<float>      dz(int out_indx, int in_index) { return 0; }
	virtual std::complex<float> dz_star(int out_indx, int in_index) { return 0; }
};

#endif /* IMPL_SEGFOURIER_H_ */
