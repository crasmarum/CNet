#ifndef IMPL_TFCONV_H_
#define IMPL_TFCONV_H_
#include <cstdint>

#include <cassert>
#include <complex>
#include <string>
#include <vector>

#include "cfunc.h"
#include "vars.h"

// TimeFreqConv: a small causal convolution over the time-frequency plane with complex taps shared by every
// position -- the "shared detector of short spectral transitions" in front of the first token-mixing block.
//
// The data input holds N tokens (frames) of B bins with C_in channels per bin, channel fastest:
//     data(t, b, c) = z(t B C_in + b C_in + c).
// The taps follow the data in the same input vector (like the weight of TokenwiseLinear):
//     tap(o, c, tau, j) = z(N B C_in + ((o C_in + c) T + tau) F + j),   o < C_out, tau < T, j < F.
// The output has N tokens of B_out = ceil(B / S) bins with C_out channels, channel fastest:
//     out(t, b', o) = sum_{c, tau, j} tap(o, c, tau, j) * data(t - tau, b' S + j - F/2, c),
// with data = 0 for t - tau < 0 (causal zero padding: frame t sees frames t-T+1 .. t only) and for bins outside
// [0, B) (zero padding in frequency); S is the frequency stride (S = 1 keeps every bin).
//
// The map is linear in the data and in the taps (holomorphic in both), so the Wirtinger backward mirrors
// TokenwiseLinear: dz of a data element += dLo_z * tap, dz~ += dLo_z~ * conj(tap); dz of a tap += dLo_z * data,
// dz~ += dLo_z~ * conj(data), summed over every output that uses it.
//
// Streaming: frame t depends on the last T frames of its input, so the N = 1 twin keeps a ring of T input frames
// (impl/framenet.h); with T = 1 the layer is frequency-only and needs no state.
class TimeFreqConv: public CFunc {
	friend class ModelSaver;
	int n_tokens_, bins_, c_in_, c_out_, t_taps_, f_taps_, stride_;

	static int outBinsOf(int bins, int stride) { return stride > 0 ? (bins + stride - 1) / stride : 0; }   // 0: caught by the assert
	static int totalIn(int n, int b, int ci, int co, int t, int f) { return n * b * ci + co * ci * t * f; }
	static int totalOut(int n, int b, int co, int s) { return n * outBinsOf(b, s) * co; }

public:
	TimeFreqConv(Uid uid, int n_tokens, int bins, int c_in, int c_out, int t_taps, int f_taps, int stride)
			: CFunc(uid, InSize(totalIn(n_tokens, bins, c_in, c_out, t_taps, f_taps)),
					OutSize(totalOut(n_tokens, bins, c_out, stride))),
			  n_tokens_(n_tokens), bins_(bins), c_in_(c_in), c_out_(c_out), t_taps_(t_taps), f_taps_(f_taps), stride_(stride) {
		assert(n_tokens > 0 && bins > 0 && c_in > 0 && c_out > 0 && t_taps > 0 && f_taps > 0 && stride > 0);
	}
	TimeFreqConv(int n_tokens, int bins, int c_in, int c_out, int t_taps, int f_taps, int stride)
			: CFunc(InSize(totalIn(n_tokens, bins, c_in, c_out, t_taps, f_taps)),
					OutSize(totalOut(n_tokens, bins, c_out, stride))),
			  n_tokens_(n_tokens), bins_(bins), c_in_(c_in), c_out_(c_out), t_taps_(t_taps), f_taps_(f_taps), stride_(stride) {
		assert(n_tokens > 0 && bins > 0 && c_in > 0 && c_out > 0 && t_taps > 0 && f_taps > 0 && stride > 0);
	}
	virtual ~TimeFreqConv() {
	}
	virtual CFunc* clone(Uid uid) {
		return new TimeFreqConv(uid, n_tokens_, bins_, c_in_, c_out_, t_taps_, f_taps_, stride_);
	}
	virtual std::string getName() {
		return "TimeFreqConv_" + std::to_string(uid_);
	}

	int nTokens() const { return n_tokens_; }
	int bins()    const { return bins_; }
	int cIn()     const { return c_in_; }
	int cOut()    const { return c_out_; }
	int tTaps()   const { return t_taps_; }
	int fTaps()   const { return f_taps_; }
	int stride()  const { return stride_; }
	int outBins() const { return outBinsOf(bins_, stride_); }
	int nTaps()   const { return c_out_ * c_in_ * t_taps_ * f_taps_; }
	int tapBase() const { return n_tokens_ * bins_ * c_in_; }   // index of the first tap in the input vector
	int dataIndex(int t, int b, int c) const { return (t * bins_ + b) * c_in_ + c; }
	int tapIndex(int o, int c, int tau, int j) const { return tapBase() + ((o * c_in_ + c) * t_taps_ + tau) * f_taps_ + j; }
	int outIndex(int t, int bo, int o) const { return (t * outBins() + bo) * c_out_ + o; }

	virtual void forward() {
		assert(no_outputs());
		const int N = n_tokens_, B = bins_, Ci = c_in_, Co = c_out_, T = t_taps_, F = f_taps_, S = stride_, Bo = outBins();
		const int total = N * Bo * Co;
		std::vector<std::complex<float> > sums((size_t) total);
		#pragma omp parallel for
		for (int oi = 0; oi < total; ++oi) {
			const int o = oi % Co, bo = (oi / Co) % Bo, t = oi / (Co * Bo);
			std::complex<float> acc(0.f, 0.f);
			for (int tau = 0; tau < T; ++tau) {
				const int ts = t - tau;
				if (ts < 0) continue;
				for (int j = 0; j < F; ++j) {
					const int b = bo * S + j - F / 2;
					if (b < 0 || b >= B) continue;
					for (int c = 0; c < Ci; ++c) acc += input().z(tapIndex(o, c, tau, j)) * input().z(dataIndex(ts, b, c));
				}
			}
			sums[oi] = acc;
		}
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int oi = 0; oi < total; ++oi) {
				output(indx).real_[offset(indx) + oi] = sums[oi].real();
				output(indx).imag_[offset(indx) + oi] = sums[oi].imag();
			}
		}
	}

	virtual void backward() {
		const int N = n_tokens_, B = bins_, Ci = c_in_, Co = c_out_, T = t_taps_, F = f_taps_, S = stride_, Bo = outBins();
		for (int indx = 0; indx < no_outputs(); ++indx) {
			for (int t = 0; t < N; ++t) {
				for (int bo = 0; bo < Bo; ++bo) {
					for (int o = 0; o < Co; ++o) {
						const int oi = outIndex(t, bo, o);
						const auto dLdz = output(indx).dz(offset(indx) + oi);
						const auto dLdz_star = output(indx).dz_star(offset(indx) + oi);
						for (int tau = 0; tau < T; ++tau) {
							const int ts = t - tau;
							if (ts < 0) continue;
							for (int j = 0; j < F; ++j) {
								const int b = bo * S + j - F / 2;
								if (b < 0 || b >= B) continue;
								for (int c = 0; c < Ci; ++c) {
									const int di = dataIndex(ts, b, c), wi = tapIndex(o, c, tau, j);
									const auto w = input().z(wi), d = input().z(di);
									auto dz = dLdz * w, dz_star = dLdz_star * std::conj(w);   // data
									input().dz_real_[di]      += dz.real();
									input().dz_imag_[di]      += dz.imag();
									input().dz_star_real_[di] += dz_star.real();
									input().dz_star_imag_[di] += dz_star.imag();
									dz = dLdz * d; dz_star = dLdz_star * std::conj(d);         // tap (shared: summed)
									input().dz_real_[wi]      += dz.real();
									input().dz_imag_[wi]      += dz.imag();
									input().dz_star_real_[wi] += dz_star.real();
									input().dz_star_imag_[wi] += dz_star.imag();
								}
							}
						}
					}
				}
			}
		}
	}

	virtual void backward(int label) {
	}
	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}
};

#endif /* IMPL_TFCONV_H_ */
