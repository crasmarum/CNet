#ifndef IMPL_CHIRP_H_
#define IMPL_CHIRP_H_

#include <cmath>
#include <complex>
#include <vector>

#include "cfunc.h"
#include "vars.h"

// Radix-2 in-place FFT on complex<double>, unnormalised forward (sign -1) or inverse (sign +1, no 1/N).
inline void chirp_fft(std::vector<std::complex<double>> &a, int sign) {
	int n = (int) a.size();
	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) std::swap(a[i], a[j]);
	}
	for (int len = 2; len <= n; len <<= 1) {
		double ang = sign * 2 * M_PI / len;
		std::complex<double> wl(cos(ang), sin(ang));
		for (int i = 0; i < n; i += len) {
			std::complex<double> w(1);
			for (int j = 0; j < len / 2; ++j) {
				auto u = a[i + j], v = a[i + j + len / 2] * w;
				a[i + j] = u + v;
				a[i + j + len / 2] = u - v;
				w *= wl;
			}
		}
	}
}

// Learnable coherent dedispersion of one channel of complex baseband (CHIME convention).
//
// Input  [x (L) | f_c (1) | d (K)]:  x = complex voltages at DT seconds per sample, f_c = channel centre frequency in
//        MHz (real part; a per-example input, never trained), d = K filter parameters, DM_k = dm_scale * Re(d_k).
// Output [y_0 (L) | ... | y_{K-1} (L)],  y_k = IFFT(D_k . FFT(x)),  D_k(f) = exp(i phi(f) DM_k),
//        phi(f) = 2 pi KDM f^2 / (f_c^2 (f_c + f)),  f = FFT bin frequency (Hz) of complex baseband sampled at DT.
//
// The sign dedisperses CHIME baseband: CHIME channels have the reverse frequency orientation to the textbook
// form, so this reproduces the archive's coherently dedispersed power (rf_product/chime/check_convention.py).
// y is linear (holomorphic) in x; DM_k is a REAL parameter (only Re(d_k) enters), so d_k's imaginary part gets no
// gradient. Wirtinger backward, with g = dL/dy, h = dL/dy*:
//   dL/dx   += sum_k FFT(D_k . IFFT(g_k))          (the transpose of IFFT D FFT)
//   dL/dx*  += sum_k IFFT(conj(D_k) . FFT(h_k))    (the adjoint)
//   dL/dd_k  = dL/dd*_k = (S1 + conj(S2)) / 2,   S1 = sum_f IFFT(g_k)_f V_f,  S2 = sum_f IFFT(conj h_k)_f V_f,
//              V = i phi D_k X dm_scale,  X = FFT(x)                (with h = conj g this is Re(S1))
class ChirpFilter: public CFunc {
	int len_, nk_;
	float dm_scale_;

public:
	static constexpr double KDM = 4.148808e15;   // s Hz^2 / (pc cm^-3)
	static constexpr double DT = 2.56e-6;        // s per sample (CHIME 390.625 kHz channels)

	ChirpFilter(Uid uid, int L, int K, float dm_scale = 100.f)
			: CFunc(uid, InSize(L + 1 + K), OutSize(K * L)), len_(L), nk_(K), dm_scale_(dm_scale) {
		assert(L > 0 && (L & (L - 1)) == 0);
	}
	ChirpFilter(int L, int K, float dm_scale = 100.f)
			: CFunc(InSize(L + 1 + K), OutSize(K * L)), len_(L), nk_(K), dm_scale_(dm_scale) {
		assert(L > 0 && (L & (L - 1)) == 0);
	}
	virtual ~ChirpFilter() {}

	int L() const { return len_; }
	int K() const { return nk_; }
	float dmScale() const { return dm_scale_; }

	virtual CFunc* clone(Uid uid) { return new ChirpFilter(uid, len_, nk_, dm_scale_); }
	virtual std::string getName() { return "ChirpFilter_" + std::to_string(uid_); }

	// phi(f) for FFT bin j, channel centre fc_mhz.
	static double phase(int j, int L, double fc_mhz) {
		double f = (j < (L + 1) / 2 ? j : j - L) / (L * DT);
		double fc = fc_mhz * 1e6;
		return 2 * M_PI * KDM * f * f / (fc * fc * (fc + f));
	}

	double fcMHz() const { return input().real_[len_]; }
	double dm(int k) const { return dm_scale_ * input().real_[len_ + 1 + k]; }

	std::vector<std::complex<double>> spectrum() const {
		std::vector<std::complex<double>> X(len_);
		for (int t = 0; t < len_; ++t) X[t] = std::complex<double>(input().real_[t], input().imag_[t]);
		chirp_fft(X, -1);
		return X;
	}

	virtual void forward() {
		assert(no_outputs());
		auto X = spectrum();
		double fc = fcMHz();
		for (int k = 0; k < nk_; ++k) {
			std::vector<std::complex<double>> Y(len_);
			double d = dm(k);
			for (int j = 0; j < len_; ++j) Y[j] = X[j] * std::polar(1.0, phase(j, len_, fc) * d);
			chirp_fft(Y, +1);
			for (int o = 0; o < no_outputs(); ++o)
				for (int t = 0; t < len_; ++t) {
					output(o).real_[offset(o) + k * len_ + t] = (float) (Y[t].real() / len_);
					output(o).imag_[offset(o) + k * len_ + t] = (float) (Y[t].imag() / len_);
				}
		}
	}

	virtual void backward() {
		auto X = spectrum();
		double fc = fcMHz();
		std::vector<std::complex<double>> gx(len_, 0.0), hx(len_, 0.0);
		std::vector<std::complex<double>> gd(nk_, 0.0);
		for (int o = 0; o < no_outputs(); ++o) {
			for (int k = 0; k < nk_; ++k) {
				double d = dm(k);
				std::vector<std::complex<double>> g(len_), hc(len_), h(len_);
				for (int t = 0; t < len_; ++t) {
					g[t] = output(o).dz(offset(o) + k * len_ + t);
					h[t] = output(o).dz_star(offset(o) + k * len_ + t);
					hc[t] = std::conj(h[t]);
				}
				// IFFT (with 1/L) of g and of conj(h)
				chirp_fft(g, +1); chirp_fft(hc, +1);
				std::complex<double> S1 = 0, S2 = 0;
				std::vector<std::complex<double>> a(len_), b(len_);
				for (int j = 0; j < len_; ++j) {
					double ph = phase(j, len_, fc);
					auto D = std::polar(1.0, ph * d);
					g[j] /= len_; hc[j] /= len_;
					auto V = std::complex<double>(0, ph * dm_scale_) * D * X[j];
					S1 += g[j] * V;
					S2 += hc[j] * V;
					a[j] = D * g[j];                      // FFT(D . IFFT(g))      -> dL/dx
					b[j] = std::conj(D);                  // IFFT(conj(D) . FFT(h)) -> dL/dx*
				}
				chirp_fft(a, -1);
				chirp_fft(h, -1);
				for (int j = 0; j < len_; ++j) h[j] *= b[j];
				chirp_fft(h, +1);
				for (int t = 0; t < len_; ++t) { gx[t] += a[t]; hx[t] += h[t] / (double) len_; }
				gd[k] += (S1 + std::conj(S2)) * 0.5;
			}
		}
		for (int t = 0; t < len_; ++t) {
			input().dz_real_[t] += (float) gx[t].real();
			input().dz_imag_[t] += (float) gx[t].imag();
			input().dz_star_real_[t] += (float) hx[t].real();
			input().dz_star_imag_[t] += (float) hx[t].imag();
		}
		for (int k = 0; k < nk_; ++k) {
			int i = len_ + 1 + k;
			input().dz_real_[i] += (float) gd[k].real();
			input().dz_imag_[i] += (float) gd[k].imag();
			input().dz_star_real_[i] += (float) gd[k].real();
			input().dz_star_imag_[i] += (float) gd[k].imag();
		}
	}

	virtual void backward(int label) {}
	virtual std::complex<float> dz(int, int) { return 0; }
	virtual std::complex<float> dz_star(int, int) { return 0; }
};

// Boxcar search: for each of K rows of length L (real power, e.g. |y|^2 of ChirpFilter output, unit-mean noise) and
// each width w, out[k * W + iw] = max over start positions s (stride max(w/4, 1)) of (sum_{s..s+w-1} p - w) / sqrt(w).
// The input is treated as a holomorphic linear sum at the arg-max window (as MeanPool), so dL/dz and dL/dz* of
// each sample in that window are the output's gradients divided by sqrt(w); elsewhere zero.
class BoxcarMax: public CFunc {
	int nk_, len_;
	std::vector<int> widths_;
	std::vector<int> argmax_;   // start of the winning window per output, from the last forward()

public:
	BoxcarMax(Uid uid, int K, int L, std::vector<int> widths)
			: CFunc(uid, InSize(K * L), OutSize(K * (int) widths.size())), nk_(K), len_(L), widths_(widths),
			  argmax_(K * widths.size(), 0) {}
	BoxcarMax(int K, int L, std::vector<int> widths)
			: CFunc(InSize(K * L), OutSize(K * (int) widths.size())), nk_(K), len_(L), widths_(widths),
			  argmax_(K * widths.size(), 0) {}
	virtual ~BoxcarMax() {}

	int K() const { return nk_; }
	int L() const { return len_; }
	const std::vector<int>& widths() const { return widths_; }
	static int stride(int w) { return w / 4 > 1 ? w / 4 : 1; }

	virtual CFunc* clone(Uid uid) { return new BoxcarMax(uid, nk_, len_, widths_); }
	virtual std::string getName() { return "BoxcarMax_" + std::to_string(uid_); }

	virtual void forward() {
		assert(no_outputs());
		int W = (int) widths_.size();
		std::vector<double> c(len_ + 1);
		for (int k = 0; k < nk_; ++k) {
			c[0] = 0;
			for (int t = 0; t < len_; ++t) c[t + 1] = c[t] + input().real_[k * len_ + t];
			for (int iw = 0; iw < W; ++iw) {
				int w = widths_[iw], st = stride(w);
				double best = -1e300; int arg = 0;
				for (int s = 0; s + w <= len_; s += st) {
					double v = c[s + w] - c[s];
					if (v > best) { best = v; arg = s; }
				}
				argmax_[k * W + iw] = arg;
				float val = (float) ((best - w) / sqrt((double) w));
				for (int o = 0; o < no_outputs(); ++o) {
					output(o).real_[offset(o) + k * W + iw] = val;
					output(o).imag_[offset(o) + k * W + iw] = 0;
				}
			}
		}
	}

	virtual void backward() {
		int W = (int) widths_.size();
		for (int o = 0; o < no_outputs(); ++o)
			for (int k = 0; k < nk_; ++k)
				for (int iw = 0; iw < W; ++iw) {
					int w = widths_[iw], s = argmax_[k * W + iw];
					float sc = 1.f / sqrtf((float) w);
					auto dz = output(o).dz(offset(o) + k * W + iw) * sc;
					auto dzs = output(o).dz_star(offset(o) + k * W + iw) * sc;
					for (int t = s; t < s + w; ++t) {
						int i = k * len_ + t;
						input().dz_real_[i] += dz.real();
						input().dz_imag_[i] += dz.imag();
						input().dz_star_real_[i] += dzs.real();
						input().dz_star_imag_[i] += dzs.imag();
					}
				}
	}

	virtual void backward(int label) {}
	virtual std::complex<float> dz(int, int) { return 0; }
	virtual std::complex<float> dz_star(int, int) { return 0; }
};

#endif /* IMPL_CHIRP_H_ */
