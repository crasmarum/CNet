#ifndef IMPL_MELPOWER_H_
#define IMPL_MELPOWER_H_
#include <cassert>
#include <complex>
#include <cstdint>
#include <string>

#include <cmath>
#include <vector>

#include "cfunc.h"
#include "vars.h"
#include "relu.h"   // kLogPowerEps

// Fixed log-mel front end on the two-sided STFT of every token (N tokens x W bins, token-major, as
// SegmentedFourier writes it): for token t and band m
//   y[t, m] = (log(eps + sum_k H[m, k] |X[t, k]|^2) - mean) / scale,   imag 0,
// with H the M triangular filters on the one-sided bins k = 0..W/2 (f_k = k fs / W), equally spaced on
// the HTK mel scale between fmin and fmax, unit peak. The mirror bins k > W/2 of a real signal are not
// read (zero gradient). mean / scale standardise the features (setStandardize; stored in model files;
// default 0 / 1). Wirtinger gradients as in CLogPower, summed over the bands that read a bin:
//   dL/dz_k = s_k conj(z_k),  dL/dz~_k = s_k z_k,  s_k = sum_m H[m, k] (dLdz_m + dLdz~_m) / (scale (eps + P_m)),
// with 1 / (eps + P_m) recovered from the output as exp(-(y_m scale + mean)).
// The GPU mapping copies the filters of the first layer of its group (use one MelPower per graph, or
// give several the same W / M / fs / fmin / fmax).
class MelPower: public CFunc {
	int N_, W_, M_, K_;
	float fs_, fmin_, fmax_;
	bool linear_ = false;           // identity filters: standardised one-sided log power, M == K (no mel pooling)
	float mean_ = 0.f, scale_ = 1.f;
	std::vector<float> H_;          // M x K
	std::vector<int> lo_, hi_;      // first / last bin read by each band (inclusive)
	std::vector<int> mlo_, mhi_;    // first / last band reading each bin (mhi < 0: bin not read)

	static float hzToMel(float f) { return 2595.f * log10f(1.f + f / 700.f); }
	static float melToHz(float m) { return 700.f * (powf(10.f, m / 2595.f) - 1.f); }

	void buildFilters() {
		K_ = W_ / 2 + 1;
		H_.assign((size_t) M_ * K_, 0.f);
		lo_.assign(M_, 0);
		hi_.assign(M_, -1);
		mlo_.assign(K_, M_);
		mhi_.assign(K_, -1);
		if (linear_) {   // one band per one-sided bin
			assert(M_ == K_);
			for (int k = 0; k < K_; ++k) {
				H_[(size_t) k * K_ + k] = 1.f;
				lo_[k] = hi_[k] = k;
				mlo_[k] = mhi_[k] = k;
			}
			return;
		}
		const float ml = hzToMel(fmin_), mh = hzToMel(fmax_);
		for (int m = 0; m < M_; ++m) {
			const float fl = melToHz(ml + (mh - ml) * m / (M_ + 1));
			const float fc = melToHz(ml + (mh - ml) * (m + 1) / (M_ + 1));
			const float fh = melToHz(ml + (mh - ml) * (m + 2) / (M_ + 1));
			bool any = false;
			for (int k = 0; k < K_; ++k) {
				const float f = k * fs_ / W_;
				float h = 0.f;
				if (f > fl && f < fc) h = (f - fl) / (fc - fl);
				else if (f >= fc && f < fh) h = (fh - f) / (fh - fc);
				if (h > 0.f) {
					H_[(size_t) m * K_ + k] = h;
					if (!any) { lo_[m] = k; any = true; }
					hi_[m] = k;
				}
			}
			if (!any) {   // band narrower than a bin: the nearest bin to its centre, weight 1
				int k = (int) lroundf(fc * W_ / fs_);
				if (k >= K_) k = K_ - 1;
				if (k < 0) k = 0;
				H_[(size_t) m * K_ + k] = 1.f;
				lo_[m] = hi_[m] = k;
			}
			for (int k = lo_[m]; k <= hi_[m]; ++k) {
				if (m < mlo_[k]) mlo_[k] = m;
				if (m > mhi_[k]) mhi_[k] = m;
			}
		}
	}

public:
	// linear = true: identity filters over the W/2 + 1 one-sided bins (M must be W/2 + 1; fmin/fmax unused).
	MelPower(Uid uid, int N, int W, int M, float fs = 16000.f, float fmin = 20.f, float fmax = 8000.f, bool linear = false)
			: CFunc(uid, InSize(N * W), OutSize(N * M)), N_(N), W_(W), M_(M), fs_(fs), fmin_(fmin), fmax_(fmax), linear_(linear) {
		assert(N > 0 && W > 1 && M > 0 && fmin >= 0.f && fmax > fmin && fmax <= fs / 2.f);
		buildFilters();
	}

	MelPower(int N, int W, int M, float fs = 16000.f, float fmin = 20.f, float fmax = 8000.f, bool linear = false)
			: CFunc(InSize(N * W), OutSize(N * M)), N_(N), W_(W), M_(M), fs_(fs), fmin_(fmin), fmax_(fmax), linear_(linear) {
		assert(N > 0 && W > 1 && M > 0 && fmin >= 0.f && fmax > fmin && fmax <= fs / 2.f);
		buildFilters();
	}

	virtual ~MelPower() {
	}

	virtual CFunc* clone(Uid uid) {
		MelPower *c = new MelPower(uid, N_, W_, M_, fs_, fmin_, fmax_, linear_);
		c->setStandardize(mean_, scale_);
		return c;
	}

	bool linear() const { return linear_; }

	virtual std::string getName() {
		return "MelPower_" + std::to_string(uid_);
	}

	void setStandardize(float mean, float scale) {
		assert(scale > 0.f);
		mean_ = mean;
		scale_ = scale;
	}
	float mean() const { return mean_; }
	float scale() const { return scale_; }
	int tokens() const { return N_; }
	int window() const { return W_; }
	int bands() const { return M_; }
	int bins() const { return K_; }
	float sampleRate() const { return fs_; }
	float fmin() const { return fmin_; }
	float fmax() const { return fmax_; }
	const std::vector<float>& filters() const { return H_; }
	const std::vector<int>& bandLo() const { return lo_; }
	const std::vector<int>& bandHi() const { return hi_; }
	const std::vector<int>& binLo() const { return mlo_; }
	const std::vector<int>& binHi() const { return mhi_; }

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int o = 0; o < out_size_; ++o) {
				const int t = o / M_, m = o % M_;
				float P = 0.f;
				for (int k = lo_[m]; k <= hi_[m]; ++k) {
					const float re = input().real_[t * W_ + k], im = input().imag_[t * W_ + k];
					P += H_[(size_t) m * K_ + k] * (re * re + im * im);
				}
				output(indx).real_[offset(indx) + o] = (logf(kLogPowerEps + P) - mean_) / scale_;
				output(indx).imag_[offset(indx) + o] = 0.f;
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < N_ * K_; ++i) {
				const int t = i / K_, k = i % K_;
				if (mhi_[k] < 0) continue;
				std::complex<float> s(0.f, 0.f);
				for (int m = mlo_[k]; m <= mhi_[k]; ++m) {
					const float h = H_[(size_t) m * K_ + k];
					if (h == 0.f) continue;
					const int o = offset(indx) + t * M_ + m;
					const float inv = expf(-(output(indx).real_[o] * scale_ + mean_));   // 1 / (eps + P_m)
					s += (output(indx).dz(o) + output(indx).dz_star(o)) * (h * inv / scale_);
				}
				const int in = t * W_ + k;
				const std::complex<float> z(input().real_[in], input().imag_[in]);
				const auto dz = s * std::conj(z), dzs = s * z;
				input().dz_real_[in]      += dz.real();
				input().dz_imag_[in]      += dz.imag();
				input().dz_star_real_[in] += dzs.real();
				input().dz_star_imag_[in] += dzs.imag();
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

#endif /* IMPL_MELPOWER_H_ */
