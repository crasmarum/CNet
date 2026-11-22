#ifndef IMPL_COMPRESSED_L2_H_
#define IMPL_COMPRESSED_L2_H_

#include <cassert>
#include <cmath>
#include <complex>
#include <string>
#include "cfunc.h"
#include "vars.h"

// Compressed spectral loss for enhancement (Braun & Tashev 2021, used by PHASEN / CMGAN / MP-SENet).
//
// Input [Y (M) | T (M)] with the target spectrum S = -T (the trainer carries the clean signal negated so
// that a Residual can also form Y - S). With a = sqrt(|y|^2 + eps), b = sqrt(|s|^2 + eps), 0 < c <= 1:
//
//   u = a^{c-1} y,  v = b^{c-1} s                      (magnitude compressed to a^c, phase kept)
//   L = sum_k  beta |u - v|^2  +  (1 - beta) (a^c - b^c)^2
//
// Wirtinger gradients (dL/dz~ stored in dz_star, dL/dz = conj since L is real), with d = u - v:
//   dL/dy~ = beta [ d (a^{c-1} + (c-1)/2 |y|^2 a^{c-3}) + conj(d) (c-1)/2 y^2 a^{c-3} ]
//          + (1-beta) c (a^c - b^c) a^{c-2} y
// and symmetrically for s (with d -> -d and a <-> b), then dL/dT~ = -dL/ds~. The derivatives are exact for
// the eps-regularised magnitudes (checked against finite differences in examples/test_compressed_l2.cpp).
// The target half gets its gradient too (its path has no parameters), so the layer is symmetric.
class CompressedL2: public CFunc, public OutputFunc {
	float c_, beta_;
	float scale_ = 1.f;        // overall weight of this loss (not stored in model files; set it after restore if needed)
	static constexpr float kEps = 1e-8f;
	Vars output_;

	// gradient dL/dz~ of the pair (y, s) with respect to y (the other element is the target)
	static std::complex<float> grad(std::complex<float> y, std::complex<float> s, float c, float beta) {
		const float y2 = std::norm(y), s2 = std::norm(s);
		const float a = std::sqrt(y2 + kEps), b = std::sqrt(s2 + kEps);
		const float pa = std::pow(a, c - 1.f), pb = std::pow(b, c - 1.f);
		const float pa3 = pa / (a * a);                       // a^{c-3}
		const std::complex<float> u = pa * y, v = pb * s, d = u - v;
		const float k = 0.5f * (c - 1.f);
		std::complex<float> g1 = d * (pa + k * y2 * pa3) + std::conj(d) * (k * pa3) * (y * y);
		std::complex<float> g2 = (c * (pa * a - pb * b) * pa / a) * y;   // c (a^c - b^c) a^{c-2} y
		return beta * g1 + (1.f - beta) * g2;
	}

public:
	CompressedL2(Uid uid, InSize in_size, float c, float beta)
			: CFunc(uid, in_size, OutSize(1)), c_(c), beta_(beta), output_(1) {
		assert(in_size.value() % 2 == 0 && c > 0.f && c <= 1.f && beta >= 0.f && beta <= 1.f);
		is_output_ = true;
	}
	CompressedL2(InSize in_size, float c, float beta)
			: CFunc(in_size, OutSize(1)), c_(c), beta_(beta), output_(1) {
		assert(in_size.value() % 2 == 0 && c > 0.f && c <= 1.f && beta >= 0.f && beta <= 1.f);
		is_output_ = true;
	}
	virtual ~CompressedL2() {
	}

	float c() const { return c_; }
	float beta() const { return beta_; }
	float scale() const { return scale_; }
	void setScale(float s) { scale_ = s; }
	int half() const { return input().length_ / 2; }

	virtual CFunc* clone(Uid uid) override {
		return new CompressedL2(uid, InSize(input().length_), c_, beta_);
	}
	virtual std::string getName() override {
		return "CompressedL2_" + std::to_string(uid_);
	}

	virtual void forward() override {
		const int M = half();
		double acc = 0;
		#pragma omp parallel for reduction(+:acc)
		for (int k = 0; k < M; ++k) {
			const std::complex<float> y = input().z(k), s = -input().z(M + k);
			const float a = std::sqrt(std::norm(y) + kEps), b = std::sqrt(std::norm(s) + kEps);
			const float pa = std::pow(a, c_ - 1.f), pb = std::pow(b, c_ - 1.f);
			const std::complex<float> d = pa * y - pb * s;
			const float dm = pa * a - pb * b;
			acc += beta_ * std::norm(d) + (1.f - beta_) * dm * dm;
		}
		output_.real_[0] = scale_ * (float) acc;
		output_.imag_[0] = 0.f;
	}

	virtual void backward() override {
		const int M = half();
		#pragma omp parallel for
		for (int k = 0; k < M; ++k) {
			const std::complex<float> y = input().z(k), s = -input().z(M + k);
			const std::complex<float> gy = scale_ * grad(y, s, c_, beta_);
			const std::complex<float> gt = -scale_ * grad(s, y, c_, beta_);    // s = -T
			input().dz_star_real_[k]     += gy.real(); input().dz_star_imag_[k]     += gy.imag();
			input().dz_real_[k]          += gy.real(); input().dz_imag_[k]          -= gy.imag();
			input().dz_star_real_[M + k] += gt.real(); input().dz_star_imag_[M + k] += gt.imag();
			input().dz_real_[M + k]      += gt.real(); input().dz_imag_[M + k]      -= gt.imag();
		}
	}
	virtual void backward(int label) override {
		backward();
	}

	virtual std::complex<float> dz(int out_indx, int in_index) override {
		return std::conj(dz_star(out_indx, in_index));
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) override {
		const int M = half();
		const std::complex<float> y = input().z(in_index % M), s = -input().z(M + in_index % M);
		return scale_ * (in_index < M ? grad(y, s, c_, beta_) : -grad(s, y, c_, beta_));
	}

	virtual int sizeInBytes() const override {
		return input().size_in_bytes() + output_.size_in_bytes();
	}
	virtual int outputLength() override {
		return output_.length_;
	}
	virtual Vars* mutableOutput() override {
		return &output_;
	}
	float loss() {
		return output_.real_[0];
	}
	virtual float loss(int label) override {
		return output_.real_[0];
	}
};

#endif /* IMPL_COMPRESSED_L2_H_ */
