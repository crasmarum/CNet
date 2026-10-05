#ifndef IMPL_RELU_H_
#define IMPL_RELU_H_

#include <cfloat>

#include "cfunc.h"
#include "vars.h"
#include "utils.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

class CRelu: public CFunc {

public:
	CRelu(Uid uid, InSize in_size) : CFunc(uid, in_size, OutSize(in_size.value())) {
	}

	CRelu(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
	}

	virtual ~CRelu() {
	}

	virtual CFunc* clone(Uid uid) {
		return new CRelu(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "CRelu_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				if (input().real_[out_ind] > 0 && input().imag_[out_ind] > 0) {
					output(indx).real_[offset(indx) + out_ind] = input().real_[out_ind];
					output(indx).imag_[offset(indx) + out_ind] = input().imag_[out_ind];
				} else {
					output(indx).real_[offset(indx) + out_ind] = 0;
					output(indx).imag_[offset(indx) + out_ind] = 0;
				}
			}
		}
}

	virtual void backward() {
		for (int out_indx = 0; out_indx < no_outputs(); ++out_indx) {
			#pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				if (input().real_[in_indx] > 0 && input().imag_[in_indx] > 0) {
					auto dLdz      = output(out_indx).dz(offset(out_indx) + in_indx);
					auto dLdz_star = output(out_indx).dz_star(offset(out_indx) + in_indx);

					input().dz_real_[in_indx] += dLdz.real();
					input().dz_imag_[in_indx] += dLdz.imag();
					input().dz_star_real_[in_indx] += dLdz_star.real();
					input().dz_star_imag_[in_indx] += dLdz_star.imag();
				}
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const CRelu& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}

		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const CRelu& rhs) const
	{
		return !operator==(rhs);
	}
};

//TODO
class CGelu: public CFunc {

public:
	CGelu(Uid uid, InSize in_size) : CFunc(uid, in_size, OutSize(in_size.value())) {
	}

	CGelu(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
	}

	virtual ~CGelu() {
	}

	virtual CFunc* clone(Uid uid) {
		return new CGelu(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "CGelu_" + std::to_string(uid_);
	}

	inline float gelu(float xi) {
		float cube = 0.044715f * xi * xi * xi;
		return 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
	}

	inline float dx_d_gelu(float x) {
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        return 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				output(indx).real_[offset(indx) + out_ind] = gelu(input().real_[out_ind]);
				output(indx).imag_[offset(indx) + out_ind] = gelu(input().imag_[out_ind]);
			}
		}
}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_star_dz = 0;

				auto dLdz      = output(indx).dz(offset(indx) + in_indx);
				auto dLdz_star = output(indx).dz_star(offset(indx) + in_indx);

				float dx = 0.5 * dx_d_gelu(input().real_[in_indx]);
				float dy = 0.5 * dx_d_gelu(input().imag_[in_indx]);
				float dz = dx + dy;
				float dz_star = dx - dy;

				sum_dz += dLdz * dz + dLdz_star * dz_star;
				sum_star_dz += dLdz * dz_star + dLdz_star * dz;

				input().dz_real_[in_indx] += sum_dz.real();
				input().dz_imag_[in_indx] += sum_dz.imag();
				input().dz_star_real_[in_indx] += sum_star_dz.real();
				input().dz_star_imag_[in_indx] += sum_star_dz.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const CGelu& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}

		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const CGelu& rhs) const
	{
		return !operator==(rhs);
	}
};

// Magnitude nonlinearity: w = |z|^2 = z*conj(z), output as a real value
// (imag = 0). Phase-invariant (|z e^{i theta}|^2 = |z|^2), which is what
// same-band signal/constellation classification needs, and smooth everywhere
// (no 1/|z| singularity). Non-holomorphic, so it genuinely uses dz_star.
//   dL/dz   = (dLdz + dLdz_star) * conj(z)
//   dL/dz~  = (dLdz + dLdz_star) * z
class CModulus2: public CFunc {

public:
	CModulus2(Uid uid, InSize in_size) : CFunc(uid, in_size, OutSize(in_size.value())) {
	}

	CModulus2(InSize in_size) : CFunc(in_size, OutSize(in_size.value())) {
	}

	virtual ~CModulus2() {
	}

	virtual CFunc* clone(Uid uid) {
		return new CModulus2(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "CModulus2_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				float re = input().real_[out_ind];
				float im = input().imag_[out_ind];
				output(indx).real_[offset(indx) + out_ind] = re * re + im * im;
				output(indx).imag_[offset(indx) + out_ind] = 0.0f;
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				auto dLdz      = output(indx).dz(offset(indx) + in_indx);
				auto dLdz_star = output(indx).dz_star(offset(indx) + in_indx);

				std::complex<float> z(input().real_[in_indx], input().imag_[in_indx]);
				auto g = dLdz + dLdz_star;
				auto sum_dz      = g * std::conj(z);
				auto sum_star_dz = g * z;

				input().dz_real_[in_indx] += sum_dz.real();
				input().dz_imag_[in_indx] += sum_dz.imag();
				input().dz_star_real_[in_indx] += sum_star_dz.real();
				input().dz_star_imag_[in_indx] += sum_star_dz.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const CModulus2& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}
		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const CModulus2& rhs) const
	{
		return !operator==(rhs);
	}
};

// Holomorphic integer power w = z^M. Unlike |z|^2 this keeps phase, and raising
// an M-PSK signal to the M-th power aligns its symbol phases, so the (pooled)
// mean of z^M is a phase-order feature that separates PSK orders. Holomorphic:
// dw/dz~ = 0, so
//   dL/dz  = dLdz      * M * z^(M-1)
//   dL/dz~ = dLdz_star * M * conj(z)^(M-1)
class CPower: public CFunc {
	int power_;

	static inline std::complex<float> ipow(std::complex<float> z, int n) {
		std::complex<float> r(1.0f, 0.0f);
		for (int i = 0; i < n; ++i) {
			r *= z;
		}
		return r;
	}

public:
	CPower(Uid uid, InSize in_size, int power)
			: CFunc(uid, in_size, OutSize(in_size.value())), power_(power) {
		assert(power_ >= 1);
	}

	CPower(InSize in_size, int power)
			: CFunc(in_size, OutSize(in_size.value())), power_(power) {
		assert(power_ >= 1);
	}

	virtual ~CPower() {
	}

	int power() const {
		return power_;
	}

	virtual CFunc* clone(Uid uid) {
		return new CPower(uid, InSize(input().length_), power_);
	}

	virtual std::string getName() {
		return "CPower_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < out_size_; ++i) {
				auto w = ipow(input().z(i), power_);
				output(indx).real_[offset(indx) + i] = w.real();
				output(indx).imag_[offset(indx) + i] = w.imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < input().length_; ++i) {
				auto dLdz      = output(indx).dz(offset(indx) + i);
				auto dLdz_star = output(indx).dz_star(offset(indx) + i);

				auto zpow = ipow(input().z(i), power_ - 1);   // z^(M-1)
				auto sum_dz      = dLdz * (float) power_ * zpow;
				auto sum_star_dz = dLdz_star * (float) power_ * std::conj(zpow);

				input().dz_real_[i] += sum_dz.real();
				input().dz_imag_[i] += sum_dz.imag();
				input().dz_star_real_[i] += sum_star_dz.real();
				input().dz_star_imag_[i] += sum_star_dz.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const CPower& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->power_ != rhs.power_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}
		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const CPower& rhs) const
	{
		return !operator==(rhs);
	}
};

#endif /* IMPL_RELU_H_ */
