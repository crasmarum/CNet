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

#endif /* IMPL_RELU_H_ */
