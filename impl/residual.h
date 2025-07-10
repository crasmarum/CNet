#ifndef IMPL_RESIDUAL_H_
#define IMPL_RESIDUAL_H_

#include <algorithm>
#include <omp.h>

#include "cfunc.h"

/***
 * Residual(u, v) = u + v
 */
class Residual: public CFunc {
public:

	Residual(Uid uid, InSize in) : CFunc(uid, InSize(2 * in.value()), OutSize(in.value())) {
	}

	Residual(Uid uid, InSize in1, InSize in2) : CFunc(uid, InSize(2 * in1.value()), OutSize(in1.value())) {
		assert(in1.value() == in2.value());
	}

	Residual(InSize in) : CFunc(InSize(2 * in.value()), OutSize(in.value())) {
	}

	Residual(InSize in1, InSize in2) : CFunc(InSize(2 * in1.value()), OutSize(in1.value())) {
		assert(in1.value() == in2.value());
	}

	virtual CFunc* clone(Uid uid) {
		return new Residual(uid, InSize(out_size_), InSize(out_size_));
	}

	virtual std::string getName() {
		return "Res_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				auto sum = input().z(out_ind) + input().z(out_ind + out_size_);
				output(indx).real_[offset(indx) + out_ind] = sum.real();
				output(indx).imag_[offset(indx) + out_ind] = sum.imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_pos = 0; out_pos < out_size_; ++out_pos) {
				//auto sum = input().z(out_pos) + input().z(out_pos + out_size_);

				auto dLdz      = output(indx).dz(offset(indx) + out_pos);
				auto dLdz_star = output(indx).dz_star(offset(indx) + out_pos);
				auto o_index = out_pos + out_size_;
				//auto other = input().z(o_index);

				auto dz = dLdz;
				auto dz_star = dLdz_star;

				input().dz_real_[out_pos] += dz.real();
				input().dz_imag_[out_pos] += dz.imag();
				input().dz_star_real_[out_pos] += dz_star.real();
				input().dz_star_imag_[out_pos] += dz_star.imag();

				input().dz_real_[o_index] += dz.real();
				input().dz_imag_[o_index] += dz.imag();
				input().dz_star_real_[o_index] += dz_star.real();
				input().dz_star_imag_[o_index] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	/*
	 * d/dz  dui/dvj Residual = 1 iff ui = vj + wj.
	 */
	virtual std::complex<float> dz(int out_indx, int in_indx) {
//		if (out_indx == in_indx % out_size_) {
//			return 1;
//		}
		return 0;
	}

	/*
	 * d/dz  dui/d~vj Residual = 0
	 */
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const Residual& rhs) const
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
	bool operator!=(const Residual& rhs) const
	{
		return !operator==(rhs);
	}
};


#endif /* IMPL_RESIDUAL_H_ */
