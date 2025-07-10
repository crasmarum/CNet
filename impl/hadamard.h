#ifndef IMPL_HADAMARD_H_
#define IMPL_HADAMARD_H_


#include <algorithm>
#include <omp.h>

#include "cfunc.h"
#include "utils.h"

class Hadamard: public CFunc {
public:

	Hadamard(Uid uid, InSize in1, InSize in2) : CFunc(uid, InSize(in1.value() + in2.value()),
			OutSize(in1.value())) {
		assert(in2.value() == in1.value());
	}

	Hadamard(InSize in1, InSize in2) : CFunc(InSize(in1.value() + in2.value()), OutSize(in1.value())) {
		assert(in2.value() == in1.value());
	}

	virtual CFunc* clone(Uid uid) {
		return new Hadamard(uid, InSize(out_size_), InSize(out_size_));
	}

	virtual std::string getName() {
		return "Hadamard_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				auto prod = input().z(out_ind) * input().z(out_size_ + out_ind);
				output(indx).real_[offset(indx) + out_ind] = prod.real();
				output(indx).imag_[offset(indx) + out_ind] = prod.imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_pos = 0; out_pos < out_size_; ++out_pos) {
				auto dLdz      = output(indx).dz(offset(indx) + out_pos);
				auto dLdz_star = output(indx).dz_star(offset(indx) + out_pos);

				auto dz = dLdz * input().z(out_size_ + out_pos);
				auto dz_star = dLdz_star * std::conj(input().z(out_size_ + out_pos));
				input().dz_real_[out_pos] += dz.real();
				input().dz_imag_[out_pos] += dz.imag();
				input().dz_star_real_[out_pos] += dz_star.real();
				input().dz_star_imag_[out_pos] += dz_star.imag();

				dz = dLdz * input().z(out_pos);
				dz_star = dLdz_star * std::conj(input().z(out_pos));
				input().dz_real_[out_size_ + out_pos] += dz.real();
				input().dz_imag_[out_size_ + out_pos] += dz.imag();
				input().dz_star_real_[out_size_ + out_pos] += dz_star.real();
				input().dz_star_imag_[out_size_ + out_pos] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	/*
	 * d/dz  L2Output = other var
	 */
	virtual std::complex<float> dz(int out_indx, int in_indx) {
		return 0;
	}

	/*
	 * d/d~z L2Output = 0
	 */
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const Hadamard& rhs) const
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
	bool operator!=(const Hadamard& rhs) const
	{
		return !operator==(rhs);
	}
};



#endif /* IMPL_HADAMARD_H_ */
