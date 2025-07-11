#ifndef IMPL_LINEAR_H_
#define IMPL_LINEAR_H_

#include <algorithm>
#include <omp.h>

#include "cfunc.h"
#include "utils.h"


class Linear: public CFunc {
	friend class ModelSaver;

	int in1_len;
	int in2_len;
public:

	Linear(Uid uid, InSize in1, InSize in2) : CFunc(uid,
			InSize(in1.value() + in2.value()), OutSize(in2.value() / in1.value())) {
		assert(in2.value() % in1.value() == 0);
		in1_len = in1.value();
		in2_len = in2.value();
	}

	Linear(InSize in1, InSize in2) : CFunc(InSize(in1.value() + in2.value()), OutSize(in2.value() / in1.value())) {
		assert(in2.value() % in1.value() == 0);
		in1_len = in1.value();
		in2_len = in2.value();
	}

	virtual ~Linear() {
	}

	virtual CFunc* clone(Uid uid) {
		return new Linear(uid, InSize(in1_len), InSize(in2_len));
	}

	int firstInputLength() {
		return in1_len;
	}

	int secondInputLength() {
		return in2_len;
	}


	virtual std::string getName() {
		return "Linear_" + std::to_string(uid_);
	}

	virtual void forward() {
		std::vector<std::complex<float> > sums(out_size_);

		#pragma omp parallel for
		for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
			std::complex<float> sum = 0;
			for (int in_indx = 0; in_indx < in1_len; ++in_indx) {
				sum += input().z(in_indx) * input().z((out_ind + 1) * in1_len + in_indx);
			}
			sums[out_ind] = sum;
		}

		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int out_ind = 0; out_ind < out_size_; ++out_ind) {
				output(indx).real_[offset(indx) + out_ind] = sums[out_ind].real();
				output(indx).imag_[offset(indx) + out_ind] = sums[out_ind].imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int in_indx = 0; in_indx < in1_len; ++in_indx) {
				for (int out_pos = 0; out_pos < out_size_; ++out_pos) {
					// sum += input().z(in_indx) * input().z((out_ind + 1) * in1_len + in_indx);

					auto dLdz      = output(indx).dz(offset(indx) + out_pos);
					auto dLdz_star = output(indx).dz_star(offset(indx) + out_pos);

					auto o_index = (out_pos + 1) * in1_len + in_indx;
					auto other = input().z(o_index);

					auto dz = dLdz * other;
					auto dz_star = dLdz_star * std::conj(other);

					input().dz_real_[in_indx] += dz.real();
					input().dz_imag_[in_indx] += dz.imag();
					input().dz_star_real_[in_indx] += dz_star.real();
					input().dz_star_imag_[in_indx] += dz_star.imag();

//					std::cout << "Y oi=" << out_pos << "\tii=" << in_indx
//							  << "\tdz=" << dz;

					// the matrix vars get their gradient updated here
					other = input().z(in_indx);
					dz = dLdz * other;
					dz_star = dLdz_star * std::conj(other);
					input().dz_real_[o_index] += dz.real();
					input().dz_imag_[o_index] += dz.imag();
					input().dz_star_real_[o_index] += dz_star.real();
					input().dz_star_imag_[o_index] += dz_star.imag();

//					std::cout << "\tmt=" << dz << std::endl;
				}
			}
		}
	}

	virtual void backward(int label) {
	}

	/*
	 * d/dz  L2Output = other var
	 */
	virtual std::complex<float> dz(int out_indx, int in_indx) {
		if (in_indx < in1_len) {
			return input().z((out_indx + 1) * in1_len + in_indx);
		} else if (in_indx / in1_len == out_indx + 1) {
			return input().z(in_indx % in1_len);
		} else {
			return 0;
		}
	}

	/*
	 * d/d~z L2Output = 0
	 */
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const Linear& rhs) const
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
	bool operator!=(const Linear& rhs) const
	{
		return !operator==(rhs);
	}
};


#endif /* IMPL_LINEAR_H_ */
