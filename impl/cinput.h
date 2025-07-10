#ifndef IMPL_CINPUT_H_
#define IMPL_CINPUT_H_

#include <algorithm>
#include <omp.h>

#include "cfunc.h"
#include "batch.h"

class CInput: public CFunc, public InputFunc {
	bool isMainInput_ = false;
public:

	CInput(Uid uid, OutSize out_size) :
			CFunc(uid, InSize(out_size.value()), out_size) {
		is_input_ = true;
	}

	CInput(OutSize out_size) : CFunc(InSize(out_size.value()), out_size) {
		is_input_ = true;
	}

	bool isMainInput() {
		return isMainInput_;
	}

	void setIsMainInput(bool val) {
		isMainInput_ = val;;
	}

	virtual CFunc* clone(Uid uid) {
		return new CInput(uid, OutSize(input().length_));
	}

	virtual ~CInput() {
	}

	virtual std::string getName() {
		return "Input_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());

		for (int out_indx = 0; out_indx < no_outputs(); ++out_indx) {
            #pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				next(out_indx)->input().real_[offset(out_indx) + in_indx] = input().real_[in_indx];
				next(out_indx)->input().imag_[offset(out_indx) + in_indx] = input().imag_[in_indx];
			}
		}
	}

	virtual void backward() {

		#pragma omp parallel for
		for (int pos = 0; pos < input().length_; ++pos) {
			for (int indx = 0; indx < no_outputs(); ++indx) {
				mutable_input()->dz_star_real_[pos] +=
						next(indx)->input().dz_star_real_[offset(indx) + pos];
				mutable_input()->dz_star_imag_[pos] +=
						next(indx)->input().dz_star_imag_[offset(indx) + pos];
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual void update(float learningRate) {
		updateInput(learningRate);
	}

	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}

	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	void setInput(InputBatch& batch, int b_indx) {
		assert(batch.sampleDim() == input().length_);
		std::copy(batch.realDataPtr(b_indx), batch.realDataPtr(b_indx) + input().length_,
				mutable_input()->real_);
		std::copy(batch.imagDataPtr(b_indx), batch.imagDataPtr(b_indx) + input().length_,
				mutable_input()->imag_);
	}

	bool operator==(const CInput& rhs) const
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
		for (int var = 0; var < this->input().length_; ++var) {
			if (this->input().real_[var] != rhs.input().real_[var]) {
				return false;
			}
			if (this->input().imag_[var] != rhs.input().imag_[var]) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const CInput& rhs) const
	{
		return !operator==(rhs);
	}
};

#endif /* IMPL_CINPUT_H_ */
