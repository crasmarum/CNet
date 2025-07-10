#ifndef IMPL_L2OUT_H_
#define IMPL_L2OUT_H_

#include <algorithm>
#include "cfunc.h"

/***
 * L2Out(u, v, w) = u~u + v~v + w~v.
 */
class L2Out: public CFunc, public OutputFunc {
	Vars output_;
public:

	L2Out(Uid uid, InSize in_size) : CFunc(uid, in_size, OutSize(1)), output_(1) {
		is_output_ = true;
	}

	L2Out(InSize in_size) : CFunc(in_size, OutSize(1)), output_(1) {
		is_output_ = true;
	}

	virtual CFunc* clone(Uid uid) override {
		return new L2Out(uid, InSize(input().length_));
	}

	virtual std::string getName() override {
		return "L2Out_" + std::to_string(uid_);
	}

	virtual void forward() override {
		output_.real_[0] = 0;
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			output_.real_[0] += (float)pow(input().real_[in_indx], 2) + (float)pow(input().imag_[in_indx], 2);
		}
	}

	/*
	 * d/dz  L2Output = ~z
	 */
	virtual std::complex<float> dz(int out_indx, int in_index) override {
		assert(out_indx == 0);
		assert(in_index >= 0 && in_index < input().length_);

		return std::conj(input().z(in_index));
	}

	/*
	 * d/d~z L2Output = z
	 */
	virtual std::complex<float> dz_star(int out_indx, int in_index) override {
		assert(out_indx == 0);
		assert(in_index >= 0 && in_index < input().length_);

		return input().z(in_index);
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




#endif /* IMPL_L2OUT_H_ */
