#ifndef IMPL_CROSSENT_H_
#define IMPL_CROSSENT_H_
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>

#include <cfloat>

#include "cfunc.h"
#include "vars.h"

// Born-rule cross-entropy: p_c = |z_c|^2 / sum_j |z_j|^2, loss -log p_label. With eps > 0 the smoothed rule
// p_c = (|z_c|^2 + eps) / (sum_j |z_j|^2 + C eps) is used instead: finite at z == 0 (p = 1/C) with a zero gradient
// there, no clamps, and the same expressions on the CPU and the GPU. eps is not stored in model files.
// Only the loss, getProbability and the gradients are smoothed: get_prediction (an arg-max, unchanged by eps)
// and sample() use the exact rule. Call setEps before the GPU allocation: the kernels take eps from the first
// layer of a mapping, and the batch clones copy it when they are made.
class CrossEntropy: public CFunc, public OutputFunc {
	Vars output_;
	float square_norm_;
	float eps_ = 0.f;

public:
	void setEps(float e) { eps_ = e; }
	float eps() const { return eps_; }
	CrossEntropy(Uid uid, InSize in_size) : CFunc(uid, in_size, OutSize(1)),
		output_(in_size.value()), square_norm_(0) {
		is_output_ = true;
	}

	CrossEntropy(InSize in_size) : CFunc(in_size, OutSize(1)),
		output_(in_size.value()), square_norm_(0) {
		is_output_ = true;
	}

	virtual ~CrossEntropy() {
	}

	virtual CFunc* clone(Uid uid) override {
		CrossEntropy *c = new CrossEntropy(uid, InSize(input().length_));
		c->eps_ = eps_;   // batch clones must use the same rule as the original
		return c;
	}

	virtual std::string getName() override {
		return "CrossEntropy_" + std::to_string(uid_);
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
	}

	float getProbability(int label) {
		if (eps_ > 0.f) {
			const float sm = std::pow(input().real_[label], 2) + std::pow(input().imag_[label], 2);
			return (sm + eps_) / (square_norm_ + input().length_ * eps_);
		}
		return std::pow(output_.real_[label], 2)
				+ std::pow(output_.imag_[label], 2);
	}

	void printProbabilities() {
		for (int pos = 0; pos < input().length_; ++pos) {
			std::cout << std::setprecision(9) << pos << ": " << getProbability(pos) << ", ";
		}
		std::cout <<  std::endl;
	}

	virtual float loss(int label) override {
		assert(label >= 0 && label < input().length_);
		float val = getProbability(label);
		if (eps_ <= 0.f) val = (val < 1e-15 ? 1e-15 : val);
		return -std::log(val);
	}

	virtual void forward() override {
		double sum = 0.0;
		for (int pos = 0; pos < input().length_; ++pos) {
			sum += (std::pow(input().real_[pos], 2) + std::pow(input().imag_[pos], 2));
		}

		square_norm_ = sum;
		if(eps_ <= 0.f && square_norm_ <= 1.0e-15) {
			square_norm_ = 1.0e-15;
		}
		sum = sqrt(square_norm_);
		if(sum <= 1.0e-15) {
			sum = 1.0e-15;
		}


        #pragma omp parallel for
		for (int pos = 0; pos < input().length_; ++pos) {
			output_.real_[pos] = input().real_[pos] / sum;
			output_.imag_[pos] = input().imag_[pos] / sum;
		}
	}

	virtual void backward(int label) override {
		assert(label >= 0 && label < input().length_);

		if (eps_ > 0.f) {   // smoothed rule: dL/dz~ = z/(S + C eps) - [label] z/(|z|^2 + eps); finite everywhere
			const float denom = square_norm_ + input().length_ * eps_;
			#pragma omp parallel for
			for (int pos = 0; pos < input().length_; ++pos) {
				float gr = input().real_[pos] / denom, gi = input().imag_[pos] / denom;
				if (label == pos) {
					const float sm = std::pow(input().real_[pos], 2) + std::pow(input().imag_[pos], 2) + eps_;
					gr -= input().real_[pos] / sm;
					gi -= input().imag_[pos] / sm;
				}
				input().dz_star_real_[pos] = gr;
				input().dz_star_imag_[pos] = gi;
				input().dz_real_[pos] = gr;
				input().dz_imag_[pos] = -gi;
			}
			return;
		}

        #pragma omp parallel for
		for (int pos = 0; pos < input().length_; ++pos) {
			if (label == pos) {
				float square_mod = std::pow(input().real_[pos], 2)
						+ std::pow(input().imag_[pos], 2);
				square_mod = (square_mod < 1e-15 ? 1e-15 : square_mod);
				input().dz_star_real_[pos] = -input().real_[pos]
						* (square_norm_ - square_mod)
						/ (square_mod * square_norm_);
				input().dz_star_imag_[pos] = -input().imag_[pos]
						* (square_norm_ - square_mod)
						/ (square_mod * square_norm_);
				input().dz_real_[pos] = input().dz_star_real_[pos];
				input().dz_imag_[pos] = -input().dz_star_imag_[pos];
			} else {
				input().dz_star_real_[pos] = input().real_[pos] / square_norm_;
				input().dz_star_imag_[pos] = input().imag_[pos] / square_norm_;
				input().dz_real_[pos] = input().dz_star_real_[pos];
				input().dz_imag_[pos] = -input().dz_star_imag_[pos];
			}
		}
	}

	virtual void backward() override {

	}

	virtual std::complex<float>      dz(int out_indx, int in_index) override {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) override {
		return 0;
	}

	int get_prediction() {
		float max = std::pow(output_.real_[0], 2)
				+ std::pow(output_.imag_[0], 2);
		int ret = 0;
		for (int var = 1; var < output_.length_; ++var) {
			float c_val = std::pow(output_.real_[var], 2)
					+ std::pow(output_.imag_[var], 2);
			if (c_val > max) {
				ret = var;
				max = c_val;
			}
		}
		return ret;
	}

	int sample() {
		float coin = (float)rand_.randDouble();
		float c_val = 0.f;
		float max = 0;
		int var_max = 0.f;
		for (int var = 0; var < output_.length_; ++var) {
			auto norm = std::pow(output_.real_[var], 2)
								+ std::pow(output_.imag_[var], 2);
			c_val += norm;
			if (coin <= c_val) {
				return var;
			}
			if (max < norm) {
				max = norm;
				var_max = var;
			}
		}
		return var_max;
	}

	bool operator==(const CrossEntropy& rhs) const
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
	bool operator!=(const CrossEntropy& rhs) const
	{
		return !operator==(rhs);
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
};

#endif /* IMPL_CROSSENT_H_ */
