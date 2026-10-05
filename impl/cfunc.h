#ifndef IMPL_CFUNC_H_
#define IMPL_CFUNC_H_

#include "sizes.h"
#include "vars.h"
#include "../gpu/gpuvars.h"
#include "../utils/myrand.h"

#include <vector>
#include <iostream>
#include <iomanip>

class OutputFunc {
	bool isMainOutput_ = false;
public:
	virtual ~OutputFunc() {
	}

	virtual int outputLength() = 0;
	virtual float loss(int label) = 0;
	virtual Vars* mutableOutput() = 0;

	bool isMainOutput() {
		return isMainOutput_;
	}

	void setIsMainOutput(bool val) {
		isMainOutput_ = val;
	}
};

class InputFunc {
	bool isMainInput_ = false;
public:
	virtual ~InputFunc() {
	}

	bool isMainInput() {
		return isMainInput_;
	}

	void setIsMainInput(bool val) {
		isMainInput_ = val;
	}
};

class CFunc {
	friend class ModelSaver;
	friend class ComplexNet;
	friend class CNet;

	Vars input_;

protected:
	int uid_;
	int out_size_;

	std::vector<CFunc*> prev_func_;  // Func* not owned
	std::vector<CFunc*> next_func_;  // Func* not owned
	std::vector<int> offset_;
	int depth_ = 0;
	bool is_input_ = false;
	bool is_output_ = false;
	bool is_gpu_only_ = false;
	int  batch_indx_ = 0;
	int type_ = 0;
	SimpleRand rand_;

public:
	// used to construct execution graph on GPU
	GpuInVar gpu_var_;

	CFunc(Uid uid, InSize in_size, OutSize out_size) : input_(in_size.value()),
		uid_(uid.value()), out_size_(out_size.value()) {
	}

	CFunc(InSize in_size, OutSize out_size) : input_(in_size.value()),
		uid_(0), out_size_(out_size.value()) {
	}

	void setRandSeed(uint64_t seed) {
		rand_ = SimpleRand(seed);
	}

	bool isInput() {
		return is_input_;
	}

	bool isOutput() {
		return is_output_;
	}

	bool isGpuOnly() {
		return is_gpu_only_;
	}

	int getType() const {
		return type_;
	}

	virtual ~CFunc() {
	}

	int outSize() {
		return out_size_;
	}

	const Vars& input() const {
		return input_;
	}

	const Vars& output() {
		assert(next_func_.size());
		return next_func_[0]->input();
	}

	const Vars& output(int index) {
		assert(index < next_func_.size());
		return next_func_[index]->input();
	}

	Vars* mutable_output() {
		assert(next_func_.size());
		return next_func_[0]->mutable_input();
	}

	Vars* mutable_output(int index) {
		assert(index < next_func_.size());
		return next_func_[index]->mutable_input();
	}

	int uid() {
		return uid_;
	}

	int depth() {
		return depth_;
	}

public:
	virtual CFunc* clone(Uid uid) = 0;
	virtual std::string getName() = 0;
	virtual void forward() = 0;
	virtual std::complex<float>      dz(int out_indx, int in_indx) = 0;
	virtual std::complex<float> dz_star(int out_indx, int in_indx) = 0;

	virtual void updateInput(float learningRate) {
		for (int pos = 0; pos < input_.length_; ++pos) {
			input_.real_[pos] -= learningRate * (input_.dz_star_real_[pos]);
			input_.imag_[pos] -= learningRate * input_.dz_star_imag_[pos];
		}
	}

	Vars* mutable_input() {
		return &input_;
	}

	virtual void printInput() {
		std::cout << getName() << ": ";
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			std::cout << in_indx << ": " << input().z(in_indx) << ", ";
		}
		std::cout << std::endl;
	}

	virtual void printInput2D(int in_len) {
		std::cout << getName() << ": " << std::endl;
		assert(in_len);
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			std::cout << in_indx << ": " << input().z(in_indx) << ", ";
			if (in_indx % in_len == in_len - 1) {
				std::cout << std::endl;
			}
		}
		std::cout << std::endl;
	}

	virtual void printInput2D(int in_len, int offset, int len) {
		std::cout << getName() << ": " << std::endl;
		assert(in_len);
		for (int in_indx = 0; in_indx < len; ++in_indx) {
			std::cout << (offset + in_indx) << ": " << input().z(offset + in_indx) << ", ";
			if (in_indx % in_len == in_len - 1) {
				std::cout << std::endl;
			}
		}
		std::cout << std::endl;
	}

	virtual void init_inputs() {
		for (int indx = 0; indx < mutable_input()->length_; ++indx) {
			float re = (float)rand_.randDouble() - 0.5;
			mutable_input()->real_[indx] = re / input().length_;
			mutable_input()->imag_[indx] = sqrt(1 - re * re) / input().length_;
		}
	}

	// Circularly symmetric complex Gaussian init (arXiv:1810.11650): real and
	// imaginary parts are independent N(0, scale/length) via Box-Muller.
	virtual void gaussInit(float scale) {
		float stddev = sqrt(scale / (float) input().length_);
		for (int i = 0; i < mutable_input()->length_; ++i) {
			float u1 = (float) rand_.randDouble();
			float u2 = (float) rand_.randDouble();
			if (u1 < 1e-12f) {
				u1 = 1e-12f;
			}
			float r = stddev * sqrt(-2.0f * log(u1));
			mutable_input()->real_[i] = r * cos(6.2831853f * u2);
			mutable_input()->imag_[i] = r * sin(6.2831853f * u2);
		}
	}

	virtual void printGradient() {
		std::cout << getName() << ": ";
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			std::cout << in_indx << "\tdz: " << input().dz(in_indx) << ", ";
		}
		std::cout << std::endl;
		std::cout << getName() << ": ";
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			std::cout << in_indx << "\tdz_star: " << input().dz_star(in_indx) << ", ";
		}
		std::cout << std::endl;
	}

	void addOutput(CFunc* next, int offset) {
		assert(next && offset >= 0);
		if (next->input().length_ < offset + out_size_) {
			std::cerr << "err " << next->getName()
					<< ": " << next->input().length_  << " < " << offset + out_size_ << std::endl;
		}
		assert(next->input().length_ >= offset + out_size_);
		next_func_.push_back(next);
		offset_.push_back(offset);

		if (next->depth_ <= depth_) {
			next->depth_ = depth_ + 1;
		}
	}

	virtual int sizeInBytes() const {
		return input_.size_in_bytes();
	}

	CFunc* next(int index) const{
		assert(index >= 0 && index < next_func_.size());
		return next_func_[index];
	}

	inline int offset(int index) {
		assert(index < offset_.size());
		return offset_[index];
	}

	int no_outputs() const {
		return next_func_.size();
	}

	int no_previous_func() const {
		return prev_func_.size();
	}

	CFunc* previous(int index) {
		assert(index >= 0 && index < prev_func_.size());
		return prev_func_[index];
	}

	/*
	 * f(u0, u1, u2) = (f0, f1), L(v0, v1) = z
	 *
	 * d/du_i L(f) =   (dL/dv0  * df0/du_i + dL/dv1 * df1/du_i)
	 *               + (dL/d~v1 * ~(dv1/d~u_i) + dL/d~v2 * ~(dv2/d~u_i))
	 *
	 * i = [0, ..., input.len)
	 */
	virtual void backward() {
//		std::cout << "back: " << this->getName() << " in_len: " << input().length_  << std::endl;
		if (is_output_) {
			out_backward();
		} else {
			out_others();
		}
	}

	virtual void backward(int label) {
	}

	void adamUpdate(float l_rate, float beta1, int t) {
		#pragma omp parallel for
		for (int indx = 0; indx < input().length_; ++indx) {
			// m = beta1 * m_prev + (1 - beta1) * grad
			*momentum_real(indx) = *momentum_real(indx) * beta1
					+ (1 - beta1) * input().dz_star_real_[indx];
			*momentum_imag(indx) = *momentum_imag(indx) * beta1
								+ (1 - beta1) * input().dz_star_imag_[indx];

			input_.real_[indx] -= l_rate * (*momentum_real(indx))
								   / (1 - pow(beta1, t + 1));
			input_.imag_[indx] -= l_rate * (*momentum_imag(indx))
								  / (1 - pow(beta1, t + 1));
		}
	}

	// True Adam (CPU): first AND second moment per parameter, treating the real
	// and imaginary components as independent real parameters. The per-parameter
	// step normalisation (the sqrt(v) term) is what momentum alone lacks.
	void trueAdamUpdate(float l_rate, float beta1, float beta2, float eps, int t) {
		int L = input_.length_;
		if ((int) adam_m_re_.size() != L) {
			adam_m_re_.assign(L, 0.f);
			adam_m_im_.assign(L, 0.f);
			adam_v_re_.assign(L, 0.f);
			adam_v_im_.assign(L, 0.f);
		}
		float bc1 = 1.0f - pow(beta1, t + 1);
		float bc2 = 1.0f - pow(beta2, t + 1);
		#pragma omp parallel for
		for (int i = 0; i < L; ++i) {
			float gr = input_.dz_star_real_[i];
			float gi = input_.dz_star_imag_[i];

			adam_m_re_[i] = beta1 * adam_m_re_[i] + (1 - beta1) * gr;
			adam_m_im_[i] = beta1 * adam_m_im_[i] + (1 - beta1) * gi;
			adam_v_re_[i] = beta2 * adam_v_re_[i] + (1 - beta2) * gr * gr;
			adam_v_im_[i] = beta2 * adam_v_im_[i] + (1 - beta2) * gi * gi;

			float mhr = adam_m_re_[i] / bc1;
			float mhi = adam_m_im_[i] / bc1;
			float vhr = adam_v_re_[i] / bc2;
			float vhi = adam_v_im_[i] / bc2;

			input_.real_[i] -= l_rate * mhr / (sqrt(vhr) + eps);
			input_.imag_[i] -= l_rate * mhi / (sqrt(vhi) + eps);
		}
	}

private:
	std::vector<float> adam_m_re_, adam_m_im_, adam_v_re_, adam_v_im_;  // true-Adam state


	void out_others() {
		for (int next_func_indx = 0; next_func_indx < next_func_.size(); ++next_func_indx) {
			auto curr_output = output(next_func_indx);
			auto curr_offset = offset_[next_func_indx];

            #pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_dz_star = 0;

				for (int out_indx = 0; out_indx < out_size_; ++out_indx) {
					auto dLdz      = curr_output.dz(curr_offset + out_indx);
					auto dLdz_star = curr_output.dz_star(curr_offset + out_indx);

					std::complex<float> dz1 = dz(out_indx, in_indx);
					std::complex<float> dz_star1 = dz_star(out_indx, in_indx);

					sum_dz += dLdz * dz1;
					sum_dz += dLdz_star * std::conj(dz_star1);

					sum_dz_star += dLdz * dz_star1;
					sum_dz_star += dLdz_star * std::conj(dz1);

//					std::cout << " i_index: " << in_indx << " o_indx: " << out_indx
//							  << " dz: " << dLdz * dz1 + dLdz_star * std::conj(dz_star1)
//							  << " dz_star: " << dLdz * dz_star1 + dLdz_star * std::conj(dz1) << std::endl;
				}

				mutable_input()->dz_real_[in_indx] += sum_dz.real();
				mutable_input()->dz_imag_[in_indx] += sum_dz.imag();

				mutable_input()->dz_star_real_[in_indx] += sum_dz_star.real();
				mutable_input()->dz_star_imag_[in_indx] += sum_dz_star.imag();
			}
		}
	}

	virtual void out_backward() {
        #pragma omp parallel for
		for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
			std::complex<float> val_dz = dz(0, in_indx);
			std::complex<float> val_dz_star = dz_star(0, in_indx);

			mutable_input()->dz_real_[in_indx] += val_dz.real();
			mutable_input()->dz_imag_[in_indx] += val_dz.imag();

			mutable_input()->dz_star_real_[in_indx] += val_dz_star.real();
			mutable_input()->dz_star_imag_[in_indx] += val_dz_star.imag();
		}
	}

	float* momentum_real(int indx) {
		assert(0 <= indx && indx < input_.length_);
		return input_.dz_real_ + indx;
	}

	float* momentum_imag(int indx) {
		assert(0 <= indx && indx < input_.length_);
		return input_.dz_imag_ + indx;
	}
};


#endif /* IMPL_CFUNC_H_ */
