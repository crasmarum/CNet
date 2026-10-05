#ifndef IMPL_SOFTMAX_H_
#define IMPL_SOFTMAX_H_

#include <cfloat>

#include "cfunc.h"
#include "vars.h"

class SoftMax: public CFunc {
	float square_norm_;
	float norm_2_3;

public:
	SoftMax(Uid uid, InSize in_size) : CFunc(uid, in_size,
			OutSize(in_size.value())), square_norm_(0), norm_2_3(0) {
	}

	SoftMax(InSize in_size) : CFunc(in_size, OutSize(in_size.value())),
		square_norm_(0), norm_2_3(0) {
	}

	virtual ~SoftMax() {
	}

	virtual CFunc* clone(Uid uid) {
		return new SoftMax(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "SoftMax_" + std::to_string(uid_);
	}

	virtual void forward() {
		double sum = 0.0;

		#pragma omp parallel for reduction (+:sum)
		for (int pos = 0; pos < input().length_; ++pos) {
			sum = sum + (std::pow(input().real_[pos], 2) + std::pow(input().imag_[pos], 2));
		}

		square_norm_ = sum;
		sum = sqrt(sum);
		if(sum <= 1.0e-15) {
			sum = 1.0e-15;
		}
		norm_2_3 = pow(sum, 3);
		if(norm_2_3 < 1.0e-15) {
			norm_2_3 = 1.0e-15;   // floor the derivative denominator (matches GPU)
		}

		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int pos = 0; pos < input().length_; ++pos) {
				output(indx).real_[offset(indx) + pos] = (float) input().real_[pos] / sum;
				output(indx).imag_[offset(indx) + pos] = (float) input().imag_[pos] / sum;
			}
		}
	}

	// O(N) backward for A(z)=z/||z||, replacing the generic O(N^2) dense Jacobian.
	// With P = sum_j dLdz_j z_j, Q = sum_j dLdz*_j conj(z_j), R = (P+Q)/||z||^3:
	//   dz[i]      += dLdz_i/||z||      - 0.5 conj(z_i) R
	//   dz_star[i] += dLdz*_i/||z||     - 0.5 z_i      R
	virtual void backward() {
		float s = sqrt(square_norm_);
		if (s < 1e-15f) {
			s = 1e-15f;
		}
		float s3 = norm_2_3;   // ||z||^3, floored in forward()
		int N = input().length_;

		for (int indx = 0; indx < no_outputs(); ++indx) {
			int off = offset(indx);
			std::complex<float> P = 0.f, Q = 0.f;
			for (int j = 0; j < N; ++j) {
				auto zj = input().z(j);
				P += output(indx).dz(off + j) * zj;
				Q += output(indx).dz_star(off + j) * std::conj(zj);
			}
			std::complex<float> R = (P + Q) / s3;

			#pragma omp parallel for
			for (int i = 0; i < N; ++i) {
				auto zi = input().z(i);
				auto sum_dz = output(indx).dz(off + i) / s - 0.5f * std::conj(zi) * R;
				auto sum_dz_star = output(indx).dz_star(off + i) / s - 0.5f * zi * R;
				mutable_input()->dz_real_[i] += sum_dz.real();
				mutable_input()->dz_imag_[i] += sum_dz.imag();
				mutable_input()->dz_star_real_[i] += sum_dz_star.real();
				mutable_input()->dz_star_imag_[i] += sum_dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		if (in_index == out_indx) {
			return 0.5f * (std::complex<float>(2.0f * square_norm_, 0)
					- input().z(in_index) * std::conj(input().z(in_index))) / norm_2_3;
		}
		return -0.5f * input().z(out_indx) * std::conj(input().z(in_index)) / norm_2_3;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
        if (out_indx == in_index) {
    		return -0.5f * input().z(in_index) * input().z(in_index) / norm_2_3;
        }
		return -0.5f * input().z(out_indx) * input().z(in_index) / norm_2_3;
	}

	bool operator==(const SoftMax& rhs) const
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
	bool operator!=(const SoftMax& rhs) const
	{
		return !operator==(rhs);
	}
};



#endif /* IMPL_SOFTMAX_H_ */
