#ifndef IMPL_POOL_H_
#define IMPL_POOL_H_
#include <cassert>
#include <string>
#include <vector>

#include <complex>

#include "cfunc.h"
#include "vars.h"

// Mean pooling over time: a length-L input is split into out_size contiguous
// segments of width = L/out_size, and each output is the mean of its segment
// (global average pooling when out_size == 1). This builds shift-robust summary
// statistics, e.g. the mean constellation power after a |z|^2 layer.
//
// Mean is a real linear (holomorphic) map w = A z with A_{p,i} = 1/width for i
// in segment p, so the Wirtinger derivatives are simply:
//   dL/dz_i  = dLdz(pool(i))      / width
//   dL/dz~_i = dLdz_star(pool(i)) / width
class MeanPool: public CFunc {
	int width_;   // input samples per output bin

public:
	MeanPool(Uid uid, InSize in_size, OutSize out_size)
			: CFunc(uid, in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		width_ = in_size.value() / out_size.value();
	}

	MeanPool(InSize in_size, OutSize out_size) : CFunc(in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		width_ = in_size.value() / out_size.value();
	}

	virtual ~MeanPool() {
	}

	virtual CFunc* clone(Uid uid) {
		return new MeanPool(uid, InSize(input().length_), OutSize(out_size_));
	}

	virtual std::string getName() {
		return "MeanPool_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int p = 0; p < out_size_; ++p) {
				std::complex<float> sum = 0;
				for (int j = 0; j < width_; ++j) {
					sum += input().z(p * width_ + j);
				}
				sum /= (float) width_;
				output(indx).real_[offset(indx) + p] = sum.real();
				output(indx).imag_[offset(indx) + p] = sum.imag();
			}
		}
	}

	virtual void backward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < input().length_; ++i) {
				int p = i / width_;
				auto dz      = output(indx).dz(offset(indx) + p) / (float) width_;
				auto dz_star = output(indx).dz_star(offset(indx) + p) / (float) width_;
				input().dz_real_[i]      += dz.real();
				input().dz_imag_[i]      += dz.imag();
				input().dz_star_real_[i] += dz_star.real();
				input().dz_star_imag_[i] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const MeanPool& rhs) const {
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
	bool operator!=(const MeanPool& rhs) const {
		return !operator==(rhs);
	}
};


// Mean over tokens, per dimension: the input is N tokens of P dims (token-major, index t*P + p, the
// layout of TokenwiseLinear), the output is the P-dim average over t. Shift-invariant pooling for
// sequence classifiers. N = in_size / out_size, so the sizes define the layer (no extra state).
class TokenMeanPool: public CFunc {
	int N_;

public:
	TokenMeanPool(Uid uid, InSize in_size, OutSize out_size)
			: CFunc(uid, in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		N_ = in_size.value() / out_size.value();
	}

	TokenMeanPool(InSize in_size, OutSize out_size) : CFunc(in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		N_ = in_size.value() / out_size.value();
	}

	virtual ~TokenMeanPool() {
	}

	virtual CFunc* clone(Uid uid) {
		return new TokenMeanPool(uid, InSize(input().length_), OutSize(out_size_));
	}

	virtual std::string getName() {
		return "TokenMeanPool_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		const int P = out_size_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int p = 0; p < P; ++p) {
				std::complex<float> sum = 0;
				for (int t = 0; t < N_; ++t) {
					sum += input().z(t * P + p);
				}
				sum /= (float) N_;
				output(indx).real_[offset(indx) + p] = sum.real();
				output(indx).imag_[offset(indx) + p] = sum.imag();
			}
		}
	}

	virtual void backward() {
		const int P = out_size_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < input().length_; ++i) {
				int p = i % P;
				auto dz      = output(indx).dz(offset(indx) + p) / (float) N_;
				auto dz_star = output(indx).dz_star(offset(indx) + p) / (float) N_;
				input().dz_real_[i]      += dz.real();
				input().dz_imag_[i]      += dz.imag();
				input().dz_star_real_[i] += dz_star.real();
				input().dz_star_imag_[i] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}
};

// Max-magnitude pooling over tokens, per dimension: out[p] = z[t*, p] with t* = argmax_t |z[t, p]|
// (the first maximum on ties), so the pooled value keeps the phase of the strongest token. Token-major
// input of N tokens x P dims (in_size = N P, out_size = P), like TokenMeanPool. The gradient flows to
// the selected token only (the pooling is piecewise linear in z).
class TokenMaxPool: public CFunc {
	int N_;
	std::vector<int> arg_;   // argmax token per output dim, from the last forward

public:
	TokenMaxPool(Uid uid, InSize in_size, OutSize out_size)
			: CFunc(uid, in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		N_ = in_size.value() / out_size.value();
		arg_.assign(out_size.value(), 0);
	}

	TokenMaxPool(InSize in_size, OutSize out_size) : CFunc(in_size, out_size) {
		assert(out_size.value() > 0 && in_size.value() % out_size.value() == 0);
		N_ = in_size.value() / out_size.value();
		arg_.assign(out_size.value(), 0);
	}

	virtual ~TokenMaxPool() {
	}

	virtual CFunc* clone(Uid uid) {
		return new TokenMaxPool(uid, InSize(input().length_), OutSize(out_size_));
	}

	virtual std::string getName() {
		return "TokenMaxPool_" + std::to_string(uid_);
	}

	int tokens() const { return N_; }
	int argmax(int p) const { return arg_[p]; }

	virtual void forward() {
		assert(no_outputs());
		const int P = out_size_;
		#pragma omp parallel for
		for (int p = 0; p < P; ++p) {
			float best = -1.f;
			int bt = 0;
			for (int t = 0; t < N_; ++t) {
				const float re = input().real_[t * P + p], im = input().imag_[t * P + p];
				const float m = re * re + im * im;
				if (m > best) {
					best = m;
					bt = t;
				}
			}
			arg_[p] = bt;
		}
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int p = 0; p < P; ++p) {
				output(indx).real_[offset(indx) + p] = input().real_[arg_[p] * P + p];
				output(indx).imag_[offset(indx) + p] = input().imag_[arg_[p] * P + p];
			}
		}
	}

	virtual void backward() {
		const int P = out_size_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int p = 0; p < P; ++p) {
				const int i = arg_[p] * P + p;
				auto dz      = output(indx).dz(offset(indx) + p);
				auto dz_star = output(indx).dz_star(offset(indx) + p);
				input().dz_real_[i]      += dz.real();
				input().dz_imag_[i]      += dz.imag();
				input().dz_star_real_[i] += dz_star.real();
				input().dz_star_imag_[i] += dz_star.imag();
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}
};

#endif /* IMPL_POOL_H_ */
