#ifndef IMPL_TOKENWISE_H_
#define IMPL_TOKENWISE_H_

#include <complex>
#include <omp.h>

#include "cfunc.h"
#include "utils.h"

// Position-wise (token-wise) linear layer: the SAME e_in x e_out weight matrix
// is applied independently to each of n_tokens contiguous e_in-sized slices of
// the input. This is the FNet / Transformer feed-forward primitive -- it mixes
// FEATURES within a token but never mixes tokens, so when it follows a Fourier
// token-mixer the only cross-token coupling is the (parameter-free) DFT.
//
// Layout (one concatenated input vector, exactly like Linear):
//   [ data : n_tokens * e_in ][ weight : e_out * e_in ]
//   data(t,c)   = z(t*e_in + c)                  t in [0,n_tokens), c in [0,e_in)
//   weight(r,c) = z(n_tokens*e_in + r*e_in + c)  r in [0,e_out)
//   out(t,r)    = sum_c data(t,c) * weight(r,c)  -> index t*e_out + r
//
// Cost is O(n_tokens * e_in * e_out) with only e_in*e_out parameters, versus a
// full Linear's O((n_tokens*e_in)^2). Wirtinger gradients mirror Linear's; the
// weight gradient sums over all tokens because the weight is shared.
class TokenwiseLinear: public CFunc {
	friend class ModelSaver;

	int n_tokens_;
	int e_in_;
	int e_out_;

	static int totalIn(int n, int ein, int eout)  { return n * ein + eout * ein; }
	static int totalOut(int n, int ein, int eout) { return n * eout; }

public:
	TokenwiseLinear(Uid uid, int n_tokens, int e_in, int e_out)
			: CFunc(uid, InSize(totalIn(n_tokens, e_in, e_out)),
					OutSize(totalOut(n_tokens, e_in, e_out))),
			  n_tokens_(n_tokens), e_in_(e_in), e_out_(e_out) {
	}

	TokenwiseLinear(int n_tokens, int e_in, int e_out)
			: CFunc(InSize(totalIn(n_tokens, e_in, e_out)),
					OutSize(totalOut(n_tokens, e_in, e_out))),
			  n_tokens_(n_tokens), e_in_(e_in), e_out_(e_out) {
	}

	virtual ~TokenwiseLinear() {
	}

	virtual CFunc* clone(Uid uid) {
		return new TokenwiseLinear(uid, n_tokens_, e_in_, e_out_);
	}

	int nTokens()    const { return n_tokens_; }
	int inDim()      const { return e_in_; }
	int outDim()     const { return e_out_; }
	int weightBase() const { return n_tokens_ * e_in_; }  // index where the weight starts

	virtual std::string getName() {
		return "TokenwiseLinear_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());
		const int w_base = n_tokens_ * e_in_;
		std::vector<std::complex<float> > sums((size_t) n_tokens_ * e_out_);

		#pragma omp parallel for
		for (int o = 0; o < n_tokens_ * e_out_; ++o) {
			int t = o / e_out_;
			int r = o % e_out_;
			std::complex<float> sum = 0;
			for (int c = 0; c < e_in_; ++c) {
				sum += input().z(t * e_in_ + c) * input().z(w_base + r * e_in_ + c);
			}
			sums[o] = sum;
		}

		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int o = 0; o < n_tokens_ * e_out_; ++o) {
				output(indx).real_[offset(indx) + o] = sums[o].real();
				output(indx).imag_[offset(indx) + o] = sums[o].imag();
			}
		}
	}

	virtual void backward() {
		const int w_base = n_tokens_ * e_in_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			for (int t = 0; t < n_tokens_; ++t) {
				for (int r = 0; r < e_out_; ++r) {
					int o = t * e_out_ + r;
					auto dLdz      = output(indx).dz(offset(indx) + o);
					auto dLdz_star = output(indx).dz_star(offset(indx) + o);
					for (int c = 0; c < e_in_; ++c) {
						int d_index = t * e_in_ + c;
						int w_index = w_base + r * e_in_ + c;

						// gradient w.r.t. the data slice
						auto w = input().z(w_index);
						auto dz      = dLdz * w;
						auto dz_star = dLdz_star * std::conj(w);
						input().dz_real_[d_index]      += dz.real();
						input().dz_imag_[d_index]      += dz.imag();
						input().dz_star_real_[d_index] += dz_star.real();
						input().dz_star_imag_[d_index] += dz_star.imag();

						// gradient w.r.t. the shared weight (summed over tokens)
						auto d = input().z(d_index);
						dz      = dLdz * d;
						dz_star = dLdz_star * std::conj(d);
						input().dz_real_[w_index]      += dz.real();
						input().dz_imag_[w_index]      += dz.imag();
						input().dz_star_real_[w_index] += dz_star.real();
						input().dz_star_imag_[w_index] += dz_star.imag();
					}
				}
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

	bool operator==(const TokenwiseLinear& rhs) const {
		if (this->uid_ != rhs.uid_
				|| this->n_tokens_ != rhs.n_tokens_
				|| this->e_in_ != rhs.e_in_
				|| this->e_out_ != rhs.e_out_
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
	bool operator!=(const TokenwiseLinear& rhs) const {
		return !operator==(rhs);
	}
};

#endif /* IMPL_TOKENWISE_H_ */
