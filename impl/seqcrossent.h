#ifndef IMPL_SEQCROSSENT_H_
#define IMPL_SEQCROSSENT_H_

#include <cfloat>
#include <vector>

#include "cfunc.h"
#include "vars.h"

// Per-position Born-rule cross entropy for an autoregressive LM. The input is
// n_pos contiguous vocab-sized blocks (position-major: block p = logits for the
// token that follows position p). Each block is treated as an independent Born
// measurement p_k = |z_k|^2 / ||z||^2, and the loss is the mean over positions of
// -log p_{target_p}. Targets are set per step via setTargets(); they are the
// input sequence shifted left by one. This is the sequence analogue of
// CrossEntropy -- the same math applied to every position in one layer.
class SequenceCrossEntropy: public CFunc, public OutputFunc {
	Vars output_;
	int vocab_;
	int n_pos_;
	std::vector<int> targets_;          // per-position target token
	std::vector<float> pos_sqnorm_;     // per-position ||z_p||^2
	// Optional probability floor for the TARGET branch of the gradient only.
	// The exact Born gradient of -log(|z_t|^2/||z||^2) grows like 1/|z_t| when the
	// correct-token amplitude collapses, so a single hard example can produce an
	// explosive update that spikes training. eps_ > 0 floors the target's Born
	// probability at eps (sqmod <- max(sqmod, eps*||z||^2)) in the gradient, which
	// bounds the update (a uniform measurement-noise floor, i.e. label-smoothing
	// for the Born rule). loss()/forward() stay exact, so the reported NLL metric
	// is unchanged -- only the optimisation is regularised. eps_ == 0 is exact.
	float eps_ = 0.0f;
	float smooth_ = 0.0f;   // Laplace-smoothed Born loss: p_t=(|z_t|^2+s)/(sum+V*s)

public:
	SequenceCrossEntropy(Uid uid, InSize in_size, int vocab)
			: CFunc(uid, in_size, OutSize(1)), output_(in_size.value()),
			  vocab_(vocab), n_pos_(in_size.value() / vocab) {
		assert(in_size.value() % vocab == 0);
		is_output_ = true;
		pos_sqnorm_.assign(n_pos_, 1.0f);
		targets_.assign(n_pos_, 0);
	}

	SequenceCrossEntropy(InSize in_size, int vocab)
			: CFunc(in_size, OutSize(1)), output_(in_size.value()),
			  vocab_(vocab), n_pos_(in_size.value() / vocab) {
		assert(in_size.value() % vocab == 0);
		is_output_ = true;
		pos_sqnorm_.assign(n_pos_, 1.0f);
		targets_.assign(n_pos_, 0);
	}

	virtual ~SequenceCrossEntropy() {
	}

	virtual CFunc* clone(Uid uid) override {
		auto* c = new SequenceCrossEntropy(uid, InSize(input().length_), vocab_);
		c->eps_ = eps_;
		c->smooth_ = smooth_;
		return c;
	}

	virtual std::string getName() override {
		return "SequenceCrossEntropy_" + std::to_string(uid_);
	}

	int vocab() const { return vocab_; }
	int nPos()  const { return n_pos_; }
	const int* targetsData() const { return targets_.data(); }   // for GPU upload
	float eps() const { return eps_; }
	void setEps(float e) { eps_ = e; }
	float smooth() const { return smooth_; }
	void setSmooth(float s) { smooth_ = s; }

	// targets[p] is the token that position p must predict (the input shifted
	// left by one). Size must equal n_pos_.
	void setTargets(const std::vector<int>& targets) {
		assert((int) targets.size() == n_pos_);
		targets_ = targets;
	}

	void getOutTemplate(int *no_planes, int *plane_size) {
	}

	// Born probability the model assigned to token k at position p.
	float getProbability(int p, int k) const {
		int gi = p * vocab_ + k;
		return output_.real_[gi] * output_.real_[gi] + output_.imag_[gi] * output_.imag_[gi];
	}

	virtual void forward() override {
		#pragma omp parallel for
		for (int p = 0; p < n_pos_; ++p) {
			double sum = 0.0;
			int base = p * vocab_;
			for (int k = 0; k < vocab_; ++k) {
				sum += (double) input().real_[base + k] * input().real_[base + k]
					 + (double) input().imag_[base + k] * input().imag_[base + k];
			}
			float sqn = (sum <= 1e-15) ? 1e-15f : (float) sum;
			pos_sqnorm_[p] = sqn;
			float nrm = sqrtf(sqn);
			for (int k = 0; k < vocab_; ++k) {
				output_.real_[base + k] = input().real_[base + k] / nrm;
				output_.imag_[base + k] = input().imag_[base + k] / nrm;
			}
		}
	}

	// Mean over positions of -log(|z_{p,target_p}|^2 / ||z_p||^2).
	virtual float loss(int /*unused*/) override {
		double total = 0.0;
		for (int p = 0; p < n_pos_; ++p) {
			float prob = getProbability(p, targets_[p]);
			prob = (prob < 1e-15f) ? 1e-15f : prob;
			total += -std::log(prob);
		}
		return (float) (total / n_pos_);
	}

	virtual void backward(int /*unused*/) override {
		const float inv_n = 1.0f / (float) n_pos_;
		#pragma omp parallel for
		for (int p = 0; p < n_pos_; ++p) {
			int base = p * vocab_;
			int tgt = targets_[p];
			float sqn = pos_sqnorm_[p];
			for (int k = 0; k < vocab_; ++k) {
				int gi = base + k;
				if (k == tgt) {
					float sqmod = input().real_[gi] * input().real_[gi]
								+ input().imag_[gi] * input().imag_[gi];
					sqmod = (sqmod < 1e-15f) ? 1e-15f : sqmod;
					// probability floor: clamp the target modulus from below so the
					// 1/|z_t| blow-up of the exact gradient is bounded (eps_==0: exact).
					if (eps_ > 0.0f) {
						float floor = eps_ * sqn;
						if (sqmod < floor) sqmod = floor;
					}
					float f = (sqn - sqmod) / (sqmod * sqn);
					input().dz_star_real_[gi] = -input().real_[gi] * f * inv_n;
					input().dz_star_imag_[gi] = -input().imag_[gi] * f * inv_n;
				} else {
					input().dz_star_real_[gi] = input().real_[gi] / sqn * inv_n;
					input().dz_star_imag_[gi] = input().imag_[gi] / sqn * inv_n;
				}
				input().dz_real_[gi] = input().dz_star_real_[gi];
				input().dz_imag_[gi] = -input().dz_star_imag_[gi];
			}
		}
	}

	virtual void backward() override {
	}

	virtual std::complex<float> dz(int out_indx, int in_index) override { return 0; }
	virtual std::complex<float> dz_star(int out_indx, int in_index) override { return 0; }

	// argmax token predicted at position p (for accuracy).
	int get_prediction(int p) const {
		int base = p * vocab_, ret = 0;
		float max = getProbability(p, 0);
		for (int k = 1; k < vocab_; ++k) {
			float v = getProbability(p, k);
			if (v > max) { max = v; ret = k; }
		}
		return ret;
	}

	// sample a token from position p's Born distribution |z_{p,k}|^2/||z_p||^2
	// (the per-position outputs are already normalised by forward()).
	int sample(int p) {
		float coin = (float) rand_.randDouble();
		float cum = 0.f, mx = 0.f;
		int argmax = 0;
		for (int k = 0; k < vocab_; ++k) {
			float pr = getProbability(p, k);
			cum += pr;
			if (coin <= cum) return k;
			if (pr > mx) { mx = pr; argmax = k; }
		}
		return argmax;
	}

	bool operator==(const SequenceCrossEntropy& rhs) const {
		if (this->uid_ != rhs.uid_ || this->input().length_ != rhs.input().length_
				|| this->vocab_ != rhs.vocab_ || this->no_outputs() != rhs.no_outputs()) {
			return false;
		}
		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const SequenceCrossEntropy& rhs) const { return !operator==(rhs); }

	virtual int sizeInBytes() const override {
		return input().size_in_bytes() + output_.size_in_bytes();
	}
	virtual int outputLength() override { return output_.length_; }
	virtual Vars* mutableOutput() override { return &output_; }
};

#endif /* IMPL_SEQCROSSENT_H_ */
