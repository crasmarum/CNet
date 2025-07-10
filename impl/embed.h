#ifndef IMPL_EMBED_H_
#define IMPL_EMBED_H_

#include <algorithm>
#include <omp.h>
#include <vector>
#include <memory>

#include "cfunc.h"
#include "../utils/myrand.h"

class CEmbedding: public CFunc, public InputFunc  {
	friend class ModelSaver;
	friend class GpuNet;
	friend class CNet;

	int *gpu_tokens_;
	int embedding_dim_;
	int no_embeddings_;
	std::vector<int> tokens_;
	int no_out_tokens_;

public:

	int embeddingDim() {
		return embedding_dim_;
	}

	int noEmbeddings() {
		return no_embeddings_;
	}

	int noOutTokens() {
		return no_out_tokens_;
	}

	CEmbedding(Uid uid, int embedding_dim, int max_out_tokens, int no_embeddings) :
			CFunc(uid, InSize(no_embeddings * embedding_dim), OutSize(max_out_tokens * embedding_dim)),
			gpu_tokens_(NULL), embedding_dim_(embedding_dim), no_embeddings_(no_embeddings), no_out_tokens_(max_out_tokens) {
		is_input_ = true;
	}

	CEmbedding(int embedding_dim, int max_out_tokens, int no_embeddings) :
		CFunc(InSize(no_embeddings * embedding_dim), OutSize(max_out_tokens * embedding_dim)),
		gpu_tokens_(NULL), embedding_dim_(embedding_dim), no_embeddings_(no_embeddings), no_out_tokens_(max_out_tokens) {
		is_input_ = true;
	}

	virtual CFunc* clone(Uid uid) {
		return new CEmbedding(uid, embedding_dim_, no_out_tokens_, no_embeddings_);
	}

	void printEmbedding() {
		int emb_offset = 0;
		for (int token = 0; token < no_embeddings_; ++token) {
			std::cout << token << ":\t";
			for (int indx = 0; indx < embedding_dim_; ++indx) {
				std::cout << input().z(emb_offset + indx) << ", ";
			}
			std::cout << std::endl;
			emb_offset += embedding_dim_;
		}
	}

	void printTokens() {
		std::cout << "Tokens:\t";
		for (auto token : tokens_) {
			std::cout << token << ", ";
		}
		std::cout << std::endl;
	}

	virtual std::string getName() {
		return "Embedding " + std::to_string(uid_);
//				+ " dim " + std::to_string(embedding_dim_)
//				+ " no_embeddings " + std::to_string(no_embeddings_)
//				+ " no-out_tokens " + std::to_string(no_out_tokens_);
	}

	void setInput(std::initializer_list<int> embedding_list) {
		tokens_.clear();
		for (auto id : embedding_list) {
			assert(id < no_embeddings_);
			tokens_.push_back(id);
		}
		assert(tokens_.size() && tokens_.size() <= no_out_tokens_);
	}

	void setInput(std::vector<int> embedding_list) {
		tokens_.clear();
		for (auto id : embedding_list) {
			assert(id < no_embeddings_);
			tokens_.push_back(id);
		}
		assert(tokens_.size() && tokens_.size() <= no_out_tokens_);
	}

	virtual void init_inputs() {
		for (int indx = 0; indx < input().length_; ++indx) {
			float re = (float)rand_.randDouble() - 0.5;
			input().real_[indx] = re / sqrt(embedding_dim_);
			input().imag_[indx] = sqrt(1 - re * re) / sqrt(embedding_dim_);
		}
	}

	virtual void forward() {
		assert(no_outputs());
		for (int out_indx = 0; out_indx < no_outputs(); ++out_indx) {
			#pragma omp parallel for
			for (auto tok_pos = 0; tok_pos < tokens_.size(); ++tok_pos) {
				int tok_offset = tokens_[tok_pos] * embedding_dim_;

				std::copy(input().real_ + tok_offset, input().real_ + tok_offset + embedding_dim_,
						next(out_indx)->input().real_ + offset(out_indx) + embedding_dim_ * tok_pos);
				std::copy(input().imag_ + tok_offset, input().imag_ + tok_offset + embedding_dim_,
						next(out_indx)->input().imag_ + offset(out_indx) + embedding_dim_ * tok_pos);
			}
			if (tokens_.size() < no_out_tokens_) {
				memset(next(out_indx)->input().real_ + offset(out_indx)
						          + embedding_dim_ * tokens_.size(), 0,
								  sizeof(float) * embedding_dim_ * (no_out_tokens_ - tokens_.size()));
				memset(next(out_indx)->input().imag_ + offset(out_indx)
								  + embedding_dim_ * tokens_.size(), 0,
								  sizeof(float) * embedding_dim_ * (no_out_tokens_ - tokens_.size()));

			}
		}
	}

	virtual void backward() {
		for (auto tok_pos = 0; tok_pos < tokens_.size(); ++tok_pos) {
			int tok_offset = tokens_[tok_pos] * embedding_dim_;

			#pragma omp parallel for
			for (int pos = 0; pos < embedding_dim_; ++pos) {
				for (int out_indx = 0; out_indx < no_outputs(); ++out_indx) {
					mutable_input()->dz_star_real_[tok_offset + pos] +=
							next(out_indx)->input().dz_star_real_[offset(out_indx) + embedding_dim_ * tok_pos + pos];
					mutable_input()->dz_star_imag_[tok_offset + pos] +=
							next(out_indx)->input().dz_star_imag_[offset(out_indx) + embedding_dim_ * tok_pos + pos];

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

	bool operator==(const CEmbedding& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()
				|| this->embedding_dim_ != rhs.embedding_dim_
				|| this->no_embeddings_ != rhs.no_embeddings_
				|| this->no_out_tokens_ != rhs.no_out_tokens_) {
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

		if (this->tokens_.size() != rhs.tokens_.size()) {
			return false;
		}
		for (int var = 0; var < tokens_.size(); ++var) {
			if (this->tokens_[var] != rhs.tokens_[var]) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const CEmbedding& rhs) const
	{
		return !operator==(rhs);
	}

	virtual int sizeInBytes() const {
		return input().size_in_bytes();
	}

	virtual void printInput() {
		std::cout << getName() << " on CPU:\n";
		std::cout << "\t tokens: ";
		for (auto token : tokens_) {
			std::cout << token << ",";
		}
		std::cout << std::endl;

		std::cout << "\t embeddings: ";
		for (int in_indx = 0; in_indx < std::min(16, input().length_); ++in_indx) {
			std::cout << in_indx / embedding_dim_ << ": " << input().z(in_indx) << ", ";
		}
		std::cout << std::endl;

	}
};

#endif /* IMPL_EMBED_H_ */
