#ifndef IMPL_BATCH_H_
#define IMPL_BATCH_H_

#include <assert.h>
#include <complex>
#include <initializer_list>
#include <vector>

#include "../impl/vars.h"

const int NO_TOKEN = -1;

class Batch {
friend class CNet;

protected:
	std::vector<float> losses_;
	std::vector<int> labels_;
	std::vector<int> sizes_;

	int current_pos_ = 0;
	int batch_size_ = 0;
	int sample_dim_ = 0;

	void addLoss(float loss) {
		losses_.push_back(loss);
	}
public:

	Batch() {}
	Batch(int batch_size, int sample_dim) : batch_size_(batch_size), sample_dim_(sample_dim) {}

	virtual ~Batch() {
	}

	int size() {
		return batch_size_;
	}

	int sampleDim() {
		return sample_dim_;
	}

	float loss(int indx) {
		assert(0 <= indx <= batch_size_);
		return losses_[indx];
	}


	int noOfLabels() const {
		return labels_.size();
	}

	const int* labelsPtr() const  {
		return &labels_[0];
	}

	int label(int indx) {
		assert(0 <= indx <= batch_size_);
		return labels_[indx];
	}
};


class EmbeddingBatch : public Batch {
friend class CNet;

std::vector<int> token_;
int counter_ = 0;

public:

EmbeddingBatch() : Batch() {}
EmbeddingBatch(int batch_size, int sample_dim) : Batch(batch_size, sample_dim) {}

virtual ~EmbeddingBatch() {
}

void add(std::vector<int> sample, int label) {
	assert(current_pos_ < batch_size_);
	token_.insert(token_.end(), sample.begin(), sample.end());

	while (token_.size() < (1 + current_pos_) * sample_dim_) {	// padding with -1s
		token_.push_back(NO_TOKEN);
	}

	labels_.push_back(label);
	sizes_.push_back(sample.size());
	current_pos_ ++;
	counter_ += sample.size();
}

int noOfTokens() const {
	return token_.size();
}

const int* tokensPtr() const {
	return &token_[0];
}

std::vector<int> batch(int indx) {
	assert(0 <= indx <= batch_size_);
	std::vector<int> ret(token_.begin() + indx * sample_dim_,
			token_.begin() + indx * sample_dim_ + sizes_[indx]);
	return ret;
}

int tokenCount() {
	return counter_;
}

void scale(int mod) {
	for (int var = 0; var < token_.size(); ++var) {
		if (token_[var] < 0) {
			continue;
		}
		token_[var] = token_[var] % mod;
	}
	for (int var = 0; var < labels_.size(); ++var) {
		labels_[var] = labels_[var] % mod;
	}
}

};

class InputBatch : public Batch {
	std::vector<float> real_data_;
	std::vector<float> imag_data_;
public:

	InputBatch() : Batch() {}
	InputBatch(int batch_size, int sample_dim) : Batch(batch_size, sample_dim) {}

	virtual ~InputBatch() {
	}

	void add(std::vector<std::complex<float> > sample, int label) {
		assert(current_pos_ < batch_size_);
		for (auto cmplx : sample) {
			real_data_.push_back(cmplx.real());
			imag_data_.push_back(cmplx.imag());
		}

		while (real_data_.size() < (1 + current_pos_) * sample_dim_) {	// padding with 0s
			real_data_.push_back(0.f);
			imag_data_.push_back(0.f);
		}

		labels_.push_back(label);
		sizes_.push_back(sample.size());
		current_pos_ ++;
	}

	void add(std::initializer_list<std::complex<float> > list, int label) {
		assert(current_pos_ < batch_size_);
		for (auto cmplx : list) {
			real_data_.push_back(cmplx.real());
			imag_data_.push_back(cmplx.imag());
		}

		while (real_data_.size() < (1 + current_pos_) * sample_dim_) {	// padding with 0s
			real_data_.push_back(0.f);
			imag_data_.push_back(0.f);
		}

		labels_.push_back(label);
		sizes_.push_back(list.size());
		current_pos_ ++;
	}

	std::vector<float> realData(int indx) {
		assert(0 <= indx <= batch_size_);
		std::vector<float> ret(real_data_.begin() + indx * sample_dim_,
				real_data_.begin() + (indx + 1) * sample_dim_);
		return ret;
	}

	std::vector<float> imagData(int indx) {
		assert(0 <= indx <= batch_size_);
		std::vector<float> ret(imag_data_.begin() + indx * sample_dim_,
				imag_data_.begin() + (indx + 1) * sample_dim_);
		return ret;
	}


	const float* realDataPtr() const {
		return &real_data_[0];
	}

	const float* imagDataPtr() const {
		return &imag_data_[0];
	}

	const float* realDataPtr(int batch_indx) const {
		return &real_data_[0] + batch_indx * sample_dim_;
	}

	const float* imagDataPtr(int batch_indx) const {
		return &imag_data_[0] + batch_indx * sample_dim_;
	}

	void setInput(Vars *input, int batch_indx) {
		assert(input->length_ <= sample_dim_);
		std::copy(real_data_.begin() + batch_indx * sample_dim_,
				  real_data_.begin() + (batch_indx + 1) * sample_dim_, input->real_);
		std::copy(imag_data_.begin() + batch_indx * sample_dim_,
				  imag_data_.begin() + (batch_indx + 1) * sample_dim_, input->imag_);
	}
};

#endif /* IMPL_BATCH_H_ */
