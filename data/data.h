#ifndef TESTS_DATA_H_
#define TESTS_DATA_H_

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../utils/myrand.h"

#include <fstream>
#include <iostream>

#include <algorithm>
#include <set>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "../impl/batch.h"
#include "../impl/log.h"

struct DataRecord {
	std::vector<std::complex<float> > data_;
	int label_ = 0;

	DataRecord(std::vector<std::complex<float> > data, int label)
		: data_(data), label_(label) {
	}
};

class BatchedDataReader {
	int current_ = 0;
	std::vector<DataRecord> samples_;

protected:
	int sample_dim_ = 0;

public:
	BatchedDataReader(int sample_dim) : sample_dim_(sample_dim) {
	}

	virtual ~BatchedDataReader() {
	}

	int size() {
		return samples_.size();
	}

	void shuffle() {
		std::random_device rd;
		auto rng = std::default_random_engine { rd() };
		std::shuffle(std::begin(samples_), std::end(samples_), rng);
	}

	bool hasNextBatch(int batch_size) {
		return current_ + batch_size <= samples_.size();
	}

	InputBatch nextBatch(int batch_size) {
		assert(samples_.size());
		InputBatch batch(batch_size, sample_dim_);

		for (int i = 0; i < batch_size; ++i) {
			batch.add(samples_[current_].data_, samples_[current_].label_);
			current_ = (current_ + 1) % samples_.size();
		}
		return batch;
	}

	void add(std::vector<std::complex<float> > data, int label) {
		samples_.push_back(DataRecord(data, label));
	}

	virtual bool readData() = 0;
};


FLAG_STRING(mnist_labels, "train-labels.idx1-ubyte")
FLAG_STRING(mnist_images, "train-images.idx3-ubyte ")

class MnistDataReader : public BatchedDataReader {
	std::ifstream images_;
	std::ifstream labels_;
	bool is_open_;
	char *local_data_;
	char *local_labels_;
	int current_indx_;
	int max_count_;

	bool open(std::string image_file, std::string label_file) {
		assert(!is_open_);
	    images_.open(image_file.c_str(), std::ios::in | std::ios::binary);
	    if (images_.fail()) {
			L_(lError) << "Cannot open " << image_file;
	    	return false;
	    }
	    labels_.open(label_file.c_str(), std::ios::in | std::ios::binary );
	    if (labels_.fail()) {
			L_(lError) << "Cannot open " << label_file;
	    	return false;
	    }

		// Reading file headers
	    char number;
	    for (int i = 1; i <= 16; ++i) {
	        images_.read(&number, sizeof(char));
		}
	    for (int i = 1; i <= 8; ++i) {
	    	labels_.read(&number, sizeof(char));
	    }
	    is_open_ = true;
		return true;
	}

public:

	virtual bool readData() override {
		assert(is_open_);

		while (current_indx_ < max_count_) {
			std::vector<std::complex<float> > sample;
			auto label = (int) local_labels_[current_indx_];

			for (int indx = 0; indx <  28 * 28; ++indx) {
				auto re = (float)local_data_[current_indx_ * 28 * 28 + indx];
				sample.push_back({re, re});
			}

			add(sample, label);

			current_indx_++;
		}

		return true;
	}

	bool Open(std::string image_file, std::string label_file, int count) {
		if (!open(image_file, label_file)) return false;
		is_open_ = false;

		local_data_ = (char*) malloc(count * 28 * 28 * sizeof(char));
		if (!local_data_) return false;
		if (!images_.read(local_data_, count * 28 * 28 * sizeof(char))) return false;

		local_labels_ = (char*) malloc(count * sizeof(char));
		if (!local_labels_) return false;
		if (!labels_.read(local_labels_, count * sizeof(char))) return false;

		max_count_ = count;
		L_(lInfo) << "read: " << max_count_ << " samples from: " << image_file;
		is_open_ = true;
		return true;
	}

	MnistDataReader() : BatchedDataReader(28 * 28), is_open_(false), local_data_(NULL),
			local_labels_(NULL), current_indx_(0), max_count_(0) {}

	virtual ~MnistDataReader() {
		if (local_data_) delete(local_data_);
		if (local_labels_) delete(local_labels_);
		images_.close();
		labels_.close();
	}
};


FLAG_STRING(radioml_train, "radioml_train.bin")
FLAG_STRING(radioml_test, "radioml_test.bin")

// Reads the synthetic RadioML-style IQ binary written by gen_radioml.py:
//   int32 magic=0x524D4C31, N, L, C, then N records of {int32 label, 2L float32
//   interleaved I,Q}. Each example becomes a length-L complex vector (real=I,
//   imag=Q). seq_len / n_classes are discovered from the header.
class RadioMLDataReader : public BatchedDataReader {
	std::ifstream in_;
	int count_ = 0;
	int seq_len_ = 0;
	int n_classes_ = 0;

public:
	RadioMLDataReader() : BatchedDataReader(0) {}

	int numClasses() { return n_classes_; }
	int seqLen() { return seq_len_; }

	bool Open(std::string path) {
		in_.open(path.c_str(), std::ios::in | std::ios::binary);
		if (in_.fail()) {
			L_(lError) << "Cannot open " << path;
			return false;
		}
		int32_t magic = 0;
		if (!in_.read((char*) &magic, sizeof(int32_t))
				|| !in_.read((char*) &count_, sizeof(int32_t))
				|| !in_.read((char*) &seq_len_, sizeof(int32_t))
				|| !in_.read((char*) &n_classes_, sizeof(int32_t))) {
			L_(lError) << "Cannot read header from " << path;
			return false;
		}
		if (magic != 0x524D4C31) {
			L_(lError) << "Bad magic in " << path;
			return false;
		}
		sample_dim_ = seq_len_;
		return true;
	}

	virtual bool readData() override {
		std::vector<float> buf(2 * seq_len_);
		for (int i = 0; i < count_; ++i) {
			int32_t label = 0;
			if (!in_.read((char*) &label, sizeof(int32_t))) {
				return false;
			}
			if (!in_.read((char*) buf.data(), 2 * seq_len_ * sizeof(float))) {
				return false;
			}
			std::vector<std::complex<float> > sample(seq_len_);
			for (int k = 0; k < seq_len_; ++k) {
				sample[k] = { buf[2 * k], buf[2 * k + 1] };
			}
			add(sample, label);
		}
		L_(lInfo) << "read: " << count_ << " RadioML examples (L=" << seq_len_
				  << ", C=" << n_classes_ << ")";
		return true;
	}

	virtual ~RadioMLDataReader() {
		in_.close();
	}
};


#endif /* TESTS_DATA_H_ */
