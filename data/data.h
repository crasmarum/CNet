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


#endif /* TESTS_DATA_H_ */
