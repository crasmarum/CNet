#ifndef GPU_REDUCE_H_
#define GPU_REDUCE_H_

#include "gpumapping.h"
#include "compl.h"
#include "kernels.h"

// Flags.
extern int reduce_block_size;

class LinearGpu;
class FourierGpu;
class L2Gpu;

const int TEST_DATA_PROVIDER = 1;

const int CROSS_ENT_DATA_PROVIDER = 2;

const int LINEAR_DATA_PROVIDER = 3;

const int FFT_DATA_PROVIDER = 4;

const int SOFTMAX_DATA_PROVIDER = 5;

const int NORM_DATA_PROVIDER = 6;

const int T_FFT_DATA_PROVIDER = 7;

inline int getPaddedLength(int block_size, GpuMapping *mp) {
	int padded_segment_len =
			mp->length() % block_size == 0 ?
					mp->length() :
					(mp->length() + block_size - mp->length() % block_size);
	int b_out_length = (padded_segment_len / block_size) * mp->getNoMappings()
			* mp->length();
	return b_out_length;
}

inline int getPaddedOutLength(int block_size, GpuMapping *mp) {
	int padded_segment_len =
			mp->getOutputLength() % block_size == 0 ?
					mp->getOutputLength() :
					(mp->getOutputLength() + block_size - mp->getOutputLength() % block_size);
	int b_out_length = (padded_segment_len / block_size) * mp->getNoMappings()
			* mp->getOutputLength();
	return b_out_length;
}

class LinearGpu : public GpuMapping {

public:
	LinearGpu(int depth) : GpuMapping(depth) {
	}
	virtual ~LinearGpu() {
	}

	void gpu_linear_forward(int block_size);

	void gpu_linear_backward(int label, int block_size);

	virtual void forward() override {
		gpu_linear_forward(reduce_block_size);
	}

	virtual void backward(int label) override {
		gpu_linear_backward(label, reduce_block_size);
	}
};

class FourierGpu : public GpuMapping {

public:
	FourierGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~FourierGpu() {
	}

	void gpu_fft_forward(int block_size);

	virtual void forward() override {
		gpu_fft_forward(reduce_block_size);
	}

	void gpu_fft_backward(int label, int block_size);

	virtual void backward(int label) override {
		gpu_fft_backward(label, reduce_block_size);
	}
};

class TrianFourierGpu : public GpuMapping {

public:
	TrianFourierGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~TrianFourierGpu() {
	}

	void gpu_T_fft_forward(int block_size);

	virtual void forward() override {
		gpu_T_fft_forward(reduce_block_size);
	}

	void gpu_T_fft_backward(int label, int block_size);

	virtual void backward(int label) override {
		gpu_T_fft_backward(label, reduce_block_size);
	}
};

class SoftMaxGpu : public GpuMapping {

public:
	SoftMaxGpu(int depth) : GpuMapping(depth) {
	}

	virtual ~SoftMaxGpu() {
	}

	void gpu_soft_max_forward(int block_size);

	virtual void forward() {
		gpu_soft_max_forward(reduce_block_size);
	}

	void gpu_soft_max_backward(int label, int block_size);

	virtual void backward(int label) {
		gpu_soft_max_backward(label, reduce_block_size);
	}
};

class CrossEntropyGpu : public GpuMapping {
	friend class GpuNet;
	friend class CNet;

	int *gpu_labels_ = NULL;

public:

	CrossEntropyGpu(int depth)
		: GpuMapping(depth) {
	}
	virtual ~CrossEntropyGpu() {
	}

	void gpu_cross_ent_forward(int block_size);

	virtual void forward() {
		gpu_cross_ent_forward(reduce_block_size);
	}

	void gpu_cross_ent_backward(int label);

	virtual void backward(int label) {
		gpu_cross_ent_backward(label);
	}

};

class L2Gpu : public GpuMapping {
	friend class GpuNet;
	friend class CNet;

public:
	L2Gpu(int depth) : GpuMapping(depth) {
	}

	virtual ~L2Gpu() {
	}

	void gpu_norm_forward(int block_size);

	virtual void forward() {
		gpu_norm_forward(reduce_block_size);
	}

	void gpu_norm_backward();

	virtual void backward(int label) {
		gpu_norm_backward();
	}
};


#endif /* GPU_REDUCE_H_ */
