#ifndef GPU_GPUVARS_H_
#define GPU_GPUVARS_H_

#include "compl.h"

struct GpuInVar {
	float* input_ptr_ = 0;  // pointer to Ijk
	int input_length_ = 0;  // input var length
	int output_length_ = 0; // output function length
	cmplx_* other_ = NULL;
	int tok_offset_ = 0;
};

struct GpuOutVar {
	float* out_ptr_ = 0;   		// pointer to the out var
	int out_length_ = 0;   		// out var length
	float reduce_real_ = 0;	// place holder for reducing results e.g., norm
	float reduce_imag_ = 0;	// place holder for reducing results e.g., norm
};

struct GpuUnitRoots {
	int length_ = 0;
	float* real_ = NULL;
	float* imag_ = NULL;
};

struct GpuCloneVar {
	int no_clones_ = 0;
	int data_len_ = 0;
	float *ancestor_ptr_ = NULL;
	float **clone_array_ptr_ = NULL;
};

struct GpuBatch {
	int f_length_ = 0;
	float *f_data_ = NULL;
	int i_length_ = 0;
	float *i_data_ = NULL;
	int label_ = -1;
};

#endif /* GPU_GPUVARS_H_ */
