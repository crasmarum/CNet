#ifndef GPU_GPUMAPPING_H_
#define GPU_GPUMAPPING_H_

#include <vector>
#include <map>
#include <memory>

#include "gpuvars.h"
#include "compl.h"
#include "../impl/cfunc.h"
#include "../impl/cnet.h"

/**
 * Input: a batch X0,X1,...,Xm where each Xj is of form Ij0,Ij1,...,Ijn
 *        allocated contiguously in the GPU memory. Each Ijk has same
 *        length L.
 *
 * Output: A list of pointers Y00,Y01,...,Y0n,Y10,...., ..., Ymn
 *         where Yij points to a block of memory of length bL
 *
 * Kernels will get (GpuInVar* in, GpuInVar* out, int batch_length_).
 * Note that for inputs that have more than one output we need to add additional
 * blocks.
 * Ex: for X -> Y1, Y2 then we add (X, Y1) and (X, Y2).
 */
class GpuMapping {
	friend class GpuNet;
	friend class CNet;

protected:
	int depth_;
	std::vector<GpuInVar> in_;		   // Ijk
	std::vector<GpuOutVar> out_;       // Bjk
	std::vector<CFunc*> cpu_func_;	   // not owned
	std::vector<int> out_offset_;

public:
	// TODO: make them private
	GpuInVar *gpu_in_ptr_ = 0;
	GpuOutVar *gpu_out_ptr_ = 0;

	GpuMapping(int depth) : depth_(depth) {
	}

	std::vector<GpuOutVar> getOuts() {
		return out_;
	}

	int getNoMappings() {
		return in_.size();
	}

	int getOutputLength() {
		assert(in_.size());
		return cpu_func_.front()->outSize();
	}

	int length() {
		assert(cpu_func_.size());
		return cpu_func_.front()->input().length_;
	}

	std::vector<CFunc*>& getCpuFun() {
		return cpu_func_;
	}

	virtual ~GpuMapping() {
	}

	virtual void forward() = 0;

	virtual void backward(int label) = 0;

	void zeroGradients();
};

#endif /* GPU_GPUMAPPING_H_ */
