#ifndef GPU_DATA_H_
#define GPU_DATA_H_

#include <vector>
#include <map>
#include <memory>

#include "gpuvars.h"
#include "gpumapping.h"
#include "kernels.h"
#include "reduce.h"
#include "../impl/cfunc.h"
#include "../impl/cnet.h"
#include "compl.h"

class GpuNet {
	friend class CNet;

	std::vector<GpuMapping*> gpu_maps_;	// owned
	bool hasGradients_ = true;
	long allocated_memory_ = 0;
	std::vector<GpuMapping*> inputs_;
	std::vector<GpuMapping*> outputs_;
	std::map<int, cmplx_*> unity_roots_;

public:
	virtual ~GpuNet() {
		for (auto gpu_fc : gpu_maps_) {
			delete gpu_fc;
		}
	}

	void setAllocatedMemoryLengthBytes(long allocated_memory_bytes) {
		allocated_memory_ = allocated_memory_bytes;
	}

	bool isInitialized() {
		return gpu_maps_.size();
	}

	void add(float **gpu_ptr, std::vector<CFunc*>& block, int depth) {
		assert(block.size());
		switch (ComplexNet::getType(block.front()))
		{
		    case isInput:
		        add(new InputGpu(depth), gpu_ptr, block);
		        break;
		    case isCrossEntropy:
		    	add(new CrossEntropyGpu(depth), gpu_ptr, block);
		        break;
		    case isCrelu:
		    	add(new ReluGpu(depth), gpu_ptr, block);
		        break;
		    case isLinear:
		    	add(new LinearGpu(depth), gpu_ptr, block);
		        break;
		    case isFourier:
		    	add(new FourierGpu(depth), gpu_ptr, block);
		        break;
		    case isTrianFourier:
		    	add(new TrianFourierGpu(depth), gpu_ptr, block);
		        break;
		    case isSoftMax:
		    	add(new SoftMaxGpu(depth), gpu_ptr, block);
		        break;
		    case isL2Out:
		    	add(new L2Gpu(depth), gpu_ptr, block);
		        break;
		    case isResidual:
		    	add(new ResidualGpu(depth), gpu_ptr, block);
		        break;
		    case isHadamard:
		    	add(new HadamardGpu(depth), gpu_ptr, block);
		        break;
		    case isGelu:
		    	add(new GeluGpu(depth), gpu_ptr, block);
		        break;
		    case isEmbedding:
		    	add(new EmbeddingGpu(depth, ((CEmbedding*)block[0])->embedding_dim_,
		    				((CEmbedding*)block[0])->no_embeddings_, ((CEmbedding*)block[0])->no_out_tokens_),
		    			gpu_ptr, block);
		        break;
		    default:
		        std::cerr << "Not implemented for: " << block.front()->getName()
				          << std::endl;
		        throw std::invalid_argument("Not implemented for: " + block.front()->getName());
		}
	}

	void addGpuOuts() {
		for (auto map : gpu_maps_) {
			for (int indx = 0; indx < map->in_.size(); ++indx) {
				CFunc *cpu_fun = map->cpu_func_[indx];

				if (CrossEntropyGpu *ce = dynamic_cast<CrossEntropyGpu*>(map)) {
					GpuOutVar outVar;
					outVar.out_ptr_ = map->in_[indx].input_ptr_
							+ (hasGradients_ ? 6 : 2) * cpu_fun->input().length_;
					outVar.out_length_ = cpu_fun->input().length_;

					map->out_.push_back(outVar);
				} else if (L2Gpu *ce = dynamic_cast<L2Gpu*>(map)) {
					GpuOutVar outVar;
					outVar.out_ptr_ = map->in_[indx].input_ptr_
							+ (hasGradients_ ? 6 : 2) * cpu_fun->input().length_;
					outVar.out_length_ = cpu_fun->input().length_;
					map->out_.push_back(outVar);
				} else {
					// In this step we only add the first output.
					GpuOutVar outVar;
					GpuInVar outputIn = cpu_fun->next(0)->gpu_var_;
					outVar.out_ptr_ = outputIn.input_ptr_ + cpu_fun->offset(0);
					outVar.out_length_ = cpu_fun->next(0)->input().length_;

					map->out_.push_back(outVar);
				}
			}

			assert(map->in_.size() == map->out_.size());
		}

		for (auto map : gpu_maps_) {
			std::vector<GpuInVar> collect;
			for (int indx = 0; indx < map->in_.size(); ++indx) {
				CFunc *cpu_fun = map->cpu_func_[indx];
				GpuInVar inVar = map->in_[indx];

				// TODO: use isOutput
				if (CrossEntropyGpu *ce = dynamic_cast<CrossEntropyGpu*>(map)) {
					continue;
				} else if (cpu_fun->no_outputs() > 1) {
					for (int fc_out_indx = 1; fc_out_indx < cpu_fun->no_outputs(); ++fc_out_indx) {
						GpuOutVar outVar;
						GpuInVar outputIn = cpu_fun->next(fc_out_indx)->gpu_var_;
						outVar.out_ptr_ = outputIn.input_ptr_ + cpu_fun->offset(fc_out_indx);
						outVar.out_length_ = cpu_fun->next(fc_out_indx)->input().length_;

						collect.push_back(inVar);	// map->in_.push_back(inVar);
						map->out_.push_back(outVar);
						map->cpu_func_.push_back(cpu_fun);
						map->out_offset_.push_back(cpu_fun->offset(fc_out_indx));
					}
				}
			}

			map->in_.insert(map->in_.end(), collect.begin(), collect.end());
			assert(map->in_.size() == map->out_.size());
		}
	}

	void printGraph() {
		std::cout << std::endl << std::endl;
		std::cout << "GPU mappings: " << gpu_maps_.size() << std::endl << "\t";
		int count = 0;
		for (auto map : gpu_maps_) {
			assert(map->in_.size() == map->out_.size());
			assert(map->in_.size() == map->cpu_func_.size());
			std::cout << map->cpu_func_[0]->getName() << ":"
					  << map->in_.size() << ", ";
			count += gpu_maps_.size();
		}
		std::cout << std::endl << "\tblocks: " << count << std::endl;

		float* prev_ptr = gpu_maps_[0]->in_[0].input_ptr_;
		for (auto map : gpu_maps_) {
			for (int indx = 0; indx < map->in_.size(); ++indx) {
				std::cout << map->depth_ << ":\t"
						<< map->in_[indx].input_ptr_ << "\t"
						<< prev_ptr <<  " +\t"
						<< 4 * (int64_t)(map->in_[indx].input_ptr_  - prev_ptr) << "\t"
						<< "out: " << map->out_[indx].out_ptr_ << " of inp len: "
						           << map->out_[indx].out_length_ << "\t"
						<< map->cpu_func_[indx]->getName() << " bytes: "
						<< map->cpu_func_[indx]->input().length_ << "x" << sizeof(float)
						<< " out offset: " << map->out_offset_[indx]
						<< " total allocated " << map->cpu_func_[indx]->sizeInBytes() << " bytes"
						<< std::endl;
				prev_ptr = map->in_[indx].input_ptr_;
			}
		}
		std::cout << prev_ptr <<  " +\t" << allocated_memory_ / sizeof(float) << std::endl;
		std::cout << std::endl << std::endl;
	}

	void reset() {
		for (auto gpu_fc : gpu_maps_) {
			delete gpu_fc;
		}
		gpu_maps_.clear();
		inputs_.clear();
		outputs_.clear();
	}

	void collectVarsFromMaps(std::vector<GpuInVar>& in_vec, std::vector<GpuOutVar>& out_vec) {
		assert(gpu_maps_.size());
		for (auto map : gpu_maps_) {
			assert( map->in_.size() ==  map->out_.size());
			in_vec.insert(in_vec.end(),  map->in_.begin(),  map->in_.end());
			out_vec.insert(out_vec.end(), map->out_.begin(), map->out_.end());
		}
		assert(in_vec.size() == out_vec.size());
	}

	void assignAdresses(GpuInVar *in_ptr, GpuOutVar *out_ptr) {
		for (auto map : gpu_maps_) {
			map->gpu_in_ptr_ = in_ptr;
			in_ptr += map->in_.size();

			map->gpu_out_ptr_ = out_ptr;
			out_ptr += map->out_.size();
		}
	}

	void forward() {
		assert(gpu_maps_.size());
		for (auto map : gpu_maps_) {
			std::cout << "\nForwarding block: ";
			for (auto fun : map->cpu_func_) {
				std::cout << fun->getName() << ", ";
			}
			std::cout << std::endl;

			map->forward();
		}
	}

	void backward(int label) {
		assert(gpu_maps_.size());
				for (auto map : gpu_maps_) {
			std::cout << "\nBackward on block: ";
			for (auto fun : map->cpu_func_) {
				std::cout << fun->getName() << ", ";
			}
			std::cout << std::endl;

			map->backward(label);
		}
	}

private:
	void add(GpuMapping *map, float **gpu_ptr, std::vector<CFunc*>& collect) {
		gpu_maps_.push_back(map);
		if (collect.front()->isInput()) {
			inputs_.push_back(map);
		}
		if (collect.front()->isOutput()) {
			outputs_.push_back(map);
		}

		for (auto func : collect) {
			GpuInVar inVar;

			inVar.input_ptr_ = *gpu_ptr;
			inVar.input_length_ = func->input().length_;
			inVar.output_length_ = func->outSize();
			*gpu_ptr += func->sizeInBytes() / sizeof(float);

			if (dynamic_cast<FourierTrans*>(func)) {
				assert(unity_roots_.find(func->input().length_) != unity_roots_.end());
				inVar.other_ = unity_roots_[func->input().length_];
			} else if (dynamic_cast<TriangFourier*>(func)) {
				assert(unity_roots_.find(func->input().length_) != unity_roots_.end());
				inVar.other_ = unity_roots_[func->input().length_];
			}

			func->gpu_var_ = inVar;
			map->in_.push_back(inVar);
			map->cpu_func_.push_back(func);

			map->out_offset_.push_back(func->no_outputs() > 0 ? func->offset(0) : 0);
		}
	}
};

#endif /* GPU_DATA_H_ */
