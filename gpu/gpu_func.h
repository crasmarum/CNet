#ifndef GPU_GPU_FUNC_H_
#define GPU_GPU_FUNC_H_

#include <algorithm>
#include <cstdio>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>



#include "../impl/cnet.h"
#include "../impl/model_saver.h"
#include "../impl/batch.h"
#include "../impl/embed.h"

#include "../utils/stopwatch.h"
#include "../utils/base64.h"

#include "allocator.h"
#include "gpu_data.h"
#include "kernels.h"


#ifdef __CUDACC__
    #include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <cuda.h>
	#include <cuda_runtime_api.h>

#else

#endif


/**
 * Assume 1 GPU for now.
 */
class CNet {

	std::unique_ptr<ComplexNet> net_;
	B64Encoding b64_encoder_;
	GpuHelper helper_;
	GpuNet gpu_net_;

	std::vector<GpuMapping*> maps_;
	GpuCloneVar *gpu_clone_data_ = NULL;
	int max_clone_data_len_ = 0;
	int no_ancestors_ = 0;
	std::vector<int> labels_;
	int *gpu_labels_ = NULL;

	void createExecutionGraph() {
		std::set<GpuMapping*> inputs;
		for (int depth = 0; depth < net_->maxDepth(); ++depth) {
			std::set<GpuMapping*> maps;
			for (auto func : net_->getFuncAtDepth(depth)) {
				GpuMapping *mp = getMappingFor(func);
				maps.insert(mp);
				if (func->isInput()) {
					inputs.insert(mp);
				}
			}
			for (auto mp : maps) {
				maps_.push_back(mp);
			}
		}
	}

	void addGpuIns(float **gpu_ptr) {
		for (int depth = 0; depth < net_->maxDepth(); ++depth) {
			// net_.print_func_at_depth(depth);

			auto funcs_at_depth = net_->getFuncAtDepth(depth);
			if (!funcs_at_depth.size()) {
				throw std::invalid_argument("AllocateNet error: 0 functions at depth: "
					     + std::to_string(depth));
			}

			auto lastFunc = funcs_at_depth.front();
			std::vector<CFunc*> collect;

			for (auto func : funcs_at_depth) {
				if (func->getType() != lastFunc->getType() || func->sizeInBytes() != lastFunc->sizeInBytes()) {
					gpu_net_.add(gpu_ptr, collect, depth);
					lastFunc = func;
					collect.clear();
				}
				collect.push_back(func);
			}
			gpu_net_.add(gpu_ptr, collect, depth);
		}
	}

	int allocateUnityRoots() {
		std::set<int> collect;
		int allocated = 0;
		for (auto fc : this->net_->functionList()) {
			if (auto fft = dynamic_cast<FourierTrans*>(fc)) {
				collect.insert(fft->input().length_);
			} else if (auto t_fft = dynamic_cast<TriangFourier*>(fc)) {
				collect.insert(t_fft->input().length_);
			}
		}
		if (!collect.size()) {
			return NO_TOKEN;
		}
		for (auto len : collect) {
			auto gpu_ptr = helper_.u_roots_allocate_and_copy_on_gpu(len);
			if (!gpu_ptr) return 0;
			gpu_net_.unity_roots_[len] = gpu_ptr;
			allocated += len * sizeof(cmplx_);
		}
		return allocated;
	}

	bool testAllocatedMaps() {
		for (auto map : gpu_net_.gpu_maps_) {
			std::vector<GpuInVar> tmp(map->in_.size());

			bool ret = helper_.in_var_copy_from_gpu(tmp.size(),
					map->gpu_in_ptr_, &tmp[0]);
			if (!ret) {
				std::cout << "Failure in_var_copy_from_gpu testAllocatedMaps." << std::endl;
				return false;
			}
			assert(tmp.size() == map->in_.size());
			for (int indx = 0; indx < tmp.size(); ++indx) {
				if (tmp[indx].input_ptr_ != map->in_[indx].input_ptr_) {
					std::cout << "Data mismatch in_var_copy_from_gpu at index "
							<<  indx << std::endl;
					return false;
				}
			}
		}
		std::cout << "Test copy from GPU and check GpuInVar[] passed." << std::endl;

		for (auto map : gpu_net_.gpu_maps_) {
			std::vector<GpuOutVar> tmp(map->out_.size());

			bool ret = helper_.out_var_copy_from_gpu(tmp.size(),
					map->gpu_out_ptr_, &tmp[0]);
			if (!ret) {
				std::cout << "Failure out_var_copy_from_gpu testAllocatedMaps." << std::endl;
				return false;
			}
			assert(tmp.size() == map->out_.size());
			for (int indx = 0; indx < tmp.size(); ++indx) {
				if (tmp[indx].out_ptr_ != map->out_[indx].out_ptr_) {
					std::cout << "\nData mismatch out_var_copy_from_gpu at index "
							<<  indx << std::endl;
					std::cout << tmp[indx].out_ptr_  << " != "
							<< map->out_[indx].out_ptr_ << std::endl;
					return false;
				}
			}
		}
		std::cout << "Test copy from GPU and check GpuOutVar[] passed." << std::endl;

		return true;
	}


	bool allocateCloneData() {
		std::vector<GpuCloneVar> clones;

		std::string info = "";
		for (auto func : cpuNet().inputs()) {
			if (func->isGpuOnly()) {
				continue;
			}

			GpuCloneVar clone_var;
			clone_var.ancestor_ptr_ = func->gpu_var_.input_ptr_;
			clone_var.data_len_ = func->gpu_var_.input_length_ * 4;
			max_clone_data_len_ = std::max(clone_var.data_len_, max_clone_data_len_);

			assert(cpuNet().gpu_clones_of_.find(func->uid())
					!= cpuNet().gpu_clones_of_.end());
			auto clone_list = cpuNet().gpu_clones_of_[func->uid()];
			clone_var.no_clones_ = clone_list.size();

			std::vector<float*> clone_data;
			for (auto clone : clone_list) {
				clone_data.push_back(clone->gpu_var_.input_ptr_);
			}
			assert(clone_data.size());

			clone_var.clone_array_ptr_ = helper_.ptr_float_allocate_on_gpu(clone_data.size());
			if (!clone_var.clone_array_ptr_) {
				std::cerr << "ptr_float_allocate_on_gpu" << std::endl;
				return false;
			}
			if (!helper_.ptr_float_copy_to_gpu(clone_var.no_clones_, &clone_data[0], clone_var.clone_array_ptr_)) {
				std::cerr << "ptr_float_copy_to_gpu" << std::endl;
				return false;
			}

			clones.push_back(clone_var);
			info += func->getName() + ", ";
		}
		no_ancestors_ = clones.size();

		gpu_clone_data_ = helper_.clone_var_allocate_on_gpu(clones.size());
		if (!gpu_clone_data_) {
			std::cerr << "clone_var_allocate_on_gpu" << std::endl;
			return false;
		}
		if (!helper_.clone_var_copy_to_gpu(clones.size(), &clones[0], gpu_clone_data_)) {
			std::cerr << "clone_var_copy_to_gpu" << std::endl;
			return false;
		}

		int batch_size = clones[0].no_clones_ + 1;
		gpu_labels_ = helper_.int_allocate_on_gpu(batch_size);
		if (!gpu_labels_) {
			std::cerr << "int_allocate_on_gpu" << std::endl;
			return false;
		}

		for (auto outp : cpuNet().outputs_) {
			if (auto cpu_ce = dynamic_cast<CrossEntropy*>(outp)) {
				if (cpu_ce->is_gpu_only_) {
					continue;
				}
				auto ce = (CrossEntropyGpu*)getMappingFor(cpu_ce);
				ce->gpu_labels_ = gpu_labels_;

				int indx = 0;
				for (auto func : ce->cpu_func_) {
					assert(func->batch_indx_ == indx++);
					if (!func->isGpuOnly()) {
						info += func->getName() + ", ";
					}
				}
			}
		}

		std::cout << "Clone data allocated on GPU for: " << info << std::endl;
		return true;
	}


	bool allocateMaps(std::vector<GpuInVar>& in_vec, std::vector<GpuOutVar>& out_vec) {
		GpuInVar *in_ptr = helper_.in_var_allocate_on_gpu(in_vec.size());
		if (!in_ptr) {
			return false;
		}
		std::cout << "Allocated: " << in_vec.size() << " GpuInVar at " << in_ptr << std::endl;

		GpuOutVar *out_ptr = helper_.out_var_allocate_on_gpu(out_vec.size());
		if (!out_ptr) {
			return false;
		}
		std::cout << "Allocated: " << out_vec.size() << " GpuOutVar at " << out_ptr << std::endl;

		gpu_net_.assignAdresses(in_ptr, out_ptr);

		for (auto map : gpu_net_.gpu_maps_) {
			bool ret = helper_.in_var_copy_to_gpu(map->in_.size(),
					&(map->in_[0]), map->gpu_in_ptr_);
			if (!ret) {
				std::cout << "Failure allocateMaps in_var_copy_to_gpu." << std::endl;
				return false;
			}
		}

		for (auto map : gpu_net_.gpu_maps_) {
			bool ret = helper_.out_var_copy_to_gpu(map->out_.size(),
					&(map->out_[0]), map->gpu_out_ptr_);
			if (!ret) {
				std::cout << "Failure allocateMaps out_var_copy_to_gpu." << std::endl;
				return false;
			}
		}

		return true;
	}

	bool uploadAncestorData(CFunc* fun) {
		auto ancestor = cpuNet().ancestor_of_[fun->uid_];
		assert(ancestor);
		// initialise it with ancestor data, dZ gradients included (for ADAM)
		return helper_.float_copy_to_gpu(4 * ancestor->input().length_,
				ancestor->mutable_input()->real_, fun->gpu_var_.input_ptr_);
	}

	/**
	 * The EmbeddingGpu keeps track of an array of inputs for embeddings,
	 * e.g., the input for the emb->in_[pos] embedding is at
	 * emb->gpu_tokens_[pos + tok_offset].
	 */
	bool allocateEmbeddingInput() {
		long allocated = 0;
		for (auto map : gpu_net_.gpu_maps_) {
			int indx = 0;
			if (auto emb = dynamic_cast<EmbeddingGpu*>(map)) {
				int size = 0;
				for (auto fun : emb->cpu_func_) {
					size += ((CEmbedding*)fun)->no_out_tokens_;
				}
				emb->gpu_tokens_ = helper_.int_allocate_on_gpu(size);
				if (!emb->gpu_tokens_) {
					return false;
				}
				emb->no_allocated_tokens_ = size;
				allocated += emb->noOutTokens() * sizeof(int);

				int offset = 0;
				for (auto fun : emb->cpu_func_) {
					((CEmbedding*)fun)->gpu_tokens_ = emb->gpu_tokens_ + offset;
					fun->gpu_var_.tok_offset_ = offset;
					map->in_[indx++].tok_offset_ = offset;

					offset += ((CEmbedding*)fun)->no_out_tokens_;
				}
			}
		}
		if (allocated) {
			std::cout << "Allocated: " << allocated << " bytes for embeddings." << std::endl;
		}
		return true;
	}


public:

	CNet() : net_(new ComplexNet()) {
	}

#ifdef __CUDACC__
	std::string PrintInfo(bool& was_failure);
	bool hasGPU(){ return true; }
	static size_t getGPUMemory(int gpuId) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpuId);
		return prop.totalGlobalMem;
	}
#else
	std::string PrintInfo(bool& was_failure) {
		was_failure = true;
		return "PrintInfo: GPU not supported.";
	}
	static size_t getGPUMemory(int gpuId) {
		return 0;
	}
	bool hasGPU() { return false; }
#endif


	ComplexNet& cpuNet() {
		return *net_;
	}

	GpuMapping* getMappingFor(CFunc *func) {
		for  (auto mapping : gpu_net_.gpu_maps_) {
			if (net_->getType(func) == net_->getType(mapping->cpu_func_.front())
					&& func->input().length_ == mapping->length()
					&& func->depth() == mapping->depth_) {
				return mapping;
			}
		}
		return NULL;
	}

	void batchToGpu(InputFunc *inp, OutputFunc *outp, Batch *batch) {
		assert(outp);
		CrossEntropy* ce = dynamic_cast<CrossEntropy*>(outp);
		if (!ce) {
			std::cerr << "This type of output is not supported yet." << std::endl;
			assert(ce);
		}

		assert(inp);
		if (CEmbedding* emb = dynamic_cast<CEmbedding*>(inp)) {
			EmbeddingBatch *emb_batch = dynamic_cast<EmbeddingBatch*>(batch);
			assert(emb_batch);

			assert(cpuNet().gpu_clones_of_.size());
			assert(cpuNet().gpu_clones_of_[emb->uid()].size() == batch->size() - 1);
			auto map = dynamic_cast<EmbeddingGpu*>(getMappingFor(emb));
			assert(map && map->gpu_tokens_);

			assert(map->in_.size() == batch->batch_size_
					&& map->no_out_tokens_ == batch->sampleDim());

			assert(helper_.int_copy_to_gpu(batch->size() * batch->sampleDim(), emb_batch->tokensPtr(), map->gpu_tokens_));

			labels_ = batch->labels_;
			assert(helper_.int_copy_to_gpu(batch->size(), &batch->labels_[0], gpu_labels_));
		} else if (CInput* cinp = dynamic_cast<CInput*>(inp)) {
			InputBatch *inp_batch = dynamic_cast<InputBatch*>(batch);
			assert(inp_batch);
			assert(cpuNet().gpu_clones_of_.size());
			assert(cpuNet().gpu_clones_of_[cinp->uid()].size() == batch->size() - 1);


			auto clones = net_->gpu_clones_of_[cinp->uid()];
			assert(clones.size() == batch->batch_size_ - 1);

			assert(helper_.float_copy_to_gpu(batch->sampleDim(),
				   inp_batch->realDataPtr(),
				   cinp->gpu_var_.input_ptr_));
			assert(helper_.float_copy_to_gpu(batch->sampleDim(),
				   inp_batch->imagDataPtr(),
				   cinp->gpu_var_.input_ptr_ + cinp->gpu_var_.input_length_));

			int count = 1;
			for (auto fun : clones) {
				assert(helper_.float_copy_to_gpu(batch->sampleDim(),
					   inp_batch->realDataPtr() + batch->sampleDim() * count,
					   fun->gpu_var_.input_ptr_));
				assert(helper_.float_copy_to_gpu(batch->sampleDim(),
					   inp_batch->imagDataPtr() + batch->sampleDim() * count,
					   fun->gpu_var_.input_ptr_ + fun->gpu_var_.input_length_));
				count++;
			}

			labels_ = batch->labels_;
			assert(helper_.int_copy_to_gpu(batch->size(), &batch->labels_[0], gpu_labels_));
		} else {
			std::cerr << "This type of input is not supported yet." << std::endl;
			assert(false);
		}
	}

	std::string DownloadNet(bool& was_failure) {
		if (!net_->functionList().size()) {
			was_failure = true;
			return "DownloadNet failed: Empty net.";
		}
		ModelSaver saver;
		auto str = saver.SaveToString(*net_);
		if (str.length() == 0) {
			was_failure = true;
			return "DownloadNet failed: SaveToString.";
		}

		return b64_encoder_.base64_encode(str);
	}

	std::string UploadNet(std::map<std::string, std::string>* parameters, bool& was_failure) {
		ModelSaver saver;
		if (parameters->find("net") == parameters->end()) {
			was_failure = true;
			return "UploadNet failed: missing parameter 'net'.";
		}
		std::string savedNet = (*parameters)["net"];
		savedNet = b64_encoder_.base64_decode(savedNet);

		if (!saver.RestoreFromString(*net_, savedNet)) {
			was_failure = true;
			return "UploadNet failed: RestoreFromString";
		}
		net_->init_exec_graph();

		std::cout << net_->toString() << std::endl;

		return "Net uploaded successfully: "
				+ std::to_string(net_->functionList().size()) + " functions.";
	}

	std::string EndCompute(bool& was_failure) {
		helper_.deallocate();
		gpu_net_.reset();
		net_.reset(new ComplexNet());
		return "All reset.";
	}

	bool inputToGpu(CFunc* input) {
		assert(input->isInput());
		if (auto emb = dynamic_cast<CEmbedding*>(input)) {
			assert(emb->gpu_tokens_ && emb->tokens_.size()
					&& emb->tokens_.size() <= emb->no_out_tokens_);
			int buff[emb->no_out_tokens_];
			memset(buff, NO_TOKEN, sizeof(int) * emb->no_out_tokens_);
			std::copy(emb->tokens_.begin(), emb->tokens_.end(), buff);
			assert(helper_.int_copy_to_gpu(emb->no_out_tokens_, buff, emb->gpu_tokens_));
		} else if (auto fun = dynamic_cast<CInput*>(input)) {
			if (!helper_.float_copy_to_gpu(2 * fun->input().length_,
					fun->mutable_input()->real_, fun->gpu_var_.input_ptr_)) {
				return false;
			}
		}

		return true;
	}

	void gpuUpdateInputs(float l_rate) {
		for (auto fun : net_->inputs()) {
			gpu_update_input(fun, l_rate);
		}
	}

	void updateInputs(float l_rate) {
		for (auto fun : net_->inputs()) {
			net_->update(l_rate, 1.0);
		}
	}

	void adamUpdate(float l_rate, float beta, int t) {
		for (auto fun : net_->inputs()) {
			if (fun->isGpuOnly()) {
				continue;
			}
			gpu_update_adam_input(fun, l_rate, beta, t);
		}
	}

	bool getInputsFromGpu() {
		for (auto fun : net_->inputs()) {
			if (fun->isGpuOnly()) {
				continue;
			}
			Vars tmp(fun->input().length_);
			if (!helper_.float_copy_from_gpu(2 * tmp.length_,
					fun->gpu_var_.input_ptr_, tmp.real_)) {

				std::cerr << "Cannot copy from GPU: " + fun->getName() << std::endl;
				return false;
			}
			std::copy(tmp.real_, tmp.real_ + 2 * tmp.length_, fun->mutable_input()->real_);
		}
		return true;
	}

	/**
	 * Copying the data Z and dZ from ancestor to all clones.
	 */
	void copyInputsToClones() {
		gpu_copy_to_clones(gpu_clone_data_, no_ancestors_, max_clone_data_len_);
	}

	/**
	 * Accumulating dZ_star from clones to their respective ancestor.
	 */
	void updateGradientsFromClones() {
		gpu_grad_from_clones(gpu_clone_data_, no_ancestors_, max_clone_data_len_ / 2);
	}

	bool copyInputsToGpu() {
		for (auto fun : net_->functionList()) {
			if (!fun->isInput()) {
				continue;
			}
			if (fun->is_gpu_only_) {	// a GPU clone
				if (uploadAncestorData(fun)) {
//					std::cout << fun->getName() << " initialised on GPU." << std::endl;
					continue;
				} else {
					std::cerr << "uploadAncestorData failed." << std::endl;
					return false;
				}
			}

			if (!helper_.float_copy_to_gpu(2 * fun->input().length_,
					fun->mutable_input()->real_, fun->gpu_var_.input_ptr_)) {
				std::cerr << "copyInputsToGpu failed." << std::endl;
				return false;
			}
			if (auto emb = dynamic_cast<CEmbedding*>(fun)) {
				if (emb->tokens_.size() == 0) {
					continue;
				}
				if(!helper_.int_copy_to_gpu(emb->tokens_.size(), (int*)&emb->tokens_[0], emb->gpu_tokens_)) {
					std::cerr << "CEmbedding copyInputsToGpu failed." << std::endl;
					return false;
				}
			}
		}
		return true;
	}

	std::string AllocateNet(int no_batches, bool& was_failure, bool print_graph = false) {
		was_failure = true;
		if (net_->maxDepth() == 0) {
			return "AllocateNet error: 0 depth net.";
		}
		if (gpu_net_.isInitialized()) {
			return "Net already initialised on GPU.";
		}

		float* gpu_ptr = helper_.float_allocate_on_gpu((net_->getTotalSizeBytes() / sizeof(float)) * no_batches);
		float* old_val = gpu_ptr;
		if (!gpu_ptr) {
			return "AllocateNet error: float_allocate_on_gpu";
		}
		std::cout << "Allocated: " << std::to_string(net_->getTotalSizeBytes() * no_batches)
				  << (hasGPU() ? " bytes on GPU." : " bytes on CPU") << std::endl;
		gpu_net_.setAllocatedMemoryLengthBytes(net_->getTotalSizeBytes() * no_batches); // TODO: add the other allocation sizes

		auto u_root_alloc = allocateUnityRoots();
		if (u_root_alloc > 0) {
			std::cout << "Allocated for unity roots: " << u_root_alloc << " bytes." << std::endl;
		} else if (!u_root_alloc) {
			return "AllocateNet error: allocateUnityRoots";
		}

		addGpuIns(&gpu_ptr);
		assert(gpu_ptr = old_val + net_->getTotalSizeBytes());

		if (!allocateEmbeddingInput()) {
			throw std::invalid_argument("AllocateNet error: allocateEmbeddingInput.");
		}

		gpu_net_.addGpuOuts();
		//gpu_net_.printGraph();


		std::vector<GpuInVar> in_vec;
		std::vector<GpuOutVar> out_vec;
		gpu_net_.collectVarsFromMaps(in_vec, out_vec);
		std::cout << "Collected: "
				  << 2 * in_vec.size() << " maps." << std::endl;

		if (!allocateMaps(in_vec, out_vec)) {
			throw std::invalid_argument("AllocateNet error: allocateMaps.");
		}
		if (!testAllocatedMaps()) {
			throw std::invalid_argument("AllocateNet error: testAllocatedMaps.");
		}
		if (!copyInputsToGpu()) {
			throw std::invalid_argument("AllocateNet error: testAllocatedMaps.");
		}

		if (cpuNet().ancestor_of_.size() && !allocateCloneData()) {
			throw std::invalid_argument("AllocateNet error: allocateCloneData.");
		}

		if (print_graph) {
			gpu_net_.printGraph();
		}
		std::cout << "From: " << (uint64_t)old_val << " to "
				  << (uint64_t)(old_val + (net_->getTotalSizeBytes() / sizeof(float)) * no_batches)
				  << std::endl << std::endl;

		createExecutionGraph();

		was_failure = false;
		return "Net successfully allocated on GPU.";
	}

	std::string GpuForward(bool& was_failure) {
		//StopWatch sw;
		for (auto mp : maps_) {
//			std::cout << mp->getCpuFun()[0]->getName() << std::endl;
			mp->forward();
			if (!mp->getCpuFun().front()->isInput()) {
				mp->zeroGradients();
			}
		}
//		auto ss = "GPU Forward done in " + std::to_string(sw.ElapsedTimeMicros()) + "ms.";
//		std::cout << ss << std::endl;
		return "";
	}

	std::string GpuInfer(bool& was_failure) {
		//StopWatch sw;
		for (auto mp : maps_) {
			mp->forward();
		}
//		auto ss = "GPU GpuInfer done in " + std::to_string(sw.ElapsedTimeMicros()) + "ms.";
//		std::cout << ss << std::endl;
		return "";
	}

	std::string GpuBackward(int label, bool& was_failure) {
		//StopWatch sw;

		for (auto it = maps_.rbegin(); it != maps_.rend(); ++it)
		{
//			std::cout << "GPU bck: " << (*it)->cpu_func_.front()->getName() << std::endl;
			(*it)->backward(label);
		}

//		auto ss = "Backward done in " + std::to_string(sw.ElapsedTimeMicros()) + "ms.";
//		std::cout << ss << std::endl;
		return "";
	}

	void printMappingFromGpu(GpuMapping *mp, int from_indx = 0) {
		assert(mp && mp->gpu_in_ptr_);

		std::vector<GpuInVar> inp(mp->in_.size());
		assert(helper_.in_var_copy_from_gpu(inp.size(), mp->gpu_in_ptr_, &inp[0]));

		std::vector<GpuOutVar> out(mp->in_.size());
		assert(helper_.out_var_copy_from_gpu(out.size(), mp->gpu_out_ptr_, &out[0]));


		std::cout << "\nGpuMapping_" << mp->cpu_func_[0]->getName() << " : "
				  << mp->gpu_in_ptr_ << " no of maps: " << mp->in_.size() << " map length: " << mp->length() << std::endl;

		if (auto emb = dynamic_cast<EmbeddingGpu*>(mp)) {
			std::vector<int> tokens(emb->no_allocated_tokens_);
			assert(helper_.int_copy_from_gpu(emb->no_allocated_tokens_, emb->gpu_tokens_, &tokens[0]));

			int indx = 0;
			for (auto fun : emb->cpu_func_) {
				std::cout << indx << "\tGPU tokens: ";
				for (int var = 0; var < std::min(10, dynamic_cast<CEmbedding*>(fun)->noOutTokens()); ++var) {
					std::cout << tokens[fun->gpu_var_.tok_offset_ + var] << ", ";
				}
				std::cout << std::endl << "\tCPU tokens: ";
				auto size = (int)((CEmbedding*)fun)->tokens_.size();
				for (int var = 0; var < std::min(10, size); ++var) {
					std::cout
							<< (var < size ?
									(int) ((CEmbedding*) fun)->tokens_[var] :
									NO_TOKEN) << ", ";
				}
				std::cout << std::endl;
				indx++;
			}
			std::cout << std::endl;
		}

		int indx = 0;
		for (auto gin : inp) {
			Vars tmp(mp->length());
			auto func = mp->cpu_func_[indx];
			std::cout << indx << "\tIn ptr: " <<  gin.input_ptr_
					  << " Out ptr: " << out[indx].out_ptr_ << "\t"
					  << func->getName() << " From: ";
			if (func->prev_func_.size()) {
				for (auto prv : func->prev_func_) {
					std::cout << prv->getName() << ", ";
				}
			}
			std::cout << "\tTo: ";
			if (func->next_func_.size()) {
				for (auto nxt : func->next_func_) {
					std::cout << nxt->getName() << ", ";
				}
			}
			std::cout << std::endl;

			assert(helper_.float_copy_from_gpu(2 * tmp.length_,
					 gin.input_ptr_, tmp.real_));

			std::cout << "\tInput on GPU from indx " << from_indx << ":\t";
			for (int var = from_indx; var < (from_indx + 16 <= mp->length() ? from_indx + 16 : mp->length()); ++var) {
				std::cout << std::setprecision (3) << tmp.z(var);
			}
			std::cout << std::endl;
			if (!no_ancestors_) {
				std::cout << "\tInput on CPU from indx " << from_indx << ":\t";
				for (int var = from_indx; var < (from_indx + 16 <= mp->length() ? from_indx + 16 : mp->length()); ++var) {
					std::cout << std::setprecision (3) << mp->cpu_func_[indx]->input().z(var);
				}
				std::cout << std::endl;
			}

			if (auto ce = dynamic_cast<CrossEntropyGpu*>(mp)) {
				int label = labels_.size() ? labels_[mp->cpu_func_[indx]->batch_indx_] : 0;
				std::cout << "\tGPU loss for label: " << label << " "
						  << getLoss(label)[mp->cpu_func_[indx]->uid()] << std::endl;
				std::cout << "\tCPU loss for label: " << label << " "
						  << ((CrossEntropy*)mp->cpu_func_[indx])->loss(label) << std::endl;
			}

			indx++;
		}
	}

	bool testGradientsFromGpu(GpuMapping& mapping, float error) {
		for (auto fun : mapping.getCpuFun()) {
			//std::cout << "\t" << fun->input().zToString(16) << std::endl;

			Vars tmp(fun->input().length_);
			if (!helper_.float_copy_from_gpu(6 * tmp.length_,
					fun->gpu_var_.input_ptr_, tmp.real_)) {
				std::cerr << "Cannot copy from GPU: " + fun->getName() << std::endl;
				return false;

			}
			//std::cout << "\t" << tmp.zToString(16) << std::endl;

			for (int indx = 0; indx < tmp.length_; ++indx) {
				if (abs(tmp.dz(indx) - fun->input().dz(indx)) > error) {
					std::cerr << indx << "\tgpu_dz = " + fun->getName() << tmp.dz(indx)
							  << " != " << fun->input().dz(indx) << std::endl;
					return false;
				}
				if (abs(tmp.dz_star(indx) - fun->input().dz_star(indx)) > error) {
					std::cerr << indx << "\tgpu_dz_star = " + fun->getName() << tmp.dz_star(indx)
							  << " != " << fun->input().dz_star(indx) << std::endl;
					return false;
				}
			}
		}
		return true;
	}

	bool printGradientsFromGpu(GpuMapping& mapping) {
		for (auto fun : mapping.getCpuFun()) {
			//std::cout << "\t" << fun->input().zToString(16) << std::endl;

			std::cout << std::fixed;
			std::cout << std::setprecision(4);

			Vars tmp(fun->input().length_);
			if (!helper_.float_copy_from_gpu(6 * tmp.length_,
					fun->gpu_var_.input_ptr_, tmp.real_)) {
				std::cerr << "Cannot copy from GPU: " + fun->getName() << std::endl;
				return false;

			}

			std::cout << fun->getName() << " : " << fun->gpu_var_.input_ptr_ << std::endl
					<< "\tC_dz:\t";
			for (int indx = 0; indx < std::min(10, tmp.length_); ++indx) {
				std::cout <<  fun->input().dz(indx) << "\t";
			}
			std::cout << std::endl << "\tG_dz:\t";

			GpuInVar gpu_var;
			gpu_var.input_length_ = tmp.length_;
			gpu_var.input_ptr_ = tmp.real_;
			for (int indx = 0; indx < std::min(10, tmp.length_); ++indx) {
				std::cout << dZ_(gpu_var, indx) << "\t";
			}
			std::cout << std::endl << "\tC_dz_*:\t";

			for (int indx = 0; indx < std::min(10, tmp.length_); ++indx) {
				std::cout <<  fun->input().dz_star(indx) << "\t";
			}
			std::cout << std::endl << "\tG_dz_*:\t";

			for (int indx = 0; indx < std::min(10, tmp.length_); ++indx) {
				std::cout << dZ_star_(gpu_var, indx) << "\t";
			}
			std::cout << std::endl;
		}
		return true;
	}

	bool testDataFromGpu(GpuMapping& mapping, float error) {
		for (auto fun : mapping.getCpuFun()) {
			//std::cout << "\t" << fun->input().zToString(16) << std::endl;

			Vars tmp(fun->input().length_);
			if (!helper_.float_copy_from_gpu(2 * tmp.length_,
					fun->gpu_var_.input_ptr_, tmp.real_)) {
				std::cerr << "Cannot copy from GPU: " + fun->getName() << std::endl;
				return false;

			}
			//std::cout << "\t" << tmp.zToString(16) << std::endl;

			for (int indx = 0; indx < tmp.length_; ++indx) {
				if (abs(tmp.z(indx) - fun->input().z(indx)) > error) {
					std::cerr << indx << ": " + fun->getName() << tmp.z(indx)
							  << " != " << fun->input().z(indx) << std::endl;
					return false;
				}
			}
		}
		return true;
	}

	std::string GetInputFromGpu(bool& was_failure) {
		was_failure = false;

		for (auto fun : net_->functionList()) {
			std::cout << fun->getName() << std::endl;
//			std::cout << "\t" << fun->input().zToString(16) << std::endl;

			Vars tmp(fun->input().length_);
			if (!helper_.float_copy_from_gpu(2 * tmp.length_,
					fun->gpu_var_.input_ptr_, tmp.real_)) {
				was_failure = true;
				return "Cannot copy from GPU: " + fun->getName();
			}
//			std::cout << "\t" << tmp.zToString(16) << std::endl;

			for (int indx = 0; indx < tmp.length_; ++indx) {
				assert(abs(tmp.z(indx) - fun->input().z(indx)) <= 0.00000001);
			}

			if (auto oFun = dynamic_cast<CrossEntropy*>(fun)) {
				if (!helper_.float_copy_from_gpu(2 * tmp.length_,
						fun->gpu_var_.input_ptr_ + 6 * tmp.length_, tmp.real_)) {
					was_failure = true;
					return "Cannot copy from GPU: " + fun->getName();
				}
//				std::cout << "\t" << tmp.zToString(128) << std::endl;
//				std::cout << "\t" << oFun->mutableOutput()->zToString(128) << std::endl;

				for (int indx = 0; indx < tmp.length_; ++indx) {
					if(abs(tmp.z(indx) - oFun->mutableOutput()->z(indx)) > 0.00000001) {
						std::cout << tmp.z(indx) << " == " << oFun->mutableOutput()->z(indx) << std::endl;
					}

					assert(abs(tmp.z(indx) - oFun->mutableOutput()->z(indx)) <= 0.00000001);
				}
			}

		}

		return "Got data from GPU.";
	}

	float getLoss(OutputFunc *outp, Batch *batch) {
		float loss = 0.f;
		assert(labels_.size());
		auto c_fun = dynamic_cast<CFunc*>(outp);
		assert(c_fun);
		CrossEntropyGpu *ce = dynamic_cast<CrossEntropyGpu*>(getMappingFor(c_fun));
		if (!ce) {
			std::cerr << "Not supported for this output." << std::endl;
			assert(ce);
		}

		std::vector<GpuOutVar> tmp(ce->out_.size());
		assert(helper_.out_var_copy_from_gpu(tmp.size(), ce->gpu_out_ptr_, &tmp[0]));

		for (auto gout : tmp) {
			loss += gout.reduce_imag_;
			batch->addLoss(gout.reduce_imag_);
		}

		return loss / batch->size();
	}

	std::map<int, float> getLoss(int label) {
		assert(gpu_net_.outputs_.size());
		std::map<int, float> ret;

		for (auto outp : gpu_net_.outputs_) {
			std::vector<GpuOutVar> tmp(outp->out_.size());
			assert(helper_.out_var_copy_from_gpu(tmp.size(), outp->gpu_out_ptr_, &tmp[0]));

			if (L2Gpu *l2 = dynamic_cast<L2Gpu*>(outp)) {
				int indx = 0;
				for (auto gout : tmp) {
					ret[l2->cpu_func_[indx]->uid()] = gout.reduce_real_;
					indx++;
				}
			} else if (CrossEntropyGpu *ce = dynamic_cast<CrossEntropyGpu*>(outp)) {
				int indx = 0;
				for (auto gout : tmp) {
					auto fun = ce->cpu_func_[indx];
					if (labels_.size()) {
						assert(fun->batch_indx_ < labels_.size());
					}
					label = labels_.size() ? labels_[fun->batch_indx_] : label;


					assert(label >= 0 && label < ce->length());

					float tmp[fun->input().length_ * 2];
					helper_.float_copy_from_gpu(fun->input().length_ * 2, fun->gpu_var_.input_ptr_, tmp);
					auto z = std::complex<float>(tmp[label], tmp[fun->input().length_ + label]);

					float nrm = norm(z);
					gout.reduce_real_ = (gout.reduce_real_ < 1e-15 ? 1e-15 : gout.reduce_real_);

					float prob = nrm / gout.reduce_real_;
					prob = (prob < 1e-15 ? 1e-15 : prob);
					ret[ce->cpu_func_[indx]->uid()] = -std::log(prob);
					indx++;
				}
			}
		}
		return ret;
	}

	int sample() {
		assert(gpu_net_.outputs_.size());
		std::map<int, float> ret;

		for (auto outp : gpu_net_.outputs_) {
			std::vector<GpuOutVar> tmp(outp->out_.size());
			assert(helper_.out_var_copy_from_gpu(tmp.size(), outp->gpu_out_ptr_, &tmp[0]));

			if (CrossEntropyGpu *ce = dynamic_cast<CrossEntropyGpu*>(outp)) {
				for (auto gout : tmp) {
					auto fun = ce->cpu_func_[0];
					float tmp[fun->input().length_ * 2];
					helper_.float_copy_from_gpu(fun->input().length_ * 2, fun->gpu_var_.input_ptr_, tmp);

					for (int var = 0; var < fun->input().length_; ++var) {
						auto z = std::complex<float>(tmp[var], tmp[fun->input().length_ + var]);
						float nrm = norm(z);
						gout.reduce_real_ = (gout.reduce_real_ < 1e-15 ? 1e-15 : gout.reduce_real_);
						tmp[var] = nrm / gout.reduce_real_;
					}

					float coin = (float)cpuNet().rand_.randDouble();
					float c_val = 0.f;
					float max = 0;
					int var_max = 0.f;
					for (int var = 0; var < fun->input().length_; ++var) {
						c_val += tmp[var];
						if (coin <= c_val) {
							return var;
						}
						if (max < tmp[var]) {
							max = tmp[var];
							var_max = var;
						}
					}
					return var_max;
				}
			}
		}
		return 0;
	}

	int add(CFunc* func) {
		return net_->add(func);
	}

	int add(CFunc* func, std::initializer_list<int> in_uids) {
		return net_->add(func, in_uids);
	}

	CFunc*& operator [](int uid) {
		return net_->map_[uid];
	}

	void init_inputs() {
		net_->init_inputs();
	}

	void init_inputs(uint64_t seed) {
		net_->init_inputs(seed);
	}

	void init_exec_graph(bool print_status = false) {
		net_->init_exec_graph(print_status);
	}

	void forward() {
		net_->forward();
	}

	void backward(int label) {
		net_->backward(label);
	}

	void adamUpdate(float learning_rate, int batch_size, float beta, int t) {
		net_->adamUpdate(learning_rate, batch_size, beta, t);
	}

	CFunc* findFirstOfType(int typeId) {
		return net_->findFirstOfType(typeId);
	}

	CFunc* findFirstInput() {
		for (auto fun : net_->func_list_) {
			if (dynamic_cast<InputFunc*>(fun)) {
				return fun;
			}
		}
		return NULL;
	}

	CFunc* findFirstOutput() {
		for (auto fun : net_->func_list_) {
			if (dynamic_cast<OutputFunc*>(fun)) {
				return fun;
			}
		}
		return NULL;
	}

	bool allocateOnGpu(int batch_size) {
		InputFunc *inp = (InputFunc*)findFirstInput();
		assert(inp);
		OutputFunc *outp = (OutputFunc*)findFirstOutput();
		assert(outp);
		inp->setIsMainInput(true);
		outp->setIsMainOutput(true);

		cpuNet().cloneForGpu(batch_size - 1);
		std::cout << cpuNet().noClones() << " clones created." << std::endl;
		init_exec_graph();

		bool was_failure = false;
		AllocateNet(1, was_failure);
		return !was_failure;
	}

	bool gpuForward(InputFunc *inp, OutputFunc *outp, InputBatch &batch) {
		copyInputsToClones();
		batchToGpu(inp, outp, &batch);
		bool was_failure = false;
		GpuForward(was_failure);
		return !was_failure;
	}

	bool gpuBackward() {
		bool was_failure = false;
		GpuBackward(0, was_failure);
		updateGradientsFromClones();
		return !was_failure;
	}

	bool save(std::string file) {
		ModelSaver saver;
		return saver.Save(cpuNet(), file);
	}

	bool restore(std::string file) {
		ModelSaver saver;
		return saver.Restore(cpuNet(), file);
	}

};




#endif /* GPU_GPU_FUNC_H_ */
