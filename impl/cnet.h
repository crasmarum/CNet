#ifndef IMPL_CNET_H_
#define IMPL_CNET_H_

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <random>
#include <initializer_list>
#include <omp.h>

#include "../utils/myrand.h"

#include "cfunc.h"
#include "sizes.h"
#include "vars.h"

#include "log.h"
#include "cinput.h"
#include "cnet.h"
#include "linear.h"
#include "residual.h"
#include "crossent.h"
#include "embed.h"
#include "ft.h"
#include "relu.h"
#include "hadamard.h"
#include "softmax.h"
#include "l2out.h"

const int isHadamard = 1;
const int isEmbedding = 2;
const int isLinear = 3;
const int isFourier = 4;
const int isCrossEntropy = 5;
const int isResidual = 6;
const int isSoftMax = 7;
const int isInput = 8;
const int isCrelu = 9;
const int isGelu = 10;
const int isL2Out = 11;
const int isTrianFourier = 12;

const int isUnknown = INT_MAX;


/*
 * A net is simply a list of layers.
 */
class ComplexNet {
	friend class ModelSaver;
	friend class CNet;

	std::vector<CFunc*> func_list_; // owning these
	std::map<int, CFunc*> map_;
	SimpleRand rand_;
	std::vector<CFunc*> inputs_;
	std::vector<CFunc*> outputs_;
	int next_uid_ = 0;

	std::vector<CFunc*>* func_at_depth_ = 0;
	int max_depth_ = 0;
	unsigned long long total_size_bytes_ = 0;

	std::map<int, std::vector<CFunc*> > gpu_clones_of_;
	std::map<int, CFunc* > ancestor_of_;


	int next_uid() {
		return ++next_uid_;
	}

public:
	ComplexNet() {
	}

	int noClones() {
		int ret = 0;
		for (auto cl_list : gpu_clones_of_) {
			ret += cl_list.second.size();
		}
		return ret;
	}

	CFunc* findFirstOfType(int typeId) {
		for (auto fun : func_list_) {
			if (getType(fun) == typeId) {
				return fun;
			}
		}
		return NULL;
	}

	virtual ~ComplexNet() {
//		std::cout << "~ComplexNet(): " << net_.size() << std::endl;
		for (int var = 0; var < func_list_.size(); ++var) {
//			std::cout << "deleting: " << func_list_[var]->getName() << std::endl;
			delete func_list_[var];
		}
		if (func_at_depth_) {
			delete [] func_at_depth_;
		}
	}

	const std::vector<CFunc*>& inputs() const {
		return inputs_;
	}

	static int getType(CFunc *func) {
		if (CInput *input = dynamic_cast<CInput*>(func)) {
			return isInput;
		}
		if (Hadamard *hadm = dynamic_cast<Hadamard*>(func)) {
			return isHadamard;
		}
		if (Linear *linear = dynamic_cast<Linear*>(func)) {
			return isLinear;
		}
		if (CEmbedding *emb = dynamic_cast<CEmbedding*>(func)) {
			return isEmbedding;
		}
		if (FourierTrans *four = dynamic_cast<FourierTrans*>(func)) {
			return isFourier;
		}
		if (TriangFourier *tfour = dynamic_cast<TriangFourier*>(func)) {
			return isTrianFourier;
		}
		if (Residual *four = dynamic_cast<Residual*>(func)) {
			return isResidual;
		}

		if (CrossEntropy *out = dynamic_cast<CrossEntropy*>(func)) {
			return isCrossEntropy;
		}

		if (SoftMax *out = dynamic_cast<SoftMax*>(func)) {
			return isSoftMax;
		}

		if (CRelu *out = dynamic_cast<CRelu*>(func)) {
			return isCrelu;
		}

		if (CGelu *out = dynamic_cast<CGelu*>(func)) {
			return isGelu;
		}

		if (L2Out *out = dynamic_cast<L2Out*>(func)) {
			return isL2Out;
		}

		L_(lDebug) << "Function has no GPU implementation: " << func->getName();
		return isUnknown;
	}

	void init_exec_graph(bool print_status = false) {
		if (func_at_depth_) {
			L_(lError) << "Exec graph already initialized " << std::endl;
			return;
		}
		max_depth_ = 0;
		for (auto func : func_list_) {
			max_depth_ = func->depth_ > max_depth_ ? func->depth_ : max_depth_;
		}
		max_depth_ += 1;

		func_at_depth_ = new std::vector<CFunc*>[max_depth_];
		for (auto func : func_list_) {
			if (!(func->is_output_ || func->no_outputs())) {
				L_(lError) << func->getName() << "\n";
			}
			assert(func->is_output_ || func->no_outputs());
			func_at_depth_[func->depth_].push_back(func);
		}

		for (int depth = 0; depth < max_depth_; ++depth) {
			stable_sort(func_at_depth_[depth].begin( ), func_at_depth_[depth].end( ),
				[ ]( const CFunc* lhs, const CFunc* rhs )
				{
					if (lhs->type_ == rhs->type_) {
						return lhs->sizeInBytes() < rhs->sizeInBytes();
					}
				   return lhs->type_ > rhs->type_;
				});
		}

		if (print_status) {
			for (int depth = 0; depth < max_depth_; ++depth) {
				std::cout << "Depth " << depth << ": ";

				for (auto func : func_at_depth_[depth]) {
					std::cout << (func->isGpuOnly() ?
									"GPU_batch_indx=" + std::to_string(func->batch_indx_) + " ": "")
							  << func->getName() << " " << func->sizeInBytes() << ", ";
				}
				std::cout << std::endl;
			}
		}
	}

	void print_func_at_depth(int depth) {
		if (max_depth_ == 0) {
			std::cout << "init_exec_graph() not called yet." << std::endl;
			return;
		}
		if (depth < 0 || depth >= max_depth_) {
			std::cout << "print_func_at_depth called with wrong depth: "
					<< std::to_string(depth) << std::endl;
			return;
		}
		std::cout << std::to_string(func_at_depth_[depth].size()) << " functions: ";
		for (auto func : func_at_depth_[depth]) {
			std::cout << func->getName() << ", ";
		}
		std::cout << std::endl;
	}

	std::vector<CFunc*>& getFuncAtDepth(int depth) {
		assert(depth < max_depth_);
		assert(func_at_depth_);
		return func_at_depth_[depth];
	}

	int maxDepth() {
		return max_depth_;
	}

	int add(CFunc* func) {
		assert(func);
		if (func->uid_ == 0) {
			func->uid_ = next_uid();
		}
		func->type_ = getType(func);
		total_size_bytes_ += func->sizeInBytes();

		assert(map_.find(func->uid_) == map_.end());

		func_list_.push_back(func);
		map_[func->uid_] = func;

		if (func->is_input_) {
			inputs_.push_back(func);
		}

		if (func->is_output_) {
			outputs_.push_back(func);
		}

		return func->uid_;
	}

	unsigned long long getTotalSizeBytes() {
		return total_size_bytes_;
	}

	int add(CFunc* func, std::initializer_list<int> in_uids) {
		int f_uid = add(func);

		int offset = 0;
		for (auto in_uid : in_uids) {
			CFunc *in_func = map_[in_uid];
			assert(in_func);

			in_func->addOutput(func, offset);
			offset += in_func->out_size_;

			func->prev_func_.push_back(in_func);
		}
		assert(offset <= func->input().length_);

		return f_uid;
	}

	int add(CFunc* func, std::vector<int> in_uids) {
		int f_uid = add(func);

		int offset = 0;
		for (auto in_uid : in_uids) {
			CFunc *in_func = map_[in_uid];
			assert(in_func);
			in_func->addOutput(func, offset);
			offset += in_func->out_size_;

			func->prev_func_.push_back(in_func);
		}
		assert(offset <= func->input().length_);

		return f_uid;
	}

	void init_inputs() {
		for (auto in_func : inputs_) {
			if (in_func->isGpuOnly()) {
				continue;
			}
			in_func->init_inputs();
		}
	}

	void init_inputs(uint64_t seed) {
		assert(this->func_list_.size());
		int indx = 1;
		for (auto in_func : inputs_) {
			if (in_func->isGpuOnly()) {
				continue;
			}
			in_func->setRandSeed(seed + indx++);
			in_func->init_inputs();
		}
	}

	void print_inputs() {
		for (auto fn_inpt : inputs_) {
			std::cout << fn_inpt->getName();
			for (int indx = 0; indx < fn_inpt->input().length_; ++indx) {
				std::cout << " " << fn_inpt->input().z(indx) << ",";
			}
			std::cout << std::endl;
		}
	}

	int getNoVars() {
		int sum = 0;
		for (auto fc : inputs_) {
			sum += (fc->input().length_);
		}
		return sum;
	}

	std::string toString() {
		std::ostringstream oss;
		oss << func_list_.size() << " functions: "
			<< " vars: " << getNoVars() << std::endl;
		for (int var = 0; var < func_list_.size(); ++var) {
			oss << "depth: " << func_list_[var]->depth() << ": "
				<< (func_list_[var]->isGpuOnly() ? "GPU-Only " : "") << func_list_[var]->getName()
				<< ": from C^" << func_list_[var]->input().length_ << " to C^" << func_list_[var]->out_size_;
			int indx = 0;
			for (auto fc : func_list_[var]->next_func_) {
				oss	<< " into " << fc->getName() << " at offset " << func_list_[var]->offset(indx++);
			}
			oss	<< std::endl;
		}
		return oss.str();
	}

	CFunc* layer_at(int indx) {
		assert(indx >= 0 && indx < func_list_.size());
		return func_list_[indx];
	}

	CFunc* back() {
		assert(func_list_.size());
		return func_list_.back();
	}

	CFunc* front() {
		assert(func_list_.size());
		return func_list_.front();
	}

	CFunc*& operator [](int uid) {
		return map_[uid];
	}

	void forward() {
		for (auto func : this->func_list_) {
			if (func->isGpuOnly()) {
				continue;
			}
			L_(lDebug) << func->getName() << std::endl;
			if (!func->is_input_) {
				func->mutable_input()->zero_gradients();
			}
			func->forward();
		}
	}

	void infer() {
		assert(func_at_depth_);
		for (int depth = 0; depth < max_depth_; ++depth) {
			for (auto func : func_at_depth_[depth]) {
				if (func->isGpuOnly()) {
					continue;
				}
				func->forward();
			}
		}
	}

	void backward() {
		for (int var = func_list_.size() - 1; var >= 0; --var) {
			if (func_list_[var]->isGpuOnly()) {
				continue;
			}
			func_list_[var]->backward();
		}
	}

	void backward(int label) {
		for (int var = func_list_.size() - 1; var >= 0; --var) {
			if (func_list_[var]->isGpuOnly()) {
				continue;
			}
			if (dynamic_cast<CrossEntropy*>(func_list_[var])) {
				func_list_[var]->backward(label);
			} else {
				func_list_[var]->backward();
			}
		}
	}

	void update(float learning_rate, int batch_size) {
		for (int var = 0; var < inputs_.size(); ++var) {
			if (func_list_[var]->isGpuOnly()) {
				continue;
			}
			inputs_[var]->updateInput(learning_rate / batch_size);
			inputs_[var]->mutable_input()->zero_gradients();
		}
	}

	void adamUpdate(float learning_rate, int batch_size, float beta, int t) {
		for (int var = 0; var < inputs_.size(); ++var) {
			if (func_list_[var]->isGpuOnly()) {
				continue;
			}
			inputs_[var]->adamUpdate(learning_rate / batch_size, beta, t);
			inputs_[var]->mutable_input()->zero_dZ_star();
		}
	}

	std::vector<CFunc*>& functionList() {
		return func_list_;
	}

	void testEqual(const ComplexNet& other) {
		int indx = 0;
		for (auto layer : functionList()) {
			if(CInput* input = dynamic_cast<CInput*>(layer)) {
				CInput* other = dynamic_cast<CInput*>(functionList()[indx]);
				assert(*input ==  *other);
			} else if(Hadamard* hadm = dynamic_cast<Hadamard*>(layer)) {
				Hadamard* other = dynamic_cast<Hadamard*>(functionList()[indx]);
				assert(*hadm ==  *other);
			} else if(Linear* linear = dynamic_cast<Linear*>(layer)) {
				Linear* other = dynamic_cast<Linear*>(functionList()[indx]);
				assert(*linear ==  *other);
			} else if(CEmbedding* emb = dynamic_cast<CEmbedding*>(layer)) {
				CEmbedding* other = dynamic_cast<CEmbedding*>(functionList()[indx]);
				assert(*emb ==  *other);
			} else if(FourierTrans* four = dynamic_cast<FourierTrans*>(layer)) {
				FourierTrans* other = dynamic_cast<FourierTrans*>(functionList()[indx]);
				assert(*four ==  *other);
			} else if(TriangFourier* tfour = dynamic_cast<TriangFourier*>(layer)) {
				TriangFourier* other = dynamic_cast<TriangFourier*>(functionList()[indx]);
				assert(*tfour ==  *other);
			} else if(Residual* res = dynamic_cast<Residual*>(layer)) {
				Residual* other = dynamic_cast<Residual*>(functionList()[indx]);
				assert(*res ==  *other);
			} else if(CrossEntropy* ce = dynamic_cast<CrossEntropy*>(layer)) {
				CrossEntropy* other = dynamic_cast<CrossEntropy*>(functionList()[indx]);
				assert(*ce ==  *other);
			} else if(SoftMax* sf = dynamic_cast<SoftMax*>(layer)) {
				SoftMax* other = dynamic_cast<SoftMax*>(functionList()[indx]);
				assert(*sf ==  *other);
			} else if(CGelu* gl = dynamic_cast<CGelu*>(layer)) {
				CGelu* other = dynamic_cast<CGelu*>(functionList()[indx]);
				assert(*gl ==  *other);
			} else if(CRelu* gl = dynamic_cast<CRelu*>(layer)) {
				CRelu* other = dynamic_cast<CRelu*>(functionList()[indx]);
				assert(*gl ==  *other);
			}
//			else if(L2Out* gl = dynamic_cast<L2Out*>(layer)) {
//				L2Out* other = dynamic_cast<L2Out*>(functionList()[indx]);
//				assert(*gl == *other);
//			}
			else {
				L_(lError) << "Unexpected func: " << layer->getName();
				assert(false);
			}
			indx++;
		}
	}

	void cloneForGpu(int no_clones) {
		if (func_at_depth_) {
			L_(lError) << "Exec graph already initialized.";
			assert(!func_at_depth_);
			return;
		}

		int max_id = func_list_.front()->uid();
		std::map<int, std::vector<CFunc*> > by_depth;
		int max_depth = 0;

		for (auto fun : func_list_) {
			if (fun->uid() > max_id) max_id = fun->uid();
			if (by_depth.find(fun->depth()) == by_depth.end()) {
				std::vector<CFunc*> vec;
				by_depth[fun->depth()] = vec;
			}
			by_depth[fun->depth()].push_back(fun);
			if (max_depth < fun->depth()) {
				max_depth = fun->depth();
			}
		}
		for (int var = 0; var < no_clones; ++var) {
			bool gotMainInput = false;
			bool gotMainOutput = false;
			std::map<int, int> clone_of;
			for (int depth = 0; depth <= max_depth; ++depth) {
				for (auto fun : by_depth[depth]) {
					clone(fun, ++max_id, var + 1, clone_of, gotMainInput, gotMainOutput);
				}
			}
			if (!gotMainInput || !gotMainOutput) {
				L_(lError) << "Missing main input or output.";
				assert(gotMainInput && gotMainOutput);
			}
		}
	}

	std::vector<CFunc*>& clonesOf(CFunc* func) {
		assert(!func->is_gpu_only_);
		return gpu_clones_of_[func->uid()];
	}

private:

	void clone(CFunc *func, int uid, int batch_indx,
			std::map<int, int>& clone_of, bool& gotMainInput, bool& gotMainOutput) {
		assert(func);
		std::vector<int> prev_func;
		for (auto pfunc : func->prev_func_) {
			assert(clone_of.find(pfunc->uid()) != clone_of.end());
			prev_func.push_back(clone_of[pfunc->uid()]);
		}

		if (InputFunc *input = dynamic_cast<InputFunc*>(func)) {
//			std::cout << "\tCloning the input fc: " << func->getName() << "\n";
			if(input->isMainInput() && gotMainInput) {
				L_(lError) << "Only one main input supported.";
				assert(!(input->isMainInput() && gotMainInput));
			}
			if (input->isMainInput()) {
				add(func->clone(Uid(uid)), prev_func);
				assert(func_list_.back()->input().real_ != NULL);
			} else {
				Vars::no_data_ = true;
				add(func->clone(Uid(uid)), prev_func);
				Vars::no_data_ = false;
				gotMainInput = true;
				assert(func_list_.back()->input().real_ == NULL);
			}
		} else if (OutputFunc *outp = dynamic_cast<OutputFunc*>(func)) {
//			std::cout << "\tCloning the output fc: " << func->getName() << "\n";
			if(outp->isMainOutput() && gotMainOutput) {
				L_(lError) << "Only one main output supported.";
				assert(!(outp->isMainOutput() && gotMainOutput));
			}
			add(func->clone(Uid(uid)), prev_func);
			gotMainOutput = true;
			assert(func_list_.back()->input().real_ != NULL);
		} else {
//			std::cout << "\tCloning the fc: " << func->getName() << "\n";
			Vars::no_data_ = true;
			add(func->clone(Uid(uid)), prev_func);
			Vars::no_data_ = false;
			assert(func_list_.back()->input().real_ == NULL);
		}

		func_list_.back()->is_gpu_only_ = true;
		func_list_.back()->batch_indx_ = batch_indx;
		ancestor_of_[uid] = func;
		clone_of[func->uid()] = uid;

		if (gpu_clones_of_.find(func->uid()) == gpu_clones_of_.end()) {
			std::vector<CFunc*> vec;
			gpu_clones_of_[func->uid()] = vec;
		}
		gpu_clones_of_[func->uid()].push_back(func_list_.back());
	}
};

#endif /* IMPL_CNET_H_ */
