// cnet_app.h -- reusable training and inference for CNet token-sequence models.
//
// A problem supplies two small classes and a main():
//
//   struct MyModel : app::ModelDef {          // the layer graph
//       app::Heads build(ComplexNet &net, int vocab, int context) override { ... }
//   };
//   struct MyData : app::DataSource {          // where the windows come from
//       int vocab() const override; int context() const override;
//       void sample(bool heldout, std::mt19937_64 &rng, std::vector<int> &in, std::vector<int> &tgt) override;
//   };
//   int main(int argc, char **argv) {
//       FLAGS::Parse(argc, argv);
//       MyData data(...); MyModel model(...);
//       app::Trainer trainer(model, data);    // options from the standard flags below
//       return trainer.run();                 // init + train + final checkpoint
//   }
//
// Inference on a saved model:
//   app::Infer inf; inf.loadModel("model.mod");
//   std::vector<int> out = inf.generate(prompt_ids, 200, 0.8, 1);
//   double nats = inf.score(token_ids);
//
// The graph must take its input through one CEmbedding and end in one SequenceCrossEntropy (the
// standard CNet language-model contract: N input ids per window, one target id per position).
// Multi-GPU data-parallel training needs NCCL: build with -DWITH_NCCL and -lnccl, launch one process
// per GPU with -world W -rank r (examples/run_dp_shakespeare.sh shows the launcher). Without NCCL the
// trainer runs on one GPU.
//
// Standard flags (all optional): -steps -warmup -lr -min_lr -batch -world -rank -val_every
// -val_batches -log_every -save_path -init_model -init_std -grad_clip -grad_norm_clip -ce_eps -seed
// -nccl_id -tf32. CNet's global kernel flags (-tw_cublas, -tw_real_loss, -born_chunk, ...) apply too.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <cublas_v2.h>
#ifdef WITH_NCCL
#include <nccl.h>
#endif

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "log.h"
#include "cnet.h"
#include "vars.h"
#include "batch.h"
#include "cinput.h"
#include "embed.h"
#include "seqcrossent.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(steps, 5000)           // optimizer steps
FLAG_INT(warmup, 200)           // linear warm-up steps
FLAG_FLOAT(lr, 1e-3)            // peak learning rate (cosine decay to -min_lr)
FLAG_FLOAT(min_lr, 1e-4)        // cosine floor
FLAG_INT(batch, 8)              // GLOBAL batch (windows per step), split evenly across ranks
FLAG_INT(world, 1)              // data-parallel processes (one per GPU)
FLAG_INT(rank, 0)               // this process's rank
FLAG_INT(val_every, 500)        // validation period in steps (0 = off)
FLAG_INT(val_batches, 20)       // held-out batches per validation
FLAG_INT(log_every, 100)        // training-loss log period
FLAG_STRING(save_path, "")      // checkpoint path (best validation, then final if at least as good)
FLAG_STRING(init_model, "")     // start from these weights instead of a fresh graph
FLAG_FLOAT(init_std, 0.02)      // fresh weights: re, im ~ N(0, init_std^2); 0 = CNet's old init_inputs start
FLAG_FLOAT(grad_clip, 0.5)      // Adam clip-by-value per gradient component (0 = off)
FLAG_FLOAT(grad_norm_clip, 0.0) // global gradient-norm clip on the summed gradient (0 = off)
FLAG_FLOAT(ce_eps, 0.001)       // Born-loss probability floor (0 = exact)
FLAG_INT(seed, 1000)            // data sampler seed (rank r uses seed + r)
FLAG_STRING(nccl_id, "/tmp/cnet_app_id")  // NCCL rendezvous file base (multi-GPU only)
FLAG_BOOL(tf32, false)          // TF32 tensor cores for FP32 GEMMs (faster, slightly less exact)

namespace app {

// Node ids of the graph's input embedding and its sequence loss.
struct Heads { int embedding, loss; Heads(int e = -1, int l = -1) : embedding(e), loss(l) {} };

// The model definition: build the layer graph on `net` for this vocabulary and context length.
class ModelDef {
public:
	virtual ~ModelDef() {}
	virtual Heads build(ComplexNet &net, int vocab, int context) = 0;
};

// The data: vocabulary, context length, and one window at a time.
class DataSource {
public:
	virtual ~DataSource() {}
	virtual int vocab() const = 0;
	virtual int context() const = 0;
	// Fill `in` and `tgt` with context() ids each: tgt[i] is the token that follows in[i].
	virtual void sample(bool heldout, std::mt19937_64 &rng, std::vector<int> &in, std::vector<int> &tgt) = 0;
};

// Training options; defaults come from the flags above.
struct TrainOptions {
	int steps = ::steps, warmup = ::warmup, batch = ::batch, world = ::world, rank = ::rank;
	int val_every = ::val_every, val_batches = ::val_batches, log_every = ::log_every, seed = ::seed;
	float lr = ::lr, min_lr = ::min_lr, init_std = ::init_std, grad_clip = ::grad_clip;
	float grad_norm_clip = ::grad_norm_clip, ce_eps = ::ce_eps;
	std::string save_path = (std::string) ::save_path, init_model = (std::string) ::init_model;
	std::string nccl_id = (std::string) ::nccl_id;
	bool tf32 = ::tf32;
};

class Trainer {
public:
	Trainer(ModelDef &model, DataSource &data, TrainOptions opt = TrainOptions())
		: model_(model), data_(data), o_(opt) {}
	~Trainer() { shutdown(); }

	// Build (or restore) the model, initialize it, allocate it on the GPU and join the data-parallel group.
	bool init() {
		const int W = o_.world, R = o_.rank;
		if (W < 1 || o_.batch % W != 0) { err("-batch must be a positive multiple of -world"); return false; }
#ifndef WITH_NCCL
		if (W != 1) { err("built without NCCL: use -world 1 (or build with -DWITH_NCCL)"); return false; }
#endif
		per_ = o_.batch / W;
#ifdef __CUDACC__
		g_cnet_tf32 = o_.tf32;
		int nDev = 0; cudaGetDeviceCount(&nDev); if (nDev < 1) nDev = 1;
		cudaSetDevice(R % nDev);
#endif
		if (!joinGroup()) return false;
		const int V = data_.vocab(), N = data_.context();
		if (!o_.init_model.empty()) {
			if (!loadModel(o_.init_model)) return false;
			if (seq_->nPos() != N || seq_->vocab() != V) { err("-init_model does not match the data's context/vocab"); return false; }
		} else {
			Heads h = model_.build(net_.cpuNet(), V, N);
			em_  = dynamic_cast<CEmbedding*>(net_.cpuNet()[h.embedding]);
			seq_ = dynamic_cast<SequenceCrossEntropy*>(net_.cpuNet()[h.loss]);
			if (!em_ || !seq_) { err("ModelDef::build must return a CEmbedding and a SequenceCrossEntropy"); return false; }
			net_.cpuNet().init_inputs(1234);                            // identical on every rank
			if (o_.init_std > 0.f) net_.cpuNet().normalInit(o_.init_std, 1234);
		}
		em_->setIsMainInput(true); seq_->setIsMainOutput(true);
		seq_->setEps(o_.ce_eps);
		if (!net_.allocateOnGpu(std::max(2, per_))) { err("GPU allocation failed"); return false; }
		bufs_ = net_.ancestorGradBuffers();
		off_.assign(bufs_.size(), 0); total_ = 0;
		for (size_t i = 0; i < bufs_.size(); ++i) { off_[i] = total_; total_ += bufs_[i].second; }
		cudaMalloc(&scratch_, (size_t) total_ * sizeof(float));
		cublasCreate(&blas_); cublasSetStream(blas_, cudaStreamPerThread); cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST);
		rng_.seed(o_.seed + R);
		if (R == 0)
			std::cout << "[trainer] vocab=" << V << " context=" << N << " global_batch=" << o_.batch << " world=" << W
					  << " steps=" << o_.steps << " lr=" << o_.lr << " init_std=" << o_.init_std << std::endl;
		ready_ = true;
		return true;
	}

	// One optimizer step on a fresh training batch; returns this rank's training loss (nats/token).
	float step() {
		++t_;
		EmbeddingBatch b = makeBatch(false);
		net_.gpuForward(em_, seq_, b);
		float loss = net_.getLoss(0)[seq_->uid()];
		net_.gpuBackward();
		for (size_t i = 0; i < bufs_.size(); ++i)
			cudaMemcpyAsync(scratch_ + off_[i], bufs_[i].first, (size_t) bufs_[i].second * sizeof(float),
							cudaMemcpyDeviceToDevice, cudaStreamPerThread);
#ifdef WITH_NCCL
		if (o_.world > 1) ncclAllReduce(scratch_, scratch_, total_, ncclFloat, ncclSum, comm_, cudaStreamPerThread);
#endif
		gnorm_ = 0.f;
		if (o_.grad_norm_clip > 0.f || (o_.rank == 0 && o_.log_every > 0 && t_ % o_.log_every == 0))
			cublasSnrm2(blas_, total_, scratch_, 1, &gnorm_);
		if (o_.grad_norm_clip > 0.f && gnorm_ > o_.grad_norm_clip) {
			float s = o_.grad_norm_clip / gnorm_; cublasSscal(blas_, total_, &s, scratch_, 1);
		}
		for (size_t i = 0; i < bufs_.size(); ++i)
			cudaMemcpyAsync(bufs_[i].first, scratch_ + off_[i], (size_t) bufs_[i].second * sizeof(float),
							cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		net_.trueAdamUpdate(lrAt(t_), 0.9f, 0.999f, 1e-8f, t_, o_.grad_clip);
		return loss;
	}

	// Mean held-out loss over `batches` random batches (this rank's draws), nats/token.
	float validate(int batches = -1) {
		if (batches < 0) batches = o_.val_batches;
		double v = 0;
		for (int k = 0; k < batches; ++k) {
			EmbeddingBatch b = makeBatch(true);
			net_.gpuForward(em_, seq_, b);
			v += net_.getLoss(0)[seq_->uid()];
		}
		return (float) (v / std::max(1, batches));
	}

	// The full loop: logging, periodic validation with best-checkpoint saving, final checkpoint.
	void train() {
		const int R = o_.rank;
		StopWatch sw; double acc = 0; int accn = 0;
		for (int s = 0; s < o_.steps; ++s) {
			acc += step(); ++accn;
			if (R == 0 && o_.log_every > 0 && t_ % o_.log_every == 0) {
				std::cout << "step " << t_ << "  train " << acc / accn << " nats/token  lr " << lrAt(t_)
						  << "  |g|=" << gnorm_ << "  " << sw.ElapsedTimeMicros() / 1000.0 / accn << " ms/step" << std::endl;
				acc = 0; accn = 0; sw.Reset();
			}
			if (R == 0 && o_.val_every > 0 && t_ % o_.val_every == 0) {
				float v = validate();
				std::cout << "  [val] step " << t_ << "  val " << v << " nats/token" << std::endl;
				if (!o_.save_path.empty() && v < best_) {
					best_ = v;
					if (saveModel(o_.save_path)) std::cout << "    [ckpt] new best val " << v << " -> " << o_.save_path << std::endl;
				}
				sw.Reset();
			}
		}
		cudaStreamSynchronize(cudaStreamPerThread);
		if (R == 0 && !o_.save_path.empty()) {                          // final model unless it is worse than the best
			bool save = true;
			if (o_.val_every > 0 && best_ < 1e29f) {
				float v = validate(); save = v <= best_;
				std::cout << "final val " << v << " (best " << best_ << "); " << (save ? "saving final" : "keeping best") << std::endl;
			}
			if (save && saveModel(o_.save_path)) std::cout << "saved model to " << o_.save_path << std::endl;
		}
	}

	// init() + train(); returns a process exit code.
	int run() { if (!ready_ && !init()) return 1; train(); shutdown(); return 0; }

	bool saveModel(const std::string &path) { return net_.getInputsFromGpu() && net_.save(path); }

	// Restore a saved graph and its weights (call before init(), or let -init_model do it).
	bool loadModel(const std::string &path) {
		if (!net_.restore(path)) { err("cannot restore " + path); return false; }
		em_  = (CEmbedding*) net_.cpuNet().findFirstOfType(isEmbedding);
		seq_ = (SequenceCrossEntropy*) net_.cpuNet().findFirstOfType(isSeqCrossEntropy);
		if (!em_ || !seq_) { err(path + ": no embedding / sequence head"); return false; }
		return true;
	}

	CNet &getModel() { return net_; }
	int globalStep() const { return t_; }
	float bestValidation() const { return best_; }
	float learningRate() const { return lrAt(t_); }

private:
	float lrAt(int t) const {
		if (t < o_.warmup) return o_.lr * (float) (t + 1) / o_.warmup;
		float p = (float) (t - o_.warmup) / std::max(1, o_.steps - o_.warmup);
		return o_.min_lr + 0.5f * (o_.lr - o_.min_lr) * (1.f + std::cos(3.14159265f * p));
	}
	EmbeddingBatch makeBatch(bool heldout) {
		const int N = data_.context();
		EmbeddingBatch b(per_, N);
		std::vector<int> in, tgt;
		for (int i = 0; i < per_; ++i) {
			data_.sample(heldout, rng_, in, tgt);
			b.add(in, tgt.back());
			b.addTargets(tgt);
		}
		return b;
	}
	bool joinGroup() {
#ifdef WITH_NCCL
		if (o_.world == 1) return true;
		idf_ = o_.nccl_id + ".bin"; rdy_ = o_.nccl_id + ".ready";
		ncclUniqueId id;
		if (o_.rank == 0) {
			ncclGetUniqueId(&id);
			{ std::ofstream f(idf_, std::ios::binary); f.write((char*) &id, sizeof(id)); }
			{ std::ofstream r(rdy_); r << "1"; }
		} else {
			int tries = 0;
			while (!std::ifstream(rdy_).good()) { usleep(50000); if (++tries > 2400) { err("NCCL id timeout"); return false; } }
			usleep(100000);
			std::ifstream f(idf_, std::ios::binary); f.read((char*) &id, sizeof(id));
		}
		if (ncclCommInitRank(&comm_, o_.world, id, o_.rank) != ncclSuccess) { err("NCCL init failed"); return false; }
		haveComm_ = true;
#endif
		return true;
	}
	void shutdown() {
		if (scratch_) { cudaFree(scratch_); scratch_ = nullptr; }
		if (blas_) { cublasDestroy(blas_); blas_ = 0; }
#ifdef WITH_NCCL
		if (haveComm_) { ncclCommDestroy(comm_); haveComm_ = false; if (o_.rank == 0) { remove(idf_.c_str()); remove(rdy_.c_str()); } }
#endif
	}
	void err(const std::string &m) const { std::cerr << "[trainer rank " << o_.rank << "] " << m << std::endl; }

	ModelDef &model_; DataSource &data_; TrainOptions o_;
	CNet net_; CEmbedding *em_ = nullptr; SequenceCrossEntropy *seq_ = nullptr;
	std::vector<std::pair<float*, int>> bufs_; std::vector<int> off_; int total_ = 0;
	float *scratch_ = nullptr; cublasHandle_t blas_ = 0; std::mt19937_64 rng_;
	int per_ = 1, t_ = 0; float gnorm_ = 0.f, best_ = 1e30f; bool ready_ = false;
#ifdef WITH_NCCL
	ncclComm_t comm_; bool haveComm_ = false; std::string idf_, rdy_;
#endif
};

// Inference on a saved model, CPU, exact Born probabilities. Token ids in, token ids out.
class Infer {
public:
	bool loadModel(const std::string &path) {
		FILELog::ReportingLevel() = lWarning;
		if (!net_.restore(path)) { std::cerr << "[infer] cannot restore " << path << std::endl; return false; }
		em_  = (CEmbedding*) net_.cpuNet().findFirstOfType(isEmbedding);
		seq_ = (SequenceCrossEntropy*) net_.cpuNet().findFirstOfType(isSeqCrossEntropy);
		if (!em_ || !seq_) { std::cerr << "[infer] " << path << ": no embedding / sequence head" << std::endl; return false; }
		seq_->setEps(0); seq_->setSmooth(0);
		return true;
	}
	CNet &getModel() { return net_; }
	int context() const { return seq_ ? seq_->nPos() : 0; }
	int vocab() const { return seq_ ? seq_->vocab() : 0; }

	// Next-token distribution after the last context()-length tail of `ctx`.
	std::vector<double> nextProbs(const std::vector<int> &ctx) {
		const int N = context(), V = vocab();
		std::vector<int> win(N, 0);
		const int n = (int) ctx.size(), take = std::min(n, N);
		for (int i = 0; i < take; ++i) win[i] = ctx[n - take + i];
		em_->setInput(win);
		net_.cpuNet().forward();
		std::vector<double> p(V);
		for (int k = 0; k < V; ++k) p[k] = seq_->getProbability(std::max(0, take - 1), k);
		return p;
	}
	// Sample one token at temperature `temp` (p^(1/temp), renormalized).
	int sample(const std::vector<int> &ctx, double temp, std::mt19937 &rng) {
		std::vector<double> p = nextProbs(ctx);
		double Z = 0;
		for (double &x : p) { x = std::pow(std::max(x, 1e-12), 1.0 / temp); Z += x; }
		double c = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * Z, s = 0;
		for (int k = 0; k < (int) p.size(); ++k) { s += p[k]; if (c <= s) return k; }
		return (int) p.size() - 1;
	}
	// Continue `prompt` by `n` tokens; `onToken` (optional) sees each token as it is produced.
	std::vector<int> generate(std::vector<int> prompt, int n, double temp, unsigned seed,
							  std::function<void(int)> onToken = nullptr) {
		std::mt19937 rng(seed);
		for (int g = 0; g < n; ++g) { int k = sample(prompt, temp, rng); prompt.push_back(k); if (onToken) onToken(k); }
		return prompt;
	}
	// Exact mean loss (nats/token) of `tokens`, in consecutive context()-length windows; the first token of each
	// window is unscored, as in CNet's streaming evaluation with resets every context() tokens.
	double score(const std::vector<int> &tokens) {
		const int N = context();
		double tot = 0; long cnt = 0;
		for (size_t s = 0; s + N <= tokens.size(); s += N) {
			std::vector<int> win(tokens.begin() + s, tokens.begin() + s + N), tgt(N, 0);
			for (int i = 0; i + 1 < N; ++i) tgt[i] = win[i + 1];
			em_->setInput(win); seq_->setTargets(tgt);
			net_.cpuNet().forward();
			for (int i = 0; i + 1 < N; ++i) { tot += -std::log(std::max((double) seq_->getProbability(i, tgt[i]), 1e-30)); ++cnt; }
		}
		return cnt ? tot / cnt : 0.0;
	}

private:
	CNet net_; CEmbedding *em_ = nullptr; SequenceCrossEntropy *seq_ = nullptr;
};

}  // namespace app
