// =============================================================================
// dp_shakespeare: a standalone, multi-GPU (data-parallel) character-level
// language model on tiny-shakespeare, built with the CNet complex-valued
// framework. A fully complex FNet-style model (causal Fourier token mixer +
// position-wise feed-forward) trained with a Born-rule sequence loss.
//
// Data parallelism is multi-PROCESS (the standard DDP architecture): launch one
// process per GPU; the ranks rendezvous via a NCCL unique id written to /tmp by
// rank 0, then cross-process all-reduce the gradients each step. Deterministic
// init makes every rank start from identical weights.
//
// Build:
//   make dp_shakespeare
//
// Run on 1 GPU:
//   ./dp_shakespeare -world 1 -batch 32 -steps 3000 -data tiny_shakespear2.txt
//
// Run on N GPUs (one process per GPU) -- see examples/run_dp_shakespeare.sh:
//   rm -f /tmp/cnet_shk_id.*
//   for r in $(seq 0 $((N-1))); do
//     CUDA_VISIBLE_DEVICES=$r ./dp_shakespeare -world N -rank $r -batch 32 ... &
//   done; wait
// =============================================================================
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <nccl.h>
#include <cublas_v2.h>

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../impl/batch.h"
#include "../impl/cinput.h"
#include "../impl/embed.h"
#include "../impl/ft.h"
#include "../impl/tokenwise.h"
#include "../impl/relu.h"
#include "../impl/residual.h"
#include "../impl/seqcrossent.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(rank, 0)            // this process's rank in [0, world)
FLAG_INT(world, 1)          // number of data-parallel processes (one per GPU)
FLAG_INT(batch, 32)         // GLOBAL batch, split across ranks (per-rank = batch/world)
FLAG_INT(emb, 64)           // embedding width E
FLAG_INT(tokens, 64)        // context length N
FLAG_INT(blocks, 4)         // number of FNet blocks L
FLAG_INT(steps, 3000)       // optimizer steps
FLAG_INT(warmup, 200)       // linear-warmup steps
FLAG_INT(val_every, 500)    // validation period (0 = off)
FLAG_FLOAT(lr, 1e-3)        // peak learning rate
FLAG_FLOAT(min_lr, 1e-4)    // cosine floor
FLAG_FLOAT(grad_clip, 0.0)  // true-Adam clip-by-value per grad component (0 = off)
FLAG_FLOAT(grad_norm_clip, 0.0) // GLOBAL gradient-norm clip on the all-reduced grad (0 = off):
                                // scales the whole gradient when ||g|| exceeds this, which
                                // (unlike clip-by-value) bounds the coordinated large-norm update
                                // that the spike-prone Born loss produces. Applied identically on
                                // every rank (same summed gradient), so replicas stay in sync.
FLAG_STRING(data, "tiny_shakespear2.txt")
FLAG_BOOL(born, false)      // use Born-rule attention mixer instead of causal Fourier
FLAG_BOOL(block_norm, false) // L2-normalize (SoftMax) each block output (deep nets)
FLAG_BOOL(token_norm, false) // per-token complex RMS pre-norm (pre-LN transformer)
FLAG_BOOL(rope, false)      // rotary position embedding on Born-attention Q,K
FLAG_FLOAT(ce_eps, 0.0)     // Born-loss target-prob floor (0 = exact; ~1e-3 stabilizes)
FLAG_STRING(save_path, "")  // if set, rank 0 saves the trained model here after training
FLAG_BOOL(generate, false)  // generation mode: restore -model and sample text (no training)
FLAG_STRING(model, "")      // model file to restore in -generate mode
FLAG_STRING(prompt, "ROMEO:")// seed text for generation
FLAG_INT(gen_len, 600)      // number of characters to generate
FLAG_FLOAT(temp, 0.8)       // sampling temperature (lower = greedier)
FLAG_BOOL(print_net, false) // build the LM and print its layer graph, then exit
FLAG_BOOL(token_lm, false)  // sub-word token LM: -data is a DIR with train.bin/val.bin (uint16 ids)
FLAG_INT(vocab_size, 8192)  // vocab in -token_lm mode (must match the BPE used to make the bins)
FLAG_STRING(nccl_id, "/tmp/cnet_shk_id")  // base path for the NCCL rendezvous id file (per DP group)

// ---------------- minimal character dataset (90/10 train/val) ----------------
struct CharData {
	std::vector<int> data;
	std::map<char, int> c2i;
	int vocab = 0, ntrain = 0;
	void load(const std::string &path) {
		std::ifstream f(path);
		std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		if (s.empty()) { std::cerr << "cannot read " << path << "\n"; exit(1); }
		for (char c : s) if (!c2i.count(c)) c2i[c] = vocab++;
		for (char c : s) data.push_back(c2i[c]);
		ntrain = (int) ((data.size() / 10) * 9);
	}
};

// ---- sub-word token dataset: <dir>/train.bin + <dir>/val.bin, uint16 ids ----
// train ids occupy [0, ntrain); val ids are appended at [ntrain, end), so the
// existing held-out sampler (which draws from [ntrain, size)) reads validation.
struct TokenData {
	std::vector<int> data;
	int vocab = 0, ntrain = 0;
	static void readBin(const std::string &p, std::vector<int> &out) {
		std::ifstream f(p, std::ios::binary);
		if (!f) { std::cerr << "cannot read " << p << "\n"; exit(1); }
		f.seekg(0, std::ios::end); std::streamoff n = f.tellg(); f.seekg(0);
		std::vector<uint16_t> buf((size_t) n / 2);
		f.read((char*) buf.data(), (std::streamsize) buf.size() * 2);
		out.reserve(out.size() + buf.size());
		for (uint16_t v : buf) out.push_back((int) v);
	}
	void load(const std::string &dir, int vocab_) {
		vocab = vocab_;
		std::vector<int> tr, va;
		readBin(dir + "/train.bin", tr);
		readBin(dir + "/val.bin", va);
		ntrain = (int) tr.size();
		data = std::move(tr);
		data.insert(data.end(), va.begin(), va.end());
	}
};

// ------- fully complex FNet-AR language model (causal Fourier mixing) ---------
// embedding + learnable positional + L x [TriangFourier mix -> residual+CGelu ->
// position-wise FFN E->4E->E -> residual+CGelu] + per-token projection + Born-rule
// SequenceCrossEntropy. Returns the output (SequenceCrossEntropy) id.
static int buildLM(ComplexNet &net, int embId, int E, int N, int L, int vocab) {
	const int size = N * E, H = 4 * E;
	int pos  = net.add(new CInput(OutSize(size)));
	int last = net.add(new Residual(InSize(size), InSize(size)), {embId, pos});
	const int dh = E / 2;   // Born-rule attention head dim (complex)
	// Per-token pre-norm (complex RMSNorm). When on, each sub-block sees a
	// normalized residual-stream input; the residual still adds to the un-normed
	// stream (standard pre-LN), which is what lets depth train stably.
	auto norm = [&](int x) { return token_norm ? net.add(new TokenNorm(N, E), {x}) : x; };
	for (int b = 0; b < L; ++b) {
		int mixout;
		if (born) {                                                        // Born-rule attention
			int pre = net.add(new CGelu(InSize(size)), {norm(last)});
			int wQ = net.add(new CInput(OutSize(E * dh))); int Q  = net.add(new TokenwiseLinear(N, E, dh), {pre, wQ});
			int wK = net.add(new CInput(OutSize(E * dh))); int K  = net.add(new TokenwiseLinear(N, E, dh), {pre, wK});
			int wV = net.add(new CInput(OutSize(E * dh))); int Vv = net.add(new TokenwiseLinear(N, E, dh), {pre, wV});
			if (rope) {                                                        // RoPE on queries/keys
				Q = net.add(new RotaryEmbed(N, dh), {Q});
				K = net.add(new RotaryEmbed(N, dh), {K});
			}
			int att = net.add(new BornAttention(N, dh), {Q, K, Vv});
			int wO = net.add(new CInput(OutSize(dh * E)));
			mixout = net.add(new TokenwiseLinear(N, dh, E), {att, wO});
		} else {                                                           // causal Fourier
			mixout = net.add(new TriangFourier(InSize(size)), {norm(last)});
		}
		int r0  = net.add(new Residual(InSize(size), InSize(size)), {last, mixout});
		int n0  = net.add(new CGelu(InSize(size)), {r0});
		int w1  = net.add(new CInput(OutSize(E * H)));
		int f1  = net.add(new TokenwiseLinear(N, E, H), {norm(n0), w1});    // position-wise FFN (pre-norm)
		int g   = net.add(new CGelu(InSize(N * H)), {f1});
		int w2  = net.add(new CInput(OutSize(H * E)));
		int f2  = net.add(new TokenwiseLinear(N, H, E), {g, w2});
		int r1  = net.add(new Residual(InSize(size), InSize(size)), {n0, f2});
		last    = net.add(new CGelu(InSize(size)), {r1});
		if (block_norm) last = net.add(new SoftMax(InSize(size)), {last});   // per-block L2 norm
	}
	last       = norm(last);                                                // final pre-logits norm
	int wo     = net.add(new CInput(OutSize(E * vocab)));
	int logits = net.add(new TokenwiseLinear(N, E, vocab), {last, wo});     // per-token -> vocab
	int ce     = net.add(new SequenceCrossEntropy(InSize(N * vocab), vocab), {logits});
	return ce;
}

// Temperature sampling from a position's Born distribution p_k = |z_k|^2/||z||^2
// (SequenceCrossEntropy normalises the per-position outputs in forward()).
static int sampleNext(SequenceCrossEntropy *seq, int p, int vocab, double temp, std::mt19937 &rng) {
	std::vector<double> q(vocab); double Z = 0;
	for (int k = 0; k < vocab; ++k) {
		double pr = seq->getProbability(p, k);
		pr = std::pow(pr < 1e-12 ? 1e-12 : pr, 1.0 / temp);
		q[k] = pr; Z += pr;
	}
	std::uniform_real_distribution<double> U(0.0, 1.0);
	double c = U(rng) * Z, s = 0;
	for (int k = 0; k < vocab; ++k) { s += q[k]; if (c <= s) return k; }
	return vocab - 1;
}

// Restore a saved model and autoregressively sample text on the CPU. The mixers
// are causal, so right-filling the N-token window and reading the distribution at
// the last real position is exact (the zero padding never influences it).
static int runGenerate() {
	FILELog::ReportingLevel() = lWarning;   // quiet per-layer restore/forward debug logs
	if (((std::string) model).empty()) { std::cerr << "-generate requires -model <file>\n"; return 1; }
	CharData ds; ds.load(data);
	std::vector<char> i2c(ds.vocab);
	for (auto &kv : ds.c2i) i2c[kv.second] = kv.first;

	CNet net;
	if (!net.restore((std::string) model)) { std::cerr << "cannot restore model: " << (std::string) model << "\n"; return 1; }
	auto *em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	auto *seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model has no embedding / sequence head\n"; return 1; }
	const int Ngen = seq->nPos(), vocab = seq->vocab();

	std::mt19937 rng(std::random_device{}());
	std::vector<int> ctx;
	for (char c : (std::string) prompt) if (ds.c2i.count(c)) ctx.push_back(ds.c2i[c]);
	std::string out = prompt;
	for (int g = 0; g < gen_len; ++g) {
		std::vector<int> win(Ngen, 0);
		int ctxn = (int) ctx.size(), take = std::min(ctxn, Ngen);
		for (int i = 0; i < take; ++i) win[i] = ctx[ctxn - take + i];   // real tokens at 0..take-1
		em->setInput(win);
		net.cpuNet().forward();
		int nxt = sampleNext(seq, take - 1, vocab, temp, rng);           // next after last real token
		ctx.push_back(nxt); out += i2c[nxt];
	}
	std::cout << "=== generated (model=" << model << " temp=" << temp << ") ===\n"
			  << out << std::endl;
	return 0;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	if (generate) return runGenerate();          // standalone sampling, no training / NCCL
	if (print_net) {                              // dump the layer graph, no training / NCCL
		int vcb;
		if (token_lm) vcb = vocab_size;
		else { CharData ds; ds.load(data); vcb = ds.vocab; }
		CNet net;
		int embId = net.cpuNet().add(new CEmbedding(emb, tokens, vcb));
		buildLM(net.cpuNet(), embId, emb, tokens, blocks, vcb);
		std::cout << "config: born=" << born << " token_norm=" << token_norm
				  << " rope=" << rope << " block_norm=" << block_norm
				  << "  E=" << emb << " N=" << tokens << " L=" << blocks
				  << " vocab=" << vcb << "\n\n"
				  << net.cpuNet().toString();
		return 0;
	}
	const int W = world, R = rank, N = tokens, E = emb, L = blocks;
	if (W < 1 || batch % W != 0) {
		std::cerr << "batch (" << batch << ") must be a positive multiple of world (" << W << ")\n";
		return 1;
	}
	const int per = batch / W;
	int nDev = 0; cudaGetDeviceCount(&nDev); if (nDev < 1) nDev = 1;
	cudaSetDevice(R % nDev);

	// --- NCCL rendezvous: rank 0 writes the unique id to /tmp; others poll. ---
	// The id-file base is a flag so independent DP groups can run concurrently
	// (e.g. a Born run on GPUs 0-3 and a Fourier run on 4-7 at the same time).
	std::string idf_s = (std::string) nccl_id + ".bin", rdy_s = (std::string) nccl_id + ".ready";
	const char *idf = idf_s.c_str(), *rdy = rdy_s.c_str();
	ncclUniqueId id;
	if (R == 0) {
		ncclGetUniqueId(&id);
		{ std::ofstream f(idf, std::ios::binary); f.write((char*) &id, sizeof(id)); }
		{ std::ofstream r(rdy); r << "1"; }
	} else {
		int t = 0;
		while (!std::ifstream(rdy).good()) { usleep(50000); if (++t > 2400) { std::cerr << "rank " << R << ": id timeout\n"; return 1; } }
		usleep(100000);
		std::ifstream f(idf, std::ios::binary); f.read((char*) &id, sizeof(id));
	}
	ncclComm_t comm;
	if (ncclCommInitRank(&comm, W, id, R) != ncclSuccess) { std::cerr << "rank " << R << ": nccl init failed\n"; return 1; }

	std::vector<int> ids; int vocab, ntrainTok;
	if (token_lm) { TokenData d; d.load(data, vocab_size); ids = std::move(d.data); vocab = d.vocab; ntrainTok = d.ntrain; }
	else          { CharData  d; d.load(data);             ids = std::move(d.data); vocab = d.vocab; ntrainTok = d.ntrain; }
	if (R == 0)
		std::cout << (token_lm ? "token-LM: tokens=" : "tiny-shakespeare: chars=") << ids.size() << " vocab=" << vocab
				  << "  model E=" << E << " N=" << N << " L=" << L
				  << "  global_batch=" << batch << " world=" << W
				  << "  born=" << born << " block_norm=" << block_norm
				  << " token_norm=" << token_norm << " rope=" << rope
				  << "  grad_clip=" << grad_clip << " ce_eps=" << ce_eps
				  << "  chance=ln(vocab)=" << std::log((double) vocab) << std::endl;

	CNet net;
	int embId = net.cpuNet().add(new CEmbedding(E, N, vocab));
	int ceId  = buildLM(net.cpuNet(), embId, E, N, L, vocab);
	((CEmbedding*) net.cpuNet()[embId])->setIsMainInput(true);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setIsMainOutput(true);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setEps(ce_eps);  // 0 = exact Born
	net.cpuNet().init_inputs(1234);                // identical init on every rank
	net.allocateOnGpu(per);
	CEmbedding *em = (CEmbedding*) net.cpuNet()[embId];
	SequenceCrossEntropy *seq = (SequenceCrossEntropy*) net.cpuNet()[ceId];

	// Ancestor-gradient buffers coalesced into one scratch -> a single all-reduce.
	auto gradBufs = net.ancestorGradBuffers();
	std::vector<int> off(gradBufs.size());
	int total = 0;
	for (size_t i = 0; i < gradBufs.size(); ++i) { off[i] = total; total += gradBufs[i].second; }
	float *scratch = nullptr; cudaMalloc(&scratch, (size_t) total * sizeof(float));
	// cuBLAS handle for the global gradient-norm clip (operates on the coalesced
	// all-reduced gradient in `scratch`).
	cublasHandle_t gbh = 0;
	if (grad_norm_clip > 0.f) { cublasCreate(&gbh); cublasSetStream(gbh, cudaStreamPerThread); cublasSetPointerMode(gbh, CUBLAS_POINTER_MODE_HOST); }

	std::mt19937 rng(1000 + R);                    // each rank draws a different data shard
	auto targetsOf = [&](const std::vector<int> &win, int lab) {
		std::vector<int> t(N, 0);
		for (int i = 0; i + 1 < (int) win.size(); ++i) t[i] = win[i + 1];
		if (!win.empty()) t[win.size() - 1] = lab;
		return t;
	};
	auto buildBatch = [&](bool heldout) {
		EmbeddingBatch b(per, N);
		int lo = heldout ? ntrainTok : 0;
		int hi = (heldout ? (int) ids.size() : ntrainTok) - N - 2;
		std::uniform_int_distribution<int> d(lo, hi);
		for (int i = 0; i < per; ++i) {
			int p = d(rng);
			std::vector<int> win(ids.begin() + p, ids.begin() + p + N);
			int lab = ids[p + N];
			b.add(win, lab);
			b.addTargets(targetsOf(win, lab));
		}
		return b;
	};
	auto lr_at = [&](int t) -> float {
		if (t < warmup) return lr * (float) (t + 1) / warmup;
		float p = (float) (t - warmup) / std::max(1, steps - warmup);
		return min_lr + 0.5f * (lr - min_lr) * (1.f + std::cos(3.14159265f * p));
	};

	const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	StopWatch sw; float acc = 0; int accn = 0;
	float best_val = 1e30f;   // best held-out loss seen, for best-checkpoint saving
	for (int t = 1; t <= steps; ++t) {
		float gnorm = 0.f;
		EmbeddingBatch batch_ = buildBatch(false);
		net.gpuForward(em, seq, batch_);
		acc += net.getLoss(0)[seq->uid()]; ++accn;
		net.gpuBackward();
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // gather
			cudaMemcpyAsync(scratch + off[i], gradBufs[i].first,
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		ncclAllReduce(scratch, scratch, total, ncclFloat, ncclSum, comm, cudaStreamPerThread);
		// Global gradient-NORM clip on the summed gradient (same on every rank).
		if (grad_norm_clip > 0.f) {
			cublasSnrm2(gbh, total, scratch, 1, &gnorm);
			if (gnorm > grad_norm_clip) {
				float s = grad_norm_clip / gnorm;
				cublasSscal(gbh, total, &s, scratch, 1);
			}
		}
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // scatter
			cudaMemcpyAsync(gradBufs[i].first, scratch + off[i],
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		net.trueAdamUpdate(lr_at(t), b1, b2, eps, t, grad_clip);

		if (R == 0 && t % 100 == 0) {
			std::cout << "step " << t << "  train " << (acc / accn) << " nats/char  lr "
					  << lr_at(t) << "  |g|=" << gnorm << "  "
					  << (sw.ElapsedTimeMicros() / 1000.0 / 100) << " ms/step"
					  << std::endl;
			acc = 0; accn = 0; sw.Reset();
		}
		if (R == 0 && val_every > 0 && t % val_every == 0) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			float vv = v / K;
			std::cout << "  [val] step " << t << "  val " << vv << " nats/char" << std::endl;
			// Save the BEST checkpoint so a later Born-loss spike can't wipe out the
			// run's best model (the loss is spike-prone at large vocab).
			if (!((std::string) save_path).empty() && vv < best_val) {
				best_val = vv;
				if (net.getInputsFromGpu() && net.save((std::string) save_path))
					std::cout << "    [ckpt] new best val " << vv << " -> saved " << (std::string) save_path << std::endl;
			}
			sw.Reset();
		}
	}
	cudaStreamSynchronize(cudaStreamPerThread);

	// Rank 0 checkpoints the final model (weights pulled GPU->CPU first). Every rank
	// holds identical weights, so one save suffices. Generate later with
	// -generate -model <save_path>. When held-out validation is on, best-val
	// checkpoints were already saved during training; only overwrite with the final
	// model if it is at least as good (don't let a late spike clobber the best).
	if (R == 0 && !((std::string) save_path).empty()) {
		bool do_save = true;
		if (val_every > 0 && best_val < 1e29f) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			float vv = v / K;
			do_save = (vv <= best_val);
			std::cout << "final val " << vv << " (best " << best_val << "); "
					  << (do_save ? "saving final" : "keeping best checkpoint") << std::endl;
		}
		if (do_save) {
			if (net.getInputsFromGpu() && net.save((std::string) save_path))
				std::cout << "saved model to " << (std::string) save_path << std::endl;
			else
				std::cerr << "WARNING: model save to " << (std::string) save_path << " failed\n";
		}
	}

	cudaFree(scratch);
	ncclCommDestroy(comm);
	if (R == 0) { remove(idf); remove(rdy); }
	return 0;
}
