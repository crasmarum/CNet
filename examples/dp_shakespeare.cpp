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
FLAG_STRING(data, "tiny_shakespear2.txt")

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

// ------- fully complex FNet-AR language model (causal Fourier mixing) ---------
// embedding + learnable positional + L x [TriangFourier mix -> residual+CGelu ->
// position-wise FFN E->4E->E -> residual+CGelu] + per-token projection + Born-rule
// SequenceCrossEntropy. Returns the output (SequenceCrossEntropy) id.
static int buildLM(ComplexNet &net, int embId, int E, int N, int L, int vocab) {
	const int size = N * E, H = 4 * E;
	int pos  = net.add(new CInput(OutSize(size)));
	int last = net.add(new Residual(InSize(size), InSize(size)), {embId, pos});
	for (int b = 0; b < L; ++b) {
		int mix = net.add(new TriangFourier(InSize(size)), {last});        // causal token mix
		int r0  = net.add(new Residual(InSize(size), InSize(size)), {last, mix});
		int n0  = net.add(new CGelu(InSize(size)), {r0});
		int w1  = net.add(new CInput(OutSize(E * H)));
		int f1  = net.add(new TokenwiseLinear(N, E, H), {n0, w1});          // position-wise FFN
		int g   = net.add(new CGelu(InSize(N * H)), {f1});
		int w2  = net.add(new CInput(OutSize(H * E)));
		int f2  = net.add(new TokenwiseLinear(N, H, E), {g, w2});
		int r1  = net.add(new Residual(InSize(size), InSize(size)), {n0, f2});
		last    = net.add(new CGelu(InSize(size)), {r1});
	}
	int wo     = net.add(new CInput(OutSize(E * vocab)));
	int logits = net.add(new TokenwiseLinear(N, E, vocab), {last, wo});     // per-token -> vocab
	int ce     = net.add(new SequenceCrossEntropy(InSize(N * vocab), vocab), {logits});
	return ce;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	const int W = world, R = rank, N = tokens, E = emb, L = blocks;
	if (W < 1 || batch % W != 0) {
		std::cerr << "batch (" << batch << ") must be a positive multiple of world (" << W << ")\n";
		return 1;
	}
	const int per = batch / W;
	int nDev = 0; cudaGetDeviceCount(&nDev); if (nDev < 1) nDev = 1;
	cudaSetDevice(R % nDev);

	// --- NCCL rendezvous: rank 0 writes the unique id to /tmp; others poll. ---
	const char *idf = "/tmp/cnet_shk_id.bin", *rdy = "/tmp/cnet_shk_id.ready";
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

	CharData ds; ds.load(data);
	const int vocab = ds.vocab;
	if (R == 0)
		std::cout << "tiny-shakespeare: chars=" << ds.data.size() << " vocab=" << vocab
				  << "  model E=" << E << " N=" << N << " L=" << L
				  << "  global_batch=" << batch << " world=" << W
				  << "  chance=ln(vocab)=" << std::log((double) vocab) << std::endl;

	CNet net;
	int embId = net.cpuNet().add(new CEmbedding(E, N, vocab));
	int ceId  = buildLM(net.cpuNet(), embId, E, N, L, vocab);
	((CEmbedding*) net.cpuNet()[embId])->setIsMainInput(true);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setIsMainOutput(true);
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

	std::mt19937 rng(1000 + R);                    // each rank draws a different data shard
	auto targetsOf = [&](const std::vector<int> &win, int lab) {
		std::vector<int> t(N, 0);
		for (int i = 0; i + 1 < (int) win.size(); ++i) t[i] = win[i + 1];
		if (!win.empty()) t[win.size() - 1] = lab;
		return t;
	};
	auto buildBatch = [&](bool heldout) {
		EmbeddingBatch b(per, N);
		int lo = heldout ? ds.ntrain : 0;
		int hi = (heldout ? (int) ds.data.size() : ds.ntrain) - N - 2;
		std::uniform_int_distribution<int> d(lo, hi);
		for (int i = 0; i < per; ++i) {
			int p = d(rng);
			std::vector<int> win(ds.data.begin() + p, ds.data.begin() + p + N);
			int lab = ds.data[p + N];
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
	for (int t = 1; t <= steps; ++t) {
		EmbeddingBatch batch_ = buildBatch(false);
		net.gpuForward(em, seq, batch_);
		acc += net.getLoss(0)[seq->uid()]; ++accn;
		net.gpuBackward();
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // gather
			cudaMemcpyAsync(scratch + off[i], gradBufs[i].first,
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		ncclAllReduce(scratch, scratch, total, ncclFloat, ncclSum, comm, cudaStreamPerThread);
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // scatter
			cudaMemcpyAsync(gradBufs[i].first, scratch + off[i],
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		net.trueAdamUpdate(lr_at(t), b1, b2, eps, t, 0.0f);

		if (R == 0 && t % 100 == 0) {
			std::cout << "step " << t << "  train " << (acc / accn) << " nats/char  lr "
					  << lr_at(t) << "  " << (sw.ElapsedTimeMicros() / 1000.0 / 100) << " ms/step"
					  << std::endl;
			acc = 0; accn = 0; sw.Reset();
		}
		if (R == 0 && val_every > 0 && t % val_every == 0) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			std::cout << "  [val] step " << t << "  val " << (v / K) << " nats/char" << std::endl;
			sw.Reset();
		}
	}
	cudaStreamSynchronize(cudaStreamPerThread);
	cudaFree(scratch);
	ncclCommDestroy(comm);
	if (R == 0) { remove(idf); remove(rdy); }
	return 0;
}
