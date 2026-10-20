// Born-rule attention micro-benchmark: CNet's fused BornAttention CUDA kernel
// (eager and CUDA-graph captured) vs. a composite torch.complex64 implementation
// (examples/torch_bornattn_bench.py). Times one forward+backward on the clone batch
// and reports peak GPU memory (cudaMemGetInfo delta over allocation + first step).
//
//   net:  CInput(3*N*d)  ->  BornAttention(N, d, causal)  ->  L2Out(N*d)
#include <iostream>
#include <cstdio>
#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../impl/cinput.h"
#include "../impl/bornattn.h"
#include "../impl/l2out.h"
#include "../impl/tokenwise.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(ba_tokens, 256)     // N (context length)
FLAG_INT(ba_dim, 64)         // d (per-token feature dim)
FLAG_INT(ba_batch, 64)
FLAG_INT(ba_iters, 100)
FLAG_INT(ba_graph, 0)        // 1 = CUDA-graph replay, 0 = eager
FLAG_INT(ba_heads, 1)        // heads (d must be divisible)
FLAG_INT(ba_linear, 0)       // 1 = exact chunkwise linear form, 0 = quadratic
FLAG_INT(ba_chunk, 512)      // chunk length of the linear form (N % chunk == 0)

// [Q;K;V] -> BornAttention -> TokenwiseLinear(d -> 1) -> L2Out(N). The per-token projection
// keeps the loss layer's length at N: L2Out's GPU reduction scratch grows as length^2/block
// (reduce.h getPaddedLength), which at length N*d would dominate the memory of this benchmark.
static void buildNet(ComplexNet& n, int N, int d) {
	int inp = n.add(new CInput(Uid(1), OutSize(3 * N * d)));          // [Q;K;V]
	int att = n.add(new BornAttention(Uid(2), N, d, true, 0.f, ba_heads), {inp});
	((BornAttention*) n[att])->setLinear(ba_linear != 0, ba_chunk);
	int w   = n.add(new CInput(Uid(3), OutSize(d)));
	int prj = n.add(new TokenwiseLinear(Uid(4), N, d, 1), {att, w});
	n.add(new L2Out(Uid(5), InSize(N)), {prj});
}
int main(int argc, char** argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	int N = ba_tokens, d = ba_dim, B = ba_batch;
	bool use_graph = ba_graph != 0;
	std::printf("BornAttention: N=%d d=%d heads=%d batch=%d  form=%s%s  mode=%s\n", N, d, ba_heads, B,
				ba_linear ? "linear" : "quadratic", ba_linear ? (" chunk=" + std::to_string(ba_chunk)).c_str() : "",
				use_graph ? "CUDA-graphs" : "eager");

#ifdef __CUDACC__
	size_t free0 = 0, total0 = 0; cudaMemGetInfo(&free0, &total0);
#endif
	CNet net;
	buildNet(net.cpuNet(), N, d);
	net.cpuNet().init_inputs();
	net.cpuNet().init_exec_graph();
	bool fail = false;
	net.AllocateNet(B, fail);
	if (fail) { std::printf("(alloc failed)\n"); return 1; }
#ifdef __CUDACC__
	{ size_t fr = 0, to = 0; cudaMemGetInfo(&fr, &to); std::printf("  after AllocateNet: %.1f MB\n", (double)(free0 - fr) / 1e6); }
	net.GpuForward(fail);
	cudaDeviceSynchronize();
	{ size_t fr = 0, to = 0; cudaMemGetInfo(&fr, &to); std::printf("  after first forward: %.1f MB\n", (double)(free0 - fr) / 1e6); }
	net.GpuBackward(0, fail);
	cudaDeviceSynchronize();
	{ size_t fr = 0, to = 0; cudaMemGetInfo(&fr, &to); std::printf("  after first backward: %.1f MB\n", (double)(free0 - fr) / 1e6); }
#endif

	// warmup (also triggers born-buffer allocation + graph capture)
	for (int w = 0; w < 5; ++w) {
#ifdef __CUDACC__
		if (use_graph) { net.GpuForwardGraphed(fail); net.GpuBackwardGraphed(0, fail); }
		else           { net.GpuForward(fail);        net.GpuBackward(0, fail); }
#else
		net.GpuForward(fail); net.GpuBackward(0, fail);
#endif
	}

#ifdef __CUDACC__
	cudaDeviceSynchronize();
	size_t free1 = 0, total1 = 0; cudaMemGetInfo(&free1, &total1);
	double used_mb = (double)(free0 - free1) / 1e6;
#endif

	StopWatch f;
	for (int i = 0; i < ba_iters; ++i) {
#ifdef __CUDACC__
		if (use_graph) net.GpuForwardGraphed(fail); else net.GpuForward(fail);
#else
		net.GpuForward(fail);
#endif
	}
	double fwd = (double) f.ElapsedTimeMicros() / ba_iters / 1000.0;

	StopWatch b;
	for (int i = 0; i < ba_iters; ++i) {
#ifdef __CUDACC__
		if (use_graph) net.GpuBackwardGraphed(0, fail); else net.GpuBackward(0, fail);
#else
		net.GpuBackward(0, fail);
#endif
	}
	double bwd = (double) b.ElapsedTimeMicros() / ba_iters / 1000.0;

#ifdef __CUDACC__
	std::printf("fwd=%.3f ms  bwd=%.3f ms  step=%.3f ms  peak_gpu_mem=%.1f MB\n",
				fwd, bwd, fwd + bwd, used_mb);
#else
	std::printf("fwd=%.3f ms  bwd=%.3f ms  step=%.3f ms\n", fwd, bwd, fwd + bwd);
#endif
	return 0;
}
