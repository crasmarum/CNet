// GPU-vs-CPU parity for ChirpFilter and BoxcarMax: [x | fc] , d -> ChirpFilter -> |.|^2 -> BoxcarMax -> CrossEntropy.
// Checks the forward output of every stage and the input gradients (signal and DM parameters). Needs a GPU.
#include <iostream>
#include <vector>
#include <random>

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/cinput.h"
#include "../impl/crossent.h"
#include "../impl/relu.h"
#include "../impl/chirp.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

static bool one(int L, int K, double fc, std::vector<float> dms) {
	std::vector<int> widths = {8, 32, 128};
	int W = (int) widths.size();
	CNet net;
	net.cpuNet().add(new CInput(Uid(1), OutSize(L + 1)));                    // x and fc
	net.cpuNet().add(new CInput(Uid(2), OutSize(K)));                        // DM parameters
	net.cpuNet().add(new ChirpFilter(Uid(3), L, K), {1, 2});
	net.cpuNet().add(new CModulus2(Uid(4), InSize(K * L)), {3});
	net.cpuNet().add(new BoxcarMax(Uid(5), K, L, widths), {4});
	net.cpuNet().add(new CrossEntropy(Uid(6), InSize(K * W)), {5});
	net.cpuNet().init_inputs(7);
	net.cpuNet().init_exec_graph();
	CInput *x = (CInput*) net.cpuNet()[1], *d = (CInput*) net.cpuNet()[2];
	std::mt19937 rng(3); std::normal_distribution<float> N01(0.f, 1.f);
	for (int t = 0; t < L; ++t) { x->mutable_input()->real_[t] = N01(rng); x->mutable_input()->imag_[t] = N01(rng); }
	x->mutable_input()->real_[L] = (float) fc; x->mutable_input()->imag_[L] = 0;
	for (int k = 0; k < K; ++k) { d->mutable_input()->real_[k] = dms[k]; d->mutable_input()->imag_[k] = 0; }

	bool fail = false;
	net.AllocateNet(1, fail);
	if (fail) { std::cout << "  AllocateNet failed\n"; return false; }
	net.cpuNet().forward();
	net.GpuForward(fail);
	bool okChirp = net.testDataFromGpu(*net.getMappingFor(net.cpuNet()[4]), 0.01);   // ChirpFilter output
	bool okBox   = net.testDataFromGpu(*net.getMappingFor(net.cpuNet()[6]), 0.01);   // BoxcarMax output
	int label = 1;
	net.cpuNet().backward(label);
	net.GpuBackward(label, fail);
	bool okGc = net.testGradientsFromGpu(*net.getMappingFor(net.cpuNet()[3]), 0.01); // grads of x and d
	bool okGb = net.testGradientsFromGpu(*net.getMappingFor(net.cpuNet()[5]), 0.01); // grads of |y|^2
	std::cout << "  L=" << L << " K=" << K << " fc=" << fc << " MHz: chirp fwd " << (okChirp ? "ok" : "BAD")
			  << ", boxcar fwd " << (okBox ? "ok" : "BAD") << ", boxcar grad " << (okGb ? "ok" : "BAD")
			  << ", chirp grad (x, DM) " << (okGc ? "ok" : "BAD") << std::endl;
	return okChirp && okBox && okGc && okGb;
}

int main(int argc, char** argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	bool ok = true;
	ok &= one(256, 2, 600.0, {1.f, 4.f});
	ok &= one(1024, 3, 420.0, {2.f, 6.f, 11.f});
	ok &= one(16384, 4, 450.0, {1.f, 5.f, 9.f, 14.f});   // production size
	std::cout << (ok ? "PASS" : "FAIL") << std::endl;
	return ok ? 0 : 1;
}
