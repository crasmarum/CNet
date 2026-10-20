// Standalone checks for multi-head BornAttention / per-head RotaryEmbed (no NCCL):
//   1. finite-difference check of the Wirtinger backward with heads > 1 (CPU);
//   2. head-slicing oracle: the H-head operator on full inputs equals the verified
//      single-head operator run on inputs zeroed outside head h, on head h's dims;
//   3. GPU-vs-CPU parity of the forward output and the input gradients (needs a GPU);
//   4. serialization round trip of the extended (heads, eps) records.
// Build: make test_bornattn_mh   Run: ./build/test_bornattn_mh [-gpu 0]
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../impl/cinput.h"
#include "../impl/bornattn.h"
#include "../impl/rotary.h"
#include "../impl/l2out.h"
#include "../impl/model_saver.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(gpu, 1)   // 0 = skip the GPU parity checks

// ---- 1. finite differences: [Q;K;V] -> BornAttention(heads) -> L2Out ----------
static bool fdBorn(int N, int d, int heads, bool causal, bool decay = false) {
	const int M = N * d;
	ComplexNet net;
	int qid = net.add(new CInput(OutSize(M)));
	int kid = net.add(new CInput(OutSize(M)));
	int vid = net.add(new CInput(OutSize(M)));
	int ba  = net.add(new BornAttention(N, d, causal, 0.f, heads), {qid, kid, vid});
	((BornAttention*) net[ba])->setDecay(decay);
	int lo  = net.add(new L2Out(InSize(M)), {ba});
	net.init_inputs(7);
	net.forward(); net.backward(0);
	auto Lval = [&]() { net.forward(); return (double) ((L2Out*) net[lo])->loss(0); };
	const double eps = 1e-3; double worst = 0;
	int ids[3] = {qid, kid, vid};
	for (int b = 0; b < 3; ++b) {
		CFunc *in = net[ids[b]];
		for (int i = 0; i < M; ++i) {
			double anr = 2.0 * in->input().dz_star_real_[i], ani = 2.0 * in->input().dz_star_imag_[i];
			float sv = in->input().real_[i];
			in->input().real_[i] = sv + eps; double Lp = Lval();
			in->input().real_[i] = sv - eps; double Lm = Lval();
			in->input().real_[i] = sv;
			double nnr = (Lp - Lm) / (2 * eps);
			sv = in->input().imag_[i];
			in->input().imag_[i] = sv + eps; Lp = Lval();
			in->input().imag_[i] = sv - eps; Lm = Lval();
			in->input().imag_[i] = sv;
			double nni = (Lp - Lm) / (2 * eps);
			worst = std::max(worst, std::max(std::fabs(anr - nnr), std::fabs(ani - nni)));
		}
	}
	bool ok = worst < 1e-2;
	std::cout << "  FD Born N=" << N << " d=" << d << " heads=" << heads << " causal=" << causal << " decay=" << decay
			  << "  worst=" << worst << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

static bool fdRope(int N, int d, int heads) {
	const int M = N * d;
	ComplexNet net;
	int xid = net.add(new CInput(OutSize(M)));
	int ro  = net.add(new RotaryEmbed(N, d, 10000.0, heads), {xid});
	int lo  = net.add(new L2Out(InSize(M)), {ro});
	net.init_inputs(11);
	net.forward(); net.backward(0);
	auto Lval = [&]() { net.forward(); return (double) ((L2Out*) net[lo])->loss(0); };
	const double eps = 1e-3; double worst = 0;
	CFunc *in = net[xid];
	for (int i = 0; i < M; ++i) {
		double anr = 2.0 * in->input().dz_star_real_[i], ani = 2.0 * in->input().dz_star_imag_[i];
		float sv = in->input().real_[i];
		in->input().real_[i] = sv + eps; double Lp = Lval();
		in->input().real_[i] = sv - eps; double Lm = Lval();
		in->input().real_[i] = sv;
		double nnr = (Lp - Lm) / (2 * eps);
		sv = in->input().imag_[i];
		in->input().imag_[i] = sv + eps; Lp = Lval();
		in->input().imag_[i] = sv - eps; Lm = Lval();
		in->input().imag_[i] = sv;
		double nni = (Lp - Lm) / (2 * eps);
		worst = std::max(worst, std::max(std::fabs(anr - nnr), std::fabs(ani - nni)));
	}
	bool ok = worst < 1e-2;
	std::cout << "  FD RoPE N=" << N << " d=" << d << " heads=" << heads << "  worst=" << worst
			  << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 2. head-slicing oracle against the single-head operator --------------------
static bool headsVsSlices(int N, int d, int heads) {
	const int M = N * d, dh = d / heads;
	// Read Born's output through the L2Out consumer's input buffer (its only input, offset 0).
	auto build = [&](ComplexNet &net, int H, int &lo, int qkv[3]) {   // add() returns uids, not indices
		qkv[0] = net.add(new CInput(OutSize(M)));
		qkv[1] = net.add(new CInput(OutSize(M)));
		qkv[2] = net.add(new CInput(OutSize(M)));
		int ba = net.add(new BornAttention(N, d, true, 0.f, H), {qkv[0], qkv[1], qkv[2]});
		lo = net.add(new L2Out(InSize(M)), {ba});
	};
	ComplexNet full; int baF, idF[3]; build(full, heads, baF, idF); full.init_inputs(5); full.forward();
	double worst = 0;
	for (int h = 0; h < heads; ++h) {
		ComplexNet one; int ba1, id1[3]; build(one, 1, ba1, id1); one.init_inputs(5);   // same init as `full`
		for (int b = 0; b < 3; ++b) {                                        // zero dims outside head h
			CFunc *in = one[id1[b]];
			for (int t = 0; t < N; ++t) for (int dd = 0; dd < d; ++dd)
				if (dd / dh != h) { in->input().real_[t * d + dd] = 0.f; in->input().imag_[t * d + dd] = 0.f; }
		}
		one.forward();
		for (int t = 0; t < N; ++t) for (int j = 0; j < dh; ++j) {
			int p = t * d + h * dh + j;
			worst = std::max(worst, (double) std::fabs(full[baF]->input().real_[p] - one[ba1]->input().real_[p]));
			worst = std::max(worst, (double) std::fabs(full[baF]->input().imag_[p] - one[ba1]->input().imag_[p]));
		}
	}
	bool ok = worst < 1e-5;
	std::cout << "  heads-vs-slices N=" << N << " d=" << d << " heads=" << heads << "  max|diff|=" << worst
			  << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 3. GPU vs CPU: forward output (as L2Out's input) and input gradients -------
static bool gpuBorn(int N, int d, int heads, float eps, bool decay = false) {
	const int M = N * d;
	CNet net;
	net.cpuNet().add(new CInput(Uid(1), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(2), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(3), OutSize(M)));
	net.cpuNet().add(new BornAttention(Uid(4), N, d, true, eps, heads), {1, 2, 3});
	((BornAttention*) net.cpuNet()[4])->setDecay(decay);
	net.cpuNet().add(new L2Out(Uid(5), InSize(M)), {4});
	net.cpuNet().init_inputs(9);
	net.cpuNet().init_exec_graph();
	bool fail = false;
	net.AllocateNet(1, fail);
	if (fail) { std::cout << "  AllocateNet failed\n"; return false; }
	net.cpuNet().forward();
	net.GpuForward(fail);
	auto mp = net.getMappingFor(net.cpuNet()[4]);
	auto mp_out = net.getMappingFor(net.cpuNet()[5]);
	bool okF = net.testDataFromGpu(*mp_out, 1e-3);
	net.cpuNet().backward(0);
	net.GpuBackward(0, fail);
	bool okG = net.testGradientsFromGpu(*mp, 1e-3);
	std::cout << "  GPU Born N=" << N << " d=" << d << " heads=" << heads << " eps=" << eps << " decay=" << decay
			  << "  forward=" << (okF ? "ok" : "BAD") << "  grad=" << (okG ? "ok" : "BAD") << std::endl;
	return okF && okG;
}

static bool gpuRope(int N, int d, int heads) {
	const int M = N * d;
	CNet net;
	net.cpuNet().add(new CInput(Uid(1), OutSize(M)));
	net.cpuNet().add(new RotaryEmbed(Uid(2), N, d, 10000.0, heads), {1});
	net.cpuNet().add(new L2Out(Uid(3), InSize(M)), {2});
	net.cpuNet().init_inputs(13);
	net.cpuNet().init_exec_graph();
	bool fail = false;
	net.AllocateNet(1, fail);
	if (fail) { std::cout << "  AllocateNet failed\n"; return false; }
	net.cpuNet().forward();
	net.GpuForward(fail);
	bool okF = net.testDataFromGpu(*net.getMappingFor(net.cpuNet()[3]), 1e-3);
	net.cpuNet().backward(0);
	net.GpuBackward(0, fail);
	bool okG = net.testGradientsFromGpu(*net.getMappingFor(net.cpuNet()[2]), 1e-3);
	std::cout << "  GPU RoPE N=" << N << " d=" << d << " heads=" << heads
			  << "  forward=" << (okF ? "ok" : "BAD") << "  grad=" << (okG ? "ok" : "BAD") << std::endl;
	return okF && okG;
}

// ---- 3b. linear chunkwise GPU forward vs the CPU quadratic oracle (Proposition 1) ----
static bool gpuBornLinear(int N, int d, int heads, int C, float eps, float scale = 1.f, bool decay = false) {
	const int M = N * d;
	CNet net;
	net.cpuNet().add(new CInput(Uid(1), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(2), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(3), OutSize(M)));
	net.cpuNet().add(new BornAttention(Uid(4), N, d, true, eps, heads), {1, 2, 3});
	((BornAttention*) net.cpuNet()[4])->setLinear(true, C);
	((BornAttention*) net.cpuNet()[4])->setDecay(decay);
	net.cpuNet().add(new L2Out(Uid(5), InSize(M)), {4});
	net.cpuNet().init_inputs(21);
	// The framework's default init is ~1e-3 per entry, which makes |<q,k>|^2 vanish against eps
	// (attention -> 0, gradients -> 0: a vacuous check). Use O(scale) uniform inputs instead.
	uint64_t st = 0x9E3779B97F4A7C15ull + 7 * N + 13 * d + heads;
	auto uni = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float) ((st >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0); };
	for (int id = 1; id <= 3; ++id) for (int i = 0; i < M; ++i) {
		net.cpuNet()[id]->input().real_[i] = scale * uni(); net.cpuNet()[id]->input().imag_[i] = scale * uni();
	}
	net.cpuNet().init_exec_graph();
	bool fail = false;
	net.AllocateNet(1, fail);
	if (fail) { std::cout << "  AllocateNet failed\n"; return false; }
	net.cpuNet().forward();
	net.GpuForward(fail);
	bool okF = net.testDataFromGpu(*net.getMappingFor(net.cpuNet()[5]), 1e-3);
	net.cpuNet().backward(0);
	net.GpuBackward(0, fail);
	bool okG = net.testGradientsFromGpu(*net.getMappingFor(net.cpuNet()[4]), 1e-3);
	double gmax = 0, xmax = 0, omax = 0;   // scales: the 1e-3 absolute check above should read as relative
	for (int id = 1; id <= 3; ++id) for (int i = 0; i < M; ++i) {
		gmax = std::max(gmax, (double) std::abs(net.cpuNet()[id]->input().dz_star(i)));
		xmax = std::max(xmax, (double) std::abs(net.cpuNet()[id]->input().z(i)));
	}
	for (int i = 0; i < M; ++i) omax = std::max(omax, (double) std::abs(net.cpuNet()[5]->input().z(i)));
	std::cout << "    max|x|=" << xmax << " max|O|=" << omax << " loss=" << ((L2Out*) net.cpuNet()[5])->loss(0) << "\n";
	std::cout << "  GPU linear Born N=" << N << " d=" << d << " heads=" << heads << " chunk=" << C
			  << " eps=" << eps << " decay=" << decay << "  forward=" << (okF ? "ok" : "BAD") << "  grad=" << (okG ? "ok" : "BAD")
			  << " (abs tol 1e-3, max|grad|=" << gmax << ")" << std::endl;
	return okF && okG;
}

// ---- 3c. streaming step mode vs the full forward (decay kept in the running state) ----
static bool stepVsFull(int N, int d, int heads, bool decay) {
	const int M = N * d;
	ComplexNet full;
	int q = full.add(new CInput(OutSize(M))), k = full.add(new CInput(OutSize(M))), v = full.add(new CInput(OutSize(M)));
	int ba = full.add(new BornAttention(N, d, true, 0.f, heads), {q, k, v});
	((BornAttention*) full[ba])->setDecay(decay);
	int lo = full.add(new L2Out(InSize(M)), {ba});
	full.init_inputs(23);
	uint64_t st = 0xC0FFEEull + N; auto uni = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float) ((st >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0); };
	for (int id : {q, k, v}) for (int i = 0; i < M; ++i) { full[id]->input().real_[i] = uni(); full[id]->input().imag_[i] = uni(); }
	full.forward();
	const Vars &O = full[lo]->input();
	ComplexNet one;
	int q1 = one.add(new CInput(OutSize(d))), k1 = one.add(new CInput(OutSize(d))), v1 = one.add(new CInput(OutSize(d)));
	int b1 = one.add(new BornAttention(1, d, true, 0.f, heads), {q1, k1, v1});
	((BornAttention*) one[b1])->setDecay(decay); ((BornAttention*) one[b1])->setStepMode(true);
	int l1 = one.add(new L2Out(InSize(d)), {b1});
	one.init_inputs(23);
	double worst = 0;
	for (int t = 0; t < N; ++t) {
		for (int dd = 0; dd < d; ++dd) {
			one[q1]->input().real_[dd] = full[q]->input().real_[t * d + dd]; one[q1]->input().imag_[dd] = full[q]->input().imag_[t * d + dd];
			one[k1]->input().real_[dd] = full[k]->input().real_[t * d + dd]; one[k1]->input().imag_[dd] = full[k]->input().imag_[t * d + dd];
			one[v1]->input().real_[dd] = full[v]->input().real_[t * d + dd]; one[v1]->input().imag_[dd] = full[v]->input().imag_[t * d + dd];
		}
		one.forward();
		const Vars &o1 = one[l1]->input();
		for (int dd = 0; dd < d; ++dd)
			worst = std::max(worst, (double) std::max(std::fabs(o1.real_[dd] - O.real_[t * d + dd]), std::fabs(o1.imag_[dd] - O.imag_[t * d + dd])));
	}
	bool ok = worst < 1e-4;
	std::cout << "  step-vs-full Born N=" << N << " d=" << d << " heads=" << heads << " decay=" << decay << "  max|dO|=" << worst << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 4. serialization of the extended records ---------------------------------
static bool serialize(int N, int d, int heads, float eps, bool decay = false) {
	const int M = N * d;
	ComplexNet net;
	int q = net.add(new CInput(OutSize(M)));
	int k = net.add(new CInput(OutSize(M)));
	int v = net.add(new CInput(OutSize(M)));
	int rq = net.add(new RotaryEmbed(N, d, 10000.0, heads), {q});
	int rk = net.add(new RotaryEmbed(N, d, 10000.0, heads), {k});
	int ba = net.add(new BornAttention(N, d, true, eps, heads), {rq, rk, v});
	((BornAttention*) net[ba])->setDecay(decay);
	int lo = net.add(new L2Out(InSize(M)), {ba});
	net.init_inputs(17); net.forward();
	float L1 = ((L2Out*) net[lo])->loss(0);
	ModelSaver saver;
	std::string blob = saver.SaveToString(net);
	ComplexNet net2;
	bool ok = saver.RestoreFromString(net2, blob);
	auto *b2 = ok ? (BornAttention*) net2.findFirstOfType(isBornAttention) : nullptr;
	auto *r2 = ok ? (RotaryEmbed*) net2.findFirstOfType(isRotary) : nullptr;
	float L2 = -1.f;
	if (ok) { net2.forward(); L2 = ((L2Out*) net2.findFirstOfType(isL2Out))->loss(0); }
	bool pass = ok && b2 && r2 && b2->heads() == heads && r2->heads() == heads && b2->decay() == decay
			&& std::fabs(b2->eps() - eps) < 1e-9f && std::fabs(L1 - L2) < 1e-5f * std::max(1.f, std::fabs(L1));
	std::cout << "  serialize heads=" << heads << " eps=" << eps << "  restore=" << ok
			  << "  heads(born/rope)=" << (b2 ? b2->heads() : -1) << "/" << (r2 ? r2->heads() : -1)
			  << "  L1=" << L1 << " L2=" << L2 << "  " << (pass ? "ok" : "BAD") << std::endl;
	return pass;
}

int main(int argc, char** argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	bool ok = true;
	std::cout << "finite differences\n";
	ok &= fdBorn(5, 3, 1, true);      // the original single-head case
	ok &= fdBorn(5, 4, 2, true);
	ok &= fdBorn(6, 6, 3, true);
	ok &= fdBorn(5, 4, 2, false);
	ok &= fdBorn(6, 6, 3, true, true);  // decay
	ok &= fdBorn(5, 4, 2, false, true);
	ok &= fdRope(5, 4, 2);
	ok &= fdRope(6, 6, 3);
	std::cout << "head slicing oracle\n";
	ok &= headsVsSlices(8, 8, 2);
	ok &= headsVsSlices(9, 12, 4);
	ok &= headsVsSlices(7, 6, 1);
	std::cout << "step mode\n";
	ok &= stepVsFull(12, 8, 2, false); ok &= stepVsFull(16, 12, 3, true);
	std::cout << "serialization\n";
	ok &= serialize(8, 8, 2, 1e-3f, true);
	ok &= serialize(8, 8, 2, 1e-3f);
	ok &= serialize(8, 8, 1, 0.f);
	ok &= serialize(8, 6, 3, 0.f);
	if (gpu) {
		std::cout << "GPU parity\n";
		ok &= gpuBorn(8, 8, 1, 0.f);
		ok &= gpuBorn(8, 8, 2, 0.f);
		ok &= gpuBorn(16, 12, 4, 1e-3f);
		ok &= gpuBorn(16, 12, 4, 1e-3f, true);   // decay
		ok &= gpuRope(8, 8, 2);
		ok &= gpuRope(16, 12, 4);
		std::cout << "GPU linear form vs quadratic oracle\n";
		ok &= gpuBornLinear(8, 8, 2, 8, 0.f, 0.5f);        // one chunk: all intra
		ok &= gpuBornLinear(8, 8, 2, 1, 0.f, 0.5f);        // chunk 1: everything through the state
		ok &= gpuBornLinear(16, 8, 2, 4, 0.f, 0.5f);
		ok &= gpuBornLinear(64, 16, 4, 16, 1e-3f, 0.5f);
		ok &= gpuBornLinear(128, 8, 1, 32, 0.f, 0.5f);
		ok &= gpuBornLinear(256, 24, 3, 64, 1.0f, 0.5f);   // eps comparable to the score sums
		ok &= gpuBornLinear(512, 32, 4, 128, 1e-2f, 0.3f);
		ok &= gpuBornLinear(1024, 32, 4, 256, 0.f, 0.2f);
		std::cout << "GPU linear form with decay vs quadratic oracle\n";
		ok &= gpuBornLinear(8, 8, 2, 8, 0.f, 0.5f, true);
		ok &= gpuBornLinear(8, 8, 2, 1, 0.f, 0.5f, true);
		ok &= gpuBornLinear(16, 8, 2, 4, 0.f, 0.5f, true);
		ok &= gpuBornLinear(64, 16, 4, 16, 1e-3f, 0.5f, true);
		ok &= gpuBornLinear(256, 24, 3, 64, 1.0f, 0.5f, true);
		ok &= gpuBornLinear(512, 32, 4, 128, 1e-2f, 0.3f, true);
	}
	std::cout << (ok ? "PASS" : "FAIL") << std::endl;
	return ok ? 0 : 1;
}
