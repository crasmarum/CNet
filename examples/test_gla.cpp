// Standalone checks for GatedLinearAttention (complex-amplitude gated linear attention):
//   1. finite differences of the Wirtinger backward on [Q;K;V;G] (CPU, 1-3 heads);
//   2. recurrence oracle: the N=1 step mode fed token by token reproduces the quadratic forward;
//   3. no-gate limit: with gate logits -> +inf the operator is plain complex linear attention;
//   4. serialization round trip.
// Build: make test_gla   Run: ./test_gla
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>
#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../impl/cinput.h"
#include "../impl/gla.h"
#include "../impl/l2out.h"
#include "../impl/model_saver.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(gpu, 1)   // 0 = skip the GPU parity checks

static void fill(CFunc *in, int M, unsigned seed, float scale) {       // O(1) inputs (init_inputs is ~1e-3)
	unsigned s = seed * 2654435761u + 12345u;
	auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.f * 2.f - 1.f; };
	for (int i = 0; i < M; ++i) { in->input().real_[i] = scale * rnd(); in->input().imag_[i] = scale * rnd(); }
}

// ---- 1. finite differences: [Q;K;V;G] -> GLA(heads) -> L2Out ------------------------
static bool fdGLA(int N, int d, int heads) {
	const int M = N * d;
	ComplexNet net;
	int qid = net.add(new CInput(OutSize(M)));
	int kid = net.add(new CInput(OutSize(M)));
	int vid = net.add(new CInput(OutSize(M)));
	int gid = net.add(new CInput(OutSize(N * heads)));
	int ga  = net.add(new GatedLinearAttention(N, d, heads), {qid, kid, vid, gid});
	int lo  = net.add(new L2Out(InSize(M)), {ga});
	net.init_inputs(7);
	fill(net[qid], M, 1, 1.f); fill(net[kid], M, 2, 1.f); fill(net[vid], M, 3, 1.f); fill(net[gid], N * heads, 4, 2.f);
	net.forward(); net.backward(0);
	auto Lval = [&]() { net.forward(); return (double) ((L2Out*) net[lo])->loss(0); };
	const double eps = 1e-3; double worst = 0, worstG = 0;
	int ids[4] = {qid, kid, vid, gid}; int lens[4] = {M, M, M, N * heads};
	for (int b = 0; b < 4; ++b) {
		CFunc *in = net[ids[b]];
		for (int i = 0; i < lens[b]; ++i) {
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
			double w = std::max(std::fabs(anr - nnr), std::fabs(ani - nni));
			if (b == 3) worstG = std::max(worstG, w); else worst = std::max(worst, w);
		}
	}
	bool ok = worst < 1e-2 && worstG < 1e-2;
	std::cout << "  FD GLA N=" << N << " d=" << d << " heads=" << heads << "  worst(QKV)=" << worst
			  << "  worst(G)=" << worstG << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 2. recurrence oracle: step mode token by token == quadratic forward ----------------
static bool stepVsFull(int N, int d, int heads) {
	const int M = N * d;
	ComplexNet full;
	int qid = full.add(new CInput(OutSize(M)));
	int kid = full.add(new CInput(OutSize(M)));
	int vid = full.add(new CInput(OutSize(M)));
	int gid = full.add(new CInput(OutSize(N * heads)));
	int ga  = full.add(new GatedLinearAttention(N, d, heads), {qid, kid, vid, gid});
	int lo  = full.add(new L2Out(InSize(M)), {ga});
	full.init_inputs(3);
	fill(full[qid], M, 11, 1.f); fill(full[kid], M, 12, 1.f); fill(full[vid], M, 13, 1.f); fill(full[gid], N * heads, 14, 2.f);
	full.forward();
	const Vars &O = full[lo]->input();                       // GLA output lands in L2Out's input
	ComplexNet one;
	int q1 = one.add(new CInput(OutSize(d))), k1 = one.add(new CInput(OutSize(d)));
	int v1 = one.add(new CInput(OutSize(d))), g1 = one.add(new CInput(OutSize(heads)));
	int g  = one.add(new GatedLinearAttention(1, d, heads), {q1, k1, v1, g1});
	int l1 = one.add(new L2Out(InSize(d)), {g});
	one.init_inputs(3);
	((GatedLinearAttention*) one[g])->setStepMode(true);
	double worst = 0;
	for (int t = 0; t < N; ++t) {
		for (int dd = 0; dd < d; ++dd) {
			one[q1]->input().real_[dd] = full[qid]->input().real_[t * d + dd]; one[q1]->input().imag_[dd] = full[qid]->input().imag_[t * d + dd];
			one[k1]->input().real_[dd] = full[kid]->input().real_[t * d + dd]; one[k1]->input().imag_[dd] = full[kid]->input().imag_[t * d + dd];
			one[v1]->input().real_[dd] = full[vid]->input().real_[t * d + dd]; one[v1]->input().imag_[dd] = full[vid]->input().imag_[t * d + dd];
		}
		for (int h = 0; h < heads; ++h) {
			one[g1]->input().real_[h] = full[gid]->input().real_[t * heads + h]; one[g1]->input().imag_[h] = full[gid]->input().imag_[t * heads + h];
		}
		one.forward();
		const Vars &o1 = one[l1]->input();
		for (int dd = 0; dd < d; ++dd)
			worst = std::max(worst, (double) std::max(std::fabs(o1.real_[dd] - O.real_[t * d + dd]), std::fabs(o1.imag_[dd] - O.imag_[t * d + dd])));
	}
	bool ok = worst < 1e-4;
	std::cout << "  step-vs-full N=" << N << " d=" << d << " heads=" << heads << "  max|dO|=" << worst << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 3. no-gate limit: logits -> +inf gives plain complex linear attention sum_{m<=k} s <q,k> v ----
static bool noGateLimit(int N, int d) {
	const int M = N * d;
	ComplexNet net;
	int qid = net.add(new CInput(OutSize(M))), kid = net.add(new CInput(OutSize(M)));
	int vid = net.add(new CInput(OutSize(M))), gid = net.add(new CInput(OutSize(N)));
	int ga  = net.add(new GatedLinearAttention(N, d, 1), {qid, kid, vid, gid});
	int lo  = net.add(new L2Out(InSize(M)), {ga});
	net.init_inputs(5);
	fill(net[qid], M, 21, 1.f); fill(net[kid], M, 22, 1.f); fill(net[vid], M, 23, 1.f);
	for (int t = 0; t < N; ++t) { net[gid]->input().real_[t] = 40.f; net[gid]->input().imag_[t] = 0.f; }
	net.forward();
	const Vars &O = net[lo]->input(); const float s = 1.f / std::sqrt((float) d);
	double worst = 0;
	for (int k = 0; k < N; ++k) for (int j = 0; j < d; ++j) {
		std::complex<float> o = 0.f;
		for (int m = 0; m <= k; ++m) {
			std::complex<float> c = 0.f;
			for (int jj = 0; jj < d; ++jj) c += net[qid]->input().z(k * d + jj) * std::conj(net[kid]->input().z(m * d + jj));
			o += s * c * net[vid]->input().z(m * d + j);
		}
		worst = std::max(worst, (double) std::abs(o - O.z(k * d + j)));
	}
	bool ok = worst < 1e-4;
	std::cout << "  no-gate limit N=" << N << " d=" << d << "  max|dO|=" << worst << "  " << (ok ? "ok" : "BAD") << std::endl;
	return ok;
}

// ---- 5. GPU chunkwise forward/backward vs the CPU quadratic oracle ----------------------
static bool gpuGLA(int N, int d, int heads, int C, int clones = 1) {
	const int M = N * d;
	CNet net;
	net.cpuNet().add(new CInput(Uid(1), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(2), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(3), OutSize(M)));
	net.cpuNet().add(new CInput(Uid(4), OutSize(N * heads)));
	net.cpuNet().add(new GatedLinearAttention(Uid(5), N, d, heads), {1, 2, 3, 4});
	((GatedLinearAttention*) net.cpuNet()[5])->setChunk(C);
	net.cpuNet().add(new L2Out(Uid(6), InSize(M)), {5});
	net.cpuNet().init_inputs(21);
	fill(net.cpuNet()[1], M, 31, 1.f); fill(net.cpuNet()[2], M, 32, 1.f); fill(net.cpuNet()[3], M, 33, 1.f); fill(net.cpuNet()[4], N * heads, 34, 2.f);
	net.cpuNet().init_exec_graph();
	bool fail = false;
	net.AllocateNet(clones, fail);
	if (fail) { std::cout << "  AllocateNet failed\n"; return false; }
	net.cpuNet().forward();
	net.GpuForward(fail);
	bool okF = net.testDataFromGpu(*net.getMappingFor(net.cpuNet()[6]), 1e-3);
	net.cpuNet().backward(0);
	net.GpuBackward(0, fail);
	bool okG = net.testGradientsFromGpu(*net.getMappingFor(net.cpuNet()[5]), 1e-3);
	double gmax = 0, omax = 0;
	for (int id = 1; id <= 4; ++id) for (int i = 0; i < (id == 4 ? N * heads : M); ++i)
		gmax = std::max(gmax, (double) std::abs(net.cpuNet()[id]->input().dz_star(i)));
	for (int i = 0; i < M; ++i) omax = std::max(omax, (double) std::abs(net.cpuNet()[6]->input().z(i)));
	std::cout << "  GPU GLA N=" << N << " d=" << d << " heads=" << heads << " chunk=" << C << " clones=" << clones
			  << "  forward=" << (okF ? "ok" : "BAD") << "  grad=" << (okG ? "ok" : "BAD")
			  << " (abs tol 1e-3; max|O|=" << omax << " max|grad|=" << gmax << ")" << std::endl;
	return okF && okG;
}

// ---- 4. serialization round trip -----------------------------------------------------
static bool roundTrip() {
	const int N = 8, d = 12, heads = 3, M = N * d;
	ComplexNet net;
	int qid = net.add(new CInput(OutSize(M))), kid = net.add(new CInput(OutSize(M)));
	int vid = net.add(new CInput(OutSize(M))), gid = net.add(new CInput(OutSize(N * heads)));
	int ga  = net.add(new GatedLinearAttention(N, d, heads), {qid, kid, vid, gid});
	net.add(new L2Out(InSize(M)), {ga});
	net.init_inputs(9);
	const char *path = "/tmp/test_gla_roundtrip.mod";
	ModelSaver saver;
	if (!saver.Save(net, path)) { std::cout << "  save failed" << std::endl; return false; }
	ComplexNet net2;
	if (!saver.Restore(net2, path)) { std::cout << "  restore failed" << std::endl; return false; }
	auto *g2 = (GatedLinearAttention*) net2.findFirstOfType(isGLA);
	bool ok = g2 && g2->nTokens() == N && g2->dim() == d && g2->heads() == heads && std::fabs(g2->scale() - 1.f / std::sqrt(4.f)) < 1e-6;
	std::cout << "  serialization round trip  " << (ok ? "ok" : "BAD") << std::endl;
	std::remove(path);
	return ok;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	bool ok = true;
	std::cout << "finite differences:" << std::endl;
	ok &= fdGLA(6, 4, 1); ok &= fdGLA(6, 6, 2); ok &= fdGLA(8, 9, 3);
	std::cout << "recurrence oracle:" << std::endl;
	ok &= stepVsFull(16, 8, 1); ok &= stepVsFull(12, 12, 3);
	std::cout << "limits:" << std::endl;
	ok &= noGateLimit(10, 6);
	std::cout << "serialization:" << std::endl;
	ok &= roundTrip();
	if (gpu) {
		std::cout << "GPU chunkwise vs CPU oracle:" << std::endl;
		ok &= gpuGLA(16, 8, 1, 16); ok &= gpuGLA(16, 8, 1, 4); ok &= gpuGLA(32, 12, 3, 8);
		ok &= gpuGLA(64, 32, 2, 16); ok &= gpuGLA(128, 64, 4, 32, 2); ok &= gpuGLA(256, 128, 4, 64);
		ok &= gpuGLA(64, 80, 1, 64); ok &= gpuGLA(128, 160, 2, 64);   // head dims 80: the published tiny-shakespeare geometry (E=160)
	}
	std::cout << (ok ? "ALL PASS" : "FAILURES") << std::endl;
	return ok ? 0 : 1;
}
