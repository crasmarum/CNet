// CPU checks for ChirpFilter and BoxcarMax (impl/chirp.h). No GPU / NCCL needed.
//   1. forward: DM = 0 is the identity; the filter removes a dispersion applied with the opposite phase;
//   2. finite differences vs the analytic Wirtinger backward, for the signal x and the DM parameters d:
//      x, fc, d -> ChirpFilter -> CrossEntropy, and x, fc, d -> ChirpFilter -> |.|^2 -> BoxcarMax -> CrossEntropy;
//   3. save / restore round trip.
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "../impl/cnet.h"
#include "../impl/cinput.h"
#include "../impl/crossent.h"
#include "../impl/relu.h"
#include "../impl/chirp.h"
#include "../impl/model_saver.h"
#include "../impl/log.h"

static const int L = 64, K = 3;
static const double FC = 600.0;                 // MHz

struct Net {
	ComplexNet net;
	int ix, ifc, id, ich, iout;
	CrossEntropy *cen = nullptr;
	CInput *x, *fc, *d;
	Net(bool with_pool) {
		ix = net.add(new CInput(Uid(1), OutSize(L)));
		ifc = net.add(new CInput(Uid(2), OutSize(1)));
		id = net.add(new CInput(Uid(3), OutSize(K)));
		ich = net.add(new ChirpFilter(Uid(4), L, K), {ix, ifc, id});
		if (with_pool) {
			int m2 = net.add(new CModulus2(Uid(5), InSize(K * L)), {ich});
			int bm = net.add(new BoxcarMax(Uid(6), K, L, {4, 8, 16}), {m2});
			iout = net.add(new CrossEntropy(Uid(7), InSize(K * 3)), {bm});
		} else {
			iout = net.add(new CrossEntropy(Uid(5), InSize(K * L)), {ich});
		}
		net.init_inputs(3);
		net.init_exec_graph();
		x = (CInput*) net[ix]; fc = (CInput*) net[ifc]; d = (CInput*) net[id];
		cen = (CrossEntropy*) net[iout];
		std::mt19937 rng(7);
		std::normal_distribution<float> N01(0.f, 1.f);
		for (int t = 0; t < L; ++t) { x->mutable_input()->real_[t] = N01(rng); x->mutable_input()->imag_[t] = N01(rng); }
		fc->mutable_input()->real_[0] = (float) FC; fc->mutable_input()->imag_[0] = 0;
		for (int k = 0; k < K; ++k) { d->mutable_input()->real_[k] = 4.0f + 3.0f * k; d->mutable_input()->imag_[k] = 0; }
	}
	float loss(int label) { net.forward(); return cen->loss(label); }
};

// max |finite difference - analytic| over all real/imag parts of one input, relative to the largest gradient
static float fd_check(Net &n, CInput *in, int len, int label, bool real_only, const char *name, float eps) {
	for (auto f : {n.x, n.fc, n.d}) f->mutable_input()->zero_gradients();
	n.net.forward();
	n.net.backward(label);
	std::vector<float> ar(len), ai(len);
	for (int k = 0; k < len; ++k) { ar[k] = 2.f * in->input().dz_star(k).real(); ai[k] = 2.f * in->input().dz_star(k).imag(); }
	float err = 0, scale = 1e-6f;
	for (int k = 0; k < len; ++k) {
		for (int part = 0; part < (real_only ? 1 : 2); ++part) {
			float *v = part == 0 ? &in->mutable_input()->real_[k] : &in->mutable_input()->imag_[k];
			float s = *v;
			*v = s + eps; float Lp = n.loss(label);
			*v = s - eps; float Lm = n.loss(label);
			*v = s;
			float num = (Lp - Lm) / (2 * eps), ana = part == 0 ? ar[k] : ai[k];
			err = std::max(err, fabsf(num - ana));
			scale = std::max(scale, fabsf(num));
		}
	}
	float rel = err / scale;
	std::cout << "  " << name << ": max |fd - analytic| = " << err << " (relative " << rel << ")";
	if (real_only) std::cout << ", imag-part gradient " << fabsf(ai[0]) << " (must be 0)";
	std::cout << std::endl;
	return real_only ? std::max(rel, fabsf(ai[0])) : rel;
}

int main() {
	FILELog::ReportingLevel() = lWarning;
	float worst = 0;

	// ---- 1. forward properties
	{
		Net n(false);
		for (int k = 0; k < K; ++k) n.d->mutable_input()->real_[k] = 0;
		n.net.forward();
		ChirpFilter *cf = (ChirpFilter*) n.net[n.ich];
		float e0 = 0;
		for (int k = 0; k < K; ++k)
			for (int t = 0; t < L; ++t)
				e0 = std::max(e0, (float) std::abs(cf->output().z(k * L + t) - n.x->input().z(t)));
		std::cout << "DM = 0 is the identity: max error " << e0 << std::endl;
		worst = std::max(worst, e0);

		// disperse x with the opposite phase at DM = 400, then the filter at d = 4 (dm_scale 100) must give x back
		std::vector<std::complex<double>> X(L);
		for (int t = 0; t < L; ++t) X[t] = n.x->input().z(t);
		std::vector<std::complex<float>> orig(L);
		for (int t = 0; t < L; ++t) orig[t] = n.x->input().z(t);
		chirp_fft(X, -1);
		for (int j = 0; j < L; ++j) X[j] *= std::polar(1.0, -ChirpFilter::phase(j, L, FC) * 400.0);
		chirp_fft(X, +1);
		for (int t = 0; t < L; ++t) { n.x->mutable_input()->real_[t] = X[t].real() / L; n.x->mutable_input()->imag_[t] = X[t].imag() / L; }
		n.d->mutable_input()->real_[0] = 4.0f;
		n.net.forward();
		float e1 = 0;
		for (int t = 0; t < L; ++t) e1 = std::max(e1, (float) std::abs(cf->output().z(t) - orig[t]));
		std::cout << "dedisperses a dispersion of DM 400: max error " << e1 << std::endl;
		worst = std::max(worst, e1);
	}

	// ---- 2. gradients
	for (bool pool : {false, true}) {
		std::cout << (pool ? "ChirpFilter -> |.|^2 -> BoxcarMax -> CrossEntropy" : "ChirpFilter -> CrossEntropy") << std::endl;
		Net n(pool);
		int label = pool ? 1 : 5;
		worst = std::max(worst, fd_check(n, n.x, L, label, false, "dL/dx", 1e-2f));
		// one parameter unit is dm_scale = 100 DM; here 1 DM turns the phase by up to ~4.6 rad, so step 1e-4 (0.01 DM)
		worst = std::max(worst, fd_check(n, n.d, K, label, true, "dL/dDM", 1e-4f));
	}

	// ---- 3. save / restore
	{
		Net n(true);
		ModelSaver saver;
		assert(saver.Save(n.net, "/tmp/test_chirp.mod"));
		ComplexNet back;
		assert(saver.Restore(back, "/tmp/test_chirp.mod"));
		ChirpFilter *a = (ChirpFilter*) n.net[n.ich], *b = (ChirpFilter*) back[n.ich];
		BoxcarMax *pa = (BoxcarMax*) n.net[6], *pb = (BoxcarMax*) back[6];
		bool ok = b && pb && a->L() == b->L() && a->K() == b->K() && a->dmScale() == b->dmScale()
				&& pa->widths() == pb->widths() && pa->K() == pb->K() && pa->L() == pb->L();
		std::cout << "save / restore round trip: " << (ok ? "ok" : "MISMATCH") << std::endl;
		if (!ok) worst = 1;
	}

	bool pass = worst < 2e-2;
	std::cout << (pass ? "PASS" : "FAIL") << "  (worst " << worst << ")" << std::endl;
	return pass ? 0 : 1;
}
