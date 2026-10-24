// Detecting dispersed radio bursts in raw CHIME/FRB voltages with a tiny complex network:
//   [x | f_c] , d -> ChirpFilter (K learnable dedispersion filters) -> |.|^2 -> BoxcarMax (3 widths)
//                 -> Linear (K*3 -> 2) -> CrossEntropy (burst / noise, Born rule)
// Training: real CHIME noise windows with synthetic dispersed bursts injected on the fly (CHIME sign convention);
// test: Pd at a fixed false-alarm rate on windows from held-out events. Mirrors the PyTorch prototype in
// SatComplex/rf_product/chime/sweep.py ("chirp:K").
//
//   python rf_product/chime/export_bin.py PREP train.bin test.bin --test FRB20190103C,FRB20181213A
//   ./chime_detect -train_bin train.bin -test_bin test.bin -filters 16 -steps 3000
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <algorithm>

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/batch.h"
#include "../impl/cinput.h"
#include "../impl/chirp.h"
#include "../impl/model_saver.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_STRING(train_bin, "train.bin")
FLAG_STRING(test_bin, "test.bin")   // comma-separated: with -cal_bin, each file is scored on its own and pooled
FLAG_STRING(cal_bin, "")            // noise windows that set the threshold; empty: the scored test windows themselves
FLAG_STRING(save_path, "")
FLAG_INT(filters, 16)         // dedispersion filters K
FLAG_INT(steps, 3000)
FLAG_INT(batch_size, 64)
FLAG_FLOAT(lr, 3e-3)
FLAG_FLOAT(dm_lo, 100)        // injected and initial DM range
FLAG_FLOAT(dm_hi, 1000)
FLAG_INT(eval_n, 1500)        // test windows used for each energy
FLAG_FLOAT(pfa, 0.01)
FLAG_INT(seed, 0)

struct Windows {
	int N = 0, L = 0;
	std::vector<float> fc;                                  // MHz
	std::vector<std::vector<std::complex<float>>> x;
	bool read(const std::string &path) {
		std::ifstream f(path, std::ios::binary);
		if (!f) return false;
		int32_t hdr[2]; f.read((char*) hdr, sizeof(hdr));
		N = hdr[0]; L = hdr[1];
		fc.resize(N); x.assign(N, std::vector<std::complex<float>>(L));
		std::vector<float> rec(1 + 2 * L);
		for (int i = 0; i < N; ++i) {
			f.read((char*) rec.data(), sizeof(float) * rec.size());
			fc[i] = rec[0];
			for (int t = 0; t < L; ++t) x[i][t] = {rec[1 + 2 * t], rec[2 + 2 * t]};
		}
		return (bool) f;
	}
};

// A noise-like burst under a Gaussian envelope, dispersed with the CHIME sign (exp(-i phi DM)), total energy E,
// placed so the dispersed burst stays in the window. Added to y in place.
static void inject(std::vector<std::complex<float>> &y, double fc_mhz, double E, std::mt19937 &rng) {
	int L = (int) y.size();
	std::uniform_real_distribution<double> U(0, 1);
	std::normal_distribution<double> N01(0, 1);
	double dm = dm_lo + (dm_hi - dm_lo) * U(rng);
	double w = (0.1e-3 + 1.9e-3 * U(rng)) / ChirpFilter::DT;
	double fc = fc_mhz * 1e6, bw = 390625.0;
	double smear = ChirpFilter::KDM * dm * (1 / ((fc - bw / 2) * (fc - bw / 2)) - 1 / ((fc + bw / 2) * (fc + bw / 2))) / ChirpFilter::DT;
	double margin = std::min(smear / 2 + 3 * w + 1, L / 2.0 - 1);
	double c = margin + (L - 2 * margin) * U(rng);
	std::vector<std::complex<double>> s(L);
	for (int t = 0; t < L; ++t) s[t] = exp(-0.5 * pow((t - c) / w, 2)) * std::complex<double>(N01(rng), N01(rng));
	chirp_fft(s, -1);
	for (int j = 0; j < L; ++j) s[j] *= std::polar(1.0, -ChirpFilter::phase(j, L, fc_mhz) * dm);
	chirp_fft(s, +1);
	double e = 0;
	for (auto &v : s) e += std::norm(v);
	double sc = sqrt(E / e);
	for (int t = 0; t < L; ++t) y[t] += std::complex<float>((float) (s[t].real() * sc), (float) (s[t].imag() * sc));
}

// [x | fc] as one main-input sample
static std::vector<std::complex<float>> sample(const Windows &W, int i, double E, std::mt19937 &rng) {
	std::vector<std::complex<float>> v(W.x[i]);
	if (E > 0) inject(v, W.fc[i], E, rng);
	v.push_back({W.fc[i], 0.f});
	return v;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	Windows tr, cal;
	std::vector<Windows> tes;
	std::vector<std::string> test_names;
	for (size_t a = 0, b; a <= test_bin.size(); a = b + 1) {
		b = test_bin.find(',', a); if (b == std::string::npos) b = test_bin.size();
		test_names.push_back(test_bin.substr(a, b - a)); tes.emplace_back();
		if (!tes.back().read(test_names.back())) { std::cerr << "cannot read " << test_names.back() << "\n"; return 1; }
	}
	if (!tr.read(train_bin) || (!cal_bin.empty() && !cal.read(cal_bin))) { std::cerr << "cannot read data\n"; return 1; }
	const Windows &te = tes[0];
	const int L = tr.L, K = filters, B = batch_size;
	const std::vector<int> widths = {32, 128, 512};
	const int Wn = (int) widths.size();
	std::cout << "train " << tr.N << " windows, test " << te.N << ", L " << L << ", K " << K << std::endl;

	CNet cnet;
	ComplexNet &net = cnet.cpuNet();
	int inp = net.add(new CInput(Uid(1), OutSize(L + 1)));                        // [x | fc]
	int dms = net.add(new CInput(Uid(2), OutSize(K)));                            // DM_k / 100
	int chf = net.add(new ChirpFilter(Uid(3), L, K), {inp, dms});
	int m2  = net.add(new CModulus2(Uid(4), InSize(K * L)), {chf});
	int box = net.add(new BoxcarMax(Uid(5), K, L, widths), {m2});
	int wts = net.add(new CInput(Uid(6), OutSize(K * Wn * 2)));
	int lin = net.add(new Linear(Uid(7), InSize(K * Wn), InSize(K * Wn * 2)), {box, wts});
	int out = net.add(new CrossEntropy(Uid(8), InSize(2)), {lin});
	net.gaussInit(0.02f, seed);
	CInput *d = (CInput*) net[dms];
	for (int k = 0; k < K; ++k) {                          // DM grid over the injected range
		d->mutable_input()->real_[k] = (dm_lo + (dm_hi - dm_lo) * (K == 1 ? 0.5 : k / (K - 1.0))) / 100.f;
		d->mutable_input()->imag_[k] = 0;
	}
	CInput *in = (CInput*) net[inp];
	CrossEntropy *cen = (CrossEntropy*) net[out];
	int learned = K + K * Wn * 2 * 2;                      // DMs (real) + complex linear weights (2 reals each)
	std::cout << "learned real parameters: " << learned << " (" << K << " DMs, " << K * Wn * 2 << " complex weights)" << std::endl;
	assert(cnet.allocateOnGpu(B));

	std::mt19937 rng(seed);
	float avg = 0;
	for (int step = 0; step < steps; ++step) {
		InputBatch batch(B, L + 1);
		std::vector<std::vector<std::complex<float>>> xs(B);
		std::vector<int> lab(B);
		std::vector<uint32_t> seeds(B);
		for (int b = 0; b < B; ++b) seeds[b] = rng();
		#pragma omp parallel for
		for (int b = 0; b < B; ++b) {
			std::mt19937 r(seeds[b]);
			std::uniform_real_distribution<double> u(0, 1);
			lab[b] = u(r) < 0.5;
			double E = lab[b] ? exp(log(50.0) + (log(3000.0) - log(50.0)) * u(r)) : 0;
			int i = std::uniform_int_distribution<int>(0, tr.N - 1)(r);
			xs[b] = sample(tr, i, E, r);
		}
		for (int b = 0; b < B; ++b) batch.add(xs[b], lab[b]);
		cnet.gpuForward(in, cen, batch);
		avg += cnet.getLoss(cen, &batch);
		cnet.gpuBackward();
		float lr_t = lr * 0.5f * (1 + cosf((float) M_PI * step / steps));
		cnet.trueAdamUpdate(lr_t, 0.9f, 0.999f, 1e-8f, step, 0.f);   // Adam step = lr (not lr / batch; see CNET_NOTES.md)
		if (step % 500 == 499) { std::cout << "step " << step + 1 << " loss " << avg / 500 << std::endl; avg = 0; }
	}

	// ---- evaluation: Pd at fixed Pfa, same windows for every energy
	auto scores = [&](const Windows &W, int n, double E) {
		std::vector<float> s;
		std::mt19937 r(12345);
		for (int b0 = 0; b0 < n; b0 += B) {
			InputBatch batch(B, L + 1);
			for (int b = 0; b < B; ++b) batch.add(sample(W, b0 + b, E, r), E > 0);
			cnet.gpuForward(in, cen, batch);
			auto p = cnet.getClassProb(cen, 1);
			s.insert(s.end(), p.begin(), p.end());
		}
		return s;
	};
	auto quantile_thr = [&](std::vector<float> v) {
		std::sort(v.begin(), v.end());
		return v[std::min((int) v.size() - 1, (int) ((1 - pfa) * v.size()))];
	};
	const std::vector<double> energies = {50.0, 100.0, 200.0, 400.0, 800.0};
	if (cal_bin.empty()) {                                  // in-sample threshold on the first eval_n test windows
		int n = std::min(eval_n, te.N);
		n -= n % B;
		float thr = quantile_thr(scores(te, n, 0));
		std::cout << "Pd at Pfa " << pfa << " over " << n << " held-out windows:";
		for (double E : energies) {
			auto s = scores(te, n, E);
			int hit = 0;
			for (float v : s) hit += v > thr;
			std::cout << "  E=" << E << ": " << hit / (double) s.size();
		}
		std::cout << std::endl;
	} else {                                                // threshold from separate calibration noise
		float thr = quantile_thr(scores(cal, cal.N - cal.N % B, 0));
		std::cout << "threshold from " << cal.N - cal.N % B << " calibration windows (nominal Pfa " << pfa << ")" << std::endl;
		std::vector<int> hits(energies.size() + 1, 0); int tot = 0;
		for (size_t f = 0; f < tes.size(); ++f) {
			int n = tes[f].N - tes[f].N % B;
			std::cout << test_names[f] << " (" << n << " windows):";
			for (size_t e = 0; e <= energies.size(); ++e) {
				double E = e == 0 ? 0 : energies[e - 1];
				int hit = 0;
				for (float v : scores(tes[f], n, E)) hit += v > thr;
				hits[e] += hit;
				std::cout << (e == 0 ? "  Pfa " : "  E=" + std::to_string((int) E) + ": ") << hit / (double) n;
			}
			tot += n;
			std::cout << std::endl;
		}
		std::cout << "pooled (" << tot << "):";
		for (size_t e = 0; e <= energies.size(); ++e)
			std::cout << (e == 0 ? "  Pfa " : "  E=" + std::to_string((int) energies[e - 1]) + ": ") << hits[e] / (double) tot;
		std::cout << std::endl;
	}
	if (cnet.getInputsFromGpu()) {
		std::cout << "learned DMs:";
		for (int k = 0; k < K; ++k) std::cout << " " << (int) std::lround(100 * d->input().real_[k]);
		std::cout << std::endl;
		if (!save_path.empty()) { ModelSaver saver; saver.Save(net, save_path); }
	}
	return 0;
}
