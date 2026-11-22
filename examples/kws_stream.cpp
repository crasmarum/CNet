// kws_stream.cpp: constant-state streaming keyword spotting on the CPU with a model trained by kws.cpp.
//
//   kws_stream -model r21.mod -labels labels17.txt -eval_bin kws_test.bin -check_clips 20      (twin vs full graph)
//   kws_stream -model r21.mod -labels labels17.txt -wav long_recording.wav [-words stop,go,...]  (detections)
//   kws_stream -model r21.mod -labels labels17.txt -stdin                                       (16 kHz int16 PCM)
//
// The model's graph is rebuilt as its N = 1 twin (impl/framenet.h): one Hann frame of W samples goes in every hop,
// the causal Born attention keeps its running state, and the trained readout runs on a FrameRing of the last N
// token projections. Every hop therefore gives a Born-rule posterior over the classes with the full causal
// context, without recomputing earlier frames; the state does not grow with time. The decay schedule and the
// framing are not stored in model files: pass the training flags (defaults are those of r19/r21).
//
// Decision rule for the live modes: the posteriors are averaged over -smooth hops; a command fires when the
// averaged arg-max is a command word (not unknown/silence, and in -words when given) with probability >= -min_prob,
// at most once per -cooldown_ms.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/cinput.h"
#include "../impl/crossent.h"
#include "../impl/model_saver.h"
#include "../impl/framenet.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_STRING(model, "")
FLAG_STRING(labels, "")
FLAG_STRING(wav, "")
FLAG_BOOL(stdin_pcm, false)           // read 16 kHz mono int16 PCM from standard input
FLAG_STRING(eval_bin, "")             // KWS1 file: compare the twin with the full graph on -check_clips clips
FLAG_INT(check_clips, 20)
FLAG_INT(stft_window, 512)
FLAG_INT(stft_hop, 160)
FLAG_BOOL(hann, true)
FLAG_FLOAT(input_gain, 50.f)
FLAG_BOOL(born_decay, true)
FLAG_FLOAT(born_decay_base, 3.f)
FLAG_FLOAT(born_decay_step, 1.f)
FLAG_STRING(words, "")                // comma-separated command set (empty: every class except unknown/silence)
FLAG_FLOAT(min_prob, 0.8f)            // operating point measured on continuous audio (data/kws_continuous_eval.py): 0.8 gives
                                      // 80% recall at ~24 false accepts/hour at 20 dB SNR with the 17-class model; 0.65 gives 80% / 134 per hour
FLAG_INT(smooth, 5)                   // hops averaged before the decision (5 hops = 50 ms at hop 160)
FLAG_INT(cooldown_ms, 1000)
FLAG_FLOAT(release, 0.3f)             // hysteresis: after a detection, re-arm only once the best command posterior has
FLAG_INT(rearm_hops, 25)              // stayed below -release for this many hops (a word stays visible while it slides through the window)
FLAG_INT(print_every, 0)              // live modes: also print the arg-max every this many hops (0 = detections only)
FLAG_FLOAT(high_prob, 0.95f)          // live modes: a word whose smoothed posterior is at least this is printed in capitals
FLAG_INT(threads, 1)                  // OpenMP threads: the N = 1 layers are tiny, more threads only add overhead
FLAG_BOOL(infer_dims, false)           // allocate only z and dz per layer (Vars::INFER_DIMS): half the activation memory


// the live printouts show a word in capitals once its smoothed posterior reaches -high_prob
static std::string shown(const std::string &word, float p) {
	if (p < high_prob) return word;
	std::string u = word;
	for (auto &c : u) c = (char) std::toupper((unsigned char) c);
	return u;
}
static const int kClip = 16000, kL = 16384, kRate = 16000;

static bool readWav(const std::string &path, std::vector<int16_t> &out, int *rate) {
	std::ifstream in(path.c_str(), std::ios::binary);
	if (!in) return false;
	char riff[12];
	if (!in.read(riff, 12) || std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(riff + 8, "WAVE", 4) != 0) return false;
	int channels = 1, bits = 16;
	*rate = 0;
	while (in) {
		char id[4]; uint32_t size;
		if (!in.read(id, 4) || !in.read((char*) &size, 4)) break;
		if (std::strncmp(id, "fmt ", 4) == 0) {
			std::vector<char> fmt(size);
			in.read(fmt.data(), size);
			channels = *(uint16_t*) &fmt[2]; *rate = *(uint32_t*) &fmt[4]; bits = *(uint16_t*) &fmt[14];
		} else if (std::strncmp(id, "data", 4) == 0) {
			if (bits != 16) { std::cerr << "kws_stream: need 16-bit PCM\n"; return false; }
			std::vector<int16_t> raw(size / 2);
			in.read((char*) raw.data(), size);
			out.clear();
			for (size_t i = 0; i + channels - 1 < raw.size(); i += channels) out.push_back(raw[i]);
			return true;
		} else {
			in.seekg(size + (size & 1), std::ios::cur);
		}
	}
	return false;
}

// The streaming front end: a window of the last W samples, advanced by hop, Hann-windowed and scaled.
struct Framer {
	int W, hop;
	std::vector<float> win, buf, frame;
	int filled = 0;
	Framer(int W_, int hop_) : W(W_), hop(hop_), win(W_, 1.f), buf(W_, 0.f), frame(W_, 0.f) {
		if (hann) for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);
	}
	// pushes hop samples; returns true when a new frame (the last W samples) is ready in `frame`
	bool push(const int16_t *x, int n) {
		std::copy(buf.begin() + n, buf.end(), buf.begin());
		for (int k = 0; k < n; ++k) buf[W - n + k] = input_gain * x[k] / 32768.f;
		filled = std::min(W, filled + n);
		if (filled < W) return false;
		for (int j = 0; j < W; ++j) frame[j] = buf[j] * win[j];
		return true;
	}
};

static std::vector<std::string> readLabels() {
	std::vector<std::string> words;
	std::ifstream lf(((std::string) labels).c_str());
	std::string w;
	while (std::getline(lf, w)) if (!w.empty()) words.push_back(w);
	return words;
}

static void probabilities(CrossEntropy *ce, std::vector<float> &p) {
	const int C = ce->input().length_;
	p.resize(C);
	for (int c = 0; c < C; ++c) p[c] = ce->getProbability(c);
}

// Twin against the full graph: the clip framed as the trainer frames it (centred in 16384 samples, N frames) goes
// through the full graph once and through the twin frame by frame; the posteriors at the last frame must agree.
static int checkAgainstFull(CNet &full, CInput *inp, CrossEntropy *cen, FrameNet &sn, const std::vector<std::string> &words) {
	const std::string path = (std::string) eval_bin;
	std::ifstream in(path.c_str(), std::ios::binary);
	int32_t h[4];
	if (!in.read((char*) h, sizeof(h)) || h[0] != 0x4B575331) { std::cerr << "kws_stream: bad KWS1 file " << path << "\n"; return 1; }
	int n = h[1];
	const int L = h[2], C = h[3];
	if (L != kClip) { std::cerr << "kws_stream: expected clips of " << kClip << " samples\n"; return 1; }
	if (check_clips > 0 && check_clips < n) n = check_clips;
	const int W = sn.W, N = sn.N;
	std::vector<float> win(W, 1.f);
	if (hann) for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);
	std::vector<float> pf, pt;
	double max_diff = 0, ms_total = 0;
	long frames = 0, ok_full = 0, ok_twin = 0;
	for (int i = 0; i < n; ++i) {
		int32_t lab;
		std::vector<int16_t> a(L);
		if (!in.read((char*) &lab, 4) || !in.read((char*) a.data(), 2 * (size_t) L)) { std::cerr << "kws_stream: truncated\n"; return 1; }
		std::vector<float> x(kL, 0.f);
		const int off = (kL - kClip) / 2;
		for (int k = 0; k < kClip; ++k) x[off + k] = input_gain * a[k] / 32768.f;
		// full graph
		Vars *v = inp->mutable_input();
		for (int f = 0; f < N; ++f)
			for (int j = 0; j < W; ++j) { v->real_[f * W + j] = x[f * stft_hop + j] * win[j]; v->imag_[f * W + j] = 0.f; }
		full.cpuNet().forward();
		probabilities(cen, pf);
		// twin, one frame at a time
		frameNetReset(sn);
		std::vector<float> frame(W);
		const auto t0 = std::chrono::steady_clock::now();
		for (int f = 0; f < N; ++f) {
			for (int j = 0; j < W; ++j) frame[j] = x[f * stft_hop + j] * win[j];
			frameNetStep(sn, frame.data(), f);
		}
		ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		frames += N;
		probabilities(sn.ce, pt);
		double d = 0;
		for (int c = 0; c < C; ++c) d = std::max(d, (double) fabsf(pf[c] - pt[c]));
		max_diff = std::max(max_diff, d);
		const int af = (int) (std::max_element(pf.begin(), pf.end()) - pf.begin());
		const int at = (int) (std::max_element(pt.begin(), pt.end()) - pt.begin());
		ok_full += (af == lab); ok_twin += (at == lab);
		if (i < 5 || d > 1e-3) {
			std::cout << "clip " << i << " label " << (lab < (int) words.size() ? words[lab] : std::to_string(lab)) << ": full "
			          << (af < (int) words.size() ? words[af] : std::to_string(af)) << " " << pf[af] << ", twin "
			          << (at < (int) words.size() ? words[at] : std::to_string(at)) << " " << pt[at] << ", max |dp| " << d << std::endl;
		}
	}
	const bool pass = max_diff < 1e-3;
	std::cout << "CHECK " << n << " clips: max |p_full - p_twin| " << max_diff << " (" << (pass ? "PASS" : "FAIL") << "), accuracy full "
	          << ok_full / (double) n << " twin " << ok_twin / (double) n << ", twin " << ms_total / frames << " ms per frame ("
	          << 1000.0 * stft_hop / kRate << " ms of audio), state " << sn.stateBytes() / 1024.0 << " KiB" << std::endl;
	return pass ? 0 : 1;
}

// Live detection on a sample stream.
struct Detector {
	FrameNet &sn;
	const std::vector<std::string> &words;
	std::vector<bool> command;
	std::vector<float> acc, p;
	std::vector<std::vector<float>> hist;
	size_t hpos = 0;
	long hop_index = 0, last_fire = -1000000, detections = 0, low_hops = 0;
	bool armed = true;
	double ms_total = 0;
	Detector(FrameNet &s, const std::vector<std::string> &w) : sn(s), words(w) {
		const int C = sn.ce->input().length_;
		command.assign(C, true);
		std::vector<std::string> allowed;
		std::string list = (std::string) ::words;
		size_t pos = 0;
		while (pos <= list.size()) {
			size_t c = list.find(',', pos);
			if (c == std::string::npos) c = list.size();
			if (c > pos) allowed.push_back(list.substr(pos, c - pos));
			pos = c + 1;
		}
		for (int c = 0; c < C; ++c) {
			const std::string name = c < (int) words.size() ? words[c] : "";
			if (name == "unknown" || name == "silence") command[c] = false;
			if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), name) == allowed.end()) command[c] = false;
		}
		acc.assign(C, 0.f);
		hist.assign(std::max(1, (int) smooth), std::vector<float>(C, 0.f));
	}
	void hop(const float *frame) {
		const auto t0 = std::chrono::steady_clock::now();
		frameNetStep(sn, frame, (int) std::min<long>(hop_index, 2000000000L));
		ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		probabilities(sn.ce, p);
		const int C = (int) p.size();
		for (int c = 0; c < C; ++c) { acc[c] += p[c] - hist[hpos][c]; hist[hpos][c] = p[c]; }
		hpos = (hpos + 1) % hist.size();
		const float inv = 1.f / hist.size();
		int best = 0;
		for (int c = 1; c < C; ++c) if (acc[c] > acc[best]) best = c;
		const float pb = acc[best] * inv;
		const double t = hop_index * (double) stft_hop / kRate;
		const long cooldown = (long) ((double) cooldown_ms * kRate / 1000.0 / stft_hop);
		float pcmd = 0.f;                                             // best command posterior (for the release test)
		for (int c = 0; c < C; ++c) if (command[c]) pcmd = std::max(pcmd, acc[c] * inv);
		if (!armed) {
			if (pcmd < release) { if (++low_hops >= rearm_hops) { armed = true; low_hops = 0; } }
			else low_hops = 0;
		}
		if (armed && command[best] && pb >= min_prob && hop_index - last_fire >= cooldown) {
			std::printf("t=%7.2f s  %-10s %.2f\n", t, shown(best < (int) words.size() ? words[best] : "?", pb).c_str(), pb);
			std::fflush(stdout);
			last_fire = hop_index; detections++;
			armed = false; low_hops = 0;
		} else if (print_every > 0 && hop_index % print_every == 0) {
			std::printf("t=%7.2f s  (%s %.2f)\n", t, shown(best < (int) words.size() ? words[best] : "?", pb).c_str(), pb);
		}
		hop_index++;
	}
};

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	if (infer_dims) Vars::dims_ = Vars::INFER_DIMS;   // before any layer is built: z and dz only (the forward's scratch)
#ifdef _OPENMP
	omp_set_num_threads(std::max(1, std::min<int>(threads, omp_get_max_threads())));
#endif
	BornAttention::setDecaySchedule(born_decay_base, born_decay_step);
	const bool live = !((std::string) wav).empty() || stdin_pcm;
	if (((std::string) model).empty() || (!live && ((std::string) eval_bin).empty())) {
		std::cerr << "usage: kws_stream -model m.mod -labels l.txt (-eval_bin kws_test.bin | -wav file.wav | -stdin_pcm true)\n";
		return 1;
	}
	const std::vector<std::string> words = readLabels();

	CNet full;
	if (!full.restore((std::string) model)) { std::cerr << "kws_stream: cannot restore " << (std::string) model << "\n"; return 1; }
	for (auto *f : full.cpuNet().funcs())
		if (auto *b = dynamic_cast<BornAttention*>(f)) {
			b->setDecay(born_decay); b->setLinear(false);
			if (!b->causal()) { std::cerr << "kws_stream: " << b->getName() << " is not causal; streaming needs a causal model\n"; return 1; }
		}
	full.cpuNet().init_exec_graph();
	CInput *inp = dynamic_cast<CInput*>(full.findFirstInput());
	CrossEntropy *cen = dynamic_cast<CrossEntropy*>(full.findFirstOutput());
	if (!inp || !cen) { std::cerr << "kws_stream: unexpected graph\n"; return 1; }

	FrameNet sn;
	if (!buildFrameNet(full.cpuNet(), inp->uid(), cen->uid(), sn)) { std::cerr << "kws_stream: " << sn.error << "\n"; return 1; }
	if (sn.W != stft_window || inp->input().length_ != sn.N * sn.W || (kL - sn.W) / stft_hop + 1 != sn.N) {
		std::cerr << "kws_stream: the model has N=" << sn.N << " frames of W=" << sn.W << "; the flags give W=" << stft_window
		          << " and " << (kL - stft_window) / stft_hop + 1 << " frames at hop " << stft_hop << "\n";
		return 1;
	}
	std::cout << "kws_stream: twin of " << sn.layers << " layers, W=" << sn.W << " hop=" << stft_hop << " N=" << sn.N
	          << ", Born layers " << sn.borns.size() << ", rings " << sn.rings.size() << ", state " << sn.stateBytes() / 1024.0
	          << " KiB, classes " << cen->input().length_ << std::endl;

	if (!live) return checkAgainstFull(full, inp, cen, sn, words);

	Detector det(sn, words);
	Framer fr(sn.W, stft_hop);
	frameNetReset(sn);
	std::vector<int16_t> block(stft_hop);
	long samples = 0;
	if (stdin_pcm) {
		while (std::cin.read((char*) block.data(), 2 * (size_t) stft_hop)) {
			samples += stft_hop;
			if (fr.push(block.data(), stft_hop)) det.hop(fr.frame.data());
		}
	} else {
		std::vector<int16_t> a;
		int rate = 0;
		if (!readWav((std::string) wav, a, &rate)) { std::cerr << "kws_stream: cannot read " << (std::string) wav << "\n"; return 1; }
		if (rate != kRate) std::cerr << "kws_stream: warning: " << rate << " Hz, expected 16000\n";
		for (size_t s = 0; s + stft_hop <= a.size(); s += stft_hop) {
			samples += stft_hop;
			if (fr.push(&a[s], stft_hop)) det.hop(fr.frame.data());
		}
	}
	std::cerr << "kws_stream: " << samples / (double) kRate << " s of audio, " << det.hop_index << " hops, " << det.detections
	          << " detections, " << det.ms_total / std::max(1L, det.hop_index) << " ms per hop (" << 1000.0 * stft_hop / kRate
	          << " ms of audio), state " << sn.stateBytes() / 1024.0 << " KiB" << std::endl;
	return 0;
}
