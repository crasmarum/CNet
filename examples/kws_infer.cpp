// kws_infer.cpp: CPU inference for a keyword-spotting model trained by kws.cpp (no GPU needed).
//
//   kws_infer -model r19.mod -labels labels.txt -wav clip.wav [-stft_window 512 -stft_hop 160 -input_gain 50 -top 5]
//   kws_infer -model r19.mod -eval_bin kws_test.bin [-eval_shifts_ms "-100,0,100"] [-eval_max N]
//
// A WAV clip must be 16 kHz mono 16-bit PCM; it is cut or zero-padded to one second, centred in the 16384-sample
// buffer and framed exactly as the trainer's batch builder does (Hann frames of W samples every hop samples, gain
// applied to the waveform), then run through the CPU graph restored from the model file. The Born-attention decay
// schedule is not stored in model files and is re-applied from the flags, which must match the training run.
// Prints the top words with their Born-rule probabilities |z_c|^2 / sum |z|^2.
//
// -eval_bin evaluates the accuracy on a KWS1 file (the trainer's kws_val.bin / kws_test.bin), with every clip
// shifted in time by each entry of -eval_shifts_ms (positive = later, zero fill; the same operation as the
// trainer's shift augmentation and its -restore mode), one clip at a time on the CPU.
#include <cstdio>
#include <utility>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/cinput.h"
#include "../impl/crossent.h"
#include "../impl/model_saver.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_STRING(model, "")
FLAG_STRING(labels, "")
FLAG_STRING(wav, "")
FLAG_STRING(eval_bin, "")             // KWS1 file: report the accuracy instead of classifying one WAV
FLAG_STRING(eval_shifts_ms, "0")      // comma-separated shifts (ms) applied to every clip of -eval_bin
FLAG_INT(eval_max, 0)                 // evaluate only the first N clips (0 = all)
FLAG_BOOL(confusion, false)           // -eval_bin: per-class recall, main confusions and the accept/reject summary
FLAG_FLOAT(min_prob, 0.f)             // -eval_bin: a prediction whose Born probability is below this is a rejection
                                      // (kws_listen.py's --min-prob operating point); 0 = always accept the arg-max
FLAG_INT(stft_window, 512)
FLAG_INT(stft_hop, 160)
FLAG_BOOL(hann, true)
FLAG_FLOAT(input_gain, 50.f)
FLAG_BOOL(born_decay, true)
FLAG_FLOAT(born_decay_base, 3.f)
FLAG_FLOAT(born_decay_step, 1.f)
FLAG_BOOL(causal, true)               // -causal false for non-causal models saved before the saver stored the flag (21 Nov 2026)
FLAG_INT(top, 5)

static const int kClip = 16000, kL = 16384;

// Minimal RIFF/WAVE reader: 16-bit PCM, first channel, any rate (reported; 16 kHz expected).
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
			if (bits != 16) { std::cerr << "kws_infer: need 16-bit PCM\n"; return false; }
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

// Framing as kws.cpp clipFrames without augmentation: centre the one-second clip (shifted by `shift` samples,
// positive = later, zero fill), apply the gain, cut Hann frames when stft_hop > 0.
static void frameClip(const int16_t *a, int shift, const std::vector<float> &win, int Nf, std::vector<float> &out) {
	const int W = stft_window, off = (kL - kClip) / 2;
	std::vector<float> x(kL, 0.f);
	for (int k = 0; k < kClip; ++k) {
		const int src = k - shift;
		if (src >= 0 && src < kClip) x[off + k] = input_gain * a[src] / 32768.f;
	}
	if (stft_hop > 0) {
		out.assign((size_t) Nf * W, 0.f);
		for (int f = 0; f < Nf; ++f)
			for (int j = 0; j < W; ++j) out[(size_t) f * W + j] = x[f * stft_hop + j] * win[j];
	} else {
		out = x;
	}
}

// Arg-max class and (optionally) its Born probability |z_c|^2 / sum |z|^2.
static int predict(CNet &cnet, CInput *inp, CrossEntropy *cen, const std::vector<float> &y, float *prob = nullptr) {
	for (size_t k = 0; k < y.size(); ++k) { inp->mutable_input()->real_[k] = y[k]; inp->mutable_input()->imag_[k] = 0.f; }
	cnet.cpuNet().forward();
	const int C = cen->input().length_;
	int best = 0;
	float bm = -1.f;
	double S = 0;
	for (int c = 0; c < C; ++c) {
		const float m = std::norm(cen->input().z(c));
		S += m;
		if (m > bm) { bm = m; best = c; }
	}
	if (prob) *prob = S > 0 ? (float) (bm / S) : 0.f;
	return best;
}

// Per-class recall with the main confusions, and the accept/reject summary that matters for a command listener:
// how often a non-command clip ("unknown", "silence") is accepted as a command, and how often a command is rejected.
// conf is C x (C + 1): the last column counts rejections (top probability below -min_prob).
static void printConfusion(const std::vector<long> &conf, int C, const std::vector<std::string> &words, const std::vector<int> &labels, int n) {
	auto name = [&](int c) { return c < (int) words.size() ? words[c] : "class" + std::to_string(c); };
	int unk = -1, sil = -1;
	for (int c = 0; c < (int) words.size() && c < C; ++c) { if (words[c] == "unknown") unk = c; if (words[c] == "silence") sil = c; }
	auto isCommand = [&](int c) { return c < C && c != unk && c != sil; };
	std::cout << "class            n  recall  rejected  main confusions" << std::endl;
	long cmd_n = 0, cmd_ok = 0, cmd_rej = 0, cmd_wrong = 0, non_n = 0, non_acc = 0;
	for (int t = 0; t < C; ++t) {
		long nt = 0;
		for (int p = 0; p <= C; ++p) nt += conf[(size_t) t * (C + 1) + p];
		if (!nt) continue;
		std::vector<std::pair<long, int>> others;
		for (int p = 0; p < C; ++p) if (p != t && conf[(size_t) t * (C + 1) + p]) others.push_back({conf[(size_t) t * (C + 1) + p], p});
		std::sort(others.rbegin(), others.rend());
		char buf[64];
		snprintf(buf, sizeof(buf), "%-12s %6ld  %5.1f%%  %6.1f%%  ", name(t).substr(0, 12).c_str(), nt,
		         100.0 * conf[(size_t) t * (C + 1) + t] / nt, 100.0 * conf[(size_t) t * (C + 1) + C] / nt);
		std::cout << buf;
		for (size_t k = 0; k < others.size() && k < 3; ++k) std::cout << name(others[k].second) << " " << others[k].first << (k + 1 < std::min<size_t>(3, others.size()) ? ", " : "");
		std::cout << std::endl;
		if (isCommand(t)) {
			cmd_n += nt; cmd_ok += conf[(size_t) t * (C + 1) + t]; cmd_rej += conf[(size_t) t * (C + 1) + C];
			for (int p = 0; p < C; ++p) if (p != t && isCommand(p)) cmd_wrong += conf[(size_t) t * (C + 1) + p];
		} else {
			non_n += nt;
			for (int p = 0; p < C; ++p) if (isCommand(p)) non_acc += conf[(size_t) t * (C + 1) + p];
		}
	}
	if (unk >= 0 || sil >= 0) {
		std::cout << "SUMMARY commands: " << cmd_n << " clips, correct " << 100.0 * cmd_ok / std::max(1L, cmd_n) << "%, rejected "
		          << 100.0 * cmd_rej / std::max(1L, cmd_n) << "%, taken for another command " << 100.0 * cmd_wrong / std::max(1L, cmd_n)
		          << "%; non-commands (unknown/silence): " << non_n << " clips, FALSE ACCEPTS as a command " << 100.0 * non_acc / std::max(1L, non_n)
		          << "%" << std::endl;
	}
}

static int evaluateBin(CNet &cnet, CInput *inp, CrossEntropy *cen, const std::vector<float> &win, int Nf, const std::vector<std::string> &words) {
	const std::string path = (std::string) eval_bin;
	std::ifstream in(path.c_str(), std::ios::binary);
	int32_t h[4];
	if (!in.read((char*) h, sizeof(h)) || h[0] != 0x4B575331) { std::cerr << "kws_infer: bad KWS1 file " << path << "\n"; return 1; }
	int n = h[1];
	const int L = h[2], C = h[3];
	if (L != kClip) { std::cerr << "kws_infer: expected clips of " << kClip << " samples\n"; return 1; }
	if (eval_max > 0 && eval_max < n) n = eval_max;
	std::vector<int16_t> audio((size_t) n * L);
	std::vector<int> labels(n);
	for (int i = 0; i < n; ++i) {
		int32_t lab;
		if (!in.read((char*) &lab, 4) || !in.read((char*) &audio[(size_t) i * L], 2 * L)) { std::cerr << "kws_infer: truncated " << path << "\n"; return 1; }
		labels[i] = lab;
	}
	std::cout << "kws_infer: " << n << " clips from " << path << " (C=" << C << ")" << std::endl;
	// comma-separated shifts in ms; blank entries are skipped, anything else that is not a number is an error
	std::vector<float> shifts;
	const std::string list = (std::string) eval_shifts_ms;
	size_t pos = 0;
	while (pos <= list.size()) {
		size_t c = list.find(',', pos);
		if (c == std::string::npos) c = list.size();
		std::string tok = list.substr(pos, c - pos);
		pos = c + 1;
		const size_t a = tok.find_first_not_of(" \t"), b = tok.find_last_not_of(" \t");
		if (a == std::string::npos) continue;
		tok = tok.substr(a, b - a + 1);
		char *end = nullptr;
		const float ms = std::strtof(tok.c_str(), &end);
		if (end == tok.c_str() || *end != '\0') { std::cerr << "kws_infer: bad -eval_shifts_ms entry \"" << tok << "\"\n"; return 1; }
		shifts.push_back(ms);
	}
	std::vector<float> y;
	for (const float ms : shifts) {
		const int shift = (int) lroundf(ms * 16.f);
		const auto t0 = std::chrono::steady_clock::now();
		long correct = 0, rejected = 0;
		std::vector<long> conf((size_t) C * (C + 1), 0);
		for (int i = 0; i < n; ++i) {
			frameClip(&audio[(size_t) i * L], shift, win, Nf, y);
			float p = 0.f;
			int pred = predict(cnet, inp, cen, y, &p);
			if (min_prob > 0.f && p < min_prob) { pred = C; rejected++; }   // below the operating point: rejected
			correct += (pred == labels[i]);
			if (labels[i] >= 0 && labels[i] < C) conf[(size_t) labels[i] * (C + 1) + pred]++;
		}
		const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		std::cout << "EVAL " << path << " shift_ms " << ms << " acc " << correct / (double) n << " n " << n
		          << (min_prob > 0.f ? " min_prob " + std::to_string((float) min_prob) + " rejected " + std::to_string(rejected) : std::string())
		          << " (" << 1000.0 * s / n << " ms/clip)" << std::endl;
		if (confusion) printConfusion(conf, C, words, labels, n);
	}
	return 0;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	BornAttention::setDecaySchedule(born_decay_base, born_decay_step);
	if (((std::string) model).empty() || (((std::string) wav).empty() && ((std::string) eval_bin).empty())) {
		std::cerr << "usage: kws_infer -model m.mod -labels labels.txt -wav clip.wav | -eval_bin kws_test.bin\n";
		return 1;
	}

	std::vector<std::string> words;
	{ std::ifstream lf(((std::string) labels).c_str()); std::string w; while (std::getline(lf, w)) if (!w.empty()) words.push_back(w); }

	CNet cnet;
	if (!cnet.restore((std::string) model)) { std::cerr << "kws_infer: cannot restore " << (std::string) model << "\n"; return 1; }
	for (auto *f : cnet.cpuNet().funcs())
		if (auto *b = dynamic_cast<BornAttention*>(f)) { b->setDecay(born_decay); b->setLinear(false); if (!causal) b->setCausal(false); }
	cnet.cpuNet().init_exec_graph();
	CInput *inp = dynamic_cast<CInput*>(cnet.findFirstInput());
	CrossEntropy *cen = dynamic_cast<CrossEntropy*>(cnet.findFirstOutput());
	if (!inp || !cen) { std::cerr << "kws_infer: unexpected graph\n"; return 1; }
	const int W = stft_window, Nf = stft_hop > 0 ? (kL - W) / stft_hop + 1 : kL / W, len = Nf * W;
	if (inp->input().length_ != len) {
		std::cerr << "kws_infer: the model expects an input of " << inp->input().length_ << " samples, the flags give " << len
		          << " (W=" << W << ", hop=" << stft_hop << ")\n";
		return 1;
	}
	std::vector<float> win(W, 1.f);
	if (stft_hop > 0 && hann) for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);

	if (!((std::string) eval_bin).empty()) return evaluateBin(cnet, inp, cen, win, Nf, words);

	std::vector<int16_t> a;
	int rate = 0;
	if (!readWav((std::string) wav, a, &rate)) { std::cerr << "kws_infer: cannot read " << (std::string) wav << "\n"; return 1; }
	if (rate != 16000) std::cerr << "kws_infer: warning: " << rate << " Hz, expected 16000\n";
	a.resize(kClip, 0);                                                      // cut or zero-pad to one second

	std::vector<float> y;
	frameClip(a.data(), 0, win, Nf, y);
	for (int k = 0; k < len; ++k) { inp->mutable_input()->real_[k] = y[k]; inp->mutable_input()->imag_[k] = 0.f; }

	const auto t0 = std::chrono::steady_clock::now();
	cnet.cpuNet().forward();
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

	const int C = cen->input().length_;
	std::vector<std::pair<float, int>> p(C);
	double S = 0;
	for (int c = 0; c < C; ++c) { p[c] = {std::norm(cen->input().z(c)), c}; S += p[c].first; }
	std::sort(p.begin(), p.end(), [](const std::pair<float, int> &u, const std::pair<float, int> &v) { return u.first > v.first; });
	for (int i = 0; i < std::min(top, C); ++i) {
		const std::string w = p[i].second < (int) words.size() ? words[p[i].second] : "class" + std::to_string(p[i].second);
		std::cout << w << "\t" << p[i].first / S << std::endl;
	}
	std::cerr << "forward " << ms << " ms" << std::endl;
	return 0;
}
