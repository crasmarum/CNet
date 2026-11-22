// se_stream.cpp: constant-state streaming speech enhancement on the CPU with a model trained by se.cpp.
//
//   se_stream -model s1.mod -wav noisy.wav -out enhanced.wav [-gain running|fixed] [-check true]
//   se_stream -model s1.mod -eval_bin se_test.bin -out_bin enhanced.bin [-eval_max N]
//
// The trainer and se_infer process 2 s segments with a fresh attention state each; here the model's graph is
// rebuilt as its N = 1 twin (impl/framenet.h) and fed one Hann frame per hop: the causal Born attention carries
// its running state across the whole recording, the mask of the current frame is inverted and overlap-added, and
// each hop of output becomes final W - hop .. W samples after it was recorded (16-32 ms at W=512, hop 256) plus
// one frame of compute. The per-segment look-ahead gain of the trainer (1/rms over 2 s) is replaced by a causal
// running RMS (-gain running, time constant -gain_seconds) or a fixed gain; the model never saw either, so the
// quality must be measured, not assumed (data/se_metrics.py on the -eval_bin output).
//
// Framing and reconstruction (the rule shared with se.cpp and se_infer): frame f covers samples [f*hop, f*hop + W)
// of the recording, starting at sample 0 without pre-padding, so the first hop of samples is covered by one frame
// only; the last frame is zero-padded to W. Each enhanced frame is inverted with the unnormalised inverse FFT,
// scaled by 1/sqrt(W) (the layer's DFT is unitary), real part taken, multiplied by the synthesis window, divided by
// the gain the frame was scaled with and overlap-added; the output is divided by the accumulated squared window of
// the frames that actually cover each sample, and samples with less than 1e-3 of it (the first 30 of a recording at
// W=512) are set to zero.
//
// -check compares the twin with the full graph on the first segment of the input, both with that segment's
// trainer gain: the enhanced frames must agree to float precision.
// -identity replaces the model's mask by the identity (Y = X, read from the twin's DFT layer) and compares the
// reconstruction with the input: it must reproduce it to the int16 rounding, which verifies the DFT pair (sign and
// 1/sqrt(W) scaling), the per-frame gain reversal and the overlap-add normaliser including the boundaries.
#include <algorithm>
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
#include "../impl/l2out.h"
#include "../impl/compressed_l2.h"
#include "../impl/model_saver.h"
#include "../impl/utils.h"
#include "../impl/framenet.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_STRING(model, "")
FLAG_STRING(wav, "")
FLAG_STRING(out, "")
FLAG_STRING(eval_bin, "")              // SEN1 file: stream every noisy clip, write SEE1 to -out_bin
FLAG_STRING(out_bin, "")
FLAG_INT(eval_max, 0)
FLAG_INT(stft_window, 512)
FLAG_INT(stft_hop, 256)
FLAG_INT(seg_frames, 124)              // frames per segment of the trainer (for -check and the running-gain window)
FLAG_BOOL(born_decay, true)
FLAG_FLOAT(born_decay_base, 3.f)
FLAG_FLOAT(born_decay_step, 1.f)
FLAG_STRING(gain, "running")           // running: causal RMS with time constant -gain_seconds; fixed: -gain_fixed
FLAG_FLOAT(gain_seconds, 2.f)
FLAG_FLOAT(gain_fixed, 1.f)            // multiplies int16/32768 samples (the trainer's gain is 1/rms of the segment)
FLAG_BOOL(check, false)
FLAG_BOOL(identity, false)             // Y = X instead of the model's output: the stream must reproduce its input
FLAG_INT(threads, 1)
FLAG_BOOL(infer_dims, false)           // allocate only z and dz per layer (Vars::INFER_DIMS): half the activation memory

static const int kRate = 16000;

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
			if (bits != 16) { std::cerr << "se_stream: need 16-bit PCM\n"; return false; }
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

static bool writeWav(const std::string &path, const std::vector<int16_t> &a) {
	std::ofstream o(path.c_str(), std::ios::binary);
	if (!o) return false;
	const uint32_t data = 2 * (uint32_t) a.size(), riff = 36 + data, fmt_size = 16, rate = kRate, byte_rate = 2 * kRate;
	const uint16_t fmt_tag = 1, channels = 1, block = 2, bits = 16;
	o.write("RIFF", 4); o.write((char*) &riff, 4); o.write("WAVE", 4);
	o.write("fmt ", 4); o.write((char*) &fmt_size, 4); o.write((char*) &fmt_tag, 2); o.write((char*) &channels, 2);
	o.write((char*) &rate, 4); o.write((char*) &byte_rate, 4); o.write((char*) &block, 2); o.write((char*) &bits, 2);
	o.write("data", 4); o.write((char*) &data, 4); o.write((char*) a.data(), data);
	return (bool) o;
}

// The trainer's gain for a block of samples: 1 / (32768 * max(rms / 32768, 1e-4)).
static float trainerGain(double sum_sq, long n) {
	const float rms = n > 0 ? sqrtf((float) (sum_sq / n)) : 0.f;
	return 1.f / (32768.f * std::max(rms / 32768.f, 1e-4f));
}

// One clip through the twin, hop by hop. Returns the enhanced samples (same length as the input).
struct Streamer {
	FrameNet &sn;
	int W, H;
	std::vector<float> win;                // Hann window
	SegmentedFourier *dft = nullptr;       // the twin's DFT layer (its output is X, the spectrum of the current frame)
	Streamer(FrameNet &s) : sn(s), W(s.W), H(stft_hop), win(W) {
		for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);
		for (CFunc *f : sn.net.funcs()) if (auto *d = dynamic_cast<SegmentedFourier*>(f)) dft = d;
	}
	double ms_total = 0;
	long frames = 0;
	int zeroed_head = 0;                   // samples at the start of the last run with too little window coverage

	std::vector<int16_t> run(const std::vector<int16_t> &noisy, bool reset = true) {
		if (reset) frameNetReset(sn);
		const int len = (int) noisy.size();
		const int n_frames = (len + H - 1) / H;                        // frame f covers samples [fH, fH + W); the last is zero-padded
		std::vector<float> acc((size_t) (n_frames + 1) * H + W, 0.f);  // overlap-add accumulator (already de-scaled)
		std::vector<float> wsum(acc.size(), 0.f);                      // sum of win^2 over the frames that cover each sample
		std::vector<float> frame(W), re(W), im(W);
		const float inv = 1.f / sqrtf((float) W);
		// causal running gain: the mean square of the first frame (available when that frame is processed), then an
		// exponential average over the hops with the time constant -gain_seconds; the trainer used 1/rms of the
		// whole 2 s segment, so a stream that started from zero energy would over-amplify its first second
		double e2 = -1.0;
		const double alpha = exp(-(double) H / ((double) gain_seconds * kRate));
		for (int f = 0; f < n_frames; ++f) {
			const int start = f * H;
			float g;
			if ((std::string) gain == "fixed") {
				g = gain_fixed / 32768.f;
			} else {
				double s = 0; int cnt = 0;
				if (e2 < 0) {                                          // first frame: its own W samples
					for (int k = 0; k < W && k < len; ++k) { s += (double) noisy[k] * noisy[k]; cnt++; }
					e2 = cnt ? s / cnt : 0.0;
				} else {                                               // the hop of new samples this frame adds
					for (int k = start + W - H; k < start + W; ++k) if (k >= 0 && k < len) { s += (double) noisy[k] * noisy[k]; cnt++; }
					e2 = alpha * e2 + (1.0 - alpha) * (cnt ? s / cnt : 0.0);
				}
				g = trainerGain(e2, 1);
			}
			for (int j = 0; j < W; ++j) {
				const int s = start + j;
				frame[j] = (s < len ? g * noisy[s] : 0.f) * win[j];
			}
			const auto t0 = std::chrono::steady_clock::now();
			frameNetStep(sn, frame.data(), f);
			ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			frames++;
			if (identity) {                                            // Y = X: the twin's DFT output for this frame
				const Vars &x = dft->output(0);
				const int off = dft->offset(0);
				for (int j = 0; j < W; ++j) { re[j] = x.real_[off + j]; im[j] = x.imag_[off + j]; }
			} else {
				const Vars &y = sn.out->input();
				for (int j = 0; j < W; ++j) { re[j] = y.real_[j]; im[j] = y.imag_[j]; }
			}
			FastFourierTransform<float>::inverseTransform(W, re.data(), im.data());   // unnormalised sum_p Y[p] e^{+2 pi i pq/W}
			for (int j = 0; j < W; ++j) {
				acc[(size_t) start + j] += win[j] * re[j] * inv / g;
				wsum[(size_t) start + j] += win[j] * win[j];
			}
		}
		std::vector<int16_t> o(len);
		zeroed_head = 0;
		for (int k = 0; k < len; ++k) {
			const bool covered = wsum[k] > 1e-3f;                      // the same threshold as the offline paths
			const float v = covered ? acc[k] / wsum[k] : 0.f;
			if (!covered && zeroed_head == k) zeroed_head = k + 1;
			o[k] = (int16_t) std::max(-32768.f, std::min(32767.f, roundf(v)));
		}
		return o;
	}
};

// -identity: the stream with Y = X must give back its input. Reports the largest deviation in int16 units over the
// covered samples, the zeroed head, and the SI-SDR of the output against the input.
static int identityCheck(Streamer &st, const std::vector<int16_t> &noisy) {
	if (!st.dft) { std::cerr << "se_stream: the twin has no DFT layer\n"; return 1; }
	const std::vector<int16_t> y = st.run(noisy);
	const int len = (int) noisy.size(), head = st.zeroed_head;
	int max_diff = 0, at = -1;
	double sxy = 0, sxx = 0, see = 0;
	for (int k = head; k < len; ++k) {
		const int d = std::abs((int) y[k] - (int) noisy[k]);
		if (d > max_diff) { max_diff = d; at = k; }
		sxy += (double) y[k] * noisy[k]; sxx += (double) noisy[k] * noisy[k];
	}
	const double a = sxx > 0 ? sxy / sxx : 0;
	for (int k = head; k < len; ++k) { const double e = y[k] - a * noisy[k]; see += e * e; }
	const double sisdr = see > 0 ? 10 * log10(a * a * sxx / see) : 1e9;
	const bool pass = max_diff <= 1;
	std::cout << "IDENTITY " << len << " samples, gain " << (std::string) gain << ": first " << head << " samples zeroed (sum w^2 < 1e-3), max |y - x| "
	          << max_diff << " LSB" << (at >= 0 ? " at sample " + std::to_string(at) : std::string()) << " over the rest, SI-SDR "
	          << (sisdr > 1e8 ? std::string("inf") : std::to_string(sisdr)) << " dB " << (pass ? "PASS" : "FAIL") << std::endl;
	return pass ? 0 : 1;
}

// Twin vs full graph on the first segment of `noisy` with the trainer's segment gain.
static int checkAgainstFull(CNet &full, CInput *inp, CFunc *readback, FrameNet &sn, const std::vector<int16_t> &noisy) {
	const int W = sn.W, H = stft_hop, N = sn.N, L = (N - 1) * H + W, NW = N * W;
	if ((int) noisy.size() < L) { std::cerr << "se_stream: -check needs at least " << L << " samples\n"; return 1; }
	double e = 0;
	for (int k = 0; k < L; ++k) e += (double) noisy[k] * noisy[k];
	const float g = trainerGain(e, L);
	std::vector<float> win(W);
	for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);
	Vars *v = inp->mutable_input();
	for (int k = 0; k < 2 * NW; ++k) { v->real_[k] = 0.f; v->imag_[k] = 0.f; }
	for (int f = 0; f < N; ++f)
		for (int j = 0; j < W; ++j) v->real_[f * W + j] = win[j] * g * noisy[f * H + j];
	full.cpuNet().forward();
	std::vector<std::complex<float>> yf(NW);
	for (int k = 0; k < NW; ++k) yf[k] = readback->input().z(k);
	frameNetReset(sn);
	std::vector<float> frame(W);
	double max_diff = 0, max_val = 0, ms = 0;
	for (int f = 0; f < N; ++f) {
		for (int j = 0; j < W; ++j) frame[j] = win[j] * g * noisy[f * H + j];
		const auto t0 = std::chrono::steady_clock::now();
		frameNetStep(sn, frame.data(), f);
		ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		const Vars &y = sn.out->input();
		for (int j = 0; j < W; ++j) {
			const std::complex<float> z(y.real_[j], y.imag_[j]);
			max_diff = std::max(max_diff, (double) std::abs(z - yf[f * W + j]));
			max_val = std::max(max_val, (double) std::abs(yf[f * W + j]));
		}
	}
	const bool pass = max_diff < 1e-3 * std::max(1.0, max_val);
	std::cout << "CHECK first segment (" << N << " frames): max |Y_full - Y_twin| " << max_diff << " (max |Y| " << max_val << ") "
	          << (pass ? "PASS" : "FAIL") << ", twin " << ms / N << " ms per frame (" << 1000.0 * H / kRate << " ms of audio), state "
	          << sn.stateBytes() / 1024.0 << " KiB" << std::endl;
	return pass ? 0 : 1;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	if (infer_dims) Vars::dims_ = Vars::INFER_DIMS;   // before any layer is built: z and dz only (the forward's scratch)
#ifdef _OPENMP
	omp_set_num_threads(std::max(1, std::min<int>(threads, omp_get_max_threads())));
#endif
	BornAttention::setDecaySchedule(born_decay_base, born_decay_step);
	const bool bin_mode = !((std::string) eval_bin).empty();
	if (((std::string) model).empty() || (!bin_mode && ((std::string) wav).empty()) || (bin_mode && ((std::string) out_bin).empty())) {
		std::cerr << "usage: se_stream -model m.mod -wav noisy.wav -out enhanced.wav [-check true] | -eval_bin se_test.bin -out_bin x.bin\n";
		return 1;
	}
	CNet full;
	if (!full.restore((std::string) model)) { std::cerr << "se_stream: cannot restore " << (std::string) model << "\n"; return 1; }
	const auto &funcs = full.cpuNet().funcs();
	int outi = -1;
	for (int i = 0; i < (int) funcs.size(); ++i) {
		if (auto *b = dynamic_cast<BornAttention*>(funcs[i])) {
			b->setDecay(born_decay); b->setLinear(false);
			if (!b->causal()) { std::cerr << "se_stream: " << b->getName() << " is not causal; streaming needs a causal model\n"; return 1; }
		}
		if (dynamic_cast<L2Out*>(funcs[i]) || dynamic_cast<CompressedL2*>(funcs[i])) outi = i;
	}
	if (outi < 1) { std::cerr << "se_stream: no loss layer in the model\n"; return 1; }
	const bool compressed = dynamic_cast<CompressedL2*>(funcs[outi]) != nullptr;
	CFunc *readback = compressed ? funcs[outi] : funcs[outi - 1];       // its input is [Y | -S]
	CFunc *Y = readback->previous(0);                                    // the enhanced spectrum X + M.X
	full.cpuNet().init_exec_graph();
	CInput *inp = dynamic_cast<CInput*>(full.findFirstInput());
	const int W = stft_window, H = stft_hop, N = seg_frames, NW = N * W;
	if (!inp || inp->input().length_ != 2 * NW || Y->outSize() != NW) {
		std::cerr << "se_stream: the model expects " << (inp ? inp->input().length_ : -1) << " inputs, the flags give " << 2 * NW << "\n";
		return 1;
	}
	if (H <= 0 || H > W) { std::cerr << "se_stream: need 0 < stft_hop <= stft_window\n"; return 1; }

	FrameNet sn;
	if (!buildFrameNet(full.cpuNet(), inp->uid(), Y->uid(), sn)) { std::cerr << "se_stream: " << sn.error << "\n"; return 1; }
	std::cout << "se_stream: twin of " << sn.layers << " layers, W=" << sn.W << " hop=" << H << " (model N=" << sn.N << "), Born layers "
	          << sn.borns.size() << ", state " << sn.stateBytes() / 1024.0 << " KiB, latency " << 1000.0 * (W - H) / kRate << "-"
	          << 1000.0 * W / kRate << " ms + compute, gain " << (std::string) gain << std::endl;

	if (bin_mode) {
		std::ifstream in(((std::string) eval_bin).c_str(), std::ios::binary);
		int32_t h[3];
		if (!in.read((char*) h, sizeof(h)) || h[0] != 0x53454E31) { std::cerr << "se_stream: bad SEN1 file\n"; return 1; }
		int n = h[1];
		if (eval_max > 0 && eval_max < n) n = eval_max;
		std::ofstream o(((std::string) out_bin).c_str(), std::ios::binary);
		const int32_t magic = 0x53454531, cnt = n;
		o.write((char*) &magic, 4); o.write((char*) &cnt, 4);
		Streamer st(sn);
		double samples = 0;
		const auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < n; ++i) {
			int32_t len;
			if (!in.read((char*) &len, 4)) { std::cerr << "se_stream: truncated\n"; return 1; }
			std::vector<int16_t> noisy(len), clean(len);
			if (!in.read((char*) noisy.data(), 2 * (size_t) len) || !in.read((char*) clean.data(), 2 * (size_t) len)) { std::cerr << "se_stream: truncated\n"; return 1; }
			const std::vector<int16_t> enh = st.run(noisy);
			o.write((char*) &len, 4);
			o.write((char*) enh.data(), 2 * (size_t) len);
			samples += len;
			if ((i + 1) % 100 == 0) std::cerr << (i + 1) << "/" << n << std::endl;
		}
		const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		std::cout << "se_stream: " << n << " clips (" << samples / kRate / 3600 << " h) in " << s << " s, " << st.frames << " frames, "
		          << st.ms_total / std::max(1L, st.frames) << " ms per frame (" << 1000.0 * H / kRate << " ms of audio) -> " << (std::string) out_bin << std::endl;
		return (bool) o ? 0 : 1;
	}

	std::vector<int16_t> noisy;
	int rate = 0;
	if (!readWav((std::string) wav, noisy, &rate)) { std::cerr << "se_stream: cannot read " << (std::string) wav << "\n"; return 1; }
	if (rate != kRate) std::cerr << "se_stream: warning: " << rate << " Hz, expected 16000\n";
	if (check) return checkAgainstFull(full, inp, readback, sn, noisy);
	Streamer st(sn);
	if (identity) return identityCheck(st, noisy);
	const std::vector<int16_t> enh = st.run(noisy);
	if (((std::string) out).empty() || !writeWav((std::string) out, enh)) { std::cerr << "se_stream: cannot write " << (std::string) out << "\n"; return 1; }
	std::cerr << "se_stream: " << noisy.size() / (double) kRate << " s, " << st.frames << " frames, " << st.ms_total / std::max(1L, st.frames)
	          << " ms per frame (" << 1000.0 * H / kRate << " ms of audio), state " << sn.stateBytes() / 1024.0 << " KiB" << std::endl;
	return 0;
}
