// se_infer.cpp: CPU speech enhancement with a model trained by se.cpp (no GPU needed).
//
//   se_infer -model s1.mod -wav noisy.wav -out enhanced.wav [-stft_window 512 -stft_hop 256 -seg_frames 124]
//   se_infer -model s1.mod -eval_bin se_test.bin -out_bin enhanced.bin [-eval_max N]
//
// The clip (16 kHz mono 16-bit PCM) is processed in consecutive segments of (N-1)*hop + W samples exactly as the
// trainer's -enhance does: each segment is scaled by 1/rms, framed (Hann), run through the CPU graph restored from
// the model file, the masked spectrum is inverted and overlap-added, and the result is written as 16-bit PCM.
// The Born-attention decay schedule is re-applied from the flags (not stored in model files).
//
// -eval_bin enhances every noisy clip of a SEN1 test file (the trainer's se_test.bin) and writes them as a SEE1
// file (the format of the trainer's -enhance output), to be scored with data/se_metrics.py.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/cinput.h"
#include "../impl/l2out.h"
#include "../impl/compressed_l2.h"
#include "../impl/model_saver.h"
#include "../impl/utils.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_STRING(model, "")
FLAG_STRING(wav, "")
FLAG_STRING(out, "")
FLAG_STRING(eval_bin, "")              // SEN1 file: enhance every noisy clip and write them to -out_bin (SEE1)
FLAG_STRING(out_bin, "")
FLAG_INT(eval_max, 0)                  // only the first N clips of -eval_bin (0 = all)
FLAG_INT(stft_window, 512)
FLAG_INT(stft_hop, 256)
FLAG_INT(seg_frames, 124)
FLAG_BOOL(born_decay, true)
FLAG_FLOAT(born_decay_base, 3.f)
FLAG_FLOAT(born_decay_step, 1.f)
FLAG_BOOL(causal, true)                // -causal false for non-causal models saved before the saver stored the flag
FLAG_BOOL(overlap, true)               // half-segment stride with a raised-cosine ramp-in per segment (no joins; 2x compute)

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
			if (bits != 16) { std::cerr << "se_infer: need 16-bit PCM\n"; return false; }
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

// The restored graph and the fixed segmenting tables.
struct Enhancer {
	CNet cnet;
	CInput *inp = nullptr;
	CFunc *readback = nullptr;         // its input is [Y | -S]
	int W = 0, H = 0, N = 0, NW = 0, L = 0;
	std::vector<float> win, wsum, ramp;
	double forward_ms = 0;
	long segments = 0;

	bool init() {
		if (!cnet.restore((std::string) model)) { std::cerr << "se_infer: cannot restore " << (std::string) model << "\n"; return false; }
		const auto &funcs = cnet.cpuNet().funcs();
		int outi = -1;
		for (int i = 0; i < (int) funcs.size(); ++i) {
			if (auto *b = dynamic_cast<BornAttention*>(funcs[i])) { b->setDecay(born_decay); b->setLinear(false); if (!causal) b->setCausal(false); }
			if (dynamic_cast<L2Out*>(funcs[i]) || dynamic_cast<CompressedL2*>(funcs[i])) outi = i;
		}
		if (outi < 1) { std::cerr << "se_infer: no loss layer in the model\n"; return false; }
		const bool compressed = dynamic_cast<CompressedL2*>(funcs[outi]) != nullptr;
		readback = compressed ? funcs[outi] : funcs[outi - 1];
		cnet.cpuNet().init_exec_graph();
		inp = dynamic_cast<CInput*>(cnet.findFirstInput());
		W = stft_window; H = stft_hop; N = seg_frames; NW = N * W; L = (N - 1) * H + W;
		if (W <= 0 || H <= 0 || H > W || N <= 0) { std::cerr << "se_infer: need 0 < stft_hop <= stft_window and seg_frames > 0\n"; return false; }
		if (!inp || inp->input().length_ != 2 * NW || readback->input().length_ != 2 * NW) {
			std::cerr << "se_infer: the model expects " << (inp ? inp->input().length_ : -1) << " inputs, the flags give " << 2 * NW << "\n";
			return false;
		}
		win.resize(W);
		for (int j = 0; j < W; ++j) win[j] = 0.5f - 0.5f * cosf(2.f * (float) M_PI * j / W);
		wsum.assign(L, 0.f);
		for (int f = 0; f < N; ++f) for (int j = 0; j < W; ++j) wsum[f * H + j] += win[j] * win[j];
		ramp.assign(L, 1.f);
		if (overlap) for (int k = 0; k < L / 2; ++k) ramp[k] = 0.5f - 0.5f * cosf((float) M_PI * (k + 1) / (L / 2));
		return true;
	}

	// Enhance one clip: consecutive (or half-overlapping) segments, each scaled by 1/rms, framed, masked, inverted.
	std::vector<int16_t> run(const std::vector<int16_t> &noisy) {
		const int len = (int) noisy.size(), stride = overlap ? L / 2 : L;
		const int n_seg = overlap ? std::max(1, (std::max(0, len - L) + stride - 1) / stride + 1) : (len + L - 1) / L;
		std::vector<float> y((size_t) (n_seg - 1) * stride + L, 0.f), wacc(y.size(), 0.f), re(W), im(W), seg(L);
		const float inv = 1.f / sqrtf((float) W);
		for (int sg = 0; sg < n_seg; ++sg) {
			const int start = sg * stride;
			double e = 0;
			for (int k = 0; k < L; ++k) { const int s = start + k; if (s < len) e += (double) noisy[s] * noisy[s]; }
			const float scale = 1.f / (32768.f * std::max(sqrtf((float) (e / L)) / 32768.f, 1e-4f));
			for (int k = 0; k < 2 * NW; ++k) { inp->mutable_input()->real_[k] = 0.f; inp->mutable_input()->imag_[k] = 0.f; }
			for (int f = 0; f < N; ++f)
				for (int j = 0; j < W; ++j) {
					const int s = start + f * H + j;
					if (s < len) inp->mutable_input()->real_[f * W + j] = win[j] * scale * noisy[s];
				}
			const auto t0 = std::chrono::steady_clock::now();
			cnet.cpuNet().forward();
			forward_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			segments++;
			std::fill(seg.begin(), seg.end(), 0.f);
			for (int f = 0; f < N; ++f) {
				for (int j = 0; j < W; ++j) { const auto z = readback->input().z(f * W + j); re[j] = z.real(); im[j] = z.imag(); }
				FastFourierTransform<float>::inverseTransform(W, re.data(), im.data());
				for (int j = 0; j < W; ++j) seg[f * H + j] += win[j] * re[j] * inv;
			}
			for (int k = 0; k < L; ++k) {
				const float v = wsum[k] > 1e-3f ? seg[k] / wsum[k] / scale : 0.f;
				y[(size_t) start + k] += ramp[k] * v;
				wacc[(size_t) start + k] += ramp[k];
			}
		}
		std::vector<int16_t> o(len);
		for (int k = 0; k < len; ++k) {
			const float v = wacc[k] > 0.f ? y[k] / wacc[k] : 0.f;
			o[k] = (int16_t) std::max(-32768.f, std::min(32767.f, roundf(v)));
		}
		return o;
	}
};

// Enhance every noisy clip of a SEN1 file and write a SEE1 file.
static int enhanceBin(Enhancer &e) {
	const std::string path = (std::string) eval_bin;
	std::ifstream in(path.c_str(), std::ios::binary);
	int32_t h[3];
	if (!in.read((char*) h, sizeof(h)) || h[0] != 0x53454E31) { std::cerr << "se_infer: bad SEN1 file " << path << "\n"; return 1; }
	int n = h[1];
	if (h[2] != kRate) std::cerr << "se_infer: warning: " << h[2] << " Hz, expected 16000\n";
	if (eval_max > 0 && eval_max < n) n = eval_max;
	std::ofstream o(((std::string) out_bin).c_str(), std::ios::binary);
	if (!o) { std::cerr << "se_infer: cannot write " << (std::string) out_bin << "\n"; return 1; }
	const int32_t magic = 0x53454531, cnt = n;
	o.write((char*) &magic, 4); o.write((char*) &cnt, 4);
	double samples = 0;
	const auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < n; ++i) {
		int32_t len;
		if (!in.read((char*) &len, 4)) { std::cerr << "se_infer: truncated " << path << "\n"; return 1; }
		std::vector<int16_t> noisy(len), clean(len);
		if (!in.read((char*) noisy.data(), 2 * (size_t) len) || !in.read((char*) clean.data(), 2 * (size_t) len)) { std::cerr << "se_infer: truncated " << path << "\n"; return 1; }
		const std::vector<int16_t> enh = e.run(noisy);
		o.write((char*) &len, 4);
		o.write((char*) enh.data(), 2 * (size_t) len);
		samples += len;
		if ((i + 1) % 100 == 0) std::cerr << (i + 1) << "/" << n << std::endl;
	}
	const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "se_infer: " << n << " clips (" << samples / kRate / 3600 << " h) enhanced in " << s << " s, " << e.segments
	          << " segments, forward " << e.forward_ms / std::max(1L, e.segments) << " ms per segment, overlap=" << overlap
	          << " -> " << (std::string) out_bin << std::endl;
	return (bool) o ? 0 : 1;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	FILELog::ReportingLevel() = lWarning;
	BornAttention::setDecaySchedule(born_decay_base, born_decay_step);
	const bool bin_mode = !((std::string) eval_bin).empty();
	if (((std::string) model).empty() || (!bin_mode && (((std::string) wav).empty() || ((std::string) out).empty()))
			|| (bin_mode && ((std::string) out_bin).empty())) {
		std::cerr << "usage: se_infer -model m.mod -wav noisy.wav -out enhanced.wav | -eval_bin se_test.bin -out_bin enhanced.bin\n";
		return 1;
	}
	Enhancer e;
	if (!e.init()) return 1;
	if (bin_mode) return enhanceBin(e);

	std::vector<int16_t> noisy;
	int rate = 0;
	if (!readWav((std::string) wav, noisy, &rate)) { std::cerr << "se_infer: cannot read " << (std::string) wav << "\n"; return 1; }
	if (rate != kRate) std::cerr << "se_infer: warning: " << rate << " Hz, expected 16000\n";
	const std::vector<int16_t> o = e.run(noisy);
	if (!writeWav((std::string) out, o)) { std::cerr << "se_infer: cannot write " << (std::string) out << "\n"; return 1; }
	std::cerr << "se_infer: " << noisy.size() / (float) kRate << " s in " << e.segments << " segments, forward " << e.forward_ms
	          << " ms total (" << e.forward_ms / std::max(1L, e.segments) << " ms per " << e.L / (float) kRate << " s segment)" << std::endl;
	return 0;
}
