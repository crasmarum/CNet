// =============================================================================
// generate: standalone CPU-only text generation from a saved CNet language model.
//
// Restores a serialized model (e.g. a born + RoPE + TokenNorm char-LM), and
// autoregressively samples characters from the per-position Born distribution
// p_k = |z_k|^2/||z||^2. No GPU / CUDA / NCCL -- builds with g++.
//
// The vocabulary is reconstructed from the SAME text file the model was trained
// on (first-seen character order, matching dp_shakespeare's CharData), so pass
// the original -data file.
//
// Build:   make generate
// Run:     ./generate -model rope20k.mod -data tiny_shakespear2.txt \
//                     -prompt "ROMEO:" -gen_len 600 -temp 0.8
// =============================================================================
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <memory>

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/embed.h"
#include "../impl/stepnet.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../impl/seqcrossent.h"
#include "../gpu/gpu_func.h"

FLAG_STRING(model, "")                       // serialized model file (required)
FLAG_STRING(data, "tiny_shakespear2.txt")    // training text (for the char<->id map)
FLAG_STRING(prompt, "ROMEO:")                // seed text
FLAG_INT(gen_len, 600)                        // characters to generate
FLAG_FLOAT(temp, 0.8)                         // sampling temperature (lower = greedier)
FLAG_INT(seed, 0)                             // sampling RNG seed (0 = random); fixed seeds make checkpoints comparable
FLAG_BOOL(stream, true)                       // constant-state decoding through the N=1 twin graph (impl/stepnet.h);
                                              // false = recompute the full N-token window per token (the old path)
FLAG_INT(threads, 8)                          // OpenMP threads for streaming decode: the per-token layers are
                                              // tiny, and a 100+-thread parallel region per layer per token is
                                              // pure overhead (120 threads: 250 s / 200 tokens; 1 thread: 0.8 s)
FLAG_STRING(eval, "")                         // score this text file through the streaming decoder: nats/char and
                                              // bits/byte over consecutive N-char windows (state reset per window,
                                              // first char of each window unscored); chars outside the vocab are skipped
FLAG_STRING(eval_dump, "")                    // -eval: also write one line per scored byte, "file_offset<TAB>nats"
FLAG_INT(eval_offset, 0)                      // byte offset into -eval
FLAG_INT(eval_chars, 1000000)                 // bytes to score from the offset (shard the file across processes)
FLAG_INT(eval_window, 0)                      // bytes between state resets (0 = the training context N). A long
                                              // window (e.g. 5242880, the MBLM 5M-byte protocol) streams the
                                              // constant-state twin unbroken; shard offsets/lengths should be
                                              // multiples of it so every process sees whole windows
FLAG_FLOAT(born_decay_base, 5.f)             // Born decay schedule gamma_h = 1 - 2^-(base + step*h); not stored in the
FLAG_FLOAT(born_decay_step, 1.f)             //   model file: pass the values the model was trained with (default RetNet)
FLAG_BOOL(score_stdin, false)                 // live scoring: read raw bytes from stdin one at a time through the
                                              // streaming decoder (state never reset) and print, per byte, its surprise
                                              // -log p(byte | all earlier bytes) in nats, one line each, flushed; the
                                              // first byte prints "nan"; bytes outside the vocab print "skip"
FLAG_INT(check, 0)                            // >0: compare the twin's next-token distribution with the full
                                              // window at the first `check` prompt positions, report, exit

// Character dataset: build the id map in first-seen order (identical to the
// dp_shakespeare trainer, so ids match the trained embedding).
struct CharData {
	std::map<char, int> c2i;
	int vocab = 0;
	void load(const std::string &path) {
		std::ifstream f(path);
		std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		if (s.empty()) { std::cerr << "cannot read data file: " << path << "\n"; exit(1); }
		for (char c : s) if (!c2i.count(c)) c2i[c] = vocab++;
	}
};

// Sample a token from position p's Born distribution with temperature.
static int sampleNext(SequenceCrossEntropy *seq, int p, int vocab, double temp, std::mt19937 &rng) {
	std::vector<double> q(vocab); double Z = 0;
	for (int k = 0; k < vocab; ++k) {
		double pr = seq->getProbability(p, k);
		pr = std::pow(pr < 1e-12 ? 1e-12 : pr, 1.0 / temp);
		q[k] = pr; Z += pr;
	}
	std::uniform_real_distribution<double> U(0.0, 1.0);
	double c = U(rng) * Z, s = 0;
	for (int k = 0; k < vocab; ++k) { s += q[k]; if (c <= s) return k; }
	return vocab - 1;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	BornAttention::setDecaySchedule(born_decay_base, born_decay_step);
	FILELog::ReportingLevel() = lWarning;        // quiet the per-layer restore logs
	if (((std::string) model).empty()) {
		std::cerr << "usage: generate -model <file> [-data <txt> -prompt <s> -gen_len <n> -temp <t>]\n";
		return 1;
	}

	// Vocabulary: prefer the <model>.vocab sidecar written by the trainer (byte k = the character
	// with id k, exact and corpus-independent); fall back to deriving it from -data in first-seen order.
	CharData ds;
	{
		std::ifstream vf((std::string) model + ".vocab", std::ios::binary);
		std::string vs((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
		if (!vs.empty()) {
			for (char c : vs) if (!ds.c2i.count(c)) ds.c2i[c] = ds.vocab++;
			std::cerr << "[vocab] loaded " << ds.vocab << " chars from " << (std::string) model << ".vocab\n";
		} else {
			ds.load(data);
			std::cerr << "[vocab] no sidecar; derived " << ds.vocab << " chars from -data " << (std::string) data << "\n";
		}
	}
	std::vector<char> i2c(ds.vocab);
	for (auto &kv : ds.c2i) i2c[kv.second] = kv.first;

	std::unique_ptr<CNet> netp(new CNet);   // heap-held so -score_stdin can free it once the step twin exists
	CNet &net = *netp;
	if (!net.restore((std::string) model)) {
		std::cerr << "cannot restore model: " << (std::string) model << "\n";
		return 1;
	}
	auto *em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	auto *seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model has no embedding / sequence head\n"; return 1; }
	const int N = seq->nPos(), vocab = seq->vocab();
	if (vocab != ds.vocab)
		std::cerr << "warning: model vocab " << vocab << " != data vocab " << ds.vocab
				  << " (wrong -data file?)\n";

	std::mt19937 rng(seed != 0 ? (unsigned) seed : std::random_device{}());
	std::vector<int> ctx;
	for (char c : (std::string) prompt) if (ds.c2i.count(c)) ctx.push_back(ds.c2i[c]);

	// Streaming twin: the same model with N = 1, Born layers carrying a running state.
	StepNet sn;
	bool use_stream = stream || check > 0;
	if (use_stream && !buildStepNet(net.cpuNet(), sn)) {
		std::cerr << "streaming decode unavailable (" << sn.error << "); using the full-window path\n";
		use_stream = false;
		if (check > 0) return 1;
	}

	if (score_stdin) {
		if (!use_stream) { std::cerr << "-score_stdin needs the streaming decoder\n"; return 1; }
		// The step twin holds copies of the weights; the full N-position network (GBs of activation
		// buffers) is only needed again for a learned positional table, which it still owns.
		if (!sn.pos_src) netp.reset();
#ifdef _OPENMP
		omp_set_num_threads(std::max(1, std::min<int>(threads, omp_get_max_threads())));
#endif
		stepReset(sn);
		int prev = -1, t = 0;
		char c;
		while (std::cin.get(c)) {
			auto it = ds.c2i.find(c);
			if (it == ds.c2i.end()) { std::cout << "skip\n" << std::flush; continue; }
			const int id = it->second;
			if (prev >= 0) {                              // surprise of `id` given the state after `prev`
				stepToken(sn, prev, t++);
				const double pr = sn.seq->getProbability(0, id);
				std::cout << -std::log(std::max(pr, 1e-12)) << '\n' << std::flush;
			} else {
				std::cout << "nan\n" << std::flush;
			}
			prev = id;
		}
		return 0;
	}

	if (!((std::string) eval).empty()) {
		std::ifstream f((std::string) eval, std::ios::binary);
		if (!f) { std::cerr << "cannot read " << (std::string) eval << "\n"; return 1; }
		f.seekg(eval_offset);
		std::string buf(eval_chars, '\0');
		f.read(&buf[0], eval_chars); buf.resize(f.gcount());
		std::vector<int> ids; size_t skipped = 0;
		for (char c : buf) { auto it = ds.c2i.find(c); if (it == ds.c2i.end()) ++skipped; else ids.push_back(it->second); }
#ifdef _OPENMP
		omp_set_num_threads(std::max(1, std::min<int>(threads, omp_get_max_threads())));
#endif
		// Windows are counted in BYTES of the file (so shards and resets align with the MBLM protocol);
		// bytes outside the vocabulary are not scored but are charged 8 bits each in the "bpb_all" figure.
		const size_t W = eval_window > 0 ? (size_t) eval_window : (size_t) N;
		double nll = 0; size_t scored = 0;
		std::ofstream dump;
		if (!((std::string) eval_dump).empty()) dump.open((std::string) eval_dump);
		size_t pos = 0;                                   // byte position in buf
		while (pos < buf.size()) {
			size_t wend = std::min(buf.size(), pos + W);
			stepReset(sn);
			int prev = -1, t = 0;
			for (size_t b = pos; b < wend; ++b) {
				auto it = ds.c2i.find(buf[b]);
				if (it == ds.c2i.end()) continue;         // skipped byte (counted above)
				int id = it->second;
				if (prev >= 0) {                          // score byte `id` given the state after `prev`
					stepToken(sn, prev, t++);
					double pr = sn.seq->getProbability(0, id);
					const double l = -std::log(std::max(pr, 1e-12));
					nll += l; ++scored;
					if (dump.is_open()) dump << (eval_offset + b) << '\t' << l << '\n';
				}
				prev = id;
			}
			pos = wend;
		}
		const double bits = nll / std::log(2.0);
		std::cout << "EVAL file=" << (std::string) eval << " offset=" << eval_offset << " bytes=" << buf.size()
				  << " scored=" << scored << " skipped=" << skipped << " N=" << N << " window=" << W
				  << " nats/char=" << (scored ? nll / scored : 0.0) << " bits/char=" << (scored ? bits / scored : 0.0)
				  << " bpb_all=" << (buf.size() ? (bits + 8.0 * skipped) / buf.size() : 0.0)
				  << " sum_nll=" << nll << std::endl;
		return 0;
	}

	if (check > 0) {
		// Equivalence: feed the prompt one token at a time through the twin and compare its
		// next-token distribution with the fixed-window model's distribution at the same position.
		const int K = std::min<int>(check, std::min<int>((int) ctx.size(), N));
		stepReset(sn);
		double worst = 0;
		for (int t = 0; t < K; ++t) {
			stepToken(sn, ctx[t], t);
			std::vector<int> win(N, 0);
			for (int i = 0; i <= t; ++i) win[i] = ctx[i];
			em->setInput(win);
			net.cpuNet().forward();
			for (int k = 0; k < vocab; ++k)
				worst = std::max(worst, std::fabs((double) sn.seq->getProbability(0, k) - (double) seq->getProbability(t, k)));
		}
		std::cout << "STEPCHECK positions=" << K << " N=" << N << " max|p_stream - p_window|=" << worst
				  << " -> " << (worst < 1e-4 ? "PASS" : "FAIL") << std::endl;
		return worst < 1e-4 ? 0 : 1;
	}

	std::cout << (std::string) prompt << std::flush;      // echo the seed
	if (use_stream) {
#ifdef _OPENMP
		omp_set_num_threads(std::max(1, std::min<int>(threads, omp_get_max_threads())));
#endif
		// Prefill the last <= N prompt tokens, then one state update per generated token. When the
		// learned positional table runs out (pos == N) re-prefill from the last N/2 tokens: a sliding
		// window like the full-window path's, amortized over N/2 tokens instead of paid per token.
		auto prefill = [&](int from) {
			stepReset(sn);
			int pos = 0;
			for (size_t i = from; i < ctx.size(); ++i) stepToken(sn, ctx[i], pos++);
			return pos;
		};
		int pos = prefill(sn.pos_src ? std::max<int>(0, (int) ctx.size() - N) : 0);
		for (int g = 0; g < gen_len; ++g) {
			int nxt = sampleNext(sn.seq, 0, vocab, temp, rng);
			ctx.push_back(nxt);
			std::cout << i2c[nxt] << std::flush;
			if (sn.pos_src && pos >= N) pos = prefill((int) ctx.size() - N / 2);   // learned table: sliding window
			else stepToken(sn, nxt, pos++);                                       // RoPE only: unbounded
		}
	} else {
		// Full-window path: the mixers are causal, so right-filling the N-token window and reading
		// the last real position is exact (padding never matters); O(N) model work per token.
		for (int g = 0; g < gen_len; ++g) {
			std::vector<int> win(N, 0);
			int ctxn = (int) ctx.size(), take = std::min(ctxn, N);
			for (int i = 0; i < take; ++i) win[i] = ctx[ctxn - take + i];
			em->setInput(win);
			net.cpuNet().forward();
			int nxt = sampleNext(seq, take - 1, vocab, temp, rng);
			ctx.push_back(nxt);
			std::cout << i2c[nxt] << std::flush;              // stream each token as it is produced
		}
	}
	std::cout << std::endl;
	return 0;
}
