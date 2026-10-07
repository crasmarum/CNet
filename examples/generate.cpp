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

#include "../utils/flags.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/embed.h"
#include "../impl/seqcrossent.h"
#include "../gpu/gpu_func.h"

FLAG_STRING(model, "")                       // serialized model file (required)
FLAG_STRING(data, "tiny_shakespear2.txt")    // training text (for the char<->id map)
FLAG_STRING(prompt, "ROMEO:")                // seed text
FLAG_INT(gen_len, 600)                        // characters to generate
FLAG_FLOAT(temp, 0.8)                         // sampling temperature (lower = greedier)

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
	FILELog::ReportingLevel() = lWarning;        // quiet the per-layer restore logs
	if (((std::string) model).empty()) {
		std::cerr << "usage: generate -model <file> [-data <txt> -prompt <s> -gen_len <n> -temp <t>]\n";
		return 1;
	}

	CharData ds; ds.load(data);
	std::vector<char> i2c(ds.vocab);
	for (auto &kv : ds.c2i) i2c[kv.second] = kv.first;

	CNet net;
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

	// Autoregressive sampling. The mixers are causal, so right-filling the N-token
	// window and reading the last real position is exact (padding never matters).
	std::mt19937 rng(std::random_device{}());
	std::vector<int> ctx;
	for (char c : (std::string) prompt) if (ds.c2i.count(c)) ctx.push_back(ds.c2i[c]);
	std::cout << (std::string) prompt << std::flush;      // echo the seed
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
	std::cout << std::endl;
	return 0;
}
