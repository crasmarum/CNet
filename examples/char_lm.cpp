// char_lm.cpp -- a byte-level language model on top of impl/cnet_app.h.
//
// Everything specific to this problem is here: the layer graph (ByteLM) and where the windows come from
// (ByteText). Training, validation, checkpoints, multi-GPU and inference come from app::Trainer / app::Infer.
//
//   make app SRC=examples/char_lm.cpp                       # single GPU (add NCCL=1 for multi-GPU)
//   ./char_lm -data book.txt -steps 2000 -save_path lm.mod   # train
//   ./char_lm -data book.txt -generate true -model lm.mod -prompt "ROMEO:" -gen_len 300
//   ./char_lm -data book.txt -score true -model lm.mod       # exact held-out loss (CPU), -score_bytes of the tail
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../impl/cnet_app.h"
#include "../impl/tokenwise.h"
#include "../impl/relu.h"
#include "../impl/residual.h"
#include "../impl/tokennorm.h"
#include "../impl/rotary.h"
#include "../impl/bornattn.h"

FLAG_STRING(data, "tiny_shakespear2.txt") // training text; the last 10% is held out
FLAG_INT(emb, 256)                        // width E (complex channels)
FLAG_INT(tokens, 512)                     // context N (bytes)
FLAG_INT(blocks, 6)                       // number of blocks
FLAG_INT(heads, 4)                        // Born attention heads
FLAG_BOOL(born_decay, true)               // fixed per-head decay
FLAG_BOOL(generate, false)                // sample from -model instead of training
FLAG_BOOL(score, false)                   // exact held-out loss of -model
FLAG_STRING(model, "")                    // model file for -generate / -score
FLAG_STRING(prompt, "\n")                 // -generate seed text
FLAG_INT(gen_len, 300)                    // -generate bytes
FLAG_FLOAT(temp, 0.8)                     // -generate temperature
FLAG_INT(score_bytes, 65536)              // -score: held-out bytes to score (CPU, exact)

// ---------------- the data: a memory-mapped byte file, vocabulary = the byte values that occur ----------------
struct ByteText : app::DataSource {
	const unsigned char *p = nullptr; size_t n = 0; int lut[256]; std::vector<unsigned char> bytes; int N;
	ByteText(const std::string &path, int context) : N(context) {
		int fd = open(path.c_str(), O_RDONLY); struct stat st; fstat(fd, &st); n = st.st_size;
		p = (const unsigned char*) mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0); close(fd);
		for (int &x : lut) x = -1;
		for (size_t i = 0; i < n; ++i) if (lut[p[i]] < 0) { lut[p[i]] = (int) bytes.size(); bytes.push_back(p[i]); }
	}
	int vocab() const override { return (int) bytes.size(); }
	int context() const override { return N; }
	size_t split() const { return n / 10 * 9; }
	void sample(bool heldout, std::mt19937_64 &rng, std::vector<int> &in, std::vector<int> &tgt) override {
		const size_t lo = heldout ? split() : 0, hi = (heldout ? n : split()) - N - 2;
		const size_t s = std::uniform_int_distribution<size_t>(lo, hi)(rng);
		in.resize(N); tgt.resize(N);
		for (int i = 0; i < N; ++i) { in[i] = lut[p[s + i]]; tgt[i] = lut[p[s + i + 1]]; }
	}
	std::vector<int> encode(const std::string &t) const { std::vector<int> v; for (unsigned char c : t) if (lut[c] >= 0) v.push_back(lut[c]); return v; }
};

// ---------------- the model: Born attention (+ decay, RoPE) and a feed-forward layer per block ----------------
struct ByteLM : app::ModelDef {
	int E, L, H;
	ByteLM(int e, int l, int h) : E(e), L(l), H(h) {}
	app::Heads build(ComplexNet &net, int V, int N) override {
		const int S = N * E, dh = E / 2, F = 4 * E;
		auto lin = [&](int x, int in, int out) { int w = net.add(new CInput(OutSize(in * out))); return net.add(new TokenwiseLinear(N, in, out), {x, w}); };
		auto norm = [&](int x) { return net.add(new TokenNorm(N, E), {x}); };
		int emb = net.add(new CEmbedding(E, N, V));
		int x = norm(emb);
		for (int b = 0; b < L; ++b) {
			int pre = net.add(new CGelu(InSize(S)), {norm(x)});
			int Q = net.add(new RotaryEmbed(N, dh, 10000.0, H), {lin(pre, E, dh)});
			int K = net.add(new RotaryEmbed(N, dh, 10000.0, H), {lin(pre, E, dh)});
			int Vv = lin(pre, E, dh);
			int att = net.add(new BornAttention(N, dh, true, 0.0, H), {Q, K, Vv});
			((BornAttention*) net[att])->setLinear(true, std::min(512, N));
			((BornAttention*) net[att])->setDecay(born_decay);
			int r0 = net.add(new Residual(InSize(S), InSize(S)), {x, lin(att, dh, E)});
			int n0 = net.add(new CGelu(InSize(S)), {r0});
			int f  = lin(net.add(new CGelu(InSize(N * F)), {lin(norm(n0), E, F)}), F, E);
			x = net.add(new CGelu(InSize(S)), {net.add(new Residual(InSize(S), InSize(S)), {n0, f})});
		}
		int loss = net.add(new SequenceCrossEntropy(InSize(N * V), V), {lin(norm(x), E, V)});
		return app::Heads(emb, loss);
	}
};

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	ByteText text((std::string) data, tokens);

	if (generate || score) {
		app::Infer inf;
		if (!inf.loadModel((std::string) model)) return 1;
		if (generate) {
			std::cout << (std::string) prompt << std::flush;
			inf.generate(text.encode((std::string) prompt), gen_len, temp, 1,
						 [&](int k) { std::cout << (char) text.bytes[k] << std::flush; });
			std::cout << std::endl;
		} else {
			const size_t s = text.split(), m = std::min<size_t>(text.n - s, (size_t) score_bytes);
			std::vector<int> ids; for (size_t i = s; i < s + m; ++i) ids.push_back(text.lut[text.p[i]]);
			double nats = inf.score(ids);
			std::cout << "held-out " << nats << " nats/byte = " << nats / std::log(2.0) << " bits/byte" << std::endl;
		}
		return 0;
	}

	ByteLM lm(emb, blocks, heads);
	app::Trainer trainer(lm, text);
	return trainer.run();
}
