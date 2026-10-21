#include <chrono>
// =============================================================================
// dp_shakespeare: a standalone, multi-GPU (data-parallel) character-level
// language model on tiny-shakespeare, built with the CNet complex-valued
// framework. A fully complex FNet-style model (causal Fourier token mixer +
// position-wise feed-forward) trained with a Born-rule sequence loss.
//
// Data parallelism is multi-PROCESS (the standard DDP architecture): launch one
// process per GPU; the ranks rendezvous via a NCCL unique id written to /tmp by
// rank 0, then cross-process all-reduce the gradients each step. Deterministic
// init makes every rank start from identical weights.
//
// Build:
//   make dp_shakespeare
//
// Run on 1 GPU:
//   ./dp_shakespeare -world 1 -batch 32 -steps 3000 -data tiny_shakespear2.txt
//
// Run on N GPUs (one process per GPU) -- see examples/run_dp_shakespeare.sh:
//   rm -f /tmp/cnet_shk_id.*
//   for r in $(seq 0 $((N-1))); do
//     CUDA_VISIBLE_DEVICES=$r ./dp_shakespeare -world N -rank $r -batch 32 ... &
//   done; wait
// =============================================================================
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <map>
#include <random>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdint>
#include <nccl.h>
#include <cublas_v2.h>

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../impl/log.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../impl/batch.h"
#include "../impl/cinput.h"
#include "../impl/embed.h"
#include "../impl/ft.h"
#include "../impl/tokenwise.h"
#include "../impl/relu.h"
#include "../impl/residual.h"
#include "../impl/seqcrossent.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

FLAG_INT(rank, 0)            // this process's rank in [0, world)
FLAG_INT(world, 1)          // number of data-parallel processes (one per GPU)
FLAG_INT(batch, 32)         // GLOBAL batch, split across ranks (per-rank = batch/world)
FLAG_INT(emb, 64)           // embedding width E
FLAG_INT(heads, 1)          // Born attention heads: the E/2-wide complex head is split into
                            // `heads` heads of E/(2*heads) dims (per-head scores, per-head RoPE)
FLAG_BOOL(pos_table, true)  // learned absolute positional table (N x E) added to the embedding; false = RoPE only
                            // (21% fewer parameters at N=4096/E=256, no context ceiling for the streaming decoder)
FLAG_BOOL(born_linear, false) // GPU: exact chunkwise linear Born form (O(N) scratch; needs N % born_chunk == 0)
FLAG_INT(born_chunk, 512)   // chunk length C of the linear form
FLAG_BOOL(born_decay, false) // fixed per-head decay gamma_h^(k-m) on the Born score (gamma_h = 1 - 2^-(5+h)); exact in the linear state
FLAG_INT(tokens, 64)        // context length N
FLAG_INT(blocks, 4)         // number of FNet blocks L
FLAG_INT(steps, 3000)       // optimizer steps
FLAG_INT(warmup, 200)       // linear-warmup steps
FLAG_INT(val_every, 500)    // validation period (0 = off)
FLAG_FLOAT(lr, 1e-3)        // peak learning rate
FLAG_FLOAT(min_lr, 1e-4)    // cosine floor
FLAG_FLOAT(grad_clip, 0.0)  // true-Adam clip-by-value per grad component (0 = off)
FLAG_FLOAT(grad_norm_clip, 0.0) // GLOBAL gradient-norm clip on the all-reduced grad (0 = off):
                                // scales the whole gradient when ||g|| exceeds this, which
                                // (unlike clip-by-value) bounds the coordinated large-norm update
                                // that the spike-prone Born loss produces. Applied identically on
                                // every rank (same summed gradient), so replicas stay in sync.
FLAG_STRING(data, "tiny_shakespear2.txt")
FLAG_BOOL(born, false)      // use Born-rule attention mixer instead of causal Fourier
FLAG_BOOL(gla, false)       // gated linear attention over complex amplitudes (degree-one counterpart of Born;
                            // per-head sigmoid gates from the pre-normed stream; TokenNorm after the mixer)
FLAG_STRING(attn_at, "")    // hybrid: comma-separated block indices that use Born attention
                            // (the rest use TriangFourier). Empty = all blocks follow -born.
FLAG_BOOL(block_norm, false) // L2-normalize (SoftMax) each block output (deep nets)
FLAG_BOOL(token_norm, false) // per-token complex RMS pre-norm (pre-LN transformer)
FLAG_BOOL(rope, false)      // rotary position embedding on Born-attention Q,K
FLAG_FLOAT(ce_eps, 0.0)     // Born-loss target-prob floor (0 = exact; ~1e-3 stabilizes)
FLAG_FLOAT(ce_smooth, 0.0)  // Laplace-smoothed Born loss p_t=(|z_t|^2+s)/(sum+V*s): bounded,
                            // better-conditioned landscape (0 = exact Born)
FLAG_FLOAT(attn_eps, 0.0)   // Born-ATTENTION normalization floor A=s/(S+eps): bounds the 1/S
                            // gradient blow-up when a query's scores collapse (0 = exact)
FLAG_STRING(save_path, "")  // if set, rank 0 saves the trained model here after training
FLAG_BOOL(generate, false)  // generation mode: restore -model and sample text (no training)
FLAG_BOOL(eval_loss, false) // eval mode: restore -model, report exact-Born held-out loss
FLAG_STRING(eval_stream, "") // eval mode: stream this byte file through -model on the GPU in -eval_window byte
                             // windows (state carried across consecutive N-token forwards, reset per window;
                             // the MBLM 5M-byte protocol), -eval_batch windows in parallel; prints bits/byte
FLAG_INT(eval_window, 5242880) // bytes per state reset (multiple of the model context N)
FLAG_INT(eval_batch, 4)      // windows evaluated in parallel (GPU memory ~ batch x the training footprint)
FLAG_INT(eval_bytes, 0)      // bytes of -eval_stream to use (0 = whole file)
FLAG_STRING(model, "")      // model file to restore in -generate mode
FLAG_STRING(init_model, "") // warm-restart: restore these weights before training (continue a checkpoint)
FLAG_STRING(prompt, "ROMEO:")// seed text for generation
FLAG_INT(gen_len, 600)      // number of characters to generate
FLAG_BOOL(gen_gpu, false)   // generate using the GPU forward (fast Bluestein mixer) instead of CPU
FLAG_FLOAT(temp, 0.8)       // sampling temperature (lower = greedier)
FLAG_BOOL(print_net, false) // build the LM and print its layer graph, then exit
FLAG_INT(profile_steps, 0)  // >0: after 20 warm-up steps, profile this many steps per layer type and per phase, print, exit
FLAG_BOOL(tf32, false)      // TF32 tensor cores for the cuBLAS GEMMs (default strict FP32)
FLAG_BOOL(ksync, true)      // false: no host sync after every kernel in eager mode (same results, less launch latency)
FLAG_INT(seed, 1000)        // base seed for the per-rank data sampler (rank r uses seed+r)
FLAG_BOOL(emb_gauss, false) // symmetry-breaking zero-mean complex embedding init (vs default)
FLAG_STRING(val_data, "")  // optional held-out text file; when set, all of -data is training data (char mode)
FLAG_BOOL(token_lm, false)  // sub-word token LM: -data is a DIR with train.bin/val.bin (uint16 ids)
FLAG_INT(vocab_size, 8192)  // vocab in -token_lm mode (must match the BPE used to make the bins)
FLAG_STRING(nccl_id, "/tmp/cnet_shk_id")  // base path for the NCCL rendezvous id file (per DP group)

// ---------------- character dataset: memory-mapped bytes, one byte per character ----------------
// The text is mapped read-only (MAP_SHARED), so it costs no heap and the page cache holds ONE copy
// for all ranks on a node; ids are produced on the fly through a 256-entry byte -> id table. Ids keep
// the first-seen order of the old loader (the training file first, then -val_data), so existing
// models and .vocab sidecars stay compatible. Positions are 64-bit: the full PG-19 training set is
// 11.6 billion characters. Without -val_data the last 10% of the file is held out, as before.
struct ByteFile {
	const unsigned char *p = nullptr; size_t n = 0;
	bool open(const std::string &path) {
		int fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) return false;
		struct stat st; if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return false; }
		n = (size_t) st.st_size;
		void *m = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
		::close(fd);
		if (m == MAP_FAILED) { n = 0; return false; }
		p = (const unsigned char*) m;
		return true;
	}
};
struct CharData {
	ByteFile tr, va;
	std::map<char, int> c2i;
	int vocab = 0, lut[256];
	size_t ntrain = 0, total = 0;
	void load(const std::string &path, const std::string &valpath = "") {
		if (!tr.open(path)) { std::cerr << "cannot read " << path << "\n"; exit(1); }
		if (!valpath.empty() && !va.open(valpath)) { std::cerr << "cannot read " << valpath << "\n"; exit(1); }
		for (int b = 0; b < 256; ++b) lut[b] = -1;
		auto scan = [&](const ByteFile &f) {
			for (size_t i = 0; i < f.n && vocab < 256; ++i) {
				const unsigned char b = f.p[i];
				if (lut[b] < 0) { lut[b] = vocab++; c2i[(char) b] = lut[b]; }
			}
		};
		scan(tr); scan(va);
		if (va.n) { ntrain = tr.n; total = tr.n + va.n; }
		else      { ntrain = (tr.n / 10) * 9; total = tr.n; }
	}
	int at(size_t i) const { return i < tr.n ? lut[tr.p[i]] : lut[va.p[i - tr.n]]; }
	size_t size() const { return total; }
};

// ---- sidecar vocab: <model>.vocab stores the char<->id table so -generate/-eval
// are self-contained and do NOT depend on re-deriving the (insertion-order, hence
// corpus-order-dependent) char vocab from a -data file. Byte k of the file is the
// character whose id is k, so it round-trips every byte value exactly. ----
static std::string vocabPath(const std::string &model) { return model + ".vocab"; }

static bool writeVocabSidecar(const std::string &model, const std::vector<char> &i2c) {
	std::ofstream f(vocabPath(model), std::ios::binary);
	if (!f) return false;
	f.write(i2c.data(), (std::streamsize) i2c.size());
	return (bool) f;
}

// Fill i2c (id->char) and c2i (char->id) from the sidecar if present; false if none.
static bool readVocabSidecar(const std::string &model, std::vector<char> &i2c,
							 std::map<char, int> &c2i) {
	std::ifstream f(vocabPath(model), std::ios::binary);
	if (!f) return false;
	std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (s.empty()) return false;
	i2c.assign(s.begin(), s.end());
	c2i.clear();
	for (int k = 0; k < (int) i2c.size(); ++k) c2i[i2c[k]] = k;
	return true;
}

// ---- sub-word token dataset: <dir>/train.bin + <dir>/val.bin, uint16 ids ----
// train ids occupy [0, ntrain); val ids are appended at [ntrain, end), so the
// existing held-out sampler (which draws from [ntrain, size)) reads validation.
struct TokenData {
	std::vector<int> data;
	int vocab = 0; size_t ntrain = 0;
	int at(size_t i) const { return data[i]; }
	size_t size() const { return data.size(); }
	static void readBin(const std::string &p, std::vector<int> &out) {
		std::ifstream f(p, std::ios::binary);
		if (!f) { std::cerr << "cannot read " << p << "\n"; exit(1); }
		f.seekg(0, std::ios::end); std::streamoff n = f.tellg(); f.seekg(0);
		std::vector<uint16_t> buf((size_t) n / 2);
		f.read((char*) buf.data(), (std::streamsize) buf.size() * 2);
		out.reserve(out.size() + buf.size());
		for (uint16_t v : buf) out.push_back((int) v);
	}
	void load(const std::string &dir, int vocab_) {
		vocab = vocab_;
		std::vector<int> tr, va;
		readBin(dir + "/train.bin", tr);
		readBin(dir + "/val.bin", va);
		ntrain = tr.size();
		data = std::move(tr);
		data.insert(data.end(), va.begin(), va.end());
	}
};

// Uniform read-only view over CharData or TokenData (the trainer and the evaluator use only this).
struct Corpus {
	CharData ch; TokenData tk; bool tok = false;
	void load(bool token_mode, const std::string &path, const std::string &valpath, int vocab_) {
		tok = token_mode;
		if (tok) tk.load(path, vocab_); else ch.load(path, valpath);
	}
	int at(size_t i) const { return tok ? tk.at(i) : ch.at(i); }
	size_t size() const { return tok ? tk.size() : ch.size(); }
	size_t ntrain() const { return tok ? tk.ntrain : ch.ntrain; }
	int vocab() const { return tok ? tk.vocab : ch.vocab; }
	void window(size_t p, int n, std::vector<int> &out) const { out.resize(n); for (int i = 0; i < n; ++i) out[i] = at(p + i); }
};


// ------- fully complex FNet-AR language model (causal Fourier mixing) ---------
// embedding + learnable positional + L x [TriangFourier mix -> residual+CGelu ->
// position-wise FFN E->4E->E -> residual+CGelu] + per-token projection + Born-rule
// SequenceCrossEntropy. Returns the output (SequenceCrossEntropy) id.
static int buildLM(ComplexNet &net, int embId, int E, int N, int L, int vocab) {
	const int size = N * E, H = 4 * E;
	int last = embId;
	if (pos_table) {                                                        // learned absolute positions
		int pos = net.add(new CInput(OutSize(size)));
		last = net.add(new Residual(InSize(size), InSize(size)), {embId, pos});
	} else {
		// No positional table (RoPE only; the model then streams past N without limit). The GPU embedding
		// mapping expects a single consumer, so the residual stream starts from the normalized embedding.
		last = net.add(new TokenNorm(N, E), {embId});
	}
	const int dh = E / 2;   // Born-rule attention head dim (complex)
	// Per-token pre-norm (complex RMSNorm). When on, each sub-block sees a
	// normalized residual-stream input; the residual still adds to the un-normed
	// stream (standard pre-LN), which is what lets depth train stably.
	auto norm = [&](int x) { return token_norm ? net.add(new TokenNorm(N, E), {x}) : x; };
	// Hybrid mixer selection: attn_at lists block indices that use Born attention; the
	// remaining blocks use the cheap TriangFourier (Bluestein) mixer. Empty => all blocks
	// follow the global -born flag.
	std::set<int> attn_set;
	{ std::stringstream ss((std::string) attn_at); std::string tok;
	  while (std::getline(ss, tok, ',')) if (!tok.empty()) attn_set.insert(std::stoi(tok)); }
	for (int b = 0; b < L; ++b) {
		int mixout;
		bool useBorn = attn_set.empty() ? ((bool) born || (bool) gla) : (attn_set.count(b) > 0);
		if (useBorn) {                                                     // Born-rule attention
			int pre = net.add(new CGelu(InSize(size)), {norm(last)});
			int wQ = net.add(new CInput(OutSize(E * dh))); int Q  = net.add(new TokenwiseLinear(N, E, dh), {pre, wQ});
			int wK = net.add(new CInput(OutSize(E * dh))); int K  = net.add(new TokenwiseLinear(N, E, dh), {pre, wK});
			int wV = net.add(new CInput(OutSize(E * dh))); int Vv = net.add(new TokenwiseLinear(N, E, dh), {pre, wV});
			if (rope) {                                                        // RoPE on queries/keys
				Q = net.add(new RotaryEmbed(N, dh, 10000.0, heads), {Q});
				K = net.add(new RotaryEmbed(N, dh, 10000.0, heads), {K});
			}
			int att;
			if (gla) {                                                         // complex-amplitude gated linear attention
				int wG = net.add(new CInput(OutSize(E * heads))); int G = net.add(new TokenwiseLinear(N, E, heads), {pre, wG});
				int g = net.add(new GatedLinearAttention(N, dh, heads), {Q, K, Vv, G});
				((GatedLinearAttention*) net[g])->setChunk(born_chunk);
				att = net.add(new TokenNorm(N, dh), {g});                       // GLA's output norm (no normalizer in the mix)
			} else {
				att = net.add(new BornAttention(N, dh, true, attn_eps, heads), {Q, K, Vv});
				((BornAttention*) net[att])->setLinear(born_linear, born_chunk);
				((BornAttention*) net[att])->setDecay(born_decay);
			}
			int wO = net.add(new CInput(OutSize(dh * E)));
			mixout = net.add(new TokenwiseLinear(N, dh, E), {att, wO});
		} else {                                                           // causal Fourier
			mixout = net.add(new TriangFourier(InSize(size)), {norm(last)});
		}
		int r0  = net.add(new Residual(InSize(size), InSize(size)), {last, mixout});
		int n0  = net.add(new CGelu(InSize(size)), {r0});
		int w1  = net.add(new CInput(OutSize(E * H)));
		int f1  = net.add(new TokenwiseLinear(N, E, H), {norm(n0), w1});    // position-wise FFN (pre-norm)
		int g   = net.add(new CGelu(InSize(N * H)), {f1});
		int w2  = net.add(new CInput(OutSize(H * E)));
		int f2  = net.add(new TokenwiseLinear(N, H, E), {g, w2});
		int r1  = net.add(new Residual(InSize(size), InSize(size)), {n0, f2});
		last    = net.add(new CGelu(InSize(size)), {r1});
		if (block_norm) last = net.add(new SoftMax(InSize(size)), {last});   // per-block L2 norm
	}
	last       = norm(last);                                                // final pre-logits norm
	int wo     = net.add(new CInput(OutSize(E * vocab)));
	int logits = net.add(new TokenwiseLinear(N, E, vocab), {last, wo});     // per-token -> vocab
	int ce     = net.add(new SequenceCrossEntropy(InSize(N * vocab), vocab), {logits});
	return ce;
}

// Temperature sampling from a position's Born distribution p_k = |z_k|^2/||z||^2
// (SequenceCrossEntropy normalises the per-position outputs in forward()).
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

// Restore a saved model and autoregressively sample text on the CPU. The mixers
// are causal, so right-filling the N-token window and reading the distribution at
// the last real position is exact (the zero padding never influences it).
static int runGenerate() {
	FILELog::ReportingLevel() = lWarning;   // quiet per-layer restore/forward debug logs
	if (((std::string) model).empty()) { std::cerr << "-generate requires -model <file>\n"; return 1; }
	// Prefer the sidecar vocab saved next to the model (exact, corpus-independent);
	// fall back to deriving it from -data for older models that have no sidecar.
	std::vector<char> i2c; std::map<char, int> c2i;
	if (readVocabSidecar((std::string) model, i2c, c2i)) {
		std::cerr << "[vocab] loaded " << i2c.size() << " chars from "
				  << vocabPath((std::string) model) << "\n";
	} else {
		CharData ds; ds.load(data);
		i2c.assign(ds.vocab, 0);
		for (auto &kv : ds.c2i) i2c[kv.second] = kv.first;
		c2i = ds.c2i;
		std::cerr << "[vocab] no sidecar; derived " << i2c.size()
				  << " chars from -data " << (std::string) data << "\n";
	}

	CNet net;
	if (!net.restore((std::string) model)) { std::cerr << "cannot restore model: " << (std::string) model << "\n"; return 1; }
	auto *em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	auto *seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model has no embedding / sequence head\n"; return 1; }
	const int Ngen = seq->nPos(), vocab = seq->vocab();

	std::mt19937 rng(std::random_device{}());
	std::vector<int> ctx;
	for (char c : (std::string) prompt) if (c2i.count(c)) ctx.push_back(c2i[c]);
	// Stream the generation live: print the header + prompt, then flush each sampled
	// character as it is produced, so output appears incrementally (and survives an
	// early kill) instead of being buffered until the end.
	std::cout << "=== generated (model=" << model << " temp=" << temp << ") ===\n"
			  << prompt << std::flush;

#ifdef __CUDACC__
	// GPU generation: run the full forward on the device (fast Bluestein mixer) and
	// read the per-position logits straight off the device to sample. Avoids the
	// O(N^2) CPU TriangFourier reference, so a hybrid/FNet model samples in seconds.
	if (gen_gpu) {
		// batch 2 (>=1 clone required by the clone/batchToGpu path); feed the same
		// window to both examples so whichever the ancestor holds is the real one.
		net.allocateOnGpu(2);                                           // weights already restored
		std::uniform_real_distribution<double> U(0.0, 1.0);
		for (int g = 0; g < gen_len; ++g) {
			std::vector<int> win(Ngen, 0);
			int ctxn = (int) ctx.size(), take = std::min(ctxn, Ngen);
			for (int i = 0; i < take; ++i) win[i] = ctx[ctxn - take + i];
			EmbeddingBatch b(2, Ngen);
			b.add(win, 0); b.addTargets(std::vector<int>(Ngen, 0));     // targets unused for sampling
			b.add(win, 0); b.addTargets(std::vector<int>(Ngen, 0));
			net.gpuForward(em, seq, b);
			int nxt = net.sampleSeqPos(seq, take - 1, vocab, temp, U(rng));
			ctx.push_back(nxt);
			std::cout << i2c[nxt] << std::flush;
		}
		std::cout << std::endl;
		return 0;
	}
#endif

	for (int g = 0; g < gen_len; ++g) {
		std::vector<int> win(Ngen, 0);
		int ctxn = (int) ctx.size(), take = std::min(ctxn, Ngen);
		for (int i = 0; i < take; ++i) win[i] = ctx[ctxn - take + i];   // real tokens at 0..take-1
		em->setInput(win);
		net.cpuNet().forward();
		int nxt = sampleNext(seq, take - 1, vocab, temp, rng);           // next after last real token
		ctx.push_back(nxt);
		std::cout << i2c[nxt] << std::flush;                             // stream one char at a time
	}
	std::cout << std::endl;
	return 0;
}

// Restore a trained model and report the EXACT Born-rule held-out loss (nats/token)
// over eval_k random validation windows. CPU forward uses the exact Born probability
// regardless of how the model was trained (smoothed or not), so this is the fair,
// comparable number. -eval_loss -model <file> -data <dir/txt> [-token_lm ...].
static int runEval() {
	FILELog::ReportingLevel() = lWarning;
	if (((std::string) model).empty()) { std::cerr << "-eval_loss requires -model <file>\n"; return 1; }
	Corpus corpus; corpus.load(token_lm, data, val_data, vocab_size);
	CNet net;
	if (!net.restore((std::string) model)) { std::cerr << "cannot restore model: " << (std::string) model << "\n"; return 1; }
	auto *em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	auto *seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model missing embedding / sequence head\n"; return 1; }
	seq->setEps(0); seq->setSmooth(0);                    // exact Born for evaluation
	const int Nw = seq->nPos();
	const long long lo = (long long) corpus.ntrain(), hi = (long long) corpus.size() - Nw - 2;
	if (hi <= lo) { std::cerr << "not enough validation tokens\n"; return 1; }
	std::mt19937_64 rng(777);
	std::uniform_int_distribution<long long> U(lo, hi);
	int K = gen_len > 0 ? gen_len : 500;                  // reuse -gen_len as eval window count
	double tot = 0;
	for (int k = 0; k < K; ++k) {
		const long long p = U(rng);
		std::vector<int> win, tgt; corpus.window((size_t) p, Nw, win); corpus.window((size_t) p + 1, Nw, tgt);
		em->setInput(win); seq->setTargets(tgt);
		net.cpuNet().forward();
		tot += seq->loss(0);
	}
	std::cout << "EXACT-BORN val = " << (tot / K) << " nats/token  over " << K
			  << " windows (model=" << (std::string) model << ")" << std::endl;
	return 0;
}

// Windowed streaming evaluation on the GPU: the exact recurrence of the CPU twin (impl/stepnet.h),
// N tokens at a time. Each gated-linear-attention layer carries its state across the consecutive
// forwards of a window (GatedLinearAttentionGpu::setCarry) and RoPE uses the absolute byte position,
// so a 5,242,880-byte window is 1,280 forwards of N=4096 with one state reset at the start. Only
// whole N-token blocks are scored (a window's trailing remainder is dropped and reported). Bytes
// outside the model's vocabulary are fed as id 0, their own positions are not scored, and they are
// charged 8 bits each in bpb_all.
static int runEvalStream() {
	FILELog::ReportingLevel() = lWarning;
	if (((std::string) model).empty()) { std::cerr << "-eval_stream requires -model <file>\n"; return 1; }
	std::vector<char> i2c; std::map<char, int> c2i;
	if (!readVocabSidecar((std::string) model, i2c, c2i)) { std::cerr << "no vocab sidecar next to " << (std::string) model << "\n"; return 1; }
	std::ifstream f((std::string) eval_stream, std::ios::binary);
	if (!f) { std::cerr << "cannot read " << (std::string) eval_stream << "\n"; return 1; }
	std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (eval_bytes > 0 && (size_t) eval_bytes < buf.size()) buf.resize(eval_bytes);

	CNet net;
	if (!net.restore((std::string) model)) { std::cerr << "cannot restore model: " << (std::string) model << "\n"; return 1; }
	auto *em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	auto *seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model missing embedding / sequence head\n"; return 1; }
	seq->setEps(0); seq->setSmooth(0);                       // exact Born probabilities
	const int N = seq->nPos();
	const long W = eval_window;
	if (W % N != 0) { std::cerr << "-eval_window must be a multiple of the model context N=" << N << "\n"; return 1; }
	// The linear (chunkwise density-matrix) form is the one with a carried state; it computes the same
	// operator as the quadratic kernel, so restored Born layers are switched to it here (chunk = -born_chunk).
	for (CFunc *fn : net.cpuNet().funcs())
		if (auto *b = dynamic_cast<BornAttention*>(fn)) {
			if (N % born_chunk != 0) { std::cerr << "eval_stream: -born_chunk must divide the context N=" << N << "\n"; return 1; }
			b->setLinear(true, born_chunk);
		}
	const long NWtot = (long) (buf.size() + W - 1) / W;        // windows, the last one possibly partial
	const int Bn = std::max(2, (int) std::min<long>(eval_batch, NWtot));   // the batched GPU path needs >= 1 clone
	if (!net.allocateOnGpu(Bn)) { std::cerr << "GPU allocation failed\n"; return 1; }
	std::vector<GatedLinearAttentionGpu*> glas; std::vector<BornAttentionGpu*> borns; std::vector<RotaryEmbed*> ropes;
	for (CFunc *fn : net.cpuNet().funcs()) {
		if (dynamic_cast<GatedLinearAttention*>(fn)) { auto *g = (GatedLinearAttentionGpu*) net.getMappingFor(fn); g->setCarry(true); glas.push_back(g); }
		if (dynamic_cast<BornAttention*>(fn)) { auto *b = (BornAttentionGpu*) net.getMappingFor(fn); b->setCarry(true); borns.push_back(b); }
		if (auto *r = dynamic_cast<RotaryEmbed*>(fn)) ropes.push_back(r);
	}
	auto *sce = (SequenceCrossEntropyGpu*) net.getMappingFor(seq);
	auto idOf = [&](long b) { auto it = c2i.find(buf[b]); return it == c2i.end() ? -1 : it->second; };

	double nll_total = 0; long scored_total = 0, skipped_total = 0, dropped_total = 0;
	std::vector<float> losses;
	for (long g = 0; g < NWtot; g += Bn) {
		const int nw = (int) std::min<long>(Bn, NWtot - g);
		std::vector<long> wbytes(Bn, 0), nblk(Bn, 0);
		long maxblk = 0;
		for (int i = 0; i < nw; ++i) {
			wbytes[i] = std::min<long>(W, (long) buf.size() - (g + i) * W);
			nblk[i] = wbytes[i] / N;                              // whole blocks only
			dropped_total += wbytes[i] - nblk[i] * N - (nblk[i] * N < wbytes[i] ? 1 : 0);
			maxblk = std::max(maxblk, nblk[i]);
		}
		for (auto *gl : glas) gl->resetCarry();
		for (auto *b : borns) b->resetCarry();
		std::vector<double> wnll(Bn, 0); std::vector<long> wsc(Bn, 0), wsk(Bn, 0);
		for (long j = 0; j < maxblk; ++j) {
			for (auto *r : ropes) r->setPos((int) (j * N));
			EmbeddingBatch b(Bn, N);
			for (int i = 0; i < Bn; ++i) {
				std::vector<int> win(N, 0); int lab = 0;
				if (i < nw && j < nblk[i]) {
					const long base = (g + i) * W + j * N;
					for (int t = 0; t < N; ++t) { int id = idOf(base + t); win[t] = id < 0 ? 0 : id; }
					if (j * N + N < wbytes[i]) { int id = idOf(base + N); lab = id < 0 ? 0 : id; }
				}
				b.add(win, lab);
				std::vector<int> tg(N, 0);                            // per-position targets: next token, label last
				for (int t = 0; t + 1 < N; ++t) tg[t] = win[t + 1];
				tg[N - 1] = lab;
				b.addTargets(tg);
			}
			net.gpuForward(em, seq, b);
			sce->readLosses(losses);
			for (int i = 0; i < nw; ++i) {
				if (j >= nblk[i]) continue;
				const long base = (g + i) * W + j * N;
				for (int t = 0; t < N; ++t) {
					const long tb = j * N + t + 1;                    // target byte, window-relative
					if (tb >= wbytes[i]) continue;                     // no next byte in this window
					if (tb >= nblk[i] * N + 1) continue;               // beyond the scored blocks
					if (idOf(base + t + 1) < 0) { wsk[i]++; continue; }
					wnll[i] += losses[(size_t) i * N + t]; wsc[i]++;
				}
			}
		}
		for (int i = 0; i < nw; ++i) {
			const double bits = wnll[i] / std::log(2.0);
			std::cout << "WINDOW " << (g + i) << " bytes=" << wbytes[i] << " blocks=" << nblk[i] << " scored=" << wsc[i]
					  << " skipped=" << wsk[i] << " nats/char=" << (wsc[i] ? wnll[i] / wsc[i] : 0.0)
					  << " bits/char=" << (wsc[i] ? bits / wsc[i] : 0.0) << " sum_nll=" << wnll[i] << std::endl;
			nll_total += wnll[i]; scored_total += wsc[i]; skipped_total += wsk[i];
		}
	}
	const double bits = nll_total / std::log(2.0);
	const long all = scored_total + skipped_total;
	std::cout << "EVAL_STREAM file=" << (std::string) eval_stream << " bytes=" << buf.size() << " window=" << W << " N=" << N
			  << " windows=" << NWtot << " scored=" << scored_total << " skipped=" << skipped_total << " dropped=" << dropped_total
			  << " nats/char=" << (scored_total ? nll_total / scored_total : 0.0)
			  << " bits/char=" << (scored_total ? bits / scored_total : 0.0)
			  << " bpb_all=" << (all ? (bits + 8.0 * skipped_total) / all : 0.0) << " sum_nll=" << nll_total << std::endl;
	return 0;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
#ifdef __CUDACC__
	g_cnet_nosync = !ksync;
	g_cnet_tf32 = tf32;
#endif
	if (!((std::string) eval_stream).empty()) return runEvalStream();   // GPU windowed streaming (bits/byte)
	if (eval_loss) return runEval();              // restore a model, report exact-Born val
	if (generate) return runGenerate();          // standalone sampling, no training / NCCL
	if (print_net) {                              // dump the layer graph, no training / NCCL
		int vcb;
		if (token_lm) vcb = vocab_size;
		else { CharData ds; ds.load(data, val_data); vcb = ds.vocab; }
		CNet net;
		int embId = net.cpuNet().add(new CEmbedding(emb, tokens, vcb));
		buildLM(net.cpuNet(), embId, emb, tokens, blocks, vcb);
		std::cout << "config: born=" << born << " token_norm=" << token_norm
				  << " rope=" << rope << " pos_table=" << pos_table << " block_norm=" << block_norm
				  << "  E=" << emb << " N=" << tokens << " L=" << blocks
				  << " vocab=" << vcb << "\n\n"
				  << net.cpuNet().toString();
		return 0;
	}
	const int W = world, R = rank, N = tokens, E = emb, L = blocks;
	if (W < 1 || batch % W != 0) {
		std::cerr << "batch (" << batch << ") must be a positive multiple of world (" << W << ")\n";
		return 1;
	}
	const int per = batch / W;
	int nDev = 0; cudaGetDeviceCount(&nDev); if (nDev < 1) nDev = 1;
	cudaSetDevice(R % nDev);

	// --- NCCL rendezvous: rank 0 writes the unique id to /tmp; others poll. ---
	// The id-file base is a flag so independent DP groups can run concurrently
	// (e.g. a Born run on GPUs 0-3 and a Fourier run on 4-7 at the same time).
	std::string idf_s = (std::string) nccl_id + ".bin", rdy_s = (std::string) nccl_id + ".ready";
	const char *idf = idf_s.c_str(), *rdy = rdy_s.c_str();
	ncclUniqueId id;
	if (R == 0) {
		ncclGetUniqueId(&id);
		{ std::ofstream f(idf, std::ios::binary); f.write((char*) &id, sizeof(id)); }
		{ std::ofstream r(rdy); r << "1"; }
	} else {
		int t = 0;
		while (!std::ifstream(rdy).good()) { usleep(50000); if (++t > 2400) { std::cerr << "rank " << R << ": id timeout\n"; return 1; } }
		usleep(100000);
		std::ifstream f(idf, std::ios::binary); f.read((char*) &id, sizeof(id));
	}
	ncclComm_t comm;
	if (ncclCommInitRank(&comm, W, id, R) != ncclSuccess) { std::cerr << "rank " << R << ": nccl init failed\n"; return 1; }

	Corpus corpus; corpus.load(token_lm, data, val_data, vocab_size);
	const int vocab = corpus.vocab();
	std::vector<char> charVocab;   // id->char for char mode, persisted as the model's .vocab sidecar
	if (!token_lm) { charVocab.assign(vocab, 0); for (auto &kv : corpus.ch.c2i) charVocab[kv.second] = kv.first; }
	// Persist the char<->id table beside the model so -generate/-eval are self-contained
	// (char mode only; token_lm uses an external BPE vocab). Written once by rank 0.
	if (R == 0 && !token_lm && !((std::string) save_path).empty()) {
		if (writeVocabSidecar((std::string) save_path, charVocab))
			std::cout << "[vocab] wrote " << charVocab.size() << " chars to "
					  << vocabPath((std::string) save_path) << "\n";
		else
			std::cerr << "WARNING: could not write vocab sidecar "
					  << vocabPath((std::string) save_path) << "\n";
	}
	if (R == 0)
		std::cout << (token_lm ? "token-LM: tokens=" : "char LM: chars=") << corpus.size() << " (train " << corpus.ntrain() << ") vocab=" << vocab
				  << "  model E=" << E << " N=" << N << " L=" << L
				  << "  global_batch=" << batch << " world=" << W
				  << "  born=" << born << " heads=" << heads << " attn_at=[" << (std::string) attn_at << "]"
				  << " block_norm=" << block_norm
				  << " token_norm=" << token_norm << " rope=" << rope
				  << "  grad_clip=" << grad_clip << " ce_eps=" << ce_eps
				  << "  chance=ln(vocab)=" << std::log((double) vocab) << std::endl;

	CNet net;
	CEmbedding *em; SequenceCrossEntropy *seq;
	if (!((std::string) init_model).empty()) {
		// Warm-restart: restore() REBUILDS the net (layers + weights) from the checkpoint, so we
		// skip buildLM/init_inputs here. Only weights are restored (not Adam moments / LR), so
		// expect a brief re-warm bump. Architecture flags must match the checkpoint.
		if (!net.restore((std::string) init_model)) {
			std::cerr << "rank " << R << ": cannot restore -init_model " << (std::string) init_model << "\n";
			return 1;
		}
		em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
		seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
		if (!em || !seq) { std::cerr << "rank " << R << ": -init_model missing embedding/seq head\n"; return 1; }
		em->setIsMainInput(true);
		seq->setIsMainOutput(true);
		seq->setEps(ce_eps);
		seq->setSmooth(ce_smooth);
		if (R == 0) std::cout << "[resume] restored weights from " << (std::string) init_model << "\n";
	} else {
		int embId = net.cpuNet().add(new CEmbedding(E, N, vocab));
		int ceId  = buildLM(net.cpuNet(), embId, E, N, L, vocab);
		((CEmbedding*) net.cpuNet()[embId])->setIsMainInput(true);
		((SequenceCrossEntropy*) net.cpuNet()[ceId])->setIsMainOutput(true);
		((SequenceCrossEntropy*) net.cpuNet()[ceId])->setEps(ce_eps);  // 0 = exact Born
		((SequenceCrossEntropy*) net.cpuNet()[ceId])->setSmooth(ce_smooth);  // Laplace-smoothed Born
		net.cpuNet().init_inputs(1234);                // identical init on every rank
		// Optional symmetry-breaking embedding init: zero-mean complex (real AND imag
		// random), giving DISTINCT per-token embeddings so the model does not start pinned at
		// the unigram saddle. Deterministic across ranks (fixed seed).
		if (emb_gauss) {
			CEmbedding *e0 = (CEmbedding*) net.cpuNet()[embId];
			std::mt19937 er(98765);
			std::uniform_real_distribution<float> U(-1.f, 1.f);
			float s = 1.f / std::sqrt((float) E);
			for (int i = 0; i < e0->input().length_; ++i) {
				e0->mutable_input()->real_[i] = U(er) * s;
				e0->mutable_input()->imag_[i] = U(er) * s;
			}
		}
		em  = (CEmbedding*) net.cpuNet()[embId];
		seq = (SequenceCrossEntropy*) net.cpuNet()[ceId];
	}
	net.allocateOnGpu(per);

	// Ancestor-gradient buffers coalesced into one scratch -> a single all-reduce.
	auto gradBufs = net.ancestorGradBuffers();
	std::vector<int> off(gradBufs.size());
	int total = 0;
	for (size_t i = 0; i < gradBufs.size(); ++i) { off[i] = total; total += gradBufs[i].second; }
	float *scratch = nullptr; cudaMalloc(&scratch, (size_t) total * sizeof(float));
	// cuBLAS handle for the global gradient-norm clip (operates on the coalesced
	// all-reduced gradient in `scratch`).
	cublasHandle_t gbh = 0;
	if (grad_norm_clip > 0.f) { cublasCreate(&gbh); cublasSetStream(gbh, cudaStreamPerThread); cublasSetPointerMode(gbh, CUBLAS_POINTER_MODE_HOST); }

	std::mt19937_64 rng(seed + R);                 // each rank draws a different data shard (seed flag); 64-bit positions
	auto targetsOf = [&](const std::vector<int> &win, int lab) {
		std::vector<int> t(N, 0);
		for (int i = 0; i + 1 < (int) win.size(); ++i) t[i] = win[i + 1];
		if (!win.empty()) t[win.size() - 1] = lab;
		return t;
	};
	auto buildBatch = [&](bool heldout) {
		EmbeddingBatch b(per, N);
		const long long lo = heldout ? (long long) corpus.ntrain() : 0;
		const long long hi = (heldout ? (long long) corpus.size() : (long long) corpus.ntrain()) - N - 2;
		std::uniform_int_distribution<long long> d(lo, hi);
		for (int i = 0; i < per; ++i) {
			const long long p = d(rng);
			std::vector<int> win; corpus.window((size_t) p, N, win);
			int lab = corpus.at((size_t) p + N);
			b.add(win, lab);
			b.addTargets(targetsOf(win, lab));
		}
		return b;
	};
	auto lr_at = [&](int t) -> float {
		if (t < warmup) return lr * (float) (t + 1) / warmup;
		float p = (float) (t - warmup) / std::max(1, steps - warmup);
		return min_lr + 0.5f * (lr - min_lr) * (1.f + std::cos(3.14159265f * p));
	};

	const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	StopWatch sw; float acc = 0; int accn = 0;
	float best_val = 1e30f;   // best held-out loss seen, for best-checkpoint saving
	for (int t = 1; t <= steps; ++t) {
		float gnorm = 0.f;
		const bool prof = profile_steps > 0 && t > 20 && t <= 20 + profile_steps;
		cnetProfiler().on = prof;
		static double ph[6] = {0, 0, 0, 0, 0, 0};
		auto phase = [&](int k, std::chrono::steady_clock::time_point &t0) {
			if (!prof) return;
			cudaDeviceSynchronize();
			auto t1 = std::chrono::steady_clock::now();
			ph[k] += std::chrono::duration<double, std::milli>(t1 - t0).count(); t0 = t1;
		};
		auto tp = std::chrono::steady_clock::now();
		if (prof) cudaDeviceSynchronize();
		EmbeddingBatch batch_ = buildBatch(false);
		phase(0, tp);
		net.gpuForward(em, seq, batch_);
		phase(1, tp);
		acc += net.getLoss(0)[seq->uid()]; ++accn;
		phase(2, tp);
		net.gpuBackward();
		phase(3, tp);
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // gather
			cudaMemcpyAsync(scratch + off[i], gradBufs[i].first,
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		ncclAllReduce(scratch, scratch, total, ncclFloat, ncclSum, comm, cudaStreamPerThread);
		// Global gradient-NORM clip on the summed gradient (same on every rank).
		if (grad_norm_clip > 0.f) {
			cublasSnrm2(gbh, total, scratch, 1, &gnorm);
			if (gnorm > grad_norm_clip) {
				float s = grad_norm_clip / gnorm;
				cublasSscal(gbh, total, &s, scratch, 1);
			}
		}
		for (size_t i = 0; i < gradBufs.size(); ++i)                      // scatter
			cudaMemcpyAsync(gradBufs[i].first, scratch + off[i],
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		phase(4, tp);
		net.trueAdamUpdate(lr_at(t), b1, b2, eps, t, grad_clip);
		phase(5, tp);
		if (profile_steps > 0 && t == 20 + profile_steps && R == 0) {
			const char *nm[6] = {"batch build (CPU)", "forward", "loss readback", "backward + grad from clones", "grad gather / all-reduce / clip", "Adam update"};
			double T = 0; for (double v : ph) T += v;
			printf("PHASE per step over %d steps (ms):\n", (int) profile_steps);
			for (int k = 0; k < 6; ++k) printf("PHASE %-32s %9.2f %5.1f%%\n", nm[k], ph[k] / profile_steps, 100.0 * ph[k] / T);
			printf("PHASE %-32s %9.2f\n", "TOTAL", T / profile_steps);
			cnetProfiler().print((int) profile_steps);
			fflush(stdout);
			return 0;
		}

		if (R == 0 && t % 100 == 0) {
			std::cout << "step " << t << "  train " << (acc / accn) << " nats/char  lr "
					  << lr_at(t) << "  |g|=" << gnorm << "  "
					  << (sw.ElapsedTimeMicros() / 1000.0 / 100) << " ms/step"
					  << std::endl;
			acc = 0; accn = 0; sw.Reset();
		}
		if (R == 0 && val_every > 0 && t % val_every == 0) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			float vv = v / K;
			std::cout << "  [val] step " << t << "  val " << vv << " nats/char" << std::endl;
			// Save the BEST checkpoint so a later Born-loss spike can't wipe out the
			// run's best model (the loss is spike-prone at large vocab).
			if (!((std::string) save_path).empty() && vv < best_val) {
				best_val = vv;
				if (net.getInputsFromGpu() && net.save((std::string) save_path))
					std::cout << "    [ckpt] new best val " << vv << " -> saved " << (std::string) save_path << std::endl;
			}
			sw.Reset();
		}
	}
	cudaStreamSynchronize(cudaStreamPerThread);

	// Rank 0 checkpoints the final model (weights pulled GPU->CPU first). Every rank
	// holds identical weights, so one save suffices. Generate later with
	// -generate -model <save_path>. When held-out validation is on, best-val
	// checkpoints were already saved during training; only overwrite with the final
	// model if it is at least as good (don't let a late spike clobber the best).
	if (R == 0 && !((std::string) save_path).empty()) {
		bool do_save = true;
		if (val_every > 0 && best_val < 1e29f) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			float vv = v / K;
			do_save = (vv <= best_val);
			std::cout << "final val " << vv << " (best " << best_val << "); "
					  << (do_save ? "saving final" : "keeping best checkpoint") << std::endl;
		}
		if (do_save) {
			if (net.getInputsFromGpu() && net.save((std::string) save_path))
				std::cout << "saved model to " << (std::string) save_path << std::endl;
			else
				std::cerr << "WARNING: model save to " << (std::string) save_path << " failed\n";
		}
	}

	cudaFree(scratch);
	ncclCommDestroy(comm);
	if (R == 0) { remove(idf); remove(rdy); }
	return 0;
}
