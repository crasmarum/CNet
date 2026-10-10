// =============================================================================
// gollm: a Go move-prediction language model built with the CNet complex-valued
// framework -- the C++ counterpart of the PyTorch go_gen2.py GPT. Moves are
// tokenised exactly as in the Python scripts:
//
//   id = color*361 + 19*(col-'a') + (row-'a')   (0..360 black, 361..721 white)
//   "222" (game start) -> 722,  so vocab = 723.
//
// The model is CNet's native complex transformer (Born-rule attention + RoPE +
// per-token norm; CNet has no softmax attention), trained with the Born-rule
// SequenceCrossEntropy. Data is a directory of uint16 move-token bins produced by
// data/prep_go.py (train.bin / val.bin).
//
// Data parallelism (optional, needs NCCL): one process per GPU, grads cross-reduced
// each step. Build WITHOUT NCCL for a single-GPU box (see the Makefile `gollm`
// target / NCCL=0), which forces -world 1.
//
// Build:   make gollm            (multi-GPU, needs NCCL)
//          make gollm NCCL=0     (single-GPU, no NCCL)
//
// Train:   ./gollm -data ~/godata -steps 5000 -save_path ~/go.mod
// Play:    ./gollm -generate -model ~/go.mod -sgf_path game.sgf
// Predict: ./gollm -predict  -model ~/go.mod -moves 2220qp1dq -topk 5
// =============================================================================
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <unistd.h>
#ifdef WITH_NCCL
#include <nccl.h>
#endif
#include <cublas_v2.h>
#include <curand.h>

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

// ---- Go move vocabulary (must match data/prep_go.py and the Python scripts) ----
static const int GO_BOARD = 19;
static const int GO_VOCAB = 723;   // 19*19 black + 19*19 white + 1 start token
static const int GO_START = 722;   // "222"

FLAG_INT(rank, 0)            // this process's rank in [0, world)
FLAG_INT(world, 1)          // number of data-parallel processes (one per GPU)
FLAG_INT(batch, 64)         // GLOBAL batch, split across ranks (per-rank = batch/world)
FLAG_INT(emb, 384)          // embedding width E   (go_gen2.py n_embd=384)
FLAG_INT(tokens, 128)       // context length N    (go_gen2.py block_size=128)
FLAG_INT(blocks, 6)         // number of transformer blocks L  (go_gen2.py n_layer=6)
FLAG_INT(steps, 5000)       // optimizer steps
FLAG_INT(warmup, 200)       // linear-warmup steps
FLAG_INT(val_every, 200)    // validation period (0 = off)
FLAG_FLOAT(lr, 3e-4)        // peak learning rate  (go_gen2.py 3e-4)
FLAG_FLOAT(min_lr, 3e-5)    // cosine floor
FLAG_FLOAT(grad_clip, 0.0)  // true-Adam clip-by-value per grad component (0 = off)
FLAG_FLOAT(grad_norm_clip, 0.0) // global grad-norm clip on the all-reduced grad (0 = off)
FLAG_FLOAT(grad_noise, 0.0) // if >0: add decaying Gaussian noise N(0, (grad_noise/(1+t)^0.55)^2)
                            // to the gradient each step (Neelakantan et al.). Symmetry-breaking
                            // noise that escapes the Born unigram saddle independent of the GEMM
                            // summation order -- i.e. can let cuBLAS (tw_cublas true) train too.
FLAG_STRING(data, "godata") // directory with train.bin / val.bin (uint16 move ids)
FLAG_INT(vocab_size, GO_VOCAB)  // vocabulary size (keep at 723 for 19x19 Go)
FLAG_BOOL(born, true)       // Born-rule attention mixer (false = causal Fourier)
FLAG_BOOL(block_norm, false)// L2-normalize (SoftMax) each block output
FLAG_BOOL(token_norm, true) // per-token complex RMS pre-norm (pre-LN transformer)
FLAG_BOOL(rope, true)       // rotary position embedding on Born-attention Q,K
FLAG_FLOAT(ce_eps, 0.0)     // Born-loss target-prob floor (0 = exact; ~1e-3 stabilizes)
FLAG_FLOAT(ce_smooth, 0.0)  // Laplace-smoothed Born loss (0 = exact Born)
FLAG_FLOAT(attn_eps, 0.0)   // Born-attention normalization floor (0 = exact)
FLAG_STRING(save_path, "")  // if set, rank 0 saves the trained model here
FLAG_BOOL(generate, false)  // generation mode: restore -model and play a game -> SGF
FLAG_BOOL(predict, false)   // predict mode: restore -model, print top-k next moves for -moves
FLAG_BOOL(eval_loss, false) // eval mode: restore -model, report exact-Born held-out loss
FLAG_STRING(model, "")      // model file to restore in -generate / -predict / -eval_loss
FLAG_STRING(moves, "222")   // seed game for -generate / context for -predict (3-char tokens)
FLAG_STRING(sgf_path, "gollm_game.sgf")  // where -generate writes the SGF
FLAG_INT(gen_len, 200)      // moves to generate (-generate) / eval windows (-eval_loss)
FLAG_INT(topk, 5)           // number of candidate moves to print in -predict
FLAG_BOOL(legal_moves, true)// -generate/-predict: forbid the 222 start token and any move onto an
                            // already-occupied point (occupancy only; does not model captures/ko)
FLAG_FLOAT(dropout, 0.0)    // complex dropout prob on each sub-layer output (0 = off). Structured
                            // multiplicative noise; a candidate symmetry-breaker for the Born
                            // unigram saddle (active on GPU/training; identity at generate/eval).
FLAG_FLOAT(temp, 0.8)       // sampling temperature (lower = greedier)
FLAG_BOOL(print_net, false) // build the LM and print its layer graph, then exit
FLAG_INT(seed, 1000)        // base seed for the per-rank data sampler (rank r uses seed+r)
FLAG_BOOL(emb_gauss, false) // symmetry-breaking zero-mean complex embedding init
FLAG_FLOAT(gauss_init, 0.0) // if >0: circularly-symmetric complex Gaussian init (this scale)
                            // for ALL weights, replacing the default +imag-biased init. Removes
                            // the Born-loss unigram-saddle knife-edge so cuBLAS (not just the
                            // element-wise kernel) escapes it. 0 = default init.
FLAG_STRING(nccl_id, "/tmp/cnet_gollm_id")  // base path for the NCCL rendezvous id file

// ---- move <-> token id (identical to go_gen2.py encode()/decode()) ----
static std::string goDecode(int x) {
	if (x == GO_START) return "222";
	std::string r;
	r += (x >= 361) ? '1' : '0';
	if (x >= 361) x -= 361;
	r += (char) ('a' + x / GO_BOARD);
	r += (char) ('a' + x % GO_BOARD);
	return r;
}
static int goEncode(const std::string &t) {
	if (t == "222") return GO_START;
	return (t[0] - '0') * 361 + GO_BOARD * (t[1] - 'a') + (t[2] - 'a');
}
// Board point (0..360 = 19*col+row) a move token lands on; -1 for the 722 start.
// Black (0..360) and white (361..721) share one point space, so occupancy is
// per-point regardless of color.
static int goPoint(int id) {
	if (id == GO_START) return -1;
	return (id >= 361) ? id - 361 : id;
}

// Parse a run of 3-char tokens into ids (ignores a trailing partial group).
static std::vector<int> goParse(const std::string &s) {
	std::vector<int> ids;
	for (size_t i = 0; i + 3 <= s.size(); i += 3) ids.push_back(goEncode(s.substr(i, 3)));
	return ids;
}
// Build an SGF (19x19) from a token sequence; skips the 722 start tokens.
static std::string movesToSgf(const std::vector<int> &ids) {
	std::string sgf = "(;SZ[19];";
	for (int id : ids) {
		if (id == GO_START) continue;
		std::string d = goDecode(id);
		sgf += (d[0] == '0') ? "B[" : "W[";
		sgf += d[1]; sgf += d[2]; sgf += "];";
	}
	sgf += ")";
	return sgf;
}

// ---- move-token dataset: <dir>/train.bin + <dir>/val.bin, uint16 ids ----
struct TokenData {
	std::vector<int> data;
	int vocab = 0, ntrain = 0;
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
		ntrain = (int) tr.size();
		data = std::move(tr);
		data.insert(data.end(), va.begin(), va.end());
	}
};

// ------- complex transformer LM (Born attention + RoPE + token-norm) ---------
// embedding + learnable positional + L x [ (Born attn | causal Fourier) ->
// residual -> position-wise FFN E->4E->E -> residual ] + per-token projection +
// Born-rule SequenceCrossEntropy. Returns the SequenceCrossEntropy id.
static int buildLM(ComplexNet &net, int embId, int E, int N, int L, int vocab) {
	const int size = N * E, H = 4 * E;
	int pos  = net.add(new CInput(OutSize(size)));
	int last = net.add(new Residual(InSize(size), InSize(size)), {embId, pos});
	const int dh = E / 2;   // Born-rule attention head dim (complex)
	auto norm = [&](int x) { return token_norm ? net.add(new TokenNorm(N, E), {x}) : x; };
	auto drop = [&](int x) { return dropout > 0.f ? net.add(new ComplexDropout(InSize(size), dropout), {x}) : x; };
	for (int b = 0; b < L; ++b) {
		int mixout;
		if (born) {                                                        // Born-rule attention
			int pre = net.add(new CGelu(InSize(size)), {norm(last)});
			int wQ = net.add(new CInput(OutSize(E * dh))); int Q  = net.add(new TokenwiseLinear(N, E, dh), {pre, wQ});
			int wK = net.add(new CInput(OutSize(E * dh))); int K  = net.add(new TokenwiseLinear(N, E, dh), {pre, wK});
			int wV = net.add(new CInput(OutSize(E * dh))); int Vv = net.add(new TokenwiseLinear(N, E, dh), {pre, wV});
			if (rope) {
				Q = net.add(new RotaryEmbed(N, dh), {Q});
				K = net.add(new RotaryEmbed(N, dh), {K});
			}
			int att = net.add(new BornAttention(N, dh, true, attn_eps), {Q, K, Vv});
			int wO = net.add(new CInput(OutSize(dh * E)));
			mixout = net.add(new TokenwiseLinear(N, dh, E), {att, wO});
		} else {                                                           // causal Fourier
			mixout = net.add(new TriangFourier(InSize(size)), {norm(last)});
		}
		mixout = drop(mixout);                                             // dropout on attention output
		int r0  = net.add(new Residual(InSize(size), InSize(size)), {last, mixout});
		int n0  = net.add(new CGelu(InSize(size)), {r0});
		int w1  = net.add(new CInput(OutSize(E * H)));
		int f1  = net.add(new TokenwiseLinear(N, E, H), {norm(n0), w1});
		int g   = net.add(new CGelu(InSize(N * H)), {f1});
		int w2  = net.add(new CInput(OutSize(H * E)));
		int f2  = net.add(new TokenwiseLinear(N, H, E), {g, w2});
		f2 = drop(f2);                                                     // dropout on FFN output
		int r1  = net.add(new Residual(InSize(size), InSize(size)), {n0, f2});
		last    = net.add(new CGelu(InSize(size)), {r1});
		if (block_norm) last = net.add(new SoftMax(InSize(size)), {last});
	}
	last       = norm(last);
	int wo     = net.add(new CInput(OutSize(E * vocab)));
	int logits = net.add(new TokenwiseLinear(N, E, vocab), {last, wo});
	int ce     = net.add(new SequenceCrossEntropy(InSize(N * vocab), vocab), {logits});
	return ce;
}

// Temperature sampling from a position's Born distribution p_k = |z_k|^2/||z||^2.
// If occ != nullptr, forbid the start token and any move onto an occupied point
// (occ is indexed by board point, size 19*19).
static int sampleNext(SequenceCrossEntropy *seq, int p, int vocab, double temp,
					  std::mt19937 &rng, const std::vector<char> *occ) {
	std::vector<double> q(vocab); double Z = 0;
	for (int k = 0; k < vocab; ++k) {
		double pr;
		int pt = goPoint(k);
		if (occ && (pt < 0 || (*occ)[pt])) {      // start token, or point already played
			pr = 0.0;
		} else {
			pr = seq->getProbability(p, k);
			pr = std::pow(pr < 1e-12 ? 1e-12 : pr, 1.0 / temp);
		}
		q[k] = pr; Z += pr;
	}
	if (Z <= 0) return GO_START;                  // board full / everything masked
	std::uniform_real_distribution<double> U(0.0, 1.0);
	double c = U(rng) * Z, s = 0;
	for (int k = 0; k < vocab; ++k) { s += q[k]; if (c <= s) return k; }
	return vocab - 1;
}

// Restore a model and return (embedding, sequence-head, N, vocab); exits on error.
static bool restoreLM(CNet &net, CEmbedding *&em, SequenceCrossEntropy *&seq, int &N, int &vocab) {
	FILELog::ReportingLevel() = lWarning;
	if (((std::string) model).empty()) { std::cerr << "need -model <file>\n"; return false; }
	if (!net.restore((std::string) model)) { std::cerr << "cannot restore model: " << (std::string) model << "\n"; return false; }
	em  = (CEmbedding*) net.cpuNet().findFirstOfType(isEmbedding);
	seq = (SequenceCrossEntropy*) net.cpuNet().findFirstOfType(isSeqCrossEntropy);
	if (!em || !seq) { std::cerr << "restored model missing embedding / sequence head\n"; return false; }
	N = seq->nPos(); vocab = seq->vocab();
	return true;
}

// Fill an N-token window with the last <=N context tokens (real at 0..take-1) and
// forward; returns the position index whose output distribution predicts the next
// move (take-1). Zero-padding never influences it (the mixers are causal).
static int forwardContext(CNet &net, CEmbedding *em, int N, const std::vector<int> &ctx) {
	int take = std::min((int) ctx.size(), N);
	std::vector<int> win(N, 0);
	for (int i = 0; i < take; ++i) win[i] = ctx[(int) ctx.size() - take + i];
	em->setInput(win);
	net.cpuNet().forward();
	return take - 1;
}

// -generate: autoregressively play a game from the -moves seed, write SGF.
static int runGenerate() {
	CNet net; CEmbedding *em; SequenceCrossEntropy *seq; int N, vocab;
	if (!restoreLM(net, em, seq, N, vocab)) return 1;
	std::vector<int> ctx = goParse((std::string) moves);
	if (ctx.empty()) ctx.push_back(GO_START);
	std::vector<char> occ(GO_BOARD * GO_BOARD, 0);          // board occupancy
	for (int id : ctx) { int pt = goPoint(id); if (pt >= 0) occ[pt] = 1; }
	std::mt19937 rng(std::random_device{}());
	for (int g = 0; g < gen_len; ++g) {
		int p = forwardContext(net, em, N, ctx);
		int nxt = sampleNext(seq, p, vocab, temp, rng, legal_moves ? &occ : nullptr);
		ctx.push_back(nxt);
		int pt = goPoint(nxt); if (pt >= 0) occ[pt] = 1;
	}
	std::string sgf = movesToSgf(ctx);
	std::ofstream(( std::string) sgf_path) << sgf;
	std::cout << "=== generated game (" << ctx.size() << " tokens, model=" << (std::string) model
			  << " temp=" << temp << ") -> " << (std::string) sgf_path << " ===\n" << sgf << std::endl;
	return 0;
}

// -predict: print the top-k next moves (with Born probabilities) for the -moves context.
static int runPredict() {
	CNet net; CEmbedding *em; SequenceCrossEntropy *seq; int N, vocab;
	if (!restoreLM(net, em, seq, N, vocab)) return 1;
	std::vector<int> ctx = goParse((std::string) moves);
	if (ctx.empty()) ctx.push_back(GO_START);
	std::vector<char> occ(GO_BOARD * GO_BOARD, 0);
	for (int id : ctx) { int pt = goPoint(id); if (pt >= 0) occ[pt] = 1; }
	int p = forwardContext(net, em, N, ctx);
	std::vector<std::pair<double, int>> pr(vocab);
	for (int k = 0; k < vocab; ++k) {
		int pt = goPoint(k);
		// legal_moves: rank occupied points and the start token out of the list
		double prob = (legal_moves && (pt < 0 || occ[pt])) ? -1.0 : seq->getProbability(p, k);
		pr[k] = { prob, k };
	}
	int K = std::min(std::max(1, (int) topk), vocab);
	std::partial_sort(pr.begin(), pr.begin() + K, pr.end(), std::greater<std::pair<double,int>>());
	std::cout << "context = " << (ctx.size() == 1 && ctx[0] == GO_START ? "(game start)" : (std::string) moves)
			  << "  (" << ctx.size() << " tokens)\n=== top " << K << " next moves ===\n";
	for (int i = 0; i < K; ++i) {
		std::string d = goDecode(pr[i].second);
		std::string who = (pr[i].second == GO_START) ? "start" : (d[0] == '0' ? "B" : "W");
		std::string coord = (pr[i].second == GO_START) ? "222" : (who + "[" + d[1] + d[2] + "]");
		std::cout << "  " << (i + 1) << ". " << coord << "  (token " << d << ", id " << pr[i].second
				  << ")  p=" << pr[i].first << "\n";
	}
	return 0;
}

// -eval_loss: exact-Born held-out loss (nats/move) over gen_len random val windows.
static int runEval() {
	TokenData d; d.load((std::string) data, vocab_size);
	CNet net; CEmbedding *em; SequenceCrossEntropy *seq; int N, vocab;
	if (!restoreLM(net, em, seq, N, vocab)) return 1;
	seq->setEps(0); seq->setSmooth(0);
	int lo = d.ntrain, hi = (int) d.data.size() - N - 2;
	if (hi <= lo) { std::cerr << "not enough validation tokens\n"; return 1; }
	std::mt19937 rng(777);
	std::uniform_int_distribution<int> U(lo, hi);
	int K = gen_len > 0 ? gen_len : 500;
	double tot = 0;
	for (int k = 0; k < K; ++k) {
		int pp = U(rng);
		std::vector<int> win(d.data.begin() + pp, d.data.begin() + pp + N), tgt(N);
		for (int i = 0; i < N; ++i) tgt[i] = d.data[pp + 1 + i];
		em->setInput(win); seq->setTargets(tgt);
		net.cpuNet().forward();
		tot += seq->loss(0);
	}
	std::cout << "EXACT-BORN val = " << (tot / K) << " nats/move  over " << K
			  << " windows (model=" << (std::string) model << ")" << std::endl;
	return 0;
}

int main(int argc, char **argv) {
	FLAGS::Parse(argc, argv);
	if (eval_loss) return runEval();
	if (generate)  return runGenerate();
	if (predict)   return runPredict();
	if (print_net) {
		CNet net;
		int embId = net.cpuNet().add(new CEmbedding(emb, tokens, vocab_size));
		buildLM(net.cpuNet(), embId, emb, tokens, blocks, vocab_size);
		std::cout << "config: born=" << born << " token_norm=" << token_norm
				  << " rope=" << rope << "  E=" << emb << " N=" << tokens << " L=" << blocks
				  << " vocab=" << vocab_size << "\n\n" << net.cpuNet().toString();
		return 0;
	}

#ifndef WITH_NCCL
	if (world > 1) { std::cerr << "built without NCCL: -world must be 1 (rebuild with NCCL for multi-GPU)\n"; return 1; }
#endif
	const int W = world, R = rank, N = tokens, E = emb, L = blocks;
	if (W < 1 || batch % W != 0) {
		std::cerr << "batch (" << batch << ") must be a positive multiple of world (" << W << ")\n";
		return 1;
	}
	const int per = batch / W;
	int nDev = 0; cudaGetDeviceCount(&nDev); if (nDev < 1) nDev = 1;
	cudaSetDevice(R % nDev);

#ifdef WITH_NCCL
	// NCCL rendezvous: rank 0 writes the unique id to /tmp; others poll.
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
#endif

	TokenData d; d.load((std::string) data, vocab_size);
	std::vector<int> &ids = d.data; int vocab = d.vocab, ntrainTok = d.ntrain;
	if (R == 0)
		std::cout << "go-LM: tokens=" << ids.size() << " vocab=" << vocab
				  << "  model E=" << E << " N=" << N << " L=" << L
				  << "  global_batch=" << batch << " world=" << W
				  << "  born=" << born << " rope=" << rope << " token_norm=" << token_norm
				  << "  ce_smooth=" << ce_smooth << "  chance=ln(vocab)=" << std::log((double) vocab) << std::endl;

	CNet net;
	int embId = net.cpuNet().add(new CEmbedding(E, N, vocab));
	int ceId  = buildLM(net.cpuNet(), embId, E, N, L, vocab);
	((CEmbedding*) net.cpuNet()[embId])->setIsMainInput(true);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setIsMainOutput(true);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setEps(ce_eps);
	((SequenceCrossEntropy*) net.cpuNet()[ceId])->setSmooth(ce_smooth);
	net.cpuNet().init_inputs(1234);                // identical init on every rank
	// Symmetric-Gaussian init for ALL weights (breaks the +imag saddle bias at the
	// source, so cuBLAS's summation order also escapes the unigram saddle).
	if (gauss_init > 0.f) net.cpuNet().gaussInit(gauss_init, 4242);
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
	net.allocateOnGpu(per);
	CEmbedding *em = (CEmbedding*) net.cpuNet()[embId];
	SequenceCrossEntropy *seq = (SequenceCrossEntropy*) net.cpuNet()[ceId];

	auto gradBufs = net.ancestorGradBuffers();
	std::vector<int> off(gradBufs.size());
	int total = 0;
	for (size_t i = 0; i < gradBufs.size(); ++i) { off[i] = total; total += gradBufs[i].second; }
	float *scratch = nullptr; cudaMalloc(&scratch, (size_t) total * sizeof(float));
	cublasHandle_t gbh = 0;
	if (grad_norm_clip > 0.f) { cublasCreate(&gbh); cublasSetStream(gbh, cudaStreamPerThread); cublasSetPointerMode(gbh, CUBLAS_POINTER_MODE_HOST); }
	// Gradient-noise machinery: a device noise buffer + cuRAND generator + cuBLAS
	// handle. Seed is rank-INDEPENDENT so every DP replica adds identical noise to
	// the (identical, all-reduced) gradient and the replicas stay in sync.
	curandGenerator_t crng = 0; float *noise = nullptr; cublasHandle_t nbh = 0;
	size_t ntot = (size_t) total + ((size_t) total & 1);   // cuRAND normal wants an even count
	if (grad_noise > 0.f) {
		curandCreateGenerator(&crng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(crng, 20261011ULL);
		curandSetStream(crng, cudaStreamPerThread);
		cudaMalloc(&noise, ntot * sizeof(float));
		cublasCreate(&nbh); cublasSetStream(nbh, cudaStreamPerThread); cublasSetPointerMode(nbh, CUBLAS_POINTER_MODE_HOST);
	}

	std::mt19937 rng(seed + R);
	auto targetsOf = [&](const std::vector<int> &win, int lab) {
		std::vector<int> t(N, 0);
		for (int i = 0; i + 1 < (int) win.size(); ++i) t[i] = win[i + 1];
		if (!win.empty()) t[win.size() - 1] = lab;
		return t;
	};
	auto buildBatch = [&](bool heldout) {
		EmbeddingBatch b(per, N);
		int lo = heldout ? ntrainTok : 0;
		int hi = (heldout ? (int) ids.size() : ntrainTok) - N - 2;
		std::uniform_int_distribution<int> dd(lo, hi);
		for (int i = 0; i < per; ++i) {
			int p = dd(rng);
			std::vector<int> win(ids.begin() + p, ids.begin() + p + N);
			int lab = ids[p + N];
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
	float best_val = 1e30f;
	for (int t = 1; t <= steps; ++t) {
		float gnorm = 0.f;
		EmbeddingBatch batch_ = buildBatch(false);
		net.gpuForward(em, seq, batch_);
		acc += net.getLoss(0)[seq->uid()]; ++accn;
		net.gpuBackward();
		for (size_t i = 0; i < gradBufs.size(); ++i)
			cudaMemcpyAsync(scratch + off[i], gradBufs[i].first,
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
#ifdef WITH_NCCL
		if (W > 1) ncclAllReduce(scratch, scratch, total, ncclFloat, ncclSum, comm, cudaStreamPerThread);
#endif
		if (grad_norm_clip > 0.f) {
			cublasSnrm2(gbh, total, scratch, 1, &gnorm);
			if (gnorm > grad_norm_clip) { float s = grad_norm_clip / gnorm; cublasSscal(gbh, total, &s, scratch, 1); }
		}
		// Decaying symmetry-breaking gradient noise (identical on every rank).
		if (grad_noise > 0.f) {
			float sigma = grad_noise / std::pow(1.f + (float) t, 0.55f);
			curandGenerateNormal(crng, noise, ntot, 0.f, sigma);     // ntot is even for cuRAND
			const float one = 1.f;
			cublasSaxpy(nbh, total, &one, noise, 1, scratch, 1);     // scratch += noise
		}
		for (size_t i = 0; i < gradBufs.size(); ++i)
			cudaMemcpyAsync(gradBufs[i].first, scratch + off[i],
				(size_t) gradBufs[i].second * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
		net.trueAdamUpdate(lr_at(t), b1, b2, eps, t, grad_clip);

		if (R == 0 && t % 100 == 0) {
			std::cout << "step " << t << "  train " << (acc / accn) << " nats/move  lr "
					  << lr_at(t) << "  |g|=" << gnorm << "  "
					  << (sw.ElapsedTimeMicros() / 1000.0 / 100) << " ms/step" << std::endl;
			acc = 0; accn = 0; sw.Reset();
		}
		if (R == 0 && val_every > 0 && t % val_every == 0) {
			float v = 0; const int K = 20;
			for (int k = 0; k < K; ++k) { EmbeddingBatch vb = buildBatch(true); net.gpuForward(em, seq, vb); v += net.getLoss(0)[seq->uid()]; }
			float vv = v / K;
			std::cout << "  [val] step " << t << "  val " << vv << " nats/move" << std::endl;
			if (!((std::string) save_path).empty() && vv < best_val) {
				best_val = vv;
				if (net.getInputsFromGpu() && net.save((std::string) save_path))
					std::cout << "    [ckpt] new best val " << vv << " -> saved " << (std::string) save_path << std::endl;
			}
			sw.Reset();
		}
	}
	cudaStreamSynchronize(cudaStreamPerThread);

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
	if (grad_noise > 0.f) { curandDestroyGenerator(crng); cudaFree(noise); cublasDestroy(nbh); }
#ifdef WITH_NCCL
	ncclCommDestroy(comm);
	if (R == 0) { remove(idf); remove(rdy); }
#endif
	return 0;
}
