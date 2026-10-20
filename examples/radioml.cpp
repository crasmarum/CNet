// =============================================================================
// radioml: radio modulation classification with the CNet complex-valued
// framework. A single complex network consumes the length-128 complex IQ window
// directly (no real/imag split) and classifies it with a Born-rule measurement
// head p_k = |z_k|^2 / ||z||^2 -- the physics-native counterpart of a softmax.
//
// The classifier combines, in one graph:
//   * a learnable complex spectral convolution bank (FourierTrans -> Hadamard ->
//     pooled |.|^2 / |.|^4 amplitude features), which separates the QAM
//     amplitude distributions;
//   * fixed phase-order branches z^M (M = 2, 4) -> pool -> |.|^2, the
//     carrier-invariant M-PSK phase structure;
//   * a CYCLOSTATIONARY bank built on the ConjLagProduct operator
//     r_tau(t) = x_t * conj(x_{t-tau}): per lag we read the cyclic-correlation
//     R(tau) = mean_t r_tau (complex), |R(tau)|^2, and a scale-safe normalized
//     cyclic SPECTRUM SoftMax(FFT_t r_tau) -> |.|^2 -> pool. This is the a-priori
//     signal structure the pooled-power read-out discards, and it is the one
//     lever that lifts held-out accuracy (see README_radioml.md);
//   * an optional STFT branch (SegmentedFourier windows {16,32}); and
//   * a small complex MLP head feeding the Born-rule CrossEntropy output.
//
// This is a single-GPU example (no NCCL). Data is the RML1 .bin format read by
// RadioMLDataReader and produced by data/rml10a_to_bin.py (see that script and
// README_radioml.md for the exact byte layout and per-SNR splits).
//
// Build:   make radioml
// Train:   ./radioml -radioml_train rml_train.bin -radioml_test rml_test.bin \
//                    -arch 16 -n_classes 11 -no_epochs 30 -model_path rml.mod
// Eval:    ./radioml -infer true -model_path rml.mod -radioml_test rml_test.bin \
//                    -arch 16 -confusion true
// Per-SNR: run -infer once per per-SNR test bin (rml_test_snr<S>.bin).
// =============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../data/data.h"          // RadioMLDataReader + radioml_train/radioml_test flags
#include "../impl/log.h"
#include "../impl/cnet.h"          // all layers (incl. ConjLagProduct, SegmentedFourier)
#include "../impl/vars.h"
#include "../impl/batch.h"
#include "../impl/cinput.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"
#include "../gpu/reduce.h"

// ---- Model / data shape -----------------------------------------------------
FLAG_INT(arch, 16)          // 11 = pooled-power baseline, 15 = + cyclic-corr,
                            // 16 = + normalized cyclic spectrum (default, best)
FLAG_INT(seq_len, 128)      // complex IQ window length L (must match the .bin header)
FLAG_INT(n_classes, 11)     // modulation classes (RML2016.10a/RML22 = 11, 10b = 10)
FLAG_INT(n_channels, 128)   // learnable spectral conv-bank channels
FLAG_INT(pool_out, 0)       // mean-pool each feature to this many time bins (0 = per-arch default: 8 for arch 16)
FLAG_INT(hidden_dim, 0)     // MLP head width (0 = default 64)
FLAG_INT(head_layers, 2)    // number of hidden Linear+CGelu layers in the head (arch 16)
FLAG_BOOL(stft, false)      // arch 16: add the STFT/spectrogram branch (SegmentedFourier windows 16,32)

// ---- Optimization -----------------------------------------------------------
FLAG_INT(no_epochs, 30)     // training epochs
FLAG_INT(batch_size, 32)    // examples per gradient step
FLAG_FLOAT(l_rate, 8e-4)    // learning rate
FLAG_BOOL(true_adam, true)  // true Adam (first + second moment) on GPU
FLAG_BOOL(use_adam, false)  // momentum-only update (ignored when -true_adam true)
FLAG_FLOAT(beta, 0.9)       // Adam first-moment decay
FLAG_FLOAT(beta2, 0.999)    // true-Adam second-moment decay
FLAG_FLOAT(eps, 1e-8)       // true-Adam epsilon
FLAG_FLOAT(grad_clip, 1.0)  // true-Adam clip-by-value per grad component (0 = off)
FLAG_BOOL(gauss_init, true) // scaled complex-Gaussian init (per-input; essential for multi-channel nets)
FLAG_FLOAT(init_scale, 1.0) // init variance = init_scale / length
FLAG_INT(seed, 12345)       // RNG seed for weight init (vary across runs for mean +/- std)
FLAG_BOOL(augment, false)   // training-time random phase-rotation + time-reversal (held-out eval is NEVER augmented)
FLAG_BOOL(aug_reverse, true) // -augment: include random time reversal (off = phase rotation / CFO only)
FLAG_FLOAT(aug_cfo, 0.0)     // -augment: max random carrier-frequency offset, rad/sample (linear phase ramp)

// ---- Mode / IO --------------------------------------------------------------
FLAG_STRING(model_path, "") // where to save (train) / restore (infer) the model
FLAG_BOOL(infer, false)     // forward-only held-out inference from a restored model
FLAG_BOOL(confusion, false) // with -infer, also print a C x C confusion matrix + per-class recall
FLAG_INT(infer_dims, 4)     // Vars::dims_ for inference (4 is safe for every net)


// Build the modulation classifier into `cnet`; returns the input / output ids.
// Ports the arch 11 / 15 / 16 lineage (pooled-power -> + cyclic-correlation ->
// + normalized cyclic spectrum). arch 16 is the default and the one documented
// in the paper / README_radioml.md.
FLAG_STRING(conv_act, "u")   // arch 20 conv activation: u (per-plane TokenNorm, amplitude-removing)
                             // | crelu | cgelu (amplitude-preserving, like VT-CNN2's ReLU)
FLAG_FLOAT(dropout, 0.0)     // arch 20: ComplexDropout prob after conv activations + dense (VT-CNN2 uses 0.5)
FLAG_INT(gla_blocks, 2)        // arch 21: number of complex-GLA blocks over the unpooled sequence
FLAG_INT(gla_dim, 64)          // arch 21: stream width E (complex) per time step
FLAG_INT(gla_heads, 4)         // arch 21: GLA heads (E % heads == 0)
FLAG_INT(gla_chunk, 64)        // arch 21: GPU chunk length (L % chunk == 0)
FLAG_STRING(lags, "1,2,3,4")   // arch 21: conjugate-lag streams x_t conj(x_{t-tau}) kept at full resolution
FLAG_BOOL(cyc_branch, true)    // arch 21: also feed the arch 16 cyclostationary bank (pooled) to the head
FLAG_INT(eval_examples, 20000) // held-out examples scored after every epoch (0 = only the final full pass);
                               // a fixed random subset of -radioml_test, dropout off, never augmented
FLAG_BOOL(cosine, false)       // cosine decay of -l_rate over all steps, down to 5% of the initial value

extern bool g_dropout_infer;   // (kernels.cpp) set true around eval so dropout is identity

// arch 20 helper: the post-conv activation. `u` is the Hadamard-net L2 norm
// (TokenNorm, per-plane); crelu/cgelu are pointwise and keep amplitude.
static int addConvAct(CNet& cnet, int node, int planes, int D) {
	if (conv_act == "crelu") return cnet.add(new CRelu(InSize(planes * D)), {node});
	if (conv_act == "cgelu") return cnet.add(new CGelu(InSize(planes * D)), {node});
	return cnet.add(new TokenNorm(planes, D), {node});
}

// optional ComplexDropout after a node (identity when -dropout 0; off at eval via g_dropout_infer)
static int maybeDrop(CNet& cnet, int node, int len) {
	if (dropout > 0.0f) return cnet.add(new ComplexDropout(InSize(len), dropout), {node});
	return node;
}

static void buildRadioMLNet(CNet& cnet, int *inp, int *outp) {
	const int L = seq_len;

	if (arch == 11) {
		// Pooled-power baseline: one FFT, then n_channels learnable spectral
		// Hadamard masks (= circular convolutions) -> pooled |.|^2 / |.|^4, plus
		// fixed phase-order branches z^2,z^4 -> pool -> |.|^2, into a 3-layer MLP.
		const int bins = (pool_out > 0) ? pool_out : 4;
		std::vector<int> acts;
		*inp = cnet.add(new CInput(OutSize(L)));
		auto fft = cnet.add(new FourierTrans(InSize(L)), {*inp});

		int orders[] = {2, 4};
		for (int oi = 0; oi < 2; ++oi) {
			int pw = cnet.add(new CPower(InSize(L), orders[oi]), {*inp});
			int pl = cnet.add(new MeanPool(InSize(L), OutSize(bins)), {pw});
			acts.push_back(cnet.add(new CModulus2(InSize(bins)), {pl}));
		}
		for (int c = 0; c < n_channels; ++c) {
			int wdata = cnet.add(new CInput(OutSize(L)));    // learnable spectral filter
			int hdm   = cnet.add(new Hadamard(InSize(L), InSize(L)), {fft, wdata});
			int m2 = cnet.add(new CModulus2(InSize(L)), {hdm});
			acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {m2}));
			int m4 = cnet.add(new CModulus2(InSize(L)), {m2});
			acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {m4}));
		}
		int feat = (2 + 2 * n_channels) * bins;
		int H = (hidden_dim > 0) ? hidden_dim : 64;
		int h1data = cnet.add(new CInput(OutSize(feat * H)));
		std::vector<int> h1in = acts; h1in.push_back(h1data);
		int h1 = cnet.cpuNet().add(new Linear(InSize(feat), InSize(feat * H)), h1in);
		int g1 = cnet.add(new CGelu(InSize(H)), {h1});
		int h2data = cnet.add(new CInput(OutSize(H * H)));
		int h2 = cnet.add(new Linear(InSize(H), InSize(H * H)), {g1, h2data});
		int g2 = cnet.add(new CGelu(InSize(H)), {h2});
		int odata = cnet.add(new CInput(OutSize(H * n_classes)));
		int lin = cnet.add(new Linear(InSize(H), InSize(H * n_classes)), {g2, odata});
		*outp = cnet.add(new CrossEntropy(InSize(n_classes)), {lin});
		return;
	}

	if (arch == 15 || arch == 16) {
		// arch 11 + a cyclostationary bank. The conjugate lag product
		// r_tau(t) = x_t conj(x_{t-tau}) is the nonlinear primitive of
		// cyclostationarity; its time-mean is the autocorrelation R(tau)
		// (|R| <= 1 for power-normalised x, so well-scaled). We read R(tau)
		// (complex -- its relative phase separates PSK orders) and |R(tau)|^2.
		// arch 16 additionally reads the normalized cyclic SPECTRUM
		// SoftMax(FFT_t r_tau) -> |.|^2 (scale-safe: a density O(1/bins), unlike
		// the raw |FFT(r_tau)|^2 ~ N that saturates the Born logits).
		const int bins   = (pool_out > 0) ? pool_out : (arch == 16 ? 8 : 4);
		int lags15[] = {1, 2, 4, 8, 16};
		int lags16[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
		int *lags    = (arch == 16) ? lags16 : lags15;
		const int n_lags = (arch == 16) ? 10 : 5;

		*inp = cnet.add(new CInput(OutSize(L)));
		auto fft = cnet.add(new FourierTrans(InSize(L)), {*inp});
		std::vector<int> acts;

		// phase-order moments z^2, z^4
		int orders[] = {2, 4};
		for (int oi = 0; oi < 2; ++oi) {
			int pw = cnet.add(new CPower(InSize(L), orders[oi]), {*inp});
			int pl = cnet.add(new MeanPool(InSize(L), OutSize(bins)), {pw});
			acts.push_back(cnet.add(new CModulus2(InSize(bins)), {pl}));
		}
		// learnable complex spectral conv bank -> pooled |.|^2 and |.|^4
		for (int c = 0; c < n_channels; ++c) {
			int wdata = cnet.add(new CInput(OutSize(L)));
			int hdm   = cnet.add(new Hadamard(InSize(L), InSize(L)), {fft, wdata});
			int m2 = cnet.add(new CModulus2(InSize(L)), {hdm});
			acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {m2}));
			int m4 = cnet.add(new CModulus2(InSize(L)), {m2});
			acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {m4}));
		}
		// cyclostationary bank: R(tau), |R(tau)|^2 [, normalized cyclic spectrum]
		for (int li = 0; li < n_lags; ++li) {
			int cl = cnet.add(new ConjLagProduct(InSize(L), lags[li]), {*inp});
			int R  = cnet.add(new MeanPool(InSize(L), OutSize(bins)), {cl});
			acts.push_back(R);
			acts.push_back(cnet.add(new CModulus2(InSize(bins)), {R}));
			if (arch == 16) {
				int f  = cnet.add(new FourierTrans(InSize(L)), {cl});
				int nf = cnet.add(new SoftMax(InSize(L)), {f});       // L2-normalize (scale-safe)
				int sp = cnet.add(new CModulus2(InSize(L)), {nf});
				acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {sp}));
			}
		}
		// optional STFT / spectrogram branch (arch 16)
		if (arch == 16 && stft) {
			int windows[] = {16, 32};
			for (int wi = 0; wi < 2; ++wi) {
				if (L % windows[wi]) continue;
				int sfr = cnet.add(new SegmentedFourier(InSize(L), windows[wi]), {*inp});
				int m   = cnet.add(new CModulus2(InSize(L)), {sfr});
				acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {m}));
			}
		}

		int feat = (int) acts.size() * bins;   // every feature node is pooled to `bins`
		int H  = (hidden_dim > 0) ? hidden_dim : 64;
		const int hl = (arch == 16 && head_layers > 0) ? head_layers : 2;

		int w0 = cnet.add(new CInput(OutSize(feat * H)));
		std::vector<int> in0 = acts; in0.push_back(w0);
		int cur = cnet.cpuNet().add(new Linear(InSize(feat), InSize(feat * H)), in0);
		cur = cnet.add(new CGelu(InSize(H)), {cur});
		for (int layer = 1; layer < hl; ++layer) {
			int wk = cnet.add(new CInput(OutSize(H * H)));
			cur = cnet.add(new Linear(InSize(H), InSize(H * H)), {cur, wk});
			cur = cnet.add(new CGelu(InSize(H)), {cur});
		}
		int odata = cnet.add(new CInput(OutSize(H * n_classes)));
		int lin = cnet.add(new Linear(InSize(H), InSize(H * n_classes)), {cur, odata});
		*outp = cnet.add(new CrossEntropy(InSize(n_classes)), {lin});
		return;
	}

	if (arch == 20) {
		// VT-CNN2 (radioML notebook) translated to the complex Hadamard-conv
		// framework. Input is the 2xW IQ image flattened row-major into D=2W reals
		// ([I_0..I_{W-1}, Q_0..Q_{W-1}], imag 0; see data/rml22_to_2dbin.py), so a
		// 2D (h,w) kernel is realized by kernel_index positions spanning the layout
		// (row stride W): (1,3) -> {0,1,2}; (2,3) -> {0,1,2, W,W+1,W+2} (the 2nd row
		// mixes Q into the conv, exactly VT-CNN2's height-2 I/Q-mixing kernel).
		//   FFT -> HConv(1->256, (1,3)) -> U -> HConv(256->80, (2,3)) -> U
		//        -> Linear(80*D -> 256) -> CGelu -> Linear(256 -> classes) -> Born
		const int W = L / 2;                       // image width (128); L = seq_len = 256
		std::vector<int> k1 = {0, 1, 2};           // (1,3)
		std::vector<int> k2 = {0, 1, 2, W, W + 1, W + 2};   // (2,3): spans both IQ rows
		const int F1 = 256, F2 = 80, H = 256;

		*inp = cnet.add(new CInput(OutSize(L)));
		int fft = cnet.add(new FourierTrans(InSize(L)), {*inp});

		int w1 = cnet.add(new CInput(OutSize(F1 * 1 * (int) k1.size() + F1)));   // conv1 taps + bias
		int c1 = cnet.add(new HadamardConv(1, L, F1, k1), {fft, w1});
		int n1 = addConvAct(cnet, c1, F1, L);                                    // activation (u/crelu/cgelu)
		n1 = maybeDrop(cnet, n1, F1 * L);

		int w2 = cnet.add(new CInput(OutSize(F2 * F1 * (int) k2.size() + F2)));  // conv2 taps + bias (C=F1)
		int c2 = cnet.add(new HadamardConv(F1, L, F2, k2), {n1, w2});            // multi-plane 256 -> 80
		int n2 = addConvAct(cnet, c2, F2, L);
		n2 = maybeDrop(cnet, n2, F2 * L);

		const int feat = F2 * L;                   // 80 * 256 = 20480
		int d1 = cnet.add(new CInput(OutSize(feat * H)));
		int h1 = cnet.cpuNet().add(new Linear(InSize(feat), InSize(feat * H)), {n2, d1});
		int g1 = cnet.add(new CGelu(InSize(H)), {h1});
		g1 = maybeDrop(cnet, g1, H);
		int d2 = cnet.add(new CInput(OutSize(H * n_classes)));
		int lin = cnet.add(new Linear(InSize(H), InSize(H * n_classes)), {g1, d2});
		*outp = cnet.add(new CrossEntropy(InSize(n_classes)), {lin});
		return;
	}

	if (arch == 21) {
		// Differential-phase streams + complex-amplitude gated linear attention over the
		// UNPOOLED sequence (the Step 2 design): per time step the network sees
		//   u_t = [ x_t,  x_t conj(x_{t-tau}) for tau in -lags,  |x_t|^2 ]
		// (a constant carrier offset becomes a constant phasor in every lag stream, while the
		// symbol-to-symbol phase jumps survive), embedded token-wise into E complex channels
		// (sum of one TokenwiseLinear(L, 1, E) per stream == TokenwiseLinear(L, S, E) on the
		// interleaved streams), then -gla_blocks pre-norm blocks of
		//   complex GLA (scalar per-head gates, running dh x dh state) + TokenNorm + residual,
		//   position-wise FFN E -> 2E -> E + residual,
		// followed by a dense head over the L x E tokens, optionally concatenated with the
		// arch 16 cyclostationary bank (R(tau), |R(tau)|^2, normalized cyclic spectrum).
		// Operates on the 1-D complex IQ bins (L = 128), not the 2-D image layout of arch 20.
		const int E = gla_dim, Hd = gla_heads, size = L * E, Hf = 2 * E;
		if (E % Hd != 0 || L % gla_chunk != 0) {
			std::cerr << "radioml arch 21: need -gla_dim % -gla_heads == 0 and -seq_len % -gla_chunk == 0\n";
			exit(1);
		}
		std::vector<int> taus;
		{ std::stringstream ss((std::string) lags); std::string tok;
		  while (std::getline(ss, tok, ',')) if (!tok.empty()) taus.push_back(std::stoi(tok)); }

		*inp = cnet.add(new CInput(OutSize(L)));
		std::vector<int> streams = {*inp};
		for (int tau : taus) streams.push_back(cnet.add(new ConjLagProduct(InSize(L), tau), {*inp}));
		streams.push_back(cnet.add(new CModulus2(InSize(L)), {*inp}));

		int emb = -1;                                             // token-wise embedding, summed over streams
		for (int sid : streams) {
			int w = cnet.add(new CInput(OutSize(E)));
			int pj = cnet.add(new TokenwiseLinear(L, 1, E), {sid, w});
			emb = (emb < 0) ? pj : cnet.add(new Residual(InSize(size), InSize(size)), {emb, pj});
		}
		int last = cnet.add(new TokenNorm(L, E), {emb});
		for (int b = 0; b < gla_blocks; ++b) {
			int pre = cnet.add(new CGelu(InSize(size)), {cnet.add(new TokenNorm(L, E), {last})});
			int wQ = cnet.add(new CInput(OutSize(E * E))); int Q  = cnet.add(new TokenwiseLinear(L, E, E), {pre, wQ});
			int wK = cnet.add(new CInput(OutSize(E * E))); int K  = cnet.add(new TokenwiseLinear(L, E, E), {pre, wK});
			int wV = cnet.add(new CInput(OutSize(E * E))); int Vv = cnet.add(new TokenwiseLinear(L, E, E), {pre, wV});
			int wG = cnet.add(new CInput(OutSize(E * Hd))); int G = cnet.add(new TokenwiseLinear(L, E, Hd), {pre, wG});
			int g = cnet.add(new GatedLinearAttention(L, E, Hd), {Q, K, Vv, G});
			((GatedLinearAttention*) cnet[g])->setChunk(gla_chunk);
			int att = cnet.add(new TokenNorm(L, E), {g});
			int wO = cnet.add(new CInput(OutSize(E * E)));
			int mix = cnet.add(new TokenwiseLinear(L, E, E), {att, wO});
			int r0 = cnet.add(new Residual(InSize(size), InSize(size)), {last, mix});
			int n0 = cnet.add(new CGelu(InSize(size)), {r0});
			int w1 = cnet.add(new CInput(OutSize(E * Hf)));
			int f1 = cnet.add(new TokenwiseLinear(L, E, Hf), {cnet.add(new TokenNorm(L, E), {n0}), w1});
			int gg = cnet.add(new CGelu(InSize(L * Hf)), {f1});
			int w2 = cnet.add(new CInput(OutSize(Hf * E)));
			int f2 = cnet.add(new TokenwiseLinear(L, Hf, E), {gg, w2});
			int r1 = cnet.add(new Residual(InSize(size), InSize(size)), {n0, f2});
			last = cnet.add(new CGelu(InSize(size)), {r1});
			last = maybeDrop(cnet, last, size);
		}
		last = cnet.add(new TokenNorm(L, E), {last});

		std::vector<int> acts = {last};
		int feat = size;
		if (cyc_branch) {                                         // arch 16 bank, pooled
			const int bins = (pool_out > 0) ? pool_out : 8;
			int lags16[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
			for (int li = 0; li < 10; ++li) {
				int cl = cnet.add(new ConjLagProduct(InSize(L), lags16[li]), {*inp});
				int R  = cnet.add(new MeanPool(InSize(L), OutSize(bins)), {cl});
				acts.push_back(R);
				acts.push_back(cnet.add(new CModulus2(InSize(bins)), {R}));
				int f  = cnet.add(new FourierTrans(InSize(L)), {cl});
				int nf = cnet.add(new SoftMax(InSize(L)), {f});
				int sp = cnet.add(new CModulus2(InSize(L)), {nf});
				acts.push_back(cnet.add(new MeanPool(InSize(L), OutSize(bins)), {sp}));
				feat += 3 * bins;
			}
		}
		const int H = (hidden_dim > 0) ? hidden_dim : 256;
		int d1 = cnet.add(new CInput(OutSize(feat * H)));
		std::vector<int> in0 = acts; in0.push_back(d1);
		int h1 = cnet.cpuNet().add(new Linear(InSize(feat), InSize(feat * H)), in0);
		int g1 = cnet.add(new CGelu(InSize(H)), {h1});
		g1 = maybeDrop(cnet, g1, H);
		int d2 = cnet.add(new CInput(OutSize(H * n_classes)));
		int lin = cnet.add(new Linear(InSize(H), InSize(H * n_classes)), {g1, d2});
		*outp = cnet.add(new CrossEntropy(InSize(n_classes)), {lin});
		return;
	}

	std::cerr << "radioml: unsupported -arch " << arch
			  << " (use 11, 15, 16, 20, or 21)." << std::endl;
	exit(1);
}


// Train on radioml_train, checkpoint to model_path, then report held-out
// accuracy on radioml_test (the test split is NEVER augmented).
static void trainRadioML() {
	RadioMLDataReader train_reader;
	if (!train_reader.Open(radioml_train) || !train_reader.readData()) {
		std::cerr << "radioml: cannot read -radioml_train " << radioml_train << std::endl;
		exit(1);
	}
	if (train_reader.seqLen() != seq_len || train_reader.numClasses() != n_classes) {
		std::cerr << "radioml: flag/data mismatch -- -seq_len " << seq_len << " -n_classes "
				  << n_classes << " but file has L=" << train_reader.seqLen()
				  << " C=" << train_reader.numClasses() << std::endl;
		exit(1);
	}
	train_reader.setAugment(augment);
	train_reader.setAugmentOptions(aug_reverse, aug_cfo, arch == 20);   // arch 20 = 2-D [I row, Q row] layout

	// Held-out reader opened up front: a fixed shuffled subset is scored after every epoch.
	RadioMLDataReader test_reader;
	const bool have_test = test_reader.Open(radioml_test) && test_reader.readData();
	if (have_test) test_reader.shuffle();

	CNet cnet;
	int cinp, coutp;
	buildRadioMLNet(cnet, &cinp, &coutp);
	std::cout << cnet.cpuNet().toString() << std::endl;

	CInput *inp = (CInput*) cnet[cinp];
	CrossEntropy *cen = (CrossEntropy*) cnet[coutp];
	if (gauss_init) {
		cnet.cpuNet().gaussInit(init_scale, seed);   // distinct per-input seeding
	} else {
		cnet.init_inputs(seed);
	}
	assert(cnet.allocateOnGpu(batch_size));

	ModelSaver saver;
	float avg_loss = 0; int correct = 0, seen = 0, step = 0;
	const int steps_per_epoch = train_reader.size() / batch_size;
	const long total_steps = (long) steps_per_epoch * no_epochs;
	float best_acc = -1.f; int best_epoch = -1;
	for (int epoch = 0; epoch < no_epochs; ++epoch) {
		train_reader.shuffle();
		for (int t = 0; t < steps_per_epoch; ++t) {
			auto batch = train_reader.nextBatch(batch_size);
			cnet.gpuForward(inp, cen, batch);
			avg_loss += cnet.getLoss(cen, &batch);
			correct  += cnet.getCorrect(cen);
			seen     += batch.size();
			if (t % 100 == 99) {
				std::cout << epoch << "\t" << t << "\t loss: " << (avg_loss / 100)
						  << "\t train-acc: " << (correct / (float) seen) << std::endl;
				avg_loss = 0; correct = 0; seen = 0;
			}
			cnet.gpuBackward();
			float lr = l_rate;
			if (cosine) lr = l_rate * (0.05f + 0.95f * 0.5f * (1.f + cosf((float) M_PI * step / (float) total_steps)));
			if (true_adam) {
				cnet.trueAdamUpdate(lr / batch.size(), beta, beta2, eps, step, grad_clip);
			} else if (use_adam) {
				cnet.adamUpdate(lr / batch.size(), beta, step);
			} else {
				cnet.gpuUpdateInputs(lr / batch.size());
			}
			step++;
		}
		// Per-epoch held-out accuracy on a fixed subset; keep the best weights in model_path.
		if (have_test && eval_examples > 0) {
			g_dropout_infer = true;
			test_reader.rewind();
			int ecorrect = 0, eseen = 0;
			const int nb = std::min(test_reader.size(), eval_examples) / batch_size;
			for (int t = 0; t < nb; ++t) {
				auto tb = test_reader.nextBatch(batch_size);
				cnet.gpuForward(inp, cen, tb);
				ecorrect += cnet.getCorrect(cen);
				eseen += tb.size();
			}
			g_dropout_infer = false;
			const float acc = ecorrect / (float) eseen;
			const bool improved = acc > best_acc;
			if (improved) { best_acc = acc; best_epoch = epoch; }
			std::cout << "EPOCH " << epoch << " held-out acc " << acc << " over " << eseen
					  << " examples (best " << best_acc << " at epoch " << best_epoch << ")" << std::endl;
			if (improved && !model_path.empty() && cnet.getInputsFromGpu()) {
				saver.Save(cnet.cpuNet(), model_path + ".best");
			}
		}
	}
	if (best_epoch >= 0)
		std::cout << "BEST held-out acc " << best_acc << " at epoch " << best_epoch
				  << " (" << model_path << ".best)" << std::endl;

	if (!model_path.empty() && cnet.getInputsFromGpu()) {
		saver.Save(cnet.cpuNet(), model_path);
		std::cout << "Final model saved at: " << model_path << std::endl;
	}

	// Held-out evaluation (never augmented; dropout off via inference mode).
	g_dropout_infer = true;
	if (have_test) {
		test_reader.rewind();
		int tcorrect = 0, tseen = 0;
		for (int t = 0; t < test_reader.size() / batch_size; ++t) {
			auto tb = test_reader.nextBatch(batch_size);
			cnet.gpuForward(inp, cen, tb);
			tcorrect += cnet.getCorrect(cen);
			tseen += tb.size();
		}
		std::cout << "RadioML TEST accuracy: " << (tcorrect / (float) tseen)
				  << " over " << tseen << " examples." << std::endl;
	}
}


// Forward-only held-out inference from a restored model (compact dims_=infer_dims
// layout). Point -radioml_test at the full test bin for overall accuracy, or at a
// per-SNR bin for one point of the accuracy-vs-SNR curve. -confusion adds a
// C x C confusion matrix with per-class recall.
static void inferRadioML() {
	Vars::dims_ = infer_dims;          // set before the net is restored/built
	g_dropout_infer = true;            // dropout is identity at inference

	RadioMLDataReader reader;
	if (!reader.Open(radioml_test) || !reader.readData()) {
		std::cerr << "radioml: cannot read -radioml_test " << radioml_test << std::endl;
		exit(1);
	}

	CNet cnet;
	if (!cnet.restore(model_path)) {
		std::cerr << "radioml: cannot restore -model_path " << model_path << std::endl;
		exit(1);
	}
	CInput *inp = (CInput*) cnet.findFirstInput();
	CrossEntropy *cen = (CrossEntropy*) cnet.findFirstOutput();
	assert(inp && cen);
	assert(cnet.allocateOnGpuInfer(batch_size));

	int C = cen->input().length_;
	std::vector<std::vector<int>> conf(C, std::vector<int>(C, 0));
	int correct = 0, seen = 0;
	for (int t = 0; t < reader.size() / batch_size; ++t) {
		auto batch = reader.nextBatch(batch_size);
		cnet.gpuInfer(inp, cen, batch);
		correct += cnet.getCorrect(cen);
		if (confusion) {
			auto pred = cnet.getPredictions(cen);
			for (int i = 0; i < batch.size(); ++i) conf[batch.label(i)][pred[i]]++;
		}
		seen += batch.size();
	}
	std::cout << "INFER accuracy: " << (correct / (float) seen)
			  << " over " << seen << " examples (Vars::dims_=" << Vars::dims_ << ")" << std::endl;

	if (confusion) {
		std::cout << "CONFUSION (rows=true, cols=pred), per-row recall:\n";
		for (int r = 0; r < C; ++r) {
			int rsum = 0; for (int c = 0; c < C; ++c) rsum += conf[r][c];
			std::cout << "t" << r << ":";
			for (int c = 0; c < C; ++c) std::cout << "\t" << conf[r][c];
			std::cout << "\t| recall=" << (rsum ? conf[r][r] / (float) rsum : 0.f) << "\n";
		}
	}
}


int main(int argc, char *argv[]) {
	FLAGS::Parse(argc, argv);
	if (infer) {
		inferRadioML();
	} else {
		trainRadioML();
	}
	return 0;
}
