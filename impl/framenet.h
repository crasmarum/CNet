#ifndef IMPL_FRAMENET_H_
#define IMPL_FRAMENET_H_

// Constant-state streaming twin for the frame-based audio graphs of examples/kws.cpp and examples/se.cpp.
//
// Those graphs take N frames of W samples at once (SegmentedFourier over the frames, per-token layers, causal
// Born attention, and either a whole-window readout or a per-frame mask). Like impl/stepnet.h for language
// models, the streaming form is a SECOND graph with N = 1: the same layers in the same order and wiring, every
// per-token size divided by N, weights copied, BornAttention in step mode (its running density-matrix state,
// BornAttention::forwardStep), RoPE given the absolute frame index. One frame goes in per hop; the output is the
// class amplitudes (KWS) or the enhanced frame (SE) for that hop, with the model's full causal context and a state
// that does not grow with time.
//
// Whole-window readouts (the flat Linear over the N x P token projections, TokenMeanPool / TokenMaxPool) have no
// per-frame form; they run unchanged on a FrameRing holding the last N frames of the projection (oldest first), so
// the twin reproduces the trained readout on a sliding window exactly, at the cost of N x P complex numbers of
// state. Layers after the ring keep their full size.
//
// Only the ancestors of a chosen output layer are built: for KWS the CrossEntropy; for SE the layer whose output
// is the enhanced spectrum Y (the loss tail that needs the clean signal is dropped, and an L2Out is appended as a
// consumer so that Y can be read from its input). Non-causal models are refused: the step recurrence is causal.
//
// Not stored in model files, so the caller must set them before building: BornAttention::setDecaySchedule (kws
// and se train with base 3, step 1; the class default is 5, 1), and the framing (window, hop, Hann, input gain).
#include <complex>
#include <cstdint>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cnet.h"
#include "cinput.h"
#include "segfourier.h"
#include "slice.h"
#include "melpower.h"
#include "tfconv.h"
#include "interleave.h"
#include "relu.h"
#include "tokenwise.h"
#include "tokennorm.h"
#include "rotary.h"
#include "bornattn.h"
#include "gla.h"
#include "residual.h"
#include "hadamard.h"
#include "linear.h"
#include "pool.h"
#include "crossent.h"
#include "l2out.h"

// The last N frames of a P-vector, oldest first (token t of the full graph at index t P + p): a sliding window
// that feeds the trained whole-window readouts. Zero until N frames have been seen. Twin-only (not saved).
class FrameRing: public CFunc {
	int P_, N_;
	std::vector<float> re_, im_;
	long frames_ = 0;

public:
	FrameRing(Uid uid, int P, int N) : CFunc(uid, InSize(P), OutSize(N * P)), P_(P), N_(N),
			re_((size_t) N * P, 0.f), im_((size_t) N * P, 0.f) {
	}
	FrameRing(int P, int N) : CFunc(InSize(P), OutSize(N * P)), P_(P), N_(N),
			re_((size_t) N * P, 0.f), im_((size_t) N * P, 0.f) {
	}
	virtual ~FrameRing() {
	}
	virtual CFunc* clone(Uid uid) {
		return new FrameRing(uid, P_, N_);
	}
	virtual std::string getName() {
		return "FrameRing_" + std::to_string(uid_);
	}
	void reset() {
		std::fill(re_.begin(), re_.end(), 0.f);
		std::fill(im_.begin(), im_.end(), 0.f);
		frames_ = 0;
	}
	long frames() const { return frames_; }
	bool full() const { return frames_ >= N_; }
	int tokens() const { return N_; }
	int width() const { return P_; }

	virtual void forward() {
		std::copy(re_.begin() + P_, re_.end(), re_.begin());    // drop the oldest frame
		std::copy(im_.begin() + P_, im_.end(), im_.begin());
		const size_t last = (size_t) (N_ - 1) * P_;
		for (int p = 0; p < P_; ++p) {
			re_[last + p] = input().real_[p];
			im_[last + p] = input().imag_[p];
		}
		frames_++;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			for (size_t i = 0; i < re_.size(); ++i) {
				output(indx).real_[offset(indx) + i] = re_[i];
				output(indx).imag_[offset(indx) + i] = im_[i];
			}
		}
	}
	virtual void backward() {
	}
	virtual void backward(int label) {
	}
	virtual std::complex<float> dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}
};

struct FrameNet {
	ComplexNet net;
	CInput *frame = nullptr;                 // one frame of W samples (real), set per hop
	CFunc *out = nullptr;                    // its input() holds the result: the C class amplitudes, or the W enhanced bins
	CrossEntropy *ce = nullptr;              // KWS: the Born-rule head (== out)
	std::vector<RotaryEmbed*> ropes;
	std::vector<BornAttention*> borns;
	std::vector<GatedLinearAttention*> glas;
	std::vector<FrameRing*> rings;
	int N = 0, W = 0, layers = 0;
	std::string error;

	// Bytes of state carried between frames: Born R (d_h^2) + M (d_h^3) complex per head, GLA S (d_h^2) per head,
	// and the N x P window of each ring.
	size_t stateBytes() const {
		size_t b = 0;
		for (auto *x : borns) { const size_t dh = x->headDim(); b += (size_t) x->heads() * (dh * dh + dh * dh * dh) * sizeof(std::complex<float>); }
		for (auto *g : glas) { const size_t dh = g->headDim(); b += (size_t) g->heads() * dh * dh * sizeof(std::complex<float>); }
		for (auto *r : rings) b += (size_t) r->tokens() * r->width() * sizeof(std::complex<float>);
		return b;
	}
};

static inline void frameCopyVars(const Vars &from, Vars *to, int n) {
	for (int i = 0; i < n; ++i) { to->real_[i] = from.real_[i]; to->imag_[i] = from.imag_[i]; }
}

// Builds sn.net as the N = 1 twin of the ancestors of `out_uid` in `full`, with `main_uid` the CInput that
// receives the framed audio. Returns false with sn.error set when a layer has no constant-state form.
inline bool buildFrameNet(ComplexNet &full, int main_uid, int out_uid, FrameNet &sn) {
	CFunc *outf = full[out_uid];
	if (!outf) { sn.error = "no layer with uid " + std::to_string(out_uid); return false; }
	std::set<int> keep;                      // uids of the ancestors of the output (and the output)
	std::vector<CFunc*> stack{outf};
	while (!stack.empty()) {
		CFunc *f = stack.back(); stack.pop_back();
		if (!keep.insert(f->uid()).second) continue;
		for (int i = 0; i < f->no_previous_func(); ++i) stack.push_back(f->previous(i));
	}
	if (!keep.count(main_uid)) { sn.error = "the output does not depend on the main input"; return false; }
	int W = 0, N = 0;
	for (CFunc *f : full.funcs()) {
		if (!keep.count(f->uid())) continue;
		if (auto *s = dynamic_cast<SegmentedFourier*>(f)) { if (!W) W = s->window(); }
		if (auto *t = dynamic_cast<TokenwiseLinear*>(f)) { if (!N || t->nTokens() < N) N = t->nTokens(); }   // per-bin layers have N W tokens
	}
	if (!W) { sn.error = "no SegmentedFourier in the graph"; return false; }
	if (N <= 0) { sn.error = "no TokenwiseLinear to take the token count from"; return false; }
	sn.N = N; sn.W = W;

	std::map<int, int> uidmap;               // full uid -> twin uid
	std::map<int, int> ringFor;              // full uid of a per-token layer -> twin uid of its FrameRing
	bool post_ring = false;                  // layers after a ring keep their full size
	auto twinPrev = [&](CFunc *f, std::vector<int> &prev) -> bool {
		for (int i = 0; i < f->no_previous_func(); ++i) {
			auto it = uidmap.find(f->previous(i)->uid());
			if (it == uidmap.end()) { sn.error = "input of " + f->getName() + " precedes it"; return false; }
			prev.push_back(it->second);
		}
		return true;
	};
	auto ringOver = [&](CFunc *data, int P) -> int {   // the ring that holds the last N frames of `data` (shared)
		auto it = ringFor.find(data->uid());
		if (it != ringFor.end()) return it->second;
		auto *r = new FrameRing(P, N);
		const int uid = sn.net.add(r, std::vector<int>{uidmap[data->uid()]});
		sn.rings.push_back(r);
		ringFor[data->uid()] = uid;
		return uid;
	};

	for (CFunc *f : full.funcs()) {
		if (!keep.count(f->uid())) continue;
		std::vector<int> prev;
		if (!twinPrev(f, prev)) return false;
		const int out = f->outSize();
		CFunc *g = nullptr;
		if (auto *c = dynamic_cast<CInput*>(f)) {
			if (f->uid() == main_uid) {
				auto *nc = new CInput(OutSize(W));
				sn.net.add(nc);
				sn.frame = nc; g = nc;
			} else {
				auto *nc = new CInput(OutSize(c->outSize()));
				sn.net.add(nc);
				frameCopyVars(c->input(), nc->mutable_input(), c->outSize());
				g = nc;
			}
		} else if (auto *s = dynamic_cast<SegmentedFourier*>(f)) {
			g = new SegmentedFourier(InSize(W), s->window()); sn.net.add(g, prev);
		} else if (auto *sl = dynamic_cast<Slice*>(f)) {
			if (sl->start() != 0) { sn.error = sl->getName() + " reads a non-leading slice (the clean half?)"; return false; }
			uidmap[f->uid()] = prev[0];      // the twin input is the first stream alone: the slice is the identity
			continue;
		} else if (dynamic_cast<CLogPower*>(f)) {
			g = new CLogPower(InSize(out / N)); sn.net.add(g, prev);
		} else if (dynamic_cast<CModulus2*>(f)) {
			g = new CModulus2(InSize(out / N)); sn.net.add(g, prev);
		} else if (auto *m = dynamic_cast<MelPower*>(f)) {
			auto *nm = new MelPower(1, m->window(), m->bands(), m->sampleRate(), m->fmin(), m->fmax(), m->linear());
			nm->setStandardize(m->mean(), m->scale());
			sn.net.add(nm, prev); g = nm;
		} else if (auto *il = dynamic_cast<Interleave*>(f)) {
			if (il->length() % N != 0) { sn.error = f->getName() + ": length not a multiple of N"; return false; }
			g = new Interleave(il->length() / N, il->k()); sn.net.add(g, prev);
		} else if (auto *cv = dynamic_cast<TimeFreqConv*>(f)) {
			// frame t reads the last T input frames: with T = 1 the layer is per-frame; otherwise a ring of T input
			// frames (zero until filled, like the causal zero padding of the full graph) feeds a T-frame copy of the
			// layer whose newest output frame is sliced out
			if (cv->nTokens() != N) { sn.error = f->getName() + ": token count differs from N"; return false; }
			const int T = cv->tTaps(), per = cv->outBins() * cv->cOut();
			if (T == 1) {
				g = new TimeFreqConv(1, cv->bins(), cv->cIn(), cv->cOut(), 1, cv->fTaps(), cv->stride()); sn.net.add(g, prev);
			} else {
				auto *r = new FrameRing(cv->bins() * cv->cIn(), T);
				const int ruid = sn.net.add(r, std::vector<int>{prev[0]});
				sn.rings.push_back(r);
				auto *c = new TimeFreqConv(T, cv->bins(), cv->cIn(), cv->cOut(), T, cv->fTaps(), cv->stride());
				const int cuid = sn.net.add(c, std::vector<int>{ruid, prev[1]});
				g = new Slice(InSize(T * per), OutSize(per), (T - 1) * per);
				sn.net.add(g, std::vector<int>{cuid});
			}
		} else if (auto *t = dynamic_cast<TokenwiseLinear*>(f)) {
			// per-token layers have N tokens; a per-bin layer (the local mask path) has N W tokens -> W per frame
			if (t->nTokens() % N != 0) { sn.error = f->getName() + ": token count not a multiple of N"; return false; }
			g = new TokenwiseLinear(t->nTokens() / N, t->eIn(), t->eOut()); sn.net.add(g, prev);
		} else if (auto *n = dynamic_cast<TokenNorm*>(f)) {
			g = new TokenNorm(1, n->dim(), n->eps()); sn.net.add(g, prev);
		} else if (auto *r = dynamic_cast<RotaryEmbed*>(f)) {
			auto *nr = new RotaryEmbed(1, r->dim(), r->base(), r->heads());
			sn.net.add(nr, prev); sn.ropes.push_back(nr); g = nr;
		} else if (auto *b = dynamic_cast<BornAttention*>(f)) {
			if (!b->causal()) { sn.error = b->getName() + " is not causal: no constant-state form"; return false; }
			auto *nb = new BornAttention(1, b->dim(), true, b->eps(), b->heads());
			nb->setDecay(b->decay()); nb->setStepMode(true);
			sn.net.add(nb, prev); sn.borns.push_back(nb); g = nb;
		} else if (auto *ga = dynamic_cast<GatedLinearAttention*>(f)) {
			auto *ng = new GatedLinearAttention(1, ga->dim(), ga->heads(), ga->scale());
			ng->setStepMode(true);
			sn.net.add(ng, prev); sn.glas.push_back(ng); g = ng;
		} else if (dynamic_cast<Hadamard*>(f)) {
			const int w = post_ring ? out : out / N;
			g = new Hadamard(InSize(w), InSize(w)); sn.net.add(g, prev);
		} else if (dynamic_cast<Residual*>(f)) {
			if (!post_ring && out % N != 0) { sn.error = f->getName() + ": size not a multiple of N"; return false; }
			const int w = post_ring ? out : out / N;
			g = new Residual(InSize(w), InSize(w)); sn.net.add(g, prev);
		} else if (dynamic_cast<CGelu*>(f)) {
			if (!post_ring && out % N != 0) { sn.error = f->getName() + ": size not a multiple of N"; return false; }
			g = new CGelu(InSize(post_ring ? out : out / N)); sn.net.add(g, prev);
		} else if (dynamic_cast<Linear*>(f)) {
			// dense readout: its data inputs are every predecessor but the last (N x P when it is the flat head over
			// the token projections; P, or [mean | max] = 2P, after the pools); the weights are the last input
			CFunc *data = f->previous(0);
			int in1 = 0;
			for (int i = 0; i + 1 < f->no_previous_func(); ++i) in1 += f->previous(i)->outSize();
			const int in2 = f->input().length_ - in1;
			if (!post_ring) {
				if (f->no_previous_func() != 2) { sn.error = f->getName() + ": several data inputs before a ring"; return false; }
				if (in1 % N != 0) { sn.error = f->getName() + ": data input not a multiple of N"; return false; }
				prev[0] = ringOver(data, in1 / N);
				post_ring = true;
			}
			g = new Linear(InSize(in1), InSize(in2)); sn.net.add(g, prev);
		} else if (dynamic_cast<TokenMeanPool*>(f) || dynamic_cast<TokenMaxPool*>(f)) {
			// the mean+max head has two pools over the same projections: they share one ring
			if (post_ring && !ringFor.count(f->previous(0)->uid())) { sn.error = f->getName() + " after a readout"; return false; }
			const int P = out;
			prev[0] = ringOver(f->previous(0), P);
			post_ring = true;
			if (dynamic_cast<TokenMeanPool*>(f)) g = new TokenMeanPool(InSize(N * P), OutSize(P));
			else g = new TokenMaxPool(InSize(N * P), OutSize(P));
			sn.net.add(g, prev);
		} else if (auto *ce = dynamic_cast<CrossEntropy*>(f)) {
			auto *nc = new CrossEntropy(InSize(f->input().length_));
			nc->setEps(ce->eps());
			sn.net.add(nc, prev); sn.ce = nc; sn.out = nc; g = nc;
		} else {
			sn.error = "layer " + f->getName() + " has no constant-state form";
			return false;
		}
		uidmap[f->uid()] = g->uid();
		sn.layers++;
	}
	if (!sn.frame) { sn.error = "main input not found"; return false; }
	if (!sn.out) {                           // SE: the output is the enhanced spectrum of the current frame
		const int w = outf->outSize() / N;
		auto *l2 = new L2Out(InSize(w));
		sn.net.add(l2, std::vector<int>{uidmap[out_uid]});
		sn.out = l2;
	}
	return true;
}

inline void frameNetReset(FrameNet &sn) {
	for (auto *b : sn.borns) b->resetState();
	for (auto *g : sn.glas) g->resetState();
	for (auto *r : sn.rings) r->reset();
}

// Feeds one frame of W real samples at absolute frame index pos; afterwards sn.out->input() holds the result.
inline void frameNetStep(FrameNet &sn, const float *frame, int pos) {
	Vars *in = sn.frame->mutable_input();
	for (int k = 0; k < sn.W; ++k) { in->real_[k] = frame[k]; in->imag_[k] = 0.f; }
	for (auto *r : sn.ropes) r->setPos(pos);
	sn.net.forward();
}

#endif /* IMPL_FRAMENET_H_ */
