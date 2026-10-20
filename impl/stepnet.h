#ifndef IMPL_STEPNET_H_
#define IMPL_STEPNET_H_

// Streaming (constant-state) decoding for a CNet language model.
//
// CNet graphs are static: every layer is built with its token count N baked into its sizes.
// A streaming decoder is therefore a SECOND graph with N = 1 -- the same layers in the same
// order and wiring, every per-token size divided by N, weights copied -- in which the Born
// attention layers keep a running density-matrix state (BornAttention::forwardStep), RoPE takes
// an absolute position, and the learned positional table is sliced one row per token. Feeding the
// tokens one at a time through this graph reproduces the fixed-window model's distribution at the
// last position exactly (up to rounding), without ever recomputing earlier tokens.
//
// Supported layers: CEmbedding, CInput (weights, and the positional table feeding a Residual),
// TokenwiseLinear, TokenNorm, RotaryEmbed, BornAttention, GatedLinearAttention, Residual, CGelu, SequenceCrossEntropy.
// Not supported (not constant-state): TriangFourier / Fourier mixers, SoftMax block norm.

#include <vector>
#include <string>
#include <map>
#include "cnet.h"
#include "cinput.h"
#include "embed.h"
#include "tokenwise.h"
#include "tokennorm.h"
#include "rotary.h"
#include "bornattn.h"
#include "residual.h"
#include "relu.h"
#include "seqcrossent.h"

struct StepNet {
	ComplexNet net;
	CEmbedding *emb = nullptr;
	SequenceCrossEntropy *seq = nullptr;
	CInput *pos_src = nullptr;     // the full model's learned positional table (N*E), not owned
	CInput *pos_step = nullptr;    // its one-row twin in `net`
	std::vector<RotaryEmbed*> ropes;
	std::vector<BornAttention*> borns;
	std::vector<GatedLinearAttention*> glas;
	int N = 0, E = 0, vocab = 0;
	std::string error;
};

static inline void copyVars(const Vars &from, Vars *to, int n) {
	for (int i = 0; i < n; ++i) { to->real_[i] = from.real_[i]; to->imag_[i] = from.imag_[i]; }
}

// Builds sn.net as the N=1 twin of `full` (a restored model). Returns false with sn.error set
// when the model contains a layer that has no constant-state form.
inline bool buildStepNet(ComplexNet &full, StepNet &sn) {
	auto *seq0 = (SequenceCrossEntropy*) full.findFirstOfType(isSeqCrossEntropy);
	auto *emb0 = (CEmbedding*) full.findFirstOfType(isEmbedding);
	if (!seq0 || !emb0) { sn.error = "model has no embedding / sequence head"; return false; }
	const int N = seq0->nPos();
	sn.N = N; sn.E = emb0->embeddingDim(); sn.vocab = seq0->vocab();
	std::map<int, int> uidmap;   // full uid -> step uid
	for (CFunc *f : full.funcs()) {
		std::vector<int> prev;
		for (int i = 0; i < f->no_previous_func(); ++i) {
			auto it = uidmap.find(f->previous(i)->uid());
			if (it == uidmap.end()) { sn.error = "input of " + f->getName() + " precedes it"; return false; }
			prev.push_back(it->second);
		}
		CFunc *g = nullptr;
		if (auto *e = dynamic_cast<CEmbedding*>(f)) {
			auto *ne = new CEmbedding(e->embeddingDim(), 1, e->noEmbeddings());
			sn.net.add(ne);
			copyVars(e->input(), ne->mutable_input(), e->noEmbeddings() * e->embeddingDim());
			sn.emb = ne; g = ne;
		} else if (auto *c = dynamic_cast<CInput*>(f)) {
			const int L = c->outSize();
			bool positional = c->no_next_func() > 0 && dynamic_cast<Residual*>(c->next(0)) != nullptr;
			if (positional) {
				if (L % N != 0) { sn.error = "positional input length not a multiple of N"; return false; }
				auto *nc = new CInput(OutSize(L / N));
				sn.net.add(nc);
				sn.pos_src = c; sn.pos_step = nc; g = nc;
			} else {
				auto *nc = new CInput(OutSize(L));
				sn.net.add(nc);
				copyVars(c->input(), nc->mutable_input(), L);
				g = nc;
			}
		} else if (auto *t = dynamic_cast<TokenwiseLinear*>(f)) {
			g = new TokenwiseLinear(1, t->eIn(), t->eOut()); sn.net.add(g, prev);
		} else if (auto *n = dynamic_cast<TokenNorm*>(f)) {
			g = new TokenNorm(1, n->dim(), n->eps()); sn.net.add(g, prev);
		} else if (auto *r = dynamic_cast<RotaryEmbed*>(f)) {
			auto *nr = new RotaryEmbed(1, r->dim(), r->base(), r->heads());
			sn.net.add(nr, prev); sn.ropes.push_back(nr); g = nr;
		} else if (auto *b = dynamic_cast<BornAttention*>(f)) {
			auto *nb = new BornAttention(1, b->dim(), b->causal(), b->eps(), b->heads());
			nb->setDecay(b->decay()); nb->setStepMode(true);
			sn.net.add(nb, prev); sn.borns.push_back(nb); g = nb;
		} else if (auto *ga = dynamic_cast<GatedLinearAttention*>(f)) {
			auto *ng = new GatedLinearAttention(1, ga->dim(), ga->heads(), ga->scale());
			ng->setStepMode(true);
			sn.net.add(ng, prev); sn.glas.push_back(ng); g = ng;
		} else if (dynamic_cast<Residual*>(f)) {
			if (f->outSize() % N != 0) { sn.error = "residual size not a multiple of N"; return false; }
			g = new Residual(InSize(f->outSize() / N), InSize(f->outSize() / N)); sn.net.add(g, prev);
		} else if (dynamic_cast<CGelu*>(f)) {
			g = new CGelu(InSize(f->outSize() / N)); sn.net.add(g, prev);
		} else if (auto *q = dynamic_cast<SequenceCrossEntropy*>(f)) {
			auto *nq = new SequenceCrossEntropy(InSize(q->vocab()), q->vocab());
			sn.net.add(nq, prev); sn.seq = nq; g = nq;
		} else {
			sn.error = "layer " + f->getName() + " has no constant-state form (Fourier mixer or block norm?)";
			return false;
		}
		uidmap[f->uid()] = g->uid();
	}
	if (!sn.seq || !sn.emb) { sn.error = "twin has no embedding / sequence head"; return false; }
	return true;
}

inline void stepReset(StepNet &sn) { for (auto *b : sn.borns) b->resetState(); for (auto *g : sn.glas) g->resetState(); }

// Feeds one token at absolute position pos; afterwards sn.seq->getProbability(0, k) is the
// next-token distribution. pos must be < N when the model has a learned positional table.
inline bool stepToken(StepNet &sn, int tok, int pos) {
	if (sn.pos_src) {
		if (pos >= sn.N) return false;
		const int E = sn.pos_src->outSize() / sn.N;
		Vars *dst = sn.pos_step->mutable_input();
		for (int i = 0; i < E; ++i) {
			dst->real_[i] = sn.pos_src->input().real_[pos * E + i];
			dst->imag_[i] = sn.pos_src->input().imag_[pos * E + i];
		}
	}
	for (auto *r : sn.ropes) r->setPos(pos);
	sn.emb->setInput(std::vector<int>{tok});
	sn.net.forward();
	return true;
}

#endif /* IMPL_STEPNET_H_ */
