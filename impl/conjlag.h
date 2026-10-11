#ifndef IMPL_CONJLAG_H_
#define IMPL_CONJLAG_H_

#include <complex>
#include "cfunc.h"
#include "vars.h"

// Conjugate lag product (instantaneous autocorrelation at lag tau):
//
//     y_t = x_t * conj(x_{t-tau})      (t >= tau;  y_t = 0 for t < tau)
//
// This is the nonlinear, non-holomorphic primitive behind cyclostationary /
// higher-order modulation features: mean-pooling y over t estimates the lag-tau
// autocorrelation R(tau); an FFT of y over t gives the cyclic spectrum (spectral
// lines at the symbol-rate / carrier cycle frequencies). It is the complex,
// two-input generalization of CModulus2 (|z|^2 = x*conj(x) is the tau=0 case).
//
// Wirtinger backward. With a = x_t, b = x_{t-tau}, output y = a*conj(b) and
// ybar = conj(a)*b, so each input index i accumulates two roles:
//   role a (i = t,      needs i >= tau):  dL/dx_i  += dy_i * conj(x_{i-tau})
//                                         dL/dx~_i += dy~_i * x_{i-tau}
//   role b (i = t-tau,  needs i+tau < L): dL/dx_i  += dy~_{i+tau} * conj(x_{i+tau})
//                                         dL/dx~_i += dy_{i+tau}  * x_{i+tau}
// where dy = dL/dy and dy~ = dL/dybar are the output's dz() / dz_star().
class ConjLagProduct : public CFunc {
	int tau_;

public:
	ConjLagProduct(Uid uid, InSize in_size, int tau)
		: CFunc(uid, in_size, OutSize(in_size.value())), tau_(tau) {}

	ConjLagProduct(InSize in_size, int tau)
		: CFunc(in_size, OutSize(in_size.value())), tau_(tau) {}

	virtual ~ConjLagProduct() {}

	int tau() const { return tau_; }

	virtual CFunc* clone(Uid uid) {
		return new ConjLagProduct(uid, InSize(input().length_), tau_);
	}

	virtual std::string getName() {
		return "ConjLagProduct_" + std::to_string(uid_);
	}

	virtual void forward() {
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int t = 0; t < out_size_; ++t) {
				if (t >= tau_) {
					std::complex<float> a(input().real_[t], input().imag_[t]);
					std::complex<float> b(input().real_[t - tau_], input().imag_[t - tau_]);
					auto y = a * std::conj(b);
					output(indx).real_[offset(indx) + t] = y.real();
					output(indx).imag_[offset(indx) + t] = y.imag();
				} else {
					output(indx).real_[offset(indx) + t] = 0.0f;
					output(indx).imag_[offset(indx) + t] = 0.0f;
				}
			}
		}
	}

	virtual void backward() {
		const int L = input().length_;
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int i = 0; i < L; ++i) {
				std::complex<float> dz_acc(0, 0), dzs_acc(0, 0);
				if (i >= tau_) {                                   // role a: output t = i
					auto dy  = output(indx).dz(offset(indx) + i);
					auto dys = output(indx).dz_star(offset(indx) + i);
					std::complex<float> xlag(input().real_[i - tau_], input().imag_[i - tau_]);
					dz_acc  += dy  * std::conj(xlag);
					dzs_acc += dys * xlag;
				}
				if (i + tau_ < L) {                                // role b: output t = i + tau
					auto dy  = output(indx).dz(offset(indx) + i + tau_);
					auto dys = output(indx).dz_star(offset(indx) + i + tau_);
					std::complex<float> xfwd(input().real_[i + tau_], input().imag_[i + tau_]);
					dz_acc  += dys * std::conj(xfwd);
					dzs_acc += dy  * xfwd;
				}
				input().dz_real_[i]      += dz_acc.real();
				input().dz_imag_[i]      += dz_acc.imag();
				input().dz_star_real_[i] += dzs_acc.real();
				input().dz_star_imag_[i] += dzs_acc.imag();
			}
		}
	}

	virtual void backward(int label) {}

	virtual std::complex<float>      dz(int out_indx, int in_index) { return 0; }
	virtual std::complex<float> dz_star(int out_indx, int in_index) { return 0; }
};

#endif /* IMPL_CONJLAG_H_ */
