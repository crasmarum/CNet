#ifndef IMPL_FT_H_
#define IMPL_FT_H_

#include <cfloat>
#include <cmath>

#include "cfunc.h"
#include "utils.h"
#include "vars.h"

class FourierTrans: public CFunc {
	FastFourierTransform<float> fft_;

UnityRoots u_roots_;
Vars buff_;

public:
	FourierTrans(Uid uid, InSize in_size) : CFunc(uid, in_size,
			OutSize(in_size.value())), u_roots_(in_size.value()), buff_(in_size.value()) {
	}

	FourierTrans(InSize in_size) : CFunc(in_size, OutSize(in_size.value())),
			u_roots_(in_size.value()), buff_(in_size.value()) {
	}

	virtual ~FourierTrans() {
	}

	virtual CFunc* clone(Uid uid) {
		return new FourierTrans(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "FourierTrans_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());

		std::copy(input().real_, input().real_ + input().length_, buff_.real_);
		std::copy(input().imag_, input().imag_  + input().length_, buff_.imag_);
		fft_.transform(buff_.length_, buff_.real_ , buff_.imag_);

		for (int indx = 0; indx < buff_.length_; ++indx) {
			buff_.real_[indx] /= sqrt(buff_.length_);	// need to keep vector norm 1
			buff_.imag_[indx] /= sqrt(buff_.length_);
		}

		for (int out_indx = 0; out_indx < no_outputs(); ++out_indx) {
			std::copy(buff_.real_, buff_.real_ + buff_.length_,
					  output(out_indx).real_ + offset(out_indx));
			std::copy(buff_.imag_, buff_.imag_  + buff_.length_,
					  output(out_indx).imag_+ offset(out_indx));
		}
	}

	virtual void backward() {
		float scaling = sqrt(buff_.length_);
		for (int indx = 0; indx < no_outputs(); ++indx) {
			#pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_star_dz = 0;

				for (int out_pos = 0; out_pos < out_size_; ++out_pos) {
					int exp = in_indx * out_pos;
					auto dLdz      = output(indx).dz(offset(indx) + out_pos);
					auto dLdz_star = output(indx).dz_star(offset(indx) + out_pos);

					sum_dz += dLdz * u_roots_.root(exp);
					sum_star_dz += dLdz_star * std::conj(u_roots_.root(exp));
				}

				input().dz_real_[in_indx] += sum_dz.real() / scaling;
				input().dz_imag_[in_indx] += sum_dz.imag() / scaling;
				input().dz_star_real_[in_indx] += sum_star_dz.real() / scaling;
				input().dz_star_imag_[in_indx] += sum_star_dz.imag() / scaling;
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const FourierTrans& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}

		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const FourierTrans& rhs) const
	{
		return !operator==(rhs);
	}
};

class TriangFourier: public CFunc {

UnityRoots u_roots_;

public:
	TriangFourier(Uid uid, InSize in_size) : CFunc(uid, in_size,
			OutSize(in_size.value())), u_roots_(in_size.value()) {
	}

	TriangFourier(InSize in_size) : CFunc(in_size, OutSize(in_size.value())),
			u_roots_(in_size.value()) {
	}

	virtual ~TriangFourier() {
	}

	virtual CFunc* clone(Uid uid) {
		return new TriangFourier(uid, InSize(input().length_));
	}

	virtual std::string getName() {
		return "TriangFourier_" + std::to_string(uid_);
	}

	virtual void forward() {
		assert(no_outputs());

		float scaling = sqrt(input().length_);
		//#pragma omp parallel for
		for (int out_pos = 0; out_pos < out_size_; ++out_pos) {

			std::complex<float> sum = 0;
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				if (out_pos < in_indx) { continue; }

				int exp = in_indx * out_pos;
				sum += input().z(in_indx) * u_roots_.root(exp);
			}

			sum /= scaling;

			for (int indx = 0; indx < no_outputs(); ++indx) {
				output(indx).real_[offset(indx) + out_pos] = sum.real();
				output(indx).imag_[offset(indx) + out_pos] = sum.imag();
			}
		}
	}

	virtual void backward() {
		float scaling = sqrt(input().length_);
		for (int var = 0; var < no_outputs(); ++var) {

			//#pragma omp parallel for
			for (int in_indx = 0; in_indx < input().length_; ++in_indx) {
				std::complex<float> sum_dz = 0;
				std::complex<float> sum_star_dz = 0;

				for (int out_pos = 0; out_pos < out_size_; ++out_pos) {
					if (out_pos < in_indx) { continue;}

					int exp = in_indx * out_pos;
					auto dLdz      = output(var).dz(offset(var) + out_pos);
					auto dLdz_star = output(var).dz_star(offset(var) + out_pos);

					sum_dz += dLdz * u_roots_.root(exp);
					sum_star_dz += dLdz_star * std::conj(u_roots_.root(exp));
				}

				input().dz_real_[in_indx] += sum_dz.real() / scaling;
				input().dz_imag_[in_indx] += sum_dz.imag() / scaling;
				input().dz_star_real_[in_indx] += sum_star_dz.real() / scaling;
				input().dz_star_imag_[in_indx] += sum_star_dz.imag() / scaling;
			}
		}
	}

	virtual void backward(int label) {
	}

	virtual std::complex<float>      dz(int out_indx, int in_index) {
		return 0;
	}
	virtual std::complex<float> dz_star(int out_indx, int in_index) {
		return 0;
	}

	bool operator==(const TriangFourier& rhs) const
	{
		if (this->uid_ != rhs.uid_
				|| this->input().length_ != rhs.input().length_
				|| this->out_size_ != rhs.out_size_
				|| this->no_outputs() != rhs.no_outputs()) {
			return false;
		}

		for (int var = 0; var < this->no_outputs(); ++var) {
			if (next(var)->uid() != rhs.next(var)->uid()) {
				return false;
			}
		}

		return true;
	}
	bool operator!=(const TriangFourier& rhs) const
	{
		return !operator==(rhs);
	}
};


#endif /* IMPL_FT_H_ */
