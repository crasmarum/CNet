
#ifndef VARS_H_
#define VARS_H_

#include <algorithm>
#include <complex>
#include <cmath>
#include <iostream>

#include "assert.h"

/**
 * An array of complex numbers of size length_ stored with real parts first.
 * For example the array [z_0, z_1] is stored as [re_z_0, re_z_1, img_z_0, img_z_1] etc.
 */
class Vars {
	friend class ComplexNet;
	friend class CFunc;
	static bool no_data_;

public:
	int length_ = 0;
	float* real_ = NULL;
	float* imag_ = NULL;
	float* dz_real_ = NULL;
	float* dz_imag_ = NULL;
	float* dz_star_real_ = NULL;
	float* dz_star_imag_ = NULL;

	Vars() :
			length_(0), real_(0), imag_(0), dz_real_(
					0), dz_imag_(0), dz_star_real_(0), dz_star_imag_(0) {
	}
	Vars(int length) : length_(length) {
		assert(length > 0);
		if (no_data_) {
			return;
		}

		real_ = new float[length_ * 6];
		imag_ = real_ + length_;
		dz_real_ = imag_ + length_;
		dz_imag_ = dz_real_ + length_;
		dz_star_real_ = dz_imag_ + length_;
		dz_star_imag_ = dz_star_real_ + length_;
		std::fill(real_, real_ + length_ * 6, 0.0);
	}

	void zero_gradients() {
		if (no_data_) {
			return;
		}
		std::fill(dz_real_, dz_real_ + length_ * 4, 0.0);
	}

	void zero_input() {
		if (no_data_) {
			return;
		}
		std::fill(real_, real_ + length_ * 2, 0.0);
	}

	void zero_dZ() {
		if (no_data_) {
			return;
		}
		std::fill(dz_real_, dz_real_ + length_ * 2, 0.0);
	}

	void zero_dZ_star() {
		if (no_data_) {
			std::cerr << "Warning: this is a shell Vars" << std::endl;
			return;
		}
		std::fill(dz_star_real_, dz_star_real_ + length_ * 2, 0.0);
	}

	// Copy constructor.
	Vars(const Vars& other) : length_(other.length_), real_(0), imag_(0), dz_real_(0), dz_imag_(0),
			dz_star_real_(0), dz_star_imag_(0) {
		if (!other.real_) {
			return;
		}
		if (no_data_) {
			return;
		}

		real_ = new float[length_ * 6];
		imag_ = real_ + length_;
		dz_real_ = imag_ + length_;
		dz_imag_ = dz_real_ + length_;
		dz_star_real_ = dz_imag_ + length_;
		dz_star_imag_ = dz_star_real_ + length_;
		std::copy(other.real_, other.real_ + length_ * 6, real_);
	}

	virtual ~Vars() {
		if (real_) {
			delete[] real_;
			real_ = NULL;
		}
	}

	inline std::complex<float> z(int indx) const {
		assert(0 <= indx && indx < length_);
		return std::complex<float>(real_[indx], imag_[indx]);
	}

	inline std::complex<float> dz(int indx) const {
		assert(0 <= indx && indx < length_ && dz_real_ && dz_imag_);
		return std::complex<float>(dz_real_[indx], dz_imag_[indx]);
	}

	inline std::complex<float> dz_star(int indx) const {
		assert(0 <= indx && indx < length_ && dz_star_real_ && dz_star_imag_);
		return std::complex<float>(dz_star_real_[indx], dz_star_imag_[indx]);
	}

	unsigned long long size_in_bytes() const {
		return 6 * length_* sizeof(float);
	}

	std::string zToString(int maxLen) const {
		std::ostringstream oss;
		for (int var = 0; var < (length_ < maxLen ? length_ : maxLen); ++var) {
			oss << z(var) << ", ";
		}
		return oss.str();
	}

	std::string dzToString(int maxLen) const {
		std::ostringstream oss;
		for (int var = 0; var < (length_ < maxLen ? length_ : maxLen); ++var) {
			oss << dz(var) << ", ";
		}
		return oss.str();
	}

	std::string dz_starToString(int maxLen) const {
		std::ostringstream oss;
		for (int var = 0; var < (length_ < maxLen ? length_ : maxLen); ++var) {
			oss << dz_star(var) << ", ";
		}
		return oss.str();
	}

	void zSetValue(int indx, std::complex<float> value) {
		assert(0 <= indx && indx < length_ && dz_real_ && dz_imag_);
		real_[indx] = value.real();
		imag_[indx] = value.imag();
	}

	void dzSetValue(int indx, std::complex<float> value) {
		assert(0 <= indx && indx < length_ && dz_real_ && dz_imag_);
		dz_real_[indx] = value.real();
		dz_imag_[indx] = value.imag();
	}

	void dz_starSetValue(int indx, std::complex<float> value) {
		assert(0 <= indx && indx < length_ && dz_star_real_ && dz_star_imag_);
		dz_star_real_[indx] = value.real();
		dz_star_imag_[indx] = value.imag();
	}
};

inline float norm(Vars xx) {
	float sum = 0;
	for (int i = 0; i < xx.length_; ++i) {
		sum += xx.z(i).real() * xx.z(i).real()
				+ xx.z(i).imag() * xx.z(i).imag();
	}
	return sum;
}

inline float abs(Vars xx) {
	return sqrt(norm(xx));
}

#endif /* VARS_H_ */
