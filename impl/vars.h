
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
 *
 * Layout: 8 contiguous length_-float segments
 *   [ real, imag, dz_real, dz_imag, dz_star_real, dz_star_imag, v_real, v_imag ].
 * The last complex buffer (v_real/v_imag) holds the Adam second moment; the
 * first moment reuses the dz slots (as plain momentum does). v is NOT touched
 * by zero_gradients so it persists across updates.
 */
class Vars {
	friend class ComplexNet;
	friend class CFunc;
	static bool no_data_;

public:
	static const int TRAIN_DIMS = 8;   // z, dz, dz_star, v (Adam 2nd moment)
	// Inference: z (2) plus the 2 dz segments used as scratch by fan-out /
	// chained element-wise layers. 4 is safe for every architecture and still
	// halves the footprint vs training. (A no-fan-out net is also correct at 2.)
	static const int INFER_DIMS = 4;
	// Segments actually allocated per variable (TRAIN_DIMS by default; set to
	// INFER_DIMS before building an inference-only net to save memory).
	static int dims_;

	int length_ = 0;
	float* real_ = NULL;
	float* imag_ = NULL;
	float* dz_real_ = NULL;
	float* dz_imag_ = NULL;
	float* dz_star_real_ = NULL;
	float* dz_star_imag_ = NULL;
	float* v_real_ = NULL;
	float* v_imag_ = NULL;

	Vars() :
			length_(0), real_(0), imag_(0), dz_real_(
					0), dz_imag_(0), dz_star_real_(0), dz_star_imag_(0),
					v_real_(0), v_imag_(0) {
	}
	// Points each segment into real_'s buffer; segments beyond dims_ stay NULL
	// (e.g. inference with dims_==2 allocates only z, so the gradient/moment
	// pointers are NULL and must not be dereferenced).
	void setSegmentPointers() {
		imag_         = real_ + length_;
		dz_real_      = (dims_ > 2) ? real_ + 2 * length_ : NULL;
		dz_imag_      = (dims_ > 3) ? real_ + 3 * length_ : NULL;
		dz_star_real_ = (dims_ > 4) ? real_ + 4 * length_ : NULL;
		dz_star_imag_ = (dims_ > 5) ? real_ + 5 * length_ : NULL;
		v_real_       = (dims_ > 6) ? real_ + 6 * length_ : NULL;
		v_imag_       = (dims_ > 7) ? real_ + 7 * length_ : NULL;
	}

	Vars(int length) : length_(length) {
		assert(length > 0);
		if (no_data_) {
			return;
		}

		real_ = new float[length_ * dims_];
		setSegmentPointers();
		std::fill(real_, real_ + length_ * dims_, 0.0);
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
			dz_star_real_(0), dz_star_imag_(0), v_real_(0), v_imag_(0) {
		if (!other.real_) {
			return;
		}
		if (no_data_) {
			return;
		}

		real_ = new float[length_ * dims_];
		setSegmentPointers();
		std::copy(other.real_, other.real_ + length_ * dims_, real_);
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

	// Adam second moment (per real/imag component).
	inline std::complex<float> v(int indx) const {
		assert(0 <= indx && indx < length_ && v_real_ && v_imag_);
		return std::complex<float>(v_real_[indx], v_imag_[indx]);
	}

	unsigned long long size_in_bytes() const {
		return dims_ * length_* sizeof(float);
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
