#ifndef IMPL_UTILS_H_
#define IMPL_UTILS_H_

#include <algorithm>
#include <complex>
#include <cmath>
#include "assert.h"

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
typedef __int128 int128_t;


/**
 * Multiplies a vector with a matrix.
 */
class MatrixMult {
	int len_;
	float *vec_real_;
	float *vec_imag_;
	float *mat_real_;
	float *mat_imag_;

public:
	MatrixMult() {
		len_ = 0;
		vec_real_ = 0;
		vec_imag_ = 0;
		mat_real_ = 0;
		mat_imag_ = 0;
	}
	MatrixMult(int len, float *vec_real, float *vec_imag, float *mat_real,
			float *mat_imag) {
		len_ = len;
		vec_real_ = vec_real;
		vec_imag_ = vec_imag;
		mat_real_ = mat_real;
		mat_imag_ = mat_imag;
	}

	/**
	 * v0, ... , vk, mat_00, ..., mat0k, mat10, ..., mat1k, ... , matkk
	 */
	void multiply(float *out_real, float *out_imag) {
#pragma omp parallel for
		for (int row = 0; row < len_; ++row) {
			std::complex<float> sum = 0;
			for (int col = 0; col < len_; ++col) {
				auto mat_real = mat_real_[row * len_ + col];
				auto mat_imag = mat_imag_[row * len_ + col];
				auto vec_real = vec_real_[col];
				auto vec_imag = vec_imag_[col];

				sum += std::complex<float>(mat_real, mat_imag)
						* std::complex<float>(vec_real, vec_imag);
			}
			out_real[row] = sum.real();
			out_imag[row] = sum.imag();
		}
	}

	/**
	 * 0 <= out_indx < len_ and 0 <= in_indx < len_
	 */
	std::complex<float> vec_dz(int out_indx, int in_indx) {
		auto mat_real = mat_real_[out_indx * len_ + in_indx];
		auto mat_imag = mat_imag_[out_indx * len_ + in_indx];

		return std::complex<float>(mat_real, mat_imag);
	}

	std::complex<float> mat_dz(int out_indx, int in_indx) {
		if (in_indx / len_ != out_indx) {
			return 0;
		}

		auto vec_real = vec_real_[in_indx % len_];
		auto vec_imag = vec_imag_[in_indx % len_];

		return std::complex<float>(vec_real, vec_imag);
	}
};

template<typename T>
class FastFourierTransform {
public:
	FastFourierTransform() {
	}

	static void transform(int length, T *real, T *imag) {
		if (length == 0) {
			return;
		}
		if ((length & (length - 1)) == 0) {  // power of 2
			transform_(length, real, imag);
			return;
		}

		transformBluestein(length, real, imag);
	}

	static void inverseTransform(int length, T *real, T *imag) {
		transform(length, imag, real);
	}

	static void convolve(int length, T *xreal, T *ximag, T *yreal, T *yimag) {
		if (length == 0) {
			return;
		}
		if ((length & (length - 1)) == 0) {
			convolve_(length, xreal, ximag, yreal, yimag);
			return;
		}

		int conv_len = 1;
		while (conv_len < length * 2 - 1) {
			conv_len *= 2;
		}

		T *areal = (T*) malloc(sizeof(T) * conv_len);
		T *aimag = (T*) malloc(sizeof(T) * conv_len);
		T *breal = (T*) malloc(sizeof(T) * conv_len);
		T *bimag = (T*) malloc(sizeof(T) * conv_len);
		for (int i = 0; i < length; i++) {
			areal[i] = xreal[i];
			aimag[i] = ximag[i];
			breal[i] = yreal[i];
			bimag[i] = yimag[i];
		}
		for (int i = length; i < conv_len; i++) {
			areal[i] = 0.0;
			aimag[i] = 0.0;
			breal[i] = 0.0;
			bimag[i] = 0.0;
		}

		convolve_(conv_len, areal, aimag, breal, bimag);

		for (int i = 0; i < length; i++) {
			yreal[i] = breal[i] + breal[length + i];
			yimag[i] = bimag[i] + bimag[length + i];
		}

		free(areal);
		free(aimag);
		free(breal);
		free(bimag);
	}

private:
	static inline unsigned int reverse(unsigned int val) {
		unsigned int ret = 0;
		unsigned int mask = 1U << 31;
		for (int var = 0; var < 32; ++var) {
			if (val & mask) {
				ret |= (1 << var);
			}
			mask = mask >> 1;
		}
		return ret;
	}

	static void convolve_(int length, T *xreal, T *ximag, T *yreal, T *yimag) {
		transform_(length, xreal, ximag);
		transform_(length, yreal, yimag);
		for (int i = 0; i < length; i++) {
			T temp = xreal[i] * yreal[i] - ximag[i] * yimag[i];
			ximag[i] = ximag[i] * yreal[i] + xreal[i] * yimag[i];
			xreal[i] = temp;
		}

		transform_(length, ximag, xreal);
		for (int i = 0; i < length; i++) {
			yreal[i] = xreal[i] / length;
			yimag[i] = ximag[i] / length;
		}
	}

	static void transform_(int length, T *real, T *imag) {
		if (length <= 1)
			return;

		int levels = -1;
		for (int i = 0; i < 32; i++) {
			if (1 << i == length) {
				levels = i;
				break;
			}
		}
		if (levels == -1)
			throw("Expected power of 2");

		T temp;
		for (int i = 0; i < length; i++) {
			int j = reverse(i) >> (32 - levels);
			if (j > i) {
				SWAP(real[i], real[j]);
				SWAP(imag[i], imag[j]);
			}
		}

		T *cosTable = (T*) malloc(sizeof(T) * length / 2);
		T *sinTable = (T*) malloc(sizeof(T) * length / 2);

		cosTable[0] = 1;
		sinTable[0] = 0;
		T qc = std::cos(2 * M_PI / length);
		T qs = std::sin(2 * M_PI / length);
		for (int i = 1; i < length / 2; i++) {
			cosTable[i] = cosTable[i - 1] * qc - sinTable[i - 1] * qs;
			sinTable[i] = sinTable[i - 1] * qc + cosTable[i - 1] * qs;
		}

		for (int size = 2; size <= length; size *= 2) {
			int halfsize = size / 2;
			int tablestep = length / size;
			for (int i = 0; i < length; i += size) {
				for (int j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
					T tpre = real[j + halfsize] * cosTable[k]
							+ imag[j + halfsize] * sinTable[k];
					T tpim = -real[j + halfsize] * sinTable[k]
							+ imag[j + halfsize] * cosTable[k];
					real[j + halfsize] = real[j] - tpre;
					imag[j + halfsize] = imag[j] - tpim;
					real[j] += tpre;
					imag[j] += tpim;
				}
			}
		}

		free(sinTable);
		free(cosTable);
	}

	static void transformBluestein(int length, T *real, T *imag) {
		int conv_len = 1;
		while (conv_len < length * 2 - 1) {
			conv_len *= 2;
		}

		T *tc = (T*) malloc(sizeof(T) * 2 * length);
		T *ts = (T*) malloc(sizeof(T) * 2 * length);
		tc[0] = 1;
		ts[0] = 0;
		T qc = std::cos(M_PI / length);
		T qs = std::sin(M_PI / length);
		for (int i = 1; i < 2 * length; i++) {
			tc[i] = tc[i - 1] * qc - ts[i - 1] * qs;
			ts[i] = ts[i - 1] * qc + tc[i - 1] * qs;
		}

		T *cosTable = (T*) malloc(sizeof(T) * length);
		T *sinTable = (T*) malloc(sizeof(T) * length);
		for (int i = 0; i < length; i++) {
			int j = (int) (((long) i * i) % (length * 2));
			cosTable[i] = tc[j];
			sinTable[i] = ts[j];
		}

		T *areal = (T*) malloc(sizeof(T) * conv_len);
		T *aimag = (T*) malloc(sizeof(T) * conv_len);
		T *breal = (T*) malloc(sizeof(T) * conv_len);
		T *bimag = (T*) malloc(sizeof(T) * conv_len);
		for (int i = length; i < conv_len; i++) {
			areal[i] = 0.0;
			aimag[i] = 0.0;
			breal[i] = 0.0;
			bimag[i] = 0.0;
		}

		for (int i = 0; i < length; i++) {
			areal[i] = real[i] * cosTable[i] + imag[i] * sinTable[i];
			aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
		}

		breal[0] = cosTable[0];
		bimag[0] = sinTable[0];
		for (int i = 1; i < length; i++) {
			breal[i] = breal[conv_len - i] = cosTable[i];
			bimag[i] = bimag[conv_len - i] = sinTable[i];
		}

		convolve_(conv_len, areal, aimag, breal, bimag);

		for (int i = 0; i < length; i++) {
			real[i] = breal[i] * cosTable[i] + bimag[i] * sinTable[i];
			imag[i] = -breal[i] * sinTable[i] + bimag[i] * cosTable[i];
		}

		free(tc);
		free(ts);
		free(sinTable);
		free(cosTable);
		free(areal);
		free(aimag);
		free(breal);
		free(bimag);
	}
};

class UnityRoots {
	std::vector<std::complex<float> > root_;

public:
	UnityRoots(int N) {
		for (int var = 0; var <= N - 1; ++var) {
			std::complex<float> r(std::cos(-2 * var * M_PI / N),
					std::sin(-2 * var * M_PI / N));
			root_.push_back(r);
		}
	}

	std::complex<float> root(int indx) {
		assert(indx >= 0);
		return root_[indx % root_.size()];
	}

	int size() {
		return root_.size();
	}
};

#endif /* IMPL_UTILS_H_ */
