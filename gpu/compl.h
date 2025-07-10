#ifndef TESTS_COMPL_H_
#define TESTS_COMPL_H_

#include <iostream>
#include <cmath>
#include <complex>

#ifdef __CUDACC__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#else
#define __device__
#define __shared__
#define __host__
#endif

struct cmplx_ {
	float real = 0.f;
	float imag = 0.f;

	__host__
	__device__
	cmplx_() :
			real(0.f), imag(0.f) {
	}

	__host__
	__device__
	cmplx_(double xx, double yy) :
			real((float) xx), imag((float) yy) {
	}

	cmplx_(const std::complex<float> &cmpl) :
			real(cmpl.real()), imag(cmpl.imag()) {
	}

	__host__
	__device__
	cmplx_& operator+=(const cmplx_ &rhs) {
		real = real + rhs.real;
		imag = imag + rhs.imag;
		return *this;
	}

	__host__
	__device__
	cmplx_& operator-=(const cmplx_ &rhs) {
		real -= rhs.real;
		imag -= rhs.imag;
		return *this;
	}

	__host__
	__device__
	cmplx_& operator*=(const cmplx_ &rhs) {
		*this = { (real - imag) * rhs.imag + real * (rhs.real - rhs.imag), (real
				+ imag) * rhs.real - real * (rhs.real - rhs.imag) };
		return *this;
	}

	__host__
	__device__
	cmplx_& operator=(const cmplx_ &rhs) {
		real = rhs.real;
		imag = rhs.imag;
		return *this;
	}
};

__host__ __device__
inline cmplx_ cmplx(double xx, double yy) {
	return cmplx_ { xx, yy };
}

__host__ __device__
inline bool operator==(const cmplx_ &lhs, const cmplx_ &rhs) {
	return lhs.real == rhs.real && lhs.imag == rhs.imag;
}

__host__
__device__ inline bool operator!=(const cmplx_ &lhs, const cmplx_ &rhs) {
	return !(lhs == rhs);
}

__host__
__device__ inline cmplx_ operator+(const cmplx_ &lhs, const cmplx_ &rhs) {
	auto result = lhs;
	result += rhs;
	return result;
}

__host__
__device__ inline cmplx_ operator*(const cmplx_ &lhs, const cmplx_ &rhs) {
	auto result = lhs;
	result *= rhs;
	return result;
}

__host__
__device__ inline cmplx_ operator*(float lhs, const cmplx_ &rhs) {
	cmplx_ result = { lhs * rhs.real, lhs * rhs.imag };
	return result;
}

__host__
__device__ inline cmplx_ operator*(const cmplx_ &lhs, float rhs) {
	cmplx_ result = { rhs * lhs.real, rhs * lhs.imag };
	return result;
}

__host__
__device__ inline cmplx_ operator/(const cmplx_ &lhs, float rhs) {
	cmplx_ result = { lhs.real / rhs, lhs.imag / rhs};
	return result;
}

__host__
__device__ inline cmplx_ operator-(const cmplx_ &xx, const cmplx_ &yy) {
	return cmplx_ { xx.real - yy.real, xx.imag - yy.imag };
}

__host__
__device__ inline cmplx_ operator-(const cmplx_ &xx) {
	return cmplx_ { -xx.real, -xx.imag };
}

__host__
__device__ inline cmplx_ conj_(const cmplx_ &xx) {
	return cmplx_ { xx.real, -xx.imag };
}

__host__
__device__ inline float norm_(const cmplx_ &xx) {
	return xx.real * xx.real + xx.imag * xx.imag;
}

__host__
__device__ inline float abs_(const cmplx_ &xx) {
	return sqrt(xx.real * xx.real + xx.imag * xx.imag);
}

inline std::ostream& operator<<(std::ostream &out, cmplx_ const &zz) {
	return out << "(" << zz.real << "," << zz.imag << ")";
}

struct Grads {
	cmplx_ grad_;
	cmplx_ grad_star_;

	__host__
	__device__
	Grads(cmplx_ grad, cmplx_ grad_star) {
		grad_ = grad;
		grad_star_ = grad_star;
	}

	__host__
	__device__
	Grads() : grad_(0, 0), grad_star_(0, 0) {
	}

	__host__
	__device__
	Grads& operator+=(const Grads &rhs) {
		grad_ += rhs.grad_;
		grad_star_ += rhs.grad_star_;
		return *this;
	}
};

__host__
__device__ inline Grads  operator+ (const Grads& lhs, const Grads& rhs) {
	  auto result = lhs;
	  result += rhs;
	  return result;
}

#endif /* TESTS_COMPL_H_ */
