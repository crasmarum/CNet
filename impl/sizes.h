#ifndef SIZES_H_
#define SIZES_H_

#include <complex>

const std::complex<float> li = std::complex<float>(0, 1);

class Size {
	int value_;
public:
	Size(int value) : value_(value) {
	}

	unsigned int value() {
		return value_;
	}
};

class InSize {
	unsigned int value_;
public:
	InSize(int value) : value_(value) {
	}

	unsigned int value() {
		return value_;
	}
};

class OutSize {
	unsigned int value_;
public:
	OutSize(int value) : value_(value) {
	}

	unsigned int value() {
		return value_;
	}
};

class Uid {
	int value_;
public:
	Uid(int value) : value_(value) {
	}

	int value() {
		return value_;
	}
};

class InUid : public Uid {
public:
	InUid(int value) : Uid(value) {
	}
};


#endif /* SIZES_H_ */
