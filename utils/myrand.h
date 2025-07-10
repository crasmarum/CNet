#ifndef UTILS_MYRAND_H_
#define UTILS_MYRAND_H_

#include <time.h>
#include <limits>
#include <cmath>
#include <cstdint>

// Not thread safe!
class SimpleRand {
	uint64_t state_;
	double last_gaussian_;
	double lastVal_;
	bool gen_gaussian_;

	static void RandInit(uint64_t *state, uint64_t seed) {
		*state = 0x2545F4914F6CDD1D ^ seed;
	}

	static uint64_t Rand(uint64_t *state) {
		// Xorshift algo.
		uint64_t x = state[0];
		x ^= x >> 12; // a
		x ^= x << 25; // b
		x ^= x >> 27; // c
		state[0] = x;
		return x * 0x2545F4914F6CDD1D;
	}

public:
	SimpleRand() {
		RandInit(&state_, clock());
		last_gaussian_ = 0.0;
		lastVal_ = 0.0;
		gen_gaussian_ = false;
	}

	SimpleRand(uint64_t seed)  {
		RandInit(&state_, seed);
		last_gaussian_ = 0.0;
		lastVal_ = 0.0;
		gen_gaussian_ = false;
	}

	uint64_t randUint64() {
		return Rand(&state_);
	}

	// Random int.
	int rand() {
		// this rather than casting to int for passing some statistical tests
		return (int)(Rand(&state_) >> 32);
	}

	// random int between 0 and range - 1.
	int rand(int range) {
		int ret = (int) Rand(&state_) % range;
		return ret < 0 ? range + ret : ret;
	}

	// random double between 0.0 and 1.0
	inline double randDouble() {
		return 5.42101086242752217E-20 * Rand(&state_);
	}

	// Generates values under the Normal(mean, std_deviation) distribution.
	double normal2(double mean, double std_deviation)
	{
		// Box–Muller transform algo
		static const double epsilon = std::numeric_limits<double>::min();
		static const double two_pi = 2.0 * 3.14159265358979323846;
		gen_gaussian_ = !gen_gaussian_;
		if (!gen_gaussian_) {
			return last_gaussian_ * std_deviation + mean;
		}

		double u1, u2;
		do {
			u1 = rand() * (1.0 / std::numeric_limits<int>::max());
			u2 = rand() * (1.0 / std::numeric_limits<int>::max());
		} while (u1 <= epsilon);

		double z0;
		z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
		last_gaussian_ = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
		return z0 * std_deviation + mean;
	}

	// Generates values under the Normal(mean, std_deviation) distribution.
	double normal(double mu, double sig) {
		double v1, v2, rsq, fac;
		if (lastVal_ == 0.) {
			do {
				v1 = 2.0 * randDouble() - 1.0;
				v2 = 2.0 * randDouble() - 1.0;
				rsq = v1 * v1 + v2 * v2;
			} while (rsq >= 1.0 || rsq == 0.0);
			fac = sqrt(-2.0 * log(rsq) / rsq);
			lastVal_ = v1 * fac;
			return mu + sig * v2 * fac;
		} else {
			fac = lastVal_;
			lastVal_ = 0.;
			return mu + sig * fac;
		}
	}

	template<typename T>
	void shuffle(int length, T* array) {
		for (int var = length - 1; var >= 1; --var) {
			int indx = rand(var + 1);
			if (indx == var) continue;
			T tmp = array[var];
			array[var] = array[indx];
			array[indx] = tmp;
		}
	}
};

#endif /* UTILS_MYRAND_H_ */
