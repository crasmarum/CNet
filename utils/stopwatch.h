#ifndef UTILS_STOPWATCH_H_
#define UTILS_STOPWATCH_H_

#include <chrono>

class StopWatch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
public:
	void Reset() {
		start_time = std::chrono::high_resolution_clock::now();
	}

	long long ElapsedTimeMicros() {
		std::chrono::time_point<std::chrono::high_resolution_clock>
			end_time = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	}

	StopWatch() {
		start_time = std::chrono::high_resolution_clock::now();
	}
};


#endif /* UTILS_STOPWATCH_H_ */
