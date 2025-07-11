#ifndef TESTS_GPUTESTS_H_
#define TESTS_GPUTESTS_H_

#include <iostream>
#include <vector>
#include <assert.h>
#include <complex>

#include "../utils/flags.h"
#include "../utils/stopwatch.h"
#include "../utils/myrand.h"

#include "../data/data.h"
#include "../impl/cnet.h"
#include "../impl/vars.h"
#include "../gpu/gpuvars.h"
#include "../gpu/compl.h"
#include "../gpu/gpumapping.h"
#include "../gpu/allocator.h"
#include "../gpu/gpu_func.h"

#include "../gpu/reduce.h"

#include "test.h"


FLAG_INT(segment_len, 6);
FLAG_BOOL(check_output, true)

void testRelu() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CRelu(Uid(2), InSize(size)), { 1 }); // @suppress("Invalid arguments")
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), { 2 });
	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CRelu(Uid(5), InSize(size)), { 4 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), { 5 });
	std::cout << gpuLead.cpuNet().toString() << std::endl;

	//gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.GpuForward(was_failure);
	assert(!was_failure);
	gpuLead.cpuNet().forward();

	for (int var = 1; var < 6; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
//		gpuLead.printMappingFromGpu(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.01));
		}
	}

	//*/

	float gpuLoss = gpuLead.getLoss(0)[6];
	float cpuLoss = ((CrossEntropy*) (gpuLead.cpuNet()[6]))->loss(0);
	if (check_output) {
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	gpuLead.cpuNet().backward(1);
	gpuLead.GpuBackward(1, was_failure);

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testFfft() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new FourierTrans(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), { 2 });
	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new FourierTrans(Uid(5), InSize(size)), { 4 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), { 5 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;
	//	gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);
	int ids[] = { 1 };
	for (auto id : ids) {
		auto inp = gpuLead.getMappingFor(gpuLead.cpuNet()[id]);
//		gpuLead.printMapping(inp, 0);
		inp->forward();
		for (auto fc : inp->getCpuFun()) {
			fc->forward();
		}
	}
	StopWatch sw;
	gpuLead.cpuNet()[2]->forward();
	std::cout << "CPU FFT forward elapsed time : " << sw.ElapsedTimeMicros() << "\n\n";
	gpuLead.cpuNet()[5]->forward();

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[5]);

	sw.Reset();
	mp->forward();
	std::cout << "gpu_fft_forward elapsed time : " << sw.ElapsedTimeMicros() << "\n\n";

	GpuMapping *ret = gpuLead.getMappingFor(gpuLead.cpuNet()[6]);
//	gpuLead.printMapping(ret, 0);
	if (check_output) {
		assert(gpuLead.testDataFromGpu(*ret, 0.1));
	}

	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testSoftmax() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new SoftMax(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), { 2 });
	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new SoftMax(Uid(5), InSize(size)), { 4 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), { 5 });
//	std::cout << gpuLead.cpuNet().toString() << std::endl;

//	gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);
	int ids[] = { 1 };
	for (auto id : ids) {
		auto inp = gpuLead.getMappingFor(gpuLead.cpuNet()[id]);
//		gpuLead.printMapping(inp, 0);
		inp->forward();
		for (auto fc : inp->getCpuFun()) {
			fc->forward();
		}
	}
	gpuLead.cpuNet()[2]->forward();
	gpuLead.cpuNet()[5]->forward();

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[5]);
	mp->forward();

	GpuMapping *ret = gpuLead.getMappingFor(gpuLead.cpuNet()[6]);
//	gpuLead.printMapping(ret, 0);
	if (check_output) {
		assert(gpuLead.testDataFromGpu(*ret, 0.001));
	}

	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testL2() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new L2Out(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new CInput(Uid(3), OutSize(size)));
	gpuLead.cpuNet().add(new L2Out(Uid(4), InSize(size)), { 3 });
	//	std::cout << gpuLead.cpuNet().toString() << std::endl;

	//gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();

	gpuLead.cpuNet()[1]->mutable_input()->real_[0] = 1;
	gpuLead.cpuNet()[1]->mutable_input()->imag_[0] = 2;
	gpuLead.cpuNet()[3]->mutable_input()->real_[1] = 3;
	gpuLead.cpuNet()[3]->mutable_input()->imag_[1] = 4;

	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	int ids[] = { 1 };
	for (auto id : ids) {
		auto inp = gpuLead.getMappingFor(gpuLead.cpuNet()[id]);
		//		gpuLead.printMapping(inp, 0);
		inp->forward();
		for (auto fc : inp->getCpuFun()) {
			fc->forward();
		}
	}
	std::cout << "2" << std::endl;

	gpuLead.cpuNet()[2]->forward();
	gpuLead.cpuNet()[4]->forward();

	gpuLead.printMappingFromGpu(gpuLead.getMappingFor(gpuLead.cpuNet()[1]));
	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	gpuLead.printMappingFromGpu(mp);

	StopWatch sw;
	((L2Gpu*)mp)->gpu_norm_forward(reduce_block_size);
	std::cout << "gpu_cross_ent_forward elapsed time : "
			<< sw.ElapsedTimeMicros() << "\n\n";

	auto loss = gpuLead.getLoss(0);
	for (auto pair : loss) {
		if (check_output) {
			std::cout << "Loss GPU: " << pair.first << " : " << pair.second
					<< std::endl;
			std::cout << "Loss CPU: " << pair.first << " : "
					<< ((L2Out*) ((gpuLead.cpuNet()[pair.first])))->loss()
					<< std::endl;

			assert(
					abs(
							pair.second
									- ((L2Out* )(gpuLead.cpuNet()[pair.first]))->loss())
							< 0.01);
		}
	}

	gpuLead.cpuNet()[2]->backward();
	gpuLead.cpuNet()[4]->backward();
	mp->backward(1024);
	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[2]));

//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[4]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}


	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testCE() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new CInput(Uid(3), OutSize(size)));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(size)), { 3 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;

	//gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);
	int ids[] = { 1 };
	for (auto id : ids) {
		auto inp = gpuLead.getMappingFor(gpuLead.cpuNet()[id]);
		//gpuLead.printMapping(inp, 0);
		inp->forward();
		for (auto fc : inp->getCpuFun()) {
			fc->forward();
		}
	}
	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	//gpuLead.printMapping(mp, 0);

	mp->forward();

	auto loss = gpuLead.getLoss(1);
	gpuLead.cpuNet()[2]->forward();
	gpuLead.cpuNet()[4]->forward();

	for (auto pair : loss) {
		if (check_output) {
			std::cout << "Loss GPU: " << pair.first << " : " << pair.second
					<< std::endl;
			std::cout << "Loss CPU: " << pair.first << " : "
					<< ((CrossEntropy*) ((gpuLead.cpuNet()[pair.first])))->loss(
							1) << std::endl;
			assert(
					abs(
							pair.second
									- ((CrossEntropy* )(gpuLead.cpuNet()[pair.first]))->loss(
											1)) < 0.01);
		}
	}
	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testLinear() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	int div = 1;
	assert(size % div == 0);
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(2), OutSize(size * size / div)));
	gpuLead.cpuNet().add(
			new Linear(Uid(3), InSize(size), InSize(size * size / div)),
			{ 1, 2 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(size / div)), { 3 });
	gpuLead.cpuNet().add(new CInput(Uid(5), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(6), OutSize(size * size / div)));
	gpuLead.cpuNet().add(
			new Linear(Uid(7), InSize(size), InSize(size * size / div)),
			{ 5, 6 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(8), InSize(size / div)), { 7 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;

	std::cout << "Trying to allocate " << gpuLead.cpuNet().getTotalSizeBytes() << " bytes on GPU..." << std::endl;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	int ids[2] = { 1, 2 };
	for (auto id : ids) {
		auto inp = gpuLead.getMappingFor(gpuLead.cpuNet()[id]);
		//gpuLead.printMapping(inp, 0);
		inp->forward();
		for (auto fc : inp->getCpuFun()) {
			fc->forward();
		}
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[7]);
	StopWatch sw;
	gpuLead.cpuNet()[3]->forward();
	gpuLead.cpuNet()[7]->forward();
	std::cout << "CPU linear forward elapsed time : " << sw.ElapsedTimeMicros()
			<< "\n\n";

	sw.Reset();
	mp->forward();
	std::cout << "gpu_linear_forward elapsed time : " << sw.ElapsedTimeMicros()
			<< "\n\n";

	auto out = gpuLead.getMappingFor(gpuLead.cpuNet()[4]);
	//gpuLead.printMapping(out, 0);

	if (check_output) {
		assert(gpuLead.testDataFromGpu(*out, 0.01));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testE2E() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new FourierTrans(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new SoftMax(Uid(3), InSize(size)), { 2 });
	gpuLead.cpuNet().add(new CRelu(Uid(4), InSize(size)), { 3 });

	gpuLead.cpuNet().add(new Residual(Uid(5), InSize(size)), { 4, 1 });

	gpuLead.cpuNet().add(new CInput(Uid(6), OutSize(size * size / 2)));
	gpuLead.cpuNet().add(new Linear(Uid(7), InSize(size), InSize(size * size / 2)), { 5, 6 });

	gpuLead.cpuNet().add(new CGelu(Uid(8), InSize(size / 2)), { 7 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(9), InSize(size / 2)), { 8 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	was_failure = false;
	gpuLead.GpuForward(was_failure);
	assert(!was_failure);

	for (int var = 1; var < 10; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
		//gpuLead.printMapping(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.1));
		}
	}

	float gpuLoss = gpuLead.getLoss(1)[9];
	float cpuLoss = ((CrossEntropy*) ((gpuLead.cpuNet().back())))->loss(1);
	std::cout << gpuLoss << "\n";
	std::cout << cpuLoss << "\n";
	if (check_output) {
		assert(abs(gpuLoss - cpuLoss) < 0.1);
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testResidual() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(2), OutSize(size)));
	gpuLead.cpuNet().add(new Residual(Uid(3), InSize(size)), { 1, 2 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(size)), { 3 });
	gpuLead.cpuNet().add(new CInput(Uid(5), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(6), OutSize(size)));
	gpuLead.cpuNet().add(new Residual(Uid(7), InSize(size)), { 5, 6 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(8), InSize(size)), { 7 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;
	//gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);
	gpuLead.GpuForward(was_failure);
	assert(!was_failure);
	gpuLead.cpuNet().forward();
	for (int var = 2; var < 5; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
		//gpuLead.printMapping(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.01));
		}
	}
	float gpuLoss = gpuLead.getLoss(0)[8];
	float cpuLoss = ((CrossEntropy*) ((gpuLead.cpuNet()[8])))->loss(0);
	if (check_output) {
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}
	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testHadamard() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(2), OutSize(size)));
	gpuLead.cpuNet().add(
			new Hadamard(Uid(3), InSize(size), InSize(size)),
			{ 1, 2 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(size)), { 3 });

	gpuLead.cpuNet().add(new CInput(Uid(5), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(6), OutSize(size)));
	gpuLead.cpuNet().add(
			new Hadamard(Uid(7), InSize(size), InSize(size)),
			{ 5, 6 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(8), InSize(size)), { 7 });
//	std::cout << gpuLead.cpuNet().toString() << std::endl;

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);


	bool was_error = false;
	gpuLead.GpuForward(was_error);
	assert(!was_error);

	gpuLead.cpuNet().forward();

	for (int var = 1; var < 5; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
//		gpuLead.printMapping(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.001));
		}
	}

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[4];
		float cpuLoss = ((CrossEntropy*) ((gpuLead.cpuNet()[4])))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testGelu() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CGelu(Uid(2), InSize(size)), { 1 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), { 2 });
	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CGelu(Uid(5), InSize(size)), { 4 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), { 5 });
	std::cout << gpuLead.cpuNet().toString() << std::endl;

	//gpuLead.cpuNet().init_inputs(1234);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.GpuForward(was_failure);
	assert(!was_failure);
	gpuLead.cpuNet().forward();

	for (int var = 1; var < 6; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
		//gpuLead.printMapping(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.01));
		}
	}

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[6];
		float cpuLoss = ((CrossEntropy*) ((gpuLead.cpuNet()[6])))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	std::cout << "\nTest passed: " << __func__ << "\n\n";
}

void testTwoOutputs() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;
	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new Residual(Uid(2), InSize(size)), { 1, 1 });
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), { 2 });
	//std::cout << gpuLead.cpuNet().toString() << std::endl;
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);
	bool was_error = false;
	gpuLead.GpuForward(was_error);
	assert(!was_error);
	gpuLead.cpuNet().forward();
	for (int var = 1; var < 4; ++var) {
		auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[var]);
		//gpuLead.printMapping(mp, 0);
		if (check_output) {
			assert(gpuLead.testDataFromGpu(*mp, 0.001));
		}
	}
	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}
	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testEmbedding() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int tok_size = segment_len;
	int no_tokens = 4;
	int out_no_tokens = 3;

	gpuLead.cpuNet().add(new CEmbedding(Uid(1), tok_size, out_no_tokens, no_tokens));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(2), InSize(tok_size * out_no_tokens)), { 1 });
	gpuLead.cpuNet().add(new CEmbedding(Uid(3), tok_size, out_no_tokens, no_tokens));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(tok_size * out_no_tokens)), { 3 });

//	gpuLead.cpuNet().init_inputs(1234567);
	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	((CEmbedding*)gpuLead.cpuNet()[1])->setInput({2});
	gpuLead.inputToGpu(gpuLead.cpuNet()[1]);

	((CEmbedding*)gpuLead.cpuNet()[3])->setInput({1, 0});
	gpuLead.inputToGpu(gpuLead.cpuNet()[3]);

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	//gpuLead.printMappingFromGpu(mp, 0);

	gpuLead.cpuNet().forward();

	mp->forward();

	mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	mp->forward();
//	gpuLead.printMappingFromGpu(mp, 0);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[2];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[2]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}
	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testZeroGradient() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CInput(Uid(3), OutSize(size)));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(size)), {3});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[4];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[4]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	gpuLead.cpuNet().backward(1);
	gpuLead.GpuBackward(1, was_failure);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[4]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testSoftMaxGradient() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new SoftMax(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new SoftMax(Uid(5), InSize(size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU SoftMax backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}



void testLinearMaxGradient() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;
	assert(size % 2 == 0);

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-1), OutSize(size * size / 2)));
	gpuLead.cpuNet().add(new Linear(Uid(2), InSize(size), InSize(size * size / 2)), {1, -1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size / 2)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-4), OutSize(size * size / 2)));
	gpuLead.cpuNet().add(new Linear(Uid(5), InSize(size), InSize(size * size / 2)), {4, -4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size / 2)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU linear backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testFFTGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new FourierTrans(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new FourierTrans(Uid(5), InSize(size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU FFT backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testResidualGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-1), OutSize(size)));
	gpuLead.cpuNet().add(new Residual(Uid(2), InSize(size), InSize(size)), {1, -1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-4), OutSize(size)));
	gpuLead.cpuNet().add(new Residual(Uid(5), InSize(size), InSize(size)), {4, -4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Residual backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testHadamardGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-1), OutSize(size)));
	gpuLead.cpuNet().add(new Hadamard(Uid(2), InSize(size), InSize(size)), {1, -1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CInput(Uid(-4), OutSize(size)));
	gpuLead.cpuNet().add(new Hadamard(Uid(5), InSize(size), InSize(size)), {4, -4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Hadamard backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testGeluGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CGelu(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new CGelu(Uid(5), InSize(size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Gelu backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[2]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[5]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}


void testEmbeddingGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CEmbedding(Uid(1), 4, size, 10));
	gpuLead.cpuNet().add(new CGelu(Uid(2), InSize(4 * size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(4 * size)), {2});

	gpuLead.cpuNet().add(new CEmbedding(Uid(4), 4, size, 10));
	gpuLead.cpuNet().add(new CGelu(Uid(5), InSize(4 * size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(4 * size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	((CEmbedding*)gpuLead.cpuNet()[1])->setInput({1, 2});
	gpuLead.inputToGpu(gpuLead.cpuNet()[1]);
	((CEmbedding*)gpuLead.cpuNet()[4])->setInput({2, 3});
	gpuLead.inputToGpu(gpuLead.cpuNet()[4]);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);


	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << "GPU loss: " << gpuLoss << "\n";
		std::cout << "CPU loss: " << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Embedding backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[1]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[4]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testInputGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), size));
	gpuLead.cpuNet().add(new CGelu(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), size));
	gpuLead.cpuNet().add(new CGelu(Uid(5), InSize(size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);


	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << "GPU loss: " << gpuLoss << "\n";
		std::cout << "CPU loss: " << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Input backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	gpuLead.cpuNet()[1]->mutable_input()->zero_gradients();
	gpuLead.cpuNet()[4]->mutable_input()->zero_gradients();
	mp->zeroGradients();
	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testEmbedding2() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;

	int emb_dim = 300;
	int no_tokens = 500;


	gpuLead.cpuNet().add(new CEmbedding(Uid(1), emb_dim, no_tokens, 10));
	gpuLead.cpuNet().add(new CGelu(Uid(2), InSize(emb_dim * no_tokens)), {1});
	gpuLead.cpuNet().add(new Residual(Uid(3), InSize(emb_dim * no_tokens), InSize(emb_dim * no_tokens)), {2, 1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(emb_dim * no_tokens)), {3});

	std::cout << gpuLead.cpuNet().toString() << std::endl;

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	((CEmbedding*)gpuLead.cpuNet()[1])->setInput({1, 2});
	gpuLead.inputToGpu(gpuLead.cpuNet()[1]);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);


	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[4];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[4]))))->loss(1);
		std::cout << "GPU loss: " << gpuLoss << "\n";
		std::cout << "CPU loss: " << cpuLoss << "\n";
		std::cout.flush();
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	if (check_output) {
		assert(gpuLead.testDataFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]), 0.01));
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

//	gpuLead.printMappingFromGpu(mp, 0);
//	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU Embedding backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

//	gpuLead.printGradientsFromGpu(*gpuLead.getMappingFor(gpuLead.cpuNet()[3]));
//	gpuLead.printMappingFromGpu(mp);
//	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testUpdateInputs() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int tok_size = segment_len;
	int no_tokens = 10;
	int out_no_tokens = 16;

	gpuLead.cpuNet().add(new CEmbedding(Uid(1), tok_size, out_no_tokens, no_tokens));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(2), InSize(tok_size * out_no_tokens)), { 1 });
	gpuLead.cpuNet().add(new CInput(Uid(3), segment_len));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(4), InSize(segment_len)), { 3 });

	gpuLead.cpuNet().init_inputs(1234567);
//	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	((CEmbedding*)gpuLead.cpuNet()[1])->setInput({1, 2});
	gpuLead.inputToGpu(gpuLead.cpuNet()[1]);

	gpuLead.cpuNet().forward();
	gpuLead.cpuNet().backward(1);

	gpuLead.GpuForward(was_failure);
	gpuLead.GpuBackward(1, was_failure);

	gpuLead.cpuNet().update(0.5, 1);

	Vars tmp1 = gpuLead.cpuNet()[1]->input();
	Vars tmp2 = gpuLead.cpuNet()[3]->input();

	gpuLead.gpuUpdateInputs(0.5);
	gpuLead.getInputsFromGpu();

	for (int indx = 0; indx < tmp1.length_; ++indx) {
		assert(abs(tmp1.z(indx) - gpuLead.cpuNet()[1]->input().z(indx)) < 0.001);
	}
	for (int indx = 0; indx < tmp2.length_; ++indx) {
		assert(abs(tmp2.z(indx) - gpuLead.cpuNet()[3]->input().z(indx)) < 0.001);
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testT_FFTGradient() {
	std::cout << "\n" << __func__ << "\n\n";
	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new TriangFourier(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(3), InSize(size)), {2});

	gpuLead.cpuNet().add(new CInput(Uid(4), OutSize(size)));
	gpuLead.cpuNet().add(new TriangFourier(Uid(5), InSize(size)), {4});
	gpuLead.cpuNet().add(new CrossEntropy(Uid(6), InSize(size)), {5});

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	gpuLead.cpuNet().forward();
	gpuLead.GpuForward(was_failure);

	if (check_output) {
		float gpuLoss = gpuLead.getLoss(1)[3];
		float cpuLoss = ((CrossEntropy*) (((gpuLead.cpuNet()[3]))))->loss(1);
		std::cout << gpuLoss << "\n";
		std::cout << cpuLoss << "\n";
		assert(abs(gpuLoss - cpuLoss) < 0.001);
	}

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[2]);
	if (check_output) {
		assert(gpuLead.testDataFromGpu(*mp, 0.001));
		assert(gpuLead.testGradientsFromGpu(*mp, 0.001));
	}

	gpuLead.printMappingFromGpu(mp, 0);
	gpuLead.printGradientsFromGpu(*mp);

	StopWatch sw;
	gpuLead.cpuNet().backward(1);
	std::cout << "CPU T_FFT backward time: " << sw.ElapsedTimeMicros() << std::endl;

	gpuLead.GpuBackward(1, was_failure);

	gpuLead.printMappingFromGpu(mp);
	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testGradientsFromGpu(*mp, 0.1));
	}

	mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	gpuLead.printMappingFromGpu(mp, 0);
	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testDataFromGpu(*mp, 0.01));
		assert(gpuLead.testGradientsFromGpu(*mp, 0.01));
	}

	gpuLead.adamUpdate(0.1, 0.9, 1);
	gpuLead.cpuNet().adamUpdate(0.1, 1, 0.9, 1);

	gpuLead.printMappingFromGpu(mp, 0);
	gpuLead.printGradientsFromGpu(*mp);

	if (check_output) {
		assert(gpuLead.testDataFromGpu(*mp, 0.01));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void testSave2() {
	ComplexNet net;
	int vocab_size = 42;
	int emb_dim = 32;
	int no_tokens = 32;
	int size = emb_dim * no_tokens;

	int emb = net.add(new CEmbedding(emb_dim, no_tokens, vocab_size));
	int pos_emb = net.add(new CInput(OutSize(size)));
	int last = net.add(new Residual(InSize(size), InSize(size)), {emb, pos_emb});

	for (int var = 0; var < 5; ++var) {
		int ftt = net.add(new TriangFourier(InSize(size)), { last });

		int f_data1 = net.add(new CInput(OutSize(size * size)));
		int f1 = net.add(new Linear(InSize(size), InSize(size * size)), {ftt, f_data1});
		int res1 = net.add(new Residual(InSize(size), InSize(size)), {ftt, f1});

		int sf = net.add(new SoftMax(InSize(size)), {res1});
		int gl = net.add(new CGelu(InSize(size)), {sf});
		int hd_data = net.add(new CInput(OutSize(size)));
		int hd = net.add(new Hadamard(InSize(size), InSize(size)), { gl, hd_data });
		int sf2 = net.add(new SoftMax(InSize(size)), {hd});

		last = net.add(new Residual(InSize(size), InSize(size)), {sf, sf2});
	}

	int lin_data = net.add(new CInput(OutSize(vocab_size * size)));
	int lin = net.add(new Linear(InSize(size), InSize(size * vocab_size)), { last, lin_data });
	int ce = net.add(new CrossEntropy(InSize(vocab_size)), { lin });

	net.init_inputs();
	net.init_exec_graph();

	ModelSaver saver;
	assert(saver.Save(net, "/Users/marcelcrasmaru/marcel/tmp/x.mod"));

	ComplexNet net2;
	assert(saver.Restore(net2, "/Users/marcelcrasmaru/marcel/tmp/x.mod"));
	net2.init_exec_graph();

	net.testEqual(net2);

	CEmbedding *emb1 = (CEmbedding*)net.findFirstOfType(isEmbedding);
	assert(emb1);
	CEmbedding *emb2 = (CEmbedding*)net2.findFirstOfType(isEmbedding);
	assert(emb2);
	CrossEntropy *ce1 = (CrossEntropy*)net.findFirstOfType(isCrossEntropy);
	assert(ce1);
	CrossEntropy *ce2 = (CrossEntropy*)net2.findFirstOfType(isCrossEntropy);
	assert(ce1);

	emb1->setInput({1, 2, 3});
	emb2->setInput({1, 2, 3});

	net.forward();
	net2.forward();
	assert(abs(ce1->loss(1) - ce2->loss(1)) < 1e-10);

//	std::cout << std::setprecision(9) << ce1->loss(1) << std::endl;
//	std::cout << std::setprecision(9) << ce2->loss(1) << std::endl;

}

void testBatch() {
	std::cout << "\n" << __func__ << "\n\n";

	CNet gpuLead;
	int size = segment_len;

	gpuLead.cpuNet().add(new CInput(Uid(1), OutSize(size)));
	gpuLead.cpuNet().add(new CrossEntropy(Uid(2), InSize(size)), {1});
	gpuLead.cpuNet().cloneForGpu(2);

	gpuLead.cpuNet().init_inputs();
	gpuLead.cpuNet().init_exec_graph();
	bool was_failure = false;
	gpuLead.AllocateNet(1, was_failure);
	assert(!was_failure);

	InputBatch batch(3, size);
	batch.add({{1.0, 2.0}, {1.0, 2.0}}, 0);
	batch.add({{3.0, 4.0}, {3.0, 4.0}}, 1);
	batch.add({{5.0, 6.0}, {5.0, 6.0}}, 2);

	InputFunc *inp = dynamic_cast<InputFunc*>(gpuLead.cpuNet()[1]);
	OutputFunc *outp = dynamic_cast<OutputFunc*>(gpuLead.cpuNet()[2]);
	gpuLead.batchToGpu(inp, outp, &batch);

	auto mp = gpuLead.getMappingFor(gpuLead.cpuNet()[1]);
	gpuLead.printMappingFromGpu(mp, 0);

	int count = 0;
	for (auto fun : mp->getCpuFun()) {
		if (!fun->mutable_input()->real_) {
			fun->mutable_input()->real_ = new float[fun->input().length_ * 2];
			fun->mutable_input()->imag_ = fun->mutable_input()->real_ + fun->input().length_;
		}
		batch.setInput(fun->mutable_input(), count);
		count++;
	}

	if (check_output) {
		assert(gpuLead.testDataFromGpu(*mp, 1e-10));
	}

	gpuLead.GpuForward(was_failure);
	std::cout << "Total loss: " << gpuLead.getLoss(outp, &batch) << "\n";
	for (int var = 0; var < 3; ++var) {
		std::cout << "Loss at batch " << var << ": " << batch.loss(var) << "\n";
	}

	gpuLead.cpuNet().forward();
	std::cout << "CPU loss: " << outp->loss(0) << "\n";
	if (check_output) {
		assert(outp->loss(0) == batch.loss(0));
	}

	std::cout << "Test passed: " << __func__ << "\n\n";
}

void runAllTests() {
	CNet cnet;
	assert(cnet.hasGPU());

	testL2();
	testSoftmax();
	testCE();
	testRelu();
	testResidual();
	testHadamard();
	testGelu();
	testTwoOutputs();
	testE2E();
	testFfft();
	testLinear();
	testEmbedding();
	testZeroGradient();
	testLinearMaxGradient();
	testFFTGradient();
	testResidualGradient();
	testHadamardGradient();
	testGeluGradient();
	testEmbeddingGradient();
	testInputGradient();
	testEmbedding2();
	testUpdateInputs();
	testT_FFTGradient();
	testSoftMaxGradient();
	testBatch();
}




#endif /* TESTS_GPUTESTS_H_ */
