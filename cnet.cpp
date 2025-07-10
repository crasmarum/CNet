#include <assert.h>
#include <complex>
#include <iostream>
#include <initializer_list>
#include <vector>

#include "utils/flags.h"

#include "data/data.h"
#include "examples/sigmoid.h"
#include "impl/cinput.h"
#include "impl/cnet.h"
#include "impl/ft.h"
#include "impl/log.h"
#include "impl/vars.h"
#include "gpu/compl.h"
#include "gpu/gpu_func.h"

#include "tests/gputests.h"

FLAG_BOOL(run_tests, false)
FLAG_BOOL(mnist_cpu_train, false)
FLAG_BOOL(mnist_gpu_train, false)
FLAG_BOOL(mnist_model_accuracy, false)
FLAG_BOOL(test_sigmoid, false)

FLAG_STRING(model_path, "")

FLAG_INT(batch_size, 32)
FLAG_FLOAT(l_rate, 0.0001)
FLAG_FLOAT(beta, 0.9)
FLAG_INT(no_epochs, 10);

FLAG_INT(no_segments, 2);
FLAG_STRING(log_level, "INFO")

const char compile_date[] =
		" © ???? Ltd. All rights reserved. Build time:" __DATE__ " " __TIME__;

void createMnistNet(CNet& cnet, int *inp, int *outp) {
	*inp = cnet.add(new CInput(OutSize(28 * 28)));

	auto fft = cnet.add(new FourierTrans(InSize(28 * 28)), {*inp});
	auto hdata = cnet.add(new CInput(OutSize(28 * 28)));
	auto hdm = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft, hdata});
	auto gelu = cnet.add(new CGelu(InSize(28 * 28)), {hdm});
	auto res = cnet.add(new Residual(InSize(28 * 28), InSize(28 * 28)), {fft, gelu});

	auto fft2 = cnet.add(new FourierTrans(InSize(28 * 28)), {res});
	auto hdata1 = cnet.add(new CInput(OutSize(28 * 28)));
	auto hdm1 = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft2, hdata1});
	auto gelu1 = cnet.add(new CGelu(InSize(28 * 28)), {hdm1});
	auto res1 = cnet.add(new Residual(InSize(28 * 28), InSize(28 * 28)), {res, gelu1});

	auto fft3 = cnet.add(new FourierTrans(InSize(28 * 28)), {res1});
	auto hdata3 = cnet.add(new CInput(OutSize(28 * 28)));
	auto hdm3 = cnet.add(new Hadamard(InSize(28 * 28), InSize(28 * 28)), {fft3, hdata3});
	auto gelu3 = cnet.add(new CGelu(InSize(28 * 28)), {hdm3});
	auto res3 = cnet.add(new Residual(InSize(28 * 28), InSize(28 * 28)), {res1, gelu3});

	auto ldata = cnet.add(new CInput(OutSize(28 * 28 * 10)));
	auto lin = cnet.add(new Linear(InSize(28 * 28), InSize(28 * 28 * 10)), {res3, ldata});
	*outp = cnet.add(new CrossEntropy(InSize(10)), {lin});
}

void trainMnistCPU() {
	MnistDataReader reader;
	assert(reader.Open(mnist_images, mnist_labels, 60000));
	assert(reader.readData());

	CNet cnet;
	int cinp, coutp;
	createMnistNet(cnet, &cinp, &coutp);

	cnet.init_inputs();
	cnet.init_exec_graph(true);

	CInput *inp = (CInput*)cnet[cinp];
	CrossEntropy *cen = (CrossEntropy*)cnet[coutp];

	float avg_loss = 0;
	for (int epoch = 0; epoch < no_epochs; ++epoch) {
		reader.shuffle();
		for (int time = 0; time < reader.size(); ++time) {
			auto batch = reader.nextBatch(1);
			inp->setInput(batch, 0);

			cnet.forward();
			avg_loss +=  cen->loss(batch.label(0));

			if (time % 100 == 99) {
				std::cout << epoch << "\t" << time << "\t loss: " << (avg_loss / 100) << std::endl;
				avg_loss = 0;
			}
			cnet.backward(batch.label(0));

			if (time % 8 == 7) {
				cnet.updateInputs(0.0003);
			}

			if (time % 10000 == 9999) {
				assert(cnet.save(model_path));
				std::cout << "Model saved at: " << model_path << std::endl;
			}
		}
	}
}

void trainMnistGPU() {
	MnistDataReader reader;
	assert(reader.Open(mnist_images, mnist_labels, 60000));
	assert(reader.readData());

	CNet cnet;
	int cinp, coutp;
	createMnistNet(cnet, &cinp, &coutp);
	std::cout << cnet.cpuNet().toString() << std::endl;

	CInput *inp = (CInput*)cnet[cinp];
	CrossEntropy *cen = (CrossEntropy*)cnet[coutp];
	cnet.init_inputs();

	assert(cnet.allocateOnGpu(batch_size));

	float avg_loss = 0;
	for (int epoch = 0; epoch < no_epochs; ++epoch) {
		reader.shuffle();
		for (int time = 0; time < reader.size() / batch_size; ++time) {
			auto batch = reader.nextBatch(batch_size);

			cnet.gpuForward(inp, cen, batch);
			avg_loss += cnet.getLoss(cen, &batch);

			if (time % 100 == 99) {
				std::cout << epoch << "\t" << time << "\t loss: " << (avg_loss / 100) << std::endl;
				avg_loss = 0;
			}

			if (time % 800 == 799) {
				if (!cnet.getInputsFromGpu()) {
					std::cerr << "Warning: could get data from GPU." << std::endl;
				} else {
					assert(cnet.save(model_path));
					std::cout << "Model saved at: " << model_path << std::endl;
				}
			}

			cnet.gpuBackward();
			cnet.updateInputs(l_rate / batch.size());
		}
	}
}

void mnistModelAccuracy() {

	MnistDataReader reader;
	assert(reader.Open(mnist_images, mnist_labels, 10000));
	assert(reader.readData());

	CNet net;

	assert(net.restore(model_path));
	net.init_exec_graph(true);

	CInput *inp = (CInput*)net.findFirstOfType(isInput);
	CrossEntropy *cen = (CrossEntropy*)net.findFirstOfType(isCrossEntropy);

	float avg_loss = 0;
	float predicted = 0;
	for (int time = 0; time < reader.size(); ++time) {
		auto batch = reader.nextBatch(1);
		inp->setInput(batch, 0);

		net.forward();
		avg_loss += cen->loss(batch.label(0));
		if (cen->get_prediction() == batch.label(0)) {
			predicted++;
		}

		if (time % 100 == 99) {
			std::cout << time << "\t loss: " << (avg_loss / time)
					  << " accuracy: " << (predicted / time) << std::endl;
		}
	}

}

void testSigmoid() {
	CNet net;
	auto inp = net.add(new CInput(OutSize(128)));
	auto sigm = net.add(new CSigmoid(InSize(128)), {inp});
	auto l2 = net.add(new L2Out(InSize(128)), {sigm});

	net.init_inputs();
	net.init_exec_graph(true);
	for (int var = 0; var < 1000; ++var) {
		net.forward();
		std::cout << var << "\tLoss: " << ((L2Out*)net[l2])->loss() << std::endl;
		net.backward(0);
		net.updateInputs(0.9);
	}
}

int main(int argc, char *argv[]) {
	FLAGS::Parse(argc, argv);
	std::cout << compile_date << std::endl;
	std::cout << std::fixed << std::setprecision(3);

	FILELog::ReportingLevel() = FILELog::FromString(log_level);

	if (run_tests) {
		runAllTests();
	} else if (mnist_cpu_train) {
		trainMnistCPU();
	} else if (mnist_gpu_train) {
		trainMnistGPU();
	} else if (mnist_model_accuracy) {
		mnistModelAccuracy();
	} else if (test_sigmoid) {
		testSigmoid();
	}

	std::cout << "Done." << std::endl;
	return 0;
}






