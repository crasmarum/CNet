#ifndef TEST_H_
#define TEST_H_

#include <iostream>
#include <complex>
#include <iostream>
#include <iomanip>


#include "../impl/cfunc.h"
#include "../impl/cinput.h"
#include "../impl/cnet.h"
#include "../impl/l2out.h"
#include "../impl/linear.h"
#include "../impl/residual.h"
#include "../impl/crossent.h"
#include "../impl/softmax.h"
#include "../impl/embed.h"
#include "../impl/ft.h"
#include "../impl/relu.h"
#include "../impl/model_saver.h"

#include "../utils/stopwatch.h"

using namespace std;

CFunc* addFourierTest(ComplexNet &cnet, CFunc* x, int no_tokens, int emb_dim, int heads) {
	int fft = cnet.add(new FourierTrans(InSize(no_tokens * emb_dim)), {x->uid()});
	assert(x->outSize() == no_tokens * emb_dim);
	std::vector<int> h_list;

	for (int hd = 0; hd < heads; ++hd) {
		int hd_data = cnet.add(new CInput(OutSize(no_tokens * emb_dim )));
		int h1 = cnet.add(new Hadamard(InSize(no_tokens * emb_dim), InSize(no_tokens * emb_dim)), {fft, hd_data});
		h_list.push_back(h1);
	}

	int concat = cnet.add(new Linear(InSize(heads * no_tokens * emb_dim), InSize(heads * no_tokens * emb_dim * no_tokens * emb_dim)),
			h_list);
	int res = cnet.add(new Residual(InSize(no_tokens * emb_dim)), {concat, x->uid()});
	int relu = cnet.add(new CGelu(InSize(no_tokens * emb_dim)), {res});
	int sf = cnet.add(new SoftMax(InSize(no_tokens * emb_dim)), {relu});
	res = cnet.add(new Residual(InSize(no_tokens * emb_dim)), {res, sf});

	return cnet[res];
}

void createBigNet(ComplexNet& cnet, int no_tokens, int emb_dim, int vocab_size, int heads) {
	std::cout << "Create net..." << std::endl;

	int x = cnet.add(new CEmbedding(emb_dim, no_tokens, vocab_size));
//	int u = cnet.add(new CInput(OutSize(batch * embed * batch * embed)));
//	int y = cnet.add(new Linear(InSize(batch * embed), InSize(batch * embed * batch * embed)), {x, u});

	CFunc *ftt = addFourierTest(cnet, cnet[x], no_tokens, emb_dim, heads);
	ftt = addFourierTest(cnet, ftt, no_tokens, emb_dim, heads);
	ftt = addFourierTest(cnet, ftt, no_tokens, emb_dim, heads);
	ftt = addFourierTest(cnet, ftt, no_tokens, emb_dim, heads);
	ftt = addFourierTest(cnet, ftt, no_tokens, emb_dim, heads);
	ftt = addFourierTest(cnet, ftt, no_tokens, emb_dim, heads);

	int lin_data = cnet.add(new CInput(OutSize(no_tokens * emb_dim * vocab_size)));

	int lin = cnet.add(new Linear(InSize(no_tokens * emb_dim), InSize(no_tokens * emb_dim * vocab_size)), {ftt->uid(), lin_data});
	cnet.add(new CrossEntropy(InSize(vocab_size)), {lin});
}

void testEqual(ComplexNet& cnet, ComplexNet& cnet2) {
	assert(cnet.functionList().size() == cnet2.functionList().size());

	int indx = 0;
	for (auto layer : cnet.functionList()) {
		if(CInput* input = dynamic_cast<CInput*>(layer)) {
			CInput* other = dynamic_cast<CInput*>(cnet2.functionList()[indx]);
			assert(*input ==  *other);
		} else if(Hadamard* hadm = dynamic_cast<Hadamard*>(layer)) {
			Hadamard* other = dynamic_cast<Hadamard*>(cnet2.functionList()[indx]);
			assert(*hadm ==  *other);
		} else if(Linear* linear = dynamic_cast<Linear*>(layer)) {
			Linear* other = dynamic_cast<Linear*>(cnet2.functionList()[indx]);
			assert(*linear ==  *other);
		} else if(CEmbedding* emb = dynamic_cast<CEmbedding*>(layer)) {
			CEmbedding* other = dynamic_cast<CEmbedding*>(cnet2.functionList()[indx]);
			assert(*emb ==  *other);
		} else if(FourierTrans* four = dynamic_cast<FourierTrans*>(layer)) {
			FourierTrans* other = dynamic_cast<FourierTrans*>(cnet2.functionList()[indx]);
			assert(*four ==  *other);
		} else if(Residual* res = dynamic_cast<Residual*>(layer)) {
			Residual* other = dynamic_cast<Residual*>(cnet2.functionList()[indx]);
			assert(*res ==  *other);
		} else if(CrossEntropy* ce = dynamic_cast<CrossEntropy*>(layer)) {
			CrossEntropy* other = dynamic_cast<CrossEntropy*>(cnet2.functionList()[indx]);
			assert(*ce ==  *other);
		} else if(SoftMax* sf = dynamic_cast<SoftMax*>(layer)) {
			SoftMax* other = dynamic_cast<SoftMax*>(cnet2.functionList()[indx]);
			assert(*sf ==  *other);
		} else if(CGelu* gl = dynamic_cast<CGelu*>(layer)) {
			CGelu* other = dynamic_cast<CGelu*>(cnet2.functionList()[indx]);
			assert(*gl ==  *other);
		} else if(CRelu* gl = dynamic_cast<CRelu*>(layer)) {
			CRelu* other = dynamic_cast<CRelu*>(cnet2.functionList()[indx]);
			assert(*gl ==  *other);
		} else {
			std::cerr << "Unexpected func: " << layer->getName() << std::endl;
			assert(false);
		}
		indx++;
	}

	((CEmbedding*)cnet[1])->setInput({1, 2});
	((CEmbedding*)cnet2[1])->setInput({1, 2});
	auto old = cnet.back()->input().z(0);

	cnet.forward();
	cnet2.forward();

	assert(cnet.back()->input().z(0) == cnet2.back()->input().z(0));
	assert(old != cnet.back()->input().z(0));
}

void testEqual2(ComplexNet& cnet, ComplexNet& cnet2) {
	assert(cnet.functionList().size() == cnet2.functionList().size());

	int indx = 0;
	for (auto layer : cnet.functionList()) {
		if(CInput* input = dynamic_cast<CInput*>(layer)) {
			CInput* other = dynamic_cast<CInput*>(cnet2.functionList()[indx]);
			assert(*input ==  *other);
		} else if(Hadamard* hadm = dynamic_cast<Hadamard*>(layer)) {
			Hadamard* other = dynamic_cast<Hadamard*>(cnet2.functionList()[indx]);
			assert(*hadm ==  *other);
		} else if(Linear* linear = dynamic_cast<Linear*>(layer)) {
			Linear* other = dynamic_cast<Linear*>(cnet2.functionList()[indx]);
			assert(*linear ==  *other);
		} else if(CEmbedding* emb = dynamic_cast<CEmbedding*>(layer)) {
			CEmbedding* other = dynamic_cast<CEmbedding*>(cnet2.functionList()[indx]);
			assert(*emb ==  *other);
		} else if(FourierTrans* four = dynamic_cast<FourierTrans*>(layer)) {
			FourierTrans* other = dynamic_cast<FourierTrans*>(cnet2.functionList()[indx]);
			assert(*four ==  *other);
		} else if(Residual* res = dynamic_cast<Residual*>(layer)) {
			Residual* other = dynamic_cast<Residual*>(cnet2.functionList()[indx]);
			assert(*res ==  *other);
		} else if(CrossEntropy* ce = dynamic_cast<CrossEntropy*>(layer)) {
			CrossEntropy* other = dynamic_cast<CrossEntropy*>(cnet2.functionList()[indx]);
			assert(*ce ==  *other);
		} else if(SoftMax* sf = dynamic_cast<SoftMax*>(layer)) {
			SoftMax* other = dynamic_cast<SoftMax*>(cnet2.functionList()[indx]);
			assert(*sf ==  *other);
		} else if(CGelu* gl = dynamic_cast<CGelu*>(layer)) {
			CGelu* other = dynamic_cast<CGelu*>(cnet2.functionList()[indx]);
			assert(*gl ==  *other);
		} else if(CRelu* gl = dynamic_cast<CRelu*>(layer)) {
			CRelu* other = dynamic_cast<CRelu*>(cnet2.functionList()[indx]);
			assert(*gl ==  *other);
		} else {
			std::cerr << "Unexpected func: " << layer->getName() << std::endl;
			assert(false);
		}
		indx++;
	}

	auto old = cnet.back()->input().z(0);
	cnet.forward();
	cnet2.forward();

	assert(cnet.back()->input().z(0) == cnet2.back()->input().z(0));
	assert(old != cnet.back()->input().z(0));
}

void testSave() {
	ComplexNet cnet;
	int batch = 64;
	int embed = 8;
	int vocab_size = 65;
	int heads = 4;

	createBigNet(cnet, batch, embed, vocab_size, heads);

	std::cout << "init inputs ..." << std::endl;
	cnet.init_inputs();
	cout << cnet.toString() << endl;

	std::cout << "Save it..." << std::endl;

	ModelSaver modSave;
	assert(modSave.Save(cnet, "/Users/marcelcrasmaru/temp.mod"));

	std::cout << "Restore it..." << std::endl;

	ComplexNet cnet2;
	assert(modSave.Restore(cnet2, "/Users/marcelcrasmaru/temp.mod"));
	cout << cnet.toString() << endl;

	std::cout << "Check it..." << std::endl;
	testEqual(cnet, cnet2);

//	((CEmbedding*)cnet[1])->setInput({2, 1, 2});
//
//	std::cout << cnet.toString() << std::endl;
//	((CEmbedding*)cnet[1])->printEmbedding();
//	cnet[1]->printInput();

//	for (int var = 0; var < 100; ++var) {
//		cnet.forward();
//
//		std:: cout << "loss: " << ((CrossEntropy*)cnet.back())->loss(1) << std::endl;
//		cnet.backward(1);
//		cnet.update(0.9, 1);
//	}

	auto saved = modSave.SaveToString(cnet);
	std:: cout << "Saved to string of length: " << saved.length() << std::endl;

	ComplexNet cnet3;
    assert(modSave.RestoreFromString(cnet3, saved));
    cout << cnet.toString() << endl;

	std::cout << "Check it..." << std::endl;

	assert(cnet.functionList().size() == cnet3.functionList().size());

	int indx = 0;
	for (auto layer : cnet.functionList()) {
		if(CInput* input = dynamic_cast<CInput*>(layer)) {
			CInput* other = dynamic_cast<CInput*>(cnet3.functionList()[indx]);
			assert(*input ==  *other);
		} else if(Hadamard* hadm = dynamic_cast<Hadamard*>(layer)) {
			Hadamard* other = dynamic_cast<Hadamard*>(cnet3.functionList()[indx]);
			assert(*hadm ==  *other);
		} else if(Linear* linear = dynamic_cast<Linear*>(layer)) {
			Linear* other = dynamic_cast<Linear*>(cnet3.functionList()[indx]);
			assert(*linear ==  *other);
		} else if(CEmbedding* emb = dynamic_cast<CEmbedding*>(layer)) {
			CEmbedding* other = dynamic_cast<CEmbedding*>(cnet3.functionList()[indx]);
			assert(*emb ==  *other);
		} else if(FourierTrans* four = dynamic_cast<FourierTrans*>(layer)) {
			FourierTrans* other = dynamic_cast<FourierTrans*>(cnet3.functionList()[indx]);
			assert(*four ==  *other);
		} else if(Residual* res = dynamic_cast<Residual*>(layer)) {
			Residual* other = dynamic_cast<Residual*>(cnet3.functionList()[indx]);
			assert(*res ==  *other);
		} else if(CrossEntropy* ce = dynamic_cast<CrossEntropy*>(layer)) {
			CrossEntropy* other = dynamic_cast<CrossEntropy*>(cnet3.functionList()[indx]);
			assert(*ce ==  *other);
		} else if(SoftMax* sf = dynamic_cast<SoftMax*>(layer)) {
			SoftMax* other = dynamic_cast<SoftMax*>(cnet3.functionList()[indx]);
			assert(*sf ==  *other);
		} else if(CGelu* gl = dynamic_cast<CGelu*>(layer)) {
			CGelu* other = dynamic_cast<CGelu*>(cnet2.functionList()[indx]);
			assert(*gl ==  *other);
		} else {
			std::cerr << "Unexpected func: " << layer->getName() << std::endl;
			assert(false);
		}
		indx++;
	}

	((CEmbedding*)cnet3[1])->setInput({1, 2});
	auto old = cnet3.back()->input().z(0);

	cnet3.forward();

	assert(cnet.back()->input().z(0) == cnet3.back()->input().z(0));
	assert(old != cnet3.back()->input().z(0));

	std:: cout << "Saved to string of length: " << saved.length() << std::endl;
}

void test1() {
	ComplexNet cnet;
	cnet.add(new CInput(Uid(1), OutSize(100)));
	cnet.add(new CInput(Uid(2), OutSize(1000)));
	cnet.add(new Linear(Uid(3), InSize(100), InSize(1000)), {1, 2});
	cnet.add(new Residual(Uid(4), 100), {3, 1});
	cnet.add(new SoftMax(Uid(5), InSize(100)), {4});
	cnet.add(new CrossEntropy(Uid(6), InSize(100)), {5});
	cout << cnet.toString() << endl;

	cnet.init_inputs();

	StopWatch sw;
	for (int var = 0; var < 1000; ++var) {
		cnet.forward();
		float loss = ((CrossEntropy*) (cnet.back()))->loss(1);
		cout << "Loss: " << loss << endl;
		if (loss < 0.000000000001f || loss > pow(10, 10)) {
			break;
		}

		cnet.backward(1);
		cnet.update(0.5, 1);
	}
	auto time = sw.ElapsedTimeMicros() / 1000;

	cnet[1]->printInput();
	cnet[2]->printInput();
	cnet[3]->printInput();
	cnet[4]->printInput();


	cout << cnet.toString() << endl;

	((CrossEntropy*) (cnet.back()))->printProbabilities();
	cout << "Time: " << time << "ms" << endl;
}

void test5() {
	ComplexNet cnet;
	cnet.add(new CInput(Uid(1), OutSize(200)));
	cnet.add(new CInput(Uid(2), OutSize(1000)));
	cnet.add(new Linear(Uid(3), InSize(200), InSize(1000)), {1, 2});
	cnet.add(new CrossEntropy(Uid(4), InSize(5)), {3});

	cnet.init_inputs();
	cout << cnet.toString() << endl;

	StopWatch sw;
	for (int var = 0; var < 10000; ++var) {
		cnet.forward();
		float loss = ((CrossEntropy*) (cnet.back()))->loss(1);
		cout << "Loss: " << loss << endl;
		if (loss < 0.000000000001f || loss > pow(10, 10)) {
			break;
		}

		cnet.backward(1);
		cnet.update(0.1, 1);
	}
	auto time = sw.ElapsedTimeMicros() / 1000;

	cout << cnet.toString() << endl;
	((CrossEntropy*) cnet.back())->printProbabilities();
	cout << "Time: " << time << "ms" << endl;
}

void test6() {
	ComplexNet net;
	net.add(new CEmbedding(Uid(1), 2, 3, 5));
	net.add(new L2Out(Uid(2), 6), {1});
	net.init_inputs();

	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();

	((CEmbedding*)net[1])->setInput({2, 1, 2});
	((CEmbedding*)net[1])->printTokens();
	net[1]->printInput();

	for (int var = 0; var < 100; ++var) {
		net.forward();
//		net[1]->printInput();
//		net[2]->printInput();
		std:: cout << "loss: " << ((L2Out*)net[2])->loss() << std::endl;
		net.backward();
		net.update(0.1, 1);
	}



	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();
}

void test7() {
	ComplexNet net;
	net.add(new CEmbedding(Uid(1), 16, 32, 5));
	net.add(new FourierTrans(Uid(2), 512), {1});
	net.add(new L2Out(Uid(3), 512), {2});
	net.init_inputs();

	((CEmbedding*)net[1])->setInput({2, 1, 2});

	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();
	net[1]->printInput();

	for (int var = 0; var < 100; ++var) {
		net.forward();
//		net[1]->printInput();
//		net[2]->printInput();
//		net[3]->printInput();

		std:: cout << "loss: " << ((L2Out*)net[3])->loss() << std::endl;
		net.backward();
		net.update(0.001, 1);
	}


	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();
}

void test8() {
	ComplexNet net;
	net.add(new CInput(Uid(1), OutSize(32)));
	net.add(new CInput(Uid(2), OutSize(32)));
	net.add(new Hadamard(Uid(3), InSize(32), InSize(32)), {1, 2});
	net.add(new CrossEntropy(Uid(4), InSize(32)), {3});

	net.init_inputs();
	std::cout << net.toString() << std::endl;

	for (int var = 0; var < 200; ++var) {
		net.forward();
		std:: cout << "loss: " << ((CrossEntropy*)net[4])->loss(1) << std::endl;
		net.backward(1);
		net.update(0.9, 1);
	}

	std::cout << net.toString() << std::endl;
}

void test9() {
	ComplexNet net;
	net.add(new CEmbedding(Uid(1), 8, 16, 5));
	net.add(new CRelu(Uid(2), 128), {1});
	net.add(new L2Out(Uid(3), 128), {2});
	net.init_inputs();
	net.init_exec_graph();

	((CEmbedding*)net[1])->setInput({2, 1, 2});

	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();
	net[1]->printInput();

	for (int var = 0; var < 100; ++var) {
		net.forward();
//		net[1]->printInput();
//		net[2]->printInput();
//		net[3]->printInput();

		std:: cout << "loss: " << ((L2Out*)net[3])->loss() << std::endl;
		net.backward();
		net.update(0.01, 1);
	}


	std::cout << net.toString() << std::endl;
	((CEmbedding*)net[1])->printEmbedding();
}

void testGradEmb() {
	ComplexNet net;

	int emb_dim = 2;
	int no_out_tok = 3;
	int no_tok = 4;
	int token = 2;

	net.add(new CEmbedding(Uid(1), emb_dim, no_out_tok, no_tok));
	net.add(new L2Out(Uid(2), InSize(net[1]->outSize())), {1});
	net.add(new L2Out(Uid(3), InSize(net[1]->outSize())), {1});

	((CEmbedding*)net[1])->setInput({token});

	net[2]->mutable_input()->dz_starSetValue(0, std::complex<float>(1, 2));
	net[2]->mutable_input()->dz_starSetValue(1, std::complex<float>(3, 4));

	net[3]->mutable_input()->dz_starSetValue(0, std::complex<float>(7, 8));
	net[3]->mutable_input()->dz_starSetValue(1, std::complex<float>(9, 10));

	std::cout << net[2]->input().dz_starToString(3) << std::endl;
	std::cout << net[3]->input().dz_starToString(3) << std::endl;

	net[1]->backward();
	std::cout << net[1]->input().dz_starToString(8) << std::endl;

	assert(net[1]->input().dz_star(emb_dim * token) ==  std::complex<float>(8, 10));
	assert(net[1]->input().dz_star(emb_dim * token + 1) ==  std::complex<float>(12, 14));
}

void testEmb() {

	ComplexNet net;

	int emb_dim = 2;
	int no_out_tok = 3;
	int no_tok = 4;

	net.add(new CEmbedding(Uid(1), emb_dim, no_out_tok, no_tok));
	net.add(new L2Out(Uid(2), InSize(net[1]->outSize())), {1});
	net.add(new L2Out(Uid(3), InSize(net[1]->outSize())), {1});

	net.init_inputs(12345);
	net.init_exec_graph();
	std::cout << net.toString() << std::endl;

	net[1]->printInput();
	((CEmbedding*)net[1])->setInput({1, 3});

	net[1]->forward();


	for (auto i : {2, 3}) {
		assert(net[i]->input().z(0) == net[1]->input().z(1 * emb_dim));
		assert(net[i]->input().z(1) == net[1]->input().z(1 * emb_dim + 1));
		assert(net[i]->input().z(2) == net[1]->input().z(3 * emb_dim));
		assert(net[i]->input().z(3) == net[1]->input().z(3 * emb_dim + 1));
		assert(net[i]->input().z(4) == std::complex<float>(0, 0));
		assert(net[i]->input().z(5) == std::complex<float>(0, 0));

		net[i]->printInput();
	}

	((CEmbedding*)net[1])->setInput({0});
	net[1]->forward();

	for (auto i : {2, 3}) {
		assert(net[i]->input().z(0) == net[1]->input().z(0));
		assert(net[i]->input().z(1) == net[1]->input().z(1));
		assert(net[i]->input().z(2) == std::complex<float>(0, 0));
		assert(net[i]->input().z(3) == std::complex<float>(0, 0));
		assert(net[i]->input().z(4) == std::complex<float>(0, 0));
		assert(net[i]->input().z(5) == std::complex<float>(0, 0));

		net[i]->printInput();
	}

	((CEmbedding*)net[1])->setInput({2, 2, 2});
	net[1]->forward();

	for (auto i : {2, 3}) {
		assert(net[i]->input().z(0) == net[1]->input().z(2 * emb_dim));
		assert(net[i]->input().z(1) == net[1]->input().z(2 * emb_dim + 1));
		assert(net[i]->input().z(2) == net[1]->input().z(2 * emb_dim));
		assert(net[i]->input().z(3) == net[1]->input().z(2 * emb_dim + 1));
		assert(net[i]->input().z(4) == net[1]->input().z(2 * emb_dim));
		assert(net[i]->input().z(5) == net[1]->input().z(2 * emb_dim + 1));

		net[i]->printInput();
	}
}

void _run_all_tests() {
	//_test_linear();
}


#endif /* TEST_H_ */
