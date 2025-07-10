#ifndef CPU_MODEL_SAVER_H_
#define CPU_MODEL_SAVER_H_

#include <stdlib.h>
#include <fstream>
#include <cstring>
#include <vector>

#include "log.h"
#include "cfunc.h"
#include "cinput.h"
#include "cnet.h"
#include "linear.h"
#include "residual.h"
#include "crossent.h"
#include "embed.h"
#include "ft.h"
#include "relu.h"
#include "hadamard.h"
#include "softmax.h"
#include "l2out.h"


const int __one__ = 1;
const bool isCpuLittleEndian = 1 == *(char*) (&__one__);

const int magicNo = 0xABCDE0EE;

class BinaryReader {
public:

	virtual ~BinaryReader() {
	}

	virtual bool read_int32(int32_t *read) = 0;

	virtual bool read_int32(std::ifstream &stream, int32_t *read) = 0;

	virtual bool read_double(double *read) = 0;

	virtual bool read_float(float *floa, int count) = 0;

	virtual bool read_float(float *floa) = 0;

	virtual bool read_float(std::ifstream &stream, float *floa) = 0;

	virtual bool read_chars(char *data, int length) = 0;

	virtual bool read_unsigned_chars(unsigned char *data, int length) = 0;
};

class FileBinaryReader_: public BinaryReader {
	std::ifstream infile_;
public:

	virtual ~FileBinaryReader_() {
	}

	bool read_int32(int32_t *read) {
		char buffer[4];
		if (!infile_.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_int32(std::ifstream &stream, int32_t *read) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_double(double *read) {
		char buffer[8];
		if (!infile_.read(buffer, 8)) {
			return false;
		}

		char *pDouble = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 8; ++i) {
				pDouble[7 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 8; ++i) {
				pDouble[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_float(float *floa, int count) {
		for (int var = 0; var < count; ++var) {
			if (!read_float(floa + var))
				return false;
		}
		return true;
	}

	bool read_float(float *floa) {
		char buffer[4];
		if (!infile_.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_float(std::ifstream &stream, float *floa) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_chars(char *data, int length) {
		return (bool) infile_.read(data, length);
	}

	bool read_unsigned_chars(unsigned char *data, int length) {
		return (bool) infile_.read((char*) data, length);
	}

	bool open(std::string file) {
		infile_.open(file, std::ofstream::in | std::ofstream::binary);
		if (infile_) {
			return true;
		}
		return false;
	}

	void close() {
		infile_.close();
	}
};

class StringBinaryReader: public BinaryReader {
	std::istringstream in_string_;
public:

	StringBinaryReader(std::string input) :
			in_string_(input) {
	}

	virtual ~StringBinaryReader() {
	}

	bool read_int32(int32_t *read) {
		char buffer[4];
		if (!in_string_.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_int32(std::ifstream &stream, int32_t *read) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pInt = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pInt[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pInt[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_double(double *read) {
		char buffer[8];
		if (!in_string_.read(buffer, 8)) {
			return false;
		}

		char *pDouble = (char*) read;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 8; ++i) {
				pDouble[7 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 8; ++i) {
				pDouble[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_float(float *floa, int count) {
		for (int var = 0; var < count; ++var) {
			if (!read_float(floa + var))
				return false;
		}
		return true;
	}

	bool read_float(float *floa) {
		char buffer[4];
		if (!in_string_.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_float(std::ifstream &stream, float *floa) {
		char buffer[4];
		if (!stream.read(buffer, 4)) {
			return false;
		}
		char *pFloat = (char*) floa;
		if (isCpuLittleEndian) {
			for (int i = 0; i < 4; ++i) {
				pFloat[3 - i] = buffer[i];
			}
		} else {
			for (int i = 0; i < 4; ++i) {
				pFloat[i] = buffer[i];
			}
		}

		return true;
	}

	bool read_chars(char *data, int length) {
		return (bool) in_string_.read(data, length);
	}

	bool read_unsigned_chars(unsigned char *data, int length) {
		return (bool) in_string_.read((char*) data, length);
	}
};

class BinaryWriter {
public:
	virtual void write(double *pDouble, int size) = 0;
	virtual void write(float *pFloat, int size) = 0;
	virtual void write_double(double doubl) = 0;
	virtual void write_float(float floa) = 0;
	virtual void write_int32(int32_t v) = 0;
	virtual void write_chars(const char *data, int length) = 0;
	virtual void write_unsigned_chars(const unsigned char *data,
			int length) = 0;

	virtual ~BinaryWriter() {
	}
};

class FileBinaryWriter_: public BinaryWriter {
	std::string file_;
	std::ofstream outfile_;

public:
	virtual ~FileBinaryWriter_() {
	}

	bool open(std::string file) {
		file_ = file;
		outfile_.open(file,
				std::ofstream::out | std::ofstream::trunc
						| std::ofstream::binary);
		if (outfile_) {
			return true;
		}
		return false;
	}

	void close() {
		outfile_.close();
	}

	void write(double *pDouble, int size) {
		for (int var = 0; var < size; ++var) {
			write_double(*(pDouble + var));
		}
	}

	void write(float *pFloat, int size) {
		for (int var = 0; var < size; ++var) {
			write_float(*(pFloat + var));
		}
	}

	void write_double(double doubl) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&doubl);
			for (int i = 0; i < 8; ++i) {
				data[i] = pDouble[7 - i];
			}
			outfile_.write(data, 8);
		} else {
			outfile_.write((char*) (&doubl), 8);
		}
	}

	void write_float(float floa) {
		if (isCpuLittleEndian) {
			char data[4], *pFloat = (char*) (float*) (&floa);
			for (int i = 0; i < 4; ++i) {
				data[i] = pFloat[3 - i];
			}
			outfile_.write(data, 4);
		} else {
			outfile_.write((char*) (&floa), 4);
		}
	}

	void write_int32(int32_t v) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&v);
			for (int i = 0; i < 4; ++i) {
				data[i] = pDouble[3 - i];
			}
			outfile_.write(data, 4);
		} else {
			outfile_.write((char*) (&v), 4);
		}
	}

	void write_chars(const char *data, int length) {
		outfile_.write(data, length);
	}

	void write_unsigned_chars(const unsigned char *data, int length) {
		outfile_.write((char*) data, length);
	}

	std::string file() {
		return file_;
	}
};

class StringBinaryWriter: public BinaryWriter {
	std::stringstream out_string_;

public:
	virtual ~StringBinaryWriter() {
	}

	std::string getString() {
		return out_string_.str();
	}

	void write(double *pDouble, int size) {
		for (int var = 0; var < size; ++var) {
			write_double(*(pDouble + var));
		}
	}

	void write(float *pFloat, int size) {
		for (int var = 0; var < size; ++var) {
			write_float(*(pFloat + var));
		}
	}

	void write_double(double doubl) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&doubl);
			for (int i = 0; i < 8; ++i) {
				data[i] = pDouble[7 - i];
			}
			out_string_.write(data, 8);
		} else {
			out_string_.write((char*) (&doubl), 8);
		}
	}

	void write_float(float floa) {
		if (isCpuLittleEndian) {
			char data[4], *pFloat = (char*) (float*) (&floa);
			for (int i = 0; i < 4; ++i) {
				data[i] = pFloat[3 - i];
			}
			out_string_.write(data, 4);
		} else {
			out_string_.write((char*) (&floa), 4);
		}
	}

	void write_int32(int32_t v) {
		if (isCpuLittleEndian) {
			char data[8], *pDouble = (char*) (double*) (&v);
			for (int i = 0; i < 4; ++i) {
				data[i] = pDouble[3 - i];
			}
			out_string_.write(data, 4);
		} else {
			out_string_.write((char*) (&v), 4);
		}
	}

	void write_chars(const char *data, int length) {
		out_string_.write(data, length);
	}

	void write_unsigned_chars(const unsigned char *data, int length) {
		out_string_.write((char*) data, length);
	}
};

class ModelSaver {
private:
	int uid = 0;
	int no_prev_func = 0;
	int in_size = 0;
	int out_size = 0;

private:
	bool WriteLayer(CFunc *layer, BinaryWriter &writer) {
		if (CInput *input = dynamic_cast<CInput*>(layer)) {
			return Write(input, writer);
		}
		if (Hadamard *hadm = dynamic_cast<Hadamard*>(layer)) {
			return Write(hadm, writer);
		}
		if (Linear *linear = dynamic_cast<Linear*>(layer)) {
			return Write(linear, writer);
		}
		if (CEmbedding *emb = dynamic_cast<CEmbedding*>(layer)) {
			return Write(emb, writer);
		}
		if (FourierTrans *four = dynamic_cast<FourierTrans*>(layer)) {
			return Write(four, writer);
		}
		if (TriangFourier *tfour = dynamic_cast<TriangFourier*>(layer)) {
			return Write(tfour, writer);
		}
		if (Residual *four = dynamic_cast<Residual*>(layer)) {
			return Write(four, writer);
		}

		if (CrossEntropy *out = dynamic_cast<CrossEntropy*>(layer)) {
			return Write(out, writer);
		}

		if (L2Out *out = dynamic_cast<L2Out*>(layer)) {
			return Write(out, writer);
		}

		if (SoftMax *out = dynamic_cast<SoftMax*>(layer)) {
			return Write(out, writer);
		}

		if (CRelu *out = dynamic_cast<CRelu*>(layer)) {
			return Write(out, writer);
		}

		if (CGelu *out = dynamic_cast<CGelu*>(layer)) {
			return Write(out, writer);
		}

		L_(lError) << "Cannot save " << layer->getName();
		return false;
	}

	void writeFuncInfo(BinaryWriter &writer, CFunc *cInp) {
		writer.write_int32(cInp->uid());
		writer.write_int32(cInp->input().length_);
		writer.write_int32(cInp->out_size_);

		writer.write_int32(cInp->no_previous_func());
		for (int var = 0; var < cInp->no_previous_func(); ++var) {
			writer.write_int32(cInp->previous(var)->uid());
		}
	}

	void readFuncInfo(BinaryReader &reader, std::vector<int> &prev) {
		assert(reader.read_int32(&uid));
		assert(reader.read_int32(&in_size));
		assert(reader.read_int32(&out_size));

		assert(reader.read_int32(&no_prev_func));
		for (int var = 0; var < no_prev_func; ++var) {
			int current = 0;
			assert(reader.read_int32(&current));
			prev.push_back(current);
		}
	}

	bool Write(CInput *cInp, BinaryWriter &writer) {
		writer.write_int32(isInput);

		writeFuncInfo(writer, cInp);
		writer.write(cInp->input().real_, cInp->input().length_);
		writer.write(cInp->input().imag_, cInp->input().length_);

		return true;
	}

	bool RestoreInput(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreInput";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new CInput(Uid(uid), OutSize(in_size)), prev);

		assert(reader.read_float(net[uid]->mutable_input()->real_, in_size));
		assert(reader.read_float(net[uid]->mutable_input()->imag_, in_size));

		return true;
	}

	bool Write(CEmbedding *emb, BinaryWriter &writer) {
		writer.write_int32(isEmbedding);

		writeFuncInfo(writer, emb);
		writer.write_int32(emb->embedding_dim_);
		writer.write_int32(emb->no_out_tokens_);
		writer.write_int32(emb->no_embeddings_);

		writer.write(emb->input().real_, emb->input().length_);
		writer.write(emb->input().imag_, emb->input().length_);

		return true;
	}

	bool RestoreEmbedding(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) <<  "RestoreEmbedding";

		std::vector<int> prev;
		readFuncInfo(reader, prev);

		int embedding_dim = 0;
		int max_out_tokens = 0;
		int no_embeddings = 0;
		assert(reader.read_int32(&embedding_dim));
		assert(reader.read_int32(&max_out_tokens));
		assert(reader.read_int32(&no_embeddings));

		net.add(
				new CEmbedding(Uid(uid), embedding_dim, max_out_tokens,
						no_embeddings), prev);

		assert(reader.read_float(net[uid]->mutable_input()->real_,
				embedding_dim * no_embeddings));
		assert(reader.read_float(net[uid]->mutable_input()->imag_,
				embedding_dim * no_embeddings));

		return true;
	}

	bool Write(Linear *linear, BinaryWriter &writer) {
		writer.write_int32(isLinear);
		writeFuncInfo(writer, linear);

		writer.write_int32(linear->in1_len);
		writer.write_int32(linear->in2_len);

		return true;
	}

	bool Write(CRelu *crelu, BinaryWriter &writer) {
		writer.write_int32(isCrelu);

		writeFuncInfo(writer, crelu);

		return true;
	}

	bool Write(CGelu *gelu, BinaryWriter &writer) {
		writer.write_int32(isGelu);

		writeFuncInfo(writer, gelu);

		return true;
	}

	bool RestoreRelu(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreRelu";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new CRelu(Uid(uid), InSize(out_size)), prev);

		return true;
	}

	bool RestoreGelu(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) <<  "RestoreGelu";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new CGelu(Uid(uid), InSize(out_size)), prev);

		return true;
	}

	bool RestoreLinear(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) <<  "RestoreLinear";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		int in1_len = 0;
		int in2_len = 0;
		assert(reader.read_int32(&in1_len));
		assert(reader.read_int32(&in2_len));

		net.add(new Linear(Uid(uid), InSize(in1_len), InSize(in2_len)), prev);

		return true;
	}

	bool Write(Hadamard *hadm, BinaryWriter &writer) {
		writer.write_int32(isHadamard);

		writeFuncInfo(writer, hadm);

		return true;
	}

	bool RestoreHadamard(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreHadamard";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new Hadamard(Uid(uid), InSize(out_size), InSize(out_size)),
				prev);

		return true;
	}

	bool Write(FourierTrans *cInp, BinaryWriter &writer) {
		writer.write_int32(isFourier);
		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreFourierTrans(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) <<  "RestoreFourierTrans";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new FourierTrans(Uid(uid), InSize(in_size)), prev);
		return true;
	}

	bool Write(TriangFourier *cInp, BinaryWriter &writer) {
		writer.write_int32(isTrianFourier);
		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreTriangFourier(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreTriangFourier";

		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new TriangFourier(Uid(uid), InSize(in_size)), prev);
		return true;
	}

	bool Write(Residual *cInp, BinaryWriter &writer) {
		writer.write_int32(isResidual);

		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreResidual(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) <<  "RestoreResidual";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new Residual(Uid(uid), InSize(out_size)), prev);
		return true;
	}

	bool Write(CrossEntropy *cInp, BinaryWriter &writer) {
		writer.write_int32(isCrossEntropy);

		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreCrossEntropy(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreCrossEntropy";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new CrossEntropy(Uid(uid), InSize(in_size)), prev);
		return true;
	}

	bool Write(L2Out *cInp, BinaryWriter &writer) {
		writer.write_int32(isL2Out);

		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreL2Out(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreL2Out";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new L2Out(Uid(uid), InSize(in_size)), prev);
		return true;
	}

	bool Write(SoftMax *cInp, BinaryWriter &writer) {
		writer.write_int32(isSoftMax);

		writeFuncInfo(writer, cInp);

		return true;
	}

	bool RestoreSoftMax(ComplexNet &net, BinaryReader &reader) {
		L_(lDebug) << "RestoreSoftMax";
		std::vector<int> prev;
		readFuncInfo(reader, prev);

		net.add(new SoftMax(Uid(uid), InSize(in_size)), prev);
		return true;
	}

public:
//	bool Save(CNet& c_net, std::string file) {
//		return Save(c_net.cpuNet(), file);
//	}

	bool Save(ComplexNet& net, std::string file) {
		FileBinaryWriter_ writer;
		if (!writer.open(file)) {
			L_(lError) << "Cannot open: \"" << file << "\"";
			return false;
		}
		writer.write_int32(magicNo);

		for (auto func : net.functionList()) {
			if (func->isGpuOnly()) {
				continue;
			}
			if (!WriteLayer(func, writer)) {
				writer.close();
				return false;
			}
		}

		writer.close();
		return true;
	}

//	std::string SaveToString(CNet &c_net) {
//		return SaveToString(c_net.cpuNet());
//	}

	std::string SaveToString(ComplexNet &net) {
		StringBinaryWriter writer;

		writer.write_int32(magicNo);

		for (auto func : net.functionList()) {
			if (func->isGpuOnly()) {
				continue;
			}
			if (!WriteLayer(func, writer)) {
				L_(lError) << "Cannot save to string!";
				return "";
			}
		}

		return writer.getString();
	}

//	bool RestoreFromString(CNet& c_net, const std::string& input) {
//		return RestoreFromString(c_net.cpuNet(), input);
//	}

	bool RestoreFromString(ComplexNet& net, const std::string& input) {
		StringBinaryReader reader(input);

		int current;
		if (!reader.read_int32(&current) || current != magicNo) {
			L_(lError) << std::hex << current;
			return false;
		}
		while (reader.read_int32(&current)) {
			if (current == isHadamard && RestoreHadamard(net, reader)) {
				continue;
			}
			if (current == isInput && RestoreInput(net, reader)) {
				continue;
			}
			if (current == isEmbedding && RestoreEmbedding(net, reader)) {
				continue;
			}
			if (current == isLinear && RestoreLinear(net, reader)) {
				continue;
			}
			if (current == isFourier && RestoreFourierTrans(net, reader)) {
				continue;
			}
			if (current == isTrianFourier && RestoreTriangFourier(net, reader)) {
				continue;
			}
			if (current == isResidual && RestoreResidual(net, reader)) {
				continue;
			}
			if (current == isSoftMax && RestoreSoftMax(net, reader)) {
				continue;
			}
			if (current == isCrossEntropy && RestoreCrossEntropy(net, reader)) {
				continue;
			}
			if (current == isL2Out && RestoreL2Out(net, reader)) {
				continue;
			}
			if (current == isCrelu && RestoreRelu(net, reader)) {
				continue;
			}

			if (current == isGelu && RestoreGelu(net, reader)) {
				continue;
			}

			L_(lError) << "Cannot restore layer of ID: " << current;
			return false;
		}

		return true;
	}

//	bool Restore(CNet &c_net, std::string file) {
//		return Restore(c_net.cpuNet(), file);
//	}

	bool Restore(ComplexNet &net, std::string file) {
		FileBinaryReader_ reader;
		if (!reader.open(file)) {
			L_(lError) << "Cannot restore layer of ID: " << "cannot open: " << file;
			return false;
		}

		int current;
		if (!reader.read_int32(&current) || current != magicNo) {
			L_(lError) << std::hex << current;
			reader.close();
			return false;
		}
		while (reader.read_int32(&current)) {
			if (current == isHadamard && RestoreHadamard(net, reader)) {
				continue;
			}
			if (current == isInput && RestoreInput(net, reader)) {
				continue;
			}
			if (current == isEmbedding && RestoreEmbedding(net, reader)) {
				continue;
			}
			if (current == isLinear && RestoreLinear(net, reader)) {
				continue;
			}
			if (current == isFourier && RestoreFourierTrans(net, reader)) {
				continue;
			}
			if (current == isTrianFourier && RestoreTriangFourier(net, reader)) {
				continue;
			}
			if (current == isResidual && RestoreResidual(net, reader)) {
				continue;
			}
			if (current == isSoftMax && RestoreSoftMax(net, reader)) {
				continue;
			}
			if (current == isCrossEntropy && RestoreCrossEntropy(net, reader)) {
				continue;
			}
			if (current == isL2Out && RestoreL2Out(net, reader)) {
				continue;
			}
			if (current == isCrelu && RestoreRelu(net, reader)) {
				continue;
			}

			if (current == isGelu && RestoreGelu(net, reader)) {
				continue;
			}

			reader.close();
			L_(lError) << "Cannot restore layer of ID: " << current;
			return false;
		}

		reader.close();
		return true;
	}

	virtual ~ModelSaver() {
	}
};

#endif /* CPU_MODEL_SAVER_H_ */
