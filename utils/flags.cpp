#include "flags.h"

#include <string>
#include <vector>

std::vector<std::string> FLAGS::tokens_;

std::vector<bool*> FLAGS::flags_bool_;
std::vector<int*> FLAGS::flags_int_;
std::vector<float*> FLAGS::flags_float_;
std::vector<std::string*> FLAGS::flags_str_;

std::vector<std::string> FLAGS::flags_bool_names_;
std::vector<std::string> FLAGS::flags_int_names_;
std::vector<std::string> FLAGS::flags_float_names_;
std::vector<std::string> FLAGS::flags_str_names_;

bool FLAGS::register_bool(const char *name, bool *var, bool val) {
	std::string s_name = std::string(name);
	flags_bool_names_.push_back(s_name);
	flags_bool_.push_back(var);
	return val;
}

int FLAGS::register_int(const char *name, int *var, int val) {
	std::string s_name = std::string(name);
	flags_int_names_.push_back(s_name);
	flags_int_.push_back(var);
	return val;
}

float FLAGS::register_float(const char *name, float *var, float val) {
	std::string s_name = std::string(name);
	flags_float_names_.push_back(s_name);
	flags_float_.push_back(var);
	return val;
}

std::string FLAGS::register_str(const char *name, std::string *var, const char* val) {
	std::string s_name = std::string(name);
	flags_str_names_.push_back(s_name);
	flags_str_.push_back(var);
	return  std::string(val);
}

