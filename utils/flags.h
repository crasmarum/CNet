#ifndef UTILS_FLAGS_H_
#define UTILS_FLAGS_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

class FLAGS {
private:
	static std::vector<std::string> tokens_;

	static std::vector<bool*> flags_bool_;
	static std::vector<int*> flags_int_;
	static std::vector<float*> flags_float_;
	static std::vector<std::string*> flags_str_;

	static std::vector<std::string> flags_bool_names_;
	static std::vector<std::string> flags_int_names_;
	static std::vector<std::string> flags_float_names_;
	static std::vector<std::string> flags_str_names_;

	FLAGS() {}

    static bool cmdOptionExists(const std::string &option) {
        return std::find(tokens_.cbegin(), tokens_.cend(), option) != tokens_.cend();
    }

	static const std::string& getCmdOption(const std::string &option) {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(tokens_.cbegin(), tokens_.cend(), option);
		if (itr != tokens_.cend() && ++itr != tokens_.cend()) {
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}

	static bool stringToBool (const std::string &str)
	{
	    return !str.empty ()
	    		&& (strcasecmp(str.c_str (), "true") == 0 || atoi (str.c_str ()) != 0);
	}

public:
	static void Parse(int &argc, char **argv) {
		for (int i = 1; i < argc; ++i) {
			tokens_.push_back(std::string(argv[i]));
		}
		bool print = cmdOptionExists("-h");

		for (int var = 0; var < flags_bool_names_.size(); ++var) {
			const std::string key = "-" + flags_bool_names_[var];
			const std::string& val = getCmdOption(key);
			if (print) std::cout << key << " "
					             << (*flags_bool_[var] ? "true" : "false") << std::endl;
			if (val.empty()) continue;
			*flags_bool_[var] = stringToBool(val);
		}

		for (int var = 0; var < flags_int_names_.size(); ++var) {
			const std::string key = "-" + flags_int_names_[var];
			const std::string& val = getCmdOption(key);
			if (print) std::cout << key << " " << *flags_int_[var] << std::endl;

			if (val.empty()) continue;
			*flags_int_[var] = std::stoi(val);
		}
		for (int var = 0; var < flags_float_names_.size(); ++var) {
			const std::string key = "-" + flags_float_names_[var];
			const std::string& val = getCmdOption(key);
			if (print) std::cout << key << " " << *flags_float_[var] << std::endl;

			if (val.empty()) continue;
			*flags_float_[var] = std::stof(val);
		}
		for (int var = 0; var < flags_str_names_.size(); ++var) {
			const std::string key = "-" + flags_str_names_[var];
			const std::string& val = getCmdOption(key);
			if (print) std::cout << key << " \"" << *flags_str_[var] << "\"" << std::endl;

			if (val.empty()) continue;
			*flags_str_[var] = val;
		}

		if (print) exit(0);
	}

	static bool register_bool(const char *name, bool *var, bool val);
	static int register_int(const char *name, int *var, int val);
	static float register_float(const char *name, float *var, float val);
	static std::string register_str(const char *name, std::string *var, const char* val);
};

#define FLAG_BOOL(name, val) bool name = FLAGS::register_bool(#name, &name, val);
#define FLAG_INT(name, val) int name = FLAGS::register_int(#name, &name, val);
#define FLAG_FLOAT(name, val) float name = FLAGS::register_float(#name, &name, val);
#define FLAG_STRING(name, val) std::string name = FLAGS::register_str(#name, &name, val);

#endif /* UTILS_FLAGS_H_ */
