#ifndef GPU_BASE64_H_
#define GPU_BASE64_H_

#include <string>

class B64Encoding {
	const char* B64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	const char B64PADDING = '$';

	const int B64index[256] =
	{
	    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  62, 63, 62, 62, 63,
	    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0,  0,  0,  0,  0,  0,
	    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
	    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0,  0,  0,  0,  63,
	    0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
	};

public:
	const std::string b64encode(const void* data, const size_t &len)
	{
		std::string result((len + 2) / 3 * 4, B64PADDING);
	    unsigned char *p = (unsigned  char*) data;
	    char *str = &result[0];
	    size_t j = 0, pad = len % 3;
	    const size_t last = len - pad;

	    for (size_t i = 0; i < last; i += 3)
	    {
	        int n = int(p[i]) << 16 | int(p[i + 1]) << 8 | p[i + 2];
	        str[j++] = B64chars[n >> 18];
	        str[j++] = B64chars[n >> 12 & 0x3F];
	        str[j++] = B64chars[n >> 6 & 0x3F];
	        str[j++] = B64chars[n & 0x3F];
	    }
	    if (pad)  /// Set padding
	    {
	        int n = --pad ? int(p[last]) << 8 | p[last + 1] : p[last];
	        str[j++] = B64chars[pad ? n >> 10 & 0x3F : n >> 2];
	        str[j++] = B64chars[pad ? n >> 4 & 0x03F : n << 4 & 0x3F];
			str[j++] = pad ? B64chars[n << 2 & 0x3F] : B64PADDING;
	    }

		for (int var = 0; var < result.length(); ++var) {
			if (result[var] == '/') {
				result[var] = '_';
			} else if (result[var] == '+') {
				result[var] = '-';
			}
		}

	    return result;
	}

	const std::string b64decode(const void* data, const size_t &len)
	{
	    if (len == 0) return "";

	    unsigned char *p = (unsigned char*) data;
	    size_t j = 0,
	        pad1 = len % 4 || p[len - 1] == B64PADDING,
	        pad2 = pad1
				&& (len % 4 > 2 || p[len - 2] != B64PADDING);
	    const size_t last = (len - pad1) / 4 << 2;
	    std::string result(last / 4 * 3 + pad1 + pad2, '\0');
	    unsigned char *str = (unsigned char*) &result[0];

	    for (size_t i = 0; i < last; i += 4)
	    {
	        int n = B64index[p[i]] << 18 | B64index[p[i + 1]] << 12 | B64index[p[i + 2]] << 6 | B64index[p[i + 3]];
	        str[j++] = n >> 16;
	        str[j++] = n >> 8 & 0xFF;
	        str[j++] = n & 0xFF;
	    }
	    if (pad1)
	    {
	        int n = B64index[p[last]] << 18 | B64index[p[last + 1]] << 12;
	        str[j++] = n >> 16;
	        if (pad2)
	        {
	            n |= B64index[p[last + 2]] << 6;
	            str[j++] = n >> 8 & 0xFF;
	        }
	    }
	    return result;
	}

	std::string base64_encode(const std::string& str)
	{
	    return b64encode(str.c_str(), str.size());
	}

	std::string base64_decode(const std::string& str64)
	{
		std::string str;
		for (int var = 0; var < str64.length(); ++var) {
			char ch = str64[var] == '_' ? '/' : str64[var];
			ch = str64[var] == '-' ? '+' : str64[var];
			str += ch;
		}
	    return b64decode(str.c_str(), str.size());
	}
};


#endif /* GPU_BASE64_H_ */
