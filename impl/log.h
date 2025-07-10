
#ifndef __LOG_H__
#define __LOG_H__

#include <sstream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

inline std::string NowTime();

enum TLogLevel {lError, lWarning, lInfo, lDebug};

template <typename T>
class Log
{
public:
    Log();
    virtual ~Log();
    std::ostringstream& Get(TLogLevel level = lInfo);
public:
    static TLogLevel& ReportingLevel();
    static std::string ToString(TLogLevel level);
    static TLogLevel FromString(const std::string& level);
protected:
    std::ostringstream os;
private:
    Log(const Log&);
    Log& operator =(const Log&);
};

template <typename T>
Log<T>::Log()
{
}

template <typename T>
std::ostringstream& Log<T>::Get(TLogLevel level)
{
    os << NowTime();
    os << " " << ToString(level) << ": ";
    os << std::string(level > lDebug ? level - lDebug : 0, '\t');
    return os;
}

template <typename T>
Log<T>::~Log()
{
    os << std::endl;
    T::Output(os.str());
}

template <typename T>
TLogLevel& Log<T>::ReportingLevel()
{
    static TLogLevel reportingLevel = lDebug;
    return reportingLevel;
}

template <typename T>
std::string Log<T>::ToString(TLogLevel level)
{
	static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "DEBUG"};
    return buffer[level];
}

template <typename T>
TLogLevel Log<T>::FromString(const std::string& level)
{
    if (level == "DEBUG")
        return lDebug;
    if (level == "INFO")
        return lInfo;
    if (level == "WARNING")
        return lWarning;
    if (level == "ERROR")
        return lError;
    Log<T>().Get(lWarning) << "Unknown logging level '" << level << "'. Using INFO level as default.";
    return lInfo;
}

class Output2FILE
{
public:
    static FILE*& Stream();
    static void Output(const std::string& msg);
};

inline FILE*& Output2FILE::Stream()
{
    static FILE* pStream = stderr;
    return pStream;
}

inline void Output2FILE::Output(const std::string& msg)
{
    FILE* pStream = Stream();
    if (!pStream)
        return;
    fprintf(pStream, "%s", msg.c_str());
    fflush(pStream);
}


#define FILELOG_DECLSPEC


class FILELOG_DECLSPEC FILELog : public Log<Output2FILE> {};
//typedef Log<Output2FILE> FILELog;

#ifndef FILELOG_MAX_LEVEL
#define FILELOG_MAX_LEVEL lDebug
#endif

#define FILE_LOG(level) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel() || !Output2FILE::Stream()) ; \
    else FILELog().Get(level)

#define L_(level) \
if (level > FILELOG_MAX_LEVEL) ;\
else if (level > FILELog::ReportingLevel() || !Output2FILE::Stream()) ; \
else FILELog().Get(level)

inline std::string NowTime()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
	return oss.str();
}

inline void initLogger(const char * file, TLogLevel level)
{
    FILELog::ReportingLevel() = level;
    FILE* log_fd = fopen( file, "w" );
    Output2FILE::Stream() = log_fd;
}

inline void endLogger()
{
	if (Output2FILE::Stream())
		fclose(Output2FILE::Stream());
}

#endif //__LOG_H__
