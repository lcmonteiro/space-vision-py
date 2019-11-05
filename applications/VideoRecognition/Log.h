#ifndef LOG_H
#define LOG_H

#include <string>
#include <iostream>
#include <sstream> 
#include <fstream>
#include "Command.h"

class Log
{
public:
	Log();
	~Log();

	void LogToFile(std::string fileName);
	void AddLine(Command command, std::string comment);
	
	std::stringstream buffer;

};

#endif

