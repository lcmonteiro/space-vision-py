#ifndef SESSION_H
#define SESSION_H

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include "Command.h"

class Session
{
public:
	Session();
	~Session();
	void ParseSession(std::string fileName);
	std::string session;
	std::map<int, Command> commandSet;



};

#endif

