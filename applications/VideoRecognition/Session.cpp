#include "Session.h"



Session::Session()
{
}


Session::~Session()
{
}

void Session::ParseSession(std::string fileName)
{
	std::ifstream file;

	file.open(fileName);
	std::string line;
	int lineNumber = 0;
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line == "End " + session)
				break;
			if (lineNumber == 0)
			{
				session = line;
				lineNumber++;
				continue;
			}
			Command command;
			int commandIndex = command.ParseCommand(line);
			auto commandExist = commandSet.find(commandIndex);
			if(commandExist == commandSet.end())
				commandSet[commandIndex] = command;			
		}
		file.close();
	}
}