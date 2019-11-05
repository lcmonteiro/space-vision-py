#include "Log.h"
#include <ctime>



Log::Log()
{
}


Log::~Log()
{
}

void Log::LogToFile(std::string fileName)
{
	std::ofstream file;
	file.open(fileName);
	file << buffer.str();
	file.close();

}

void Log::AddLine(Command command, std::string comment)
{
	std::time_t seconds = std::time(nullptr);	

	buffer << seconds << " " <<command.index << " " << comment << std::endl;
}
