#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <sstream> 
#include <iostream> 
#include<opencv2/core/core.hpp>

enum AssetType
{
	image = 0, number
};

enum CommandState
{
	idle = 0, started, evaluated, finished 
};

enum TestType
{
	found_once = 0, not_found_once, found_always
};

enum VisualType
{
	visual_persistent = 0, visual_transient
};

class Command
{	
public:
	Command();
	~Command();
	int ParseCommand(const std::string &line);

	long delay;
	long duration;
	std::string label;
	AssetType assetType;
	cv::Rect RoI;
	unsigned int index;
	unsigned int scale;
	CommandState state;
	TestType testType;
	VisualType visualType;
	std::string testImage;
};

#endif