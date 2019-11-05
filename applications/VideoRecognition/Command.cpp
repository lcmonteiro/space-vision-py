#include "Command.h"

Command::Command() : delay(0), duration(1), label(""), assetType(AssetType::image), index(0), scale(0), state(CommandState::idle),testType(TestType::found_once), visualType(VisualType::visual_persistent), testImage(""), RoI(cv::Rect(0,0,0,0))
{

}

Command::~Command()
{

}

AssetType ParseAssetType(std::string token)
{
	if (token == "image")
	{
		return AssetType::image;
	}
	if (token == "number")
	{
		return AssetType::number;
	}

	return AssetType::image;

}

TestType ParseTestType(std::string token)
{
	if (token == "found_once")
	{
		return TestType::found_once;
	}
	if (token == "not_found_once")
	{
		return TestType::not_found_once;
	}

	return TestType::found_once;
	
}

VisualType ParseVisualType(std::string token)
{
	if (token == "visual_persistent")
	{
		return VisualType::visual_persistent;
	}
	if (token == "visual_transient")
	{
		return VisualType::visual_transient;
	}

	return VisualType::visual_persistent;

}

cv::Rect ParseRoI(std::string token)
{
	cv::Rect ret(0,0,0,0);

	std::string quad;
	std::istringstream tokenStream(token);
	int iteraction = 0;
	while (std::getline(tokenStream, quad, ','))
	{
		switch (iteraction)
		{
		case 0:
			ret.x = atoi(quad.c_str());
			break;
		case 1:
			ret.y = atoi(quad.c_str());
			break;
		case 2:
			ret.width = atoi(quad.c_str());
			break;
		case 3:
			ret.height = atoi(quad.c_str());
			break;
		default:
			break;
		}

		iteraction++;
	}

	return ret;

}

int Command::ParseCommand(const std::string &line)
{			
	std::string token;
	std::istringstream tokenStream(line);	
	int iteraction = 0;
	while (std::getline(tokenStream, token, ' '))
	{

		switch (iteraction)
		{
		case 0:
			index = atoi(token.c_str());
			break;
		case 1:
			assetType = ParseAssetType(token);
			break;
		case 2:
			testImage = token;
			break;
		case 3:
			delay = atoi(token.c_str());
			break;
		case 4:
			duration = atoi(token.c_str());
			break;
		case 5:
			label = token;
			break;
		case 6:
			testType = ParseTestType(token);
			break;
		case 7:
			visualType = ParseVisualType(token);
			break;
		case 8:
			scale = atoi(token.c_str());
			break;
		case 9:
			RoI = ParseRoI(token);
			break;
		default:
			break;
		}

			iteraction++;
		
	}

	return index;
}