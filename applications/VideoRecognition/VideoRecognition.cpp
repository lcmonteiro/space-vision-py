#ifndef VIDEO_RECOGNITION_H
#define VIDEO_RECOGNITION_H

#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>  
#include <iomanip>
#include <sstream>
#include "Session.h"
#include "Log.h"
#include <opencv2/videoio.hpp>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "CharacterRecognition.h"
using namespace cv;
using std::cout;
using std::endl;


const char* src_window = "Main Window";
const char* calibration_template = "calibration_template.png";

int drag = 0, select_flag = 0;

Point point1, point2 = Point(0, 0);
bool callback = false;
bool calibrated = false;
const int CalibrationInterval = 10;
int cooldownCalibration = 0;
CharacterRecognition charRec;
Mat frame;

int state = 0;
Rect area1;

std::string command;

void detectAndShow(const Mat& test, const Mat& base);
bool PixelMatch(const Mat& test, const Mat& base, String label, int match_method, float threshold, Rect roi);
bool PixelMatch(const Mat& test, const Mat& base, String label, int match_method, float threshold, Rect roi, unsigned int multiScale);
double ShapeMatch(const Mat& test, const Mat& base);
void SaveImage(std::string fileName, Mat& test);
void SaveImage(Command command, Mat& test);
cv::Size Resize(cv::Size size, float factor);

void mouseHandler(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN && !drag && !select_flag)
	{		
		point1 = cv::Point(x, y);
		point2 = cv::Point(x, y);
		drag = 1;
	}

	if (event == cv::EVENT_MOUSEMOVE && drag && !select_flag)
	{		
		point2 = cv::Point(x, y);
	}

	if (event == EVENT_LBUTTONUP && drag && !select_flag)
	{
		point2 = cv::Point(x, y);
		drag = 0;
		area1 = Rect(point1, point2);
		Mat imagePortion = Mat(frame, area1);	
		SaveImage("ImageTest", imagePortion);
	}
}
int main(int argc, char* argv[])
{				
	Log log;	
	charRec.Train("images.xml", "classifications.xml");

	VisionCapture captRefrnc(0);
	/*captRefrnc.set(CAP_PROP_FRAME_WIDTH, 1280);
	captRefrnc.set(CAP_PROP_FRAME_HEIGHT, 720);*/
	captRefrnc.set(CAP_PROP_FRAME_WIDTH, 1280);
	captRefrnc.set(CAP_PROP_FRAME_HEIGHT, 960);
	captRefrnc.set(CAP_PROP_AUTOFOCUS, 0);
	//captRefrnc.set(CAP_PROP_, 0);

	Session session;	
	

	if (!captRefrnc.isOpened())
	{
		cout << "Could not open reference " << endl;
		return -1;
	}
	namedWindow(src_window, 1);
	cv::setMouseCallback(src_window, mouseHandler, 0);	

	int frameNum = -1;

	while (true)
	{
		bool bSuccess = captRefrnc.read(frame);

		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}


		//calibration
		//cooldownCalibration--;
		//if (!calibrated || cooldownCalibration <= 0)
		//{
		//	Mat calibrationTemplate = imread(calibration_template, IMREAD_ANYCOLOR);
		//	calibrated = PixelMatch(calibrationTemplate, frame, "Calibration", TM_CCOEFF_NORMED, 0.9f);						
		//	if(calibrated)
		//		cooldownCalibration = CalibrationInterval;
		//	continue;
		//}		

		
		session.ParseSession("commands.txt");

		for (auto& comm : session.commandSet)
		{
			if (comm.second.state == CommandState::finished)
			{
				continue;
			}
			if (comm.second.state == CommandState::idle)
			{
				comm.second.delay--;
				if (comm.second.delay <= 0)
				{
					comm.second.state = CommandState::started;
				}				
			}
			if (comm.second.state == CommandState::started)
			{
				Mat testImageComm = imread(comm.second.testImage, IMREAD_ANYCOLOR);
				bool result = false;
				if(comm.second.assetType == AssetType::image)
					result = PixelMatch(testImageComm, frame, comm.second.label, TM_CCOEFF_NORMED, 0.85f, comm.second.RoI, comm.second.scale);
				else if (comm.second.assetType == AssetType::number)
				{
					std::string strResult = charRec.RecognitionArea(&frame, comm.second.RoI);
					putText(frame, comm.second.label+"_"+strResult, cv::Point(comm.second.RoI.x, comm.second.RoI.y - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
					result = comm.second.testImage == strResult;
				}
				comm.second.duration--;
				//test type found once
				if (comm.second.testType == TestType::found_once)
				{
					if (!result && comm.second.duration > 0)
					{
						continue;
					}
					if (result)
						log.AddLine(comm.second, "Success");						
					if (!result && comm.second.duration <= 0)
						log.AddLine(comm.second, "Fail");

					log.LogToFile("result.txt");
					SaveImage(comm.second, frame);
					comm.second.state = CommandState::evaluated;
				}			
				else if (comm.second.testType == TestType::not_found_once)
				{
					if (result && comm.second.duration > 0)
						continue;					
					if (!result)
						log.AddLine(comm.second, "Success");
					else if(result && comm.second.duration <= 0)
						log.AddLine(comm.second, "Fail");

					log.LogToFile("result.txt");
					SaveImage(comm.second, frame);
					comm.second.state = CommandState::evaluated;
				}
				else if (comm.second.testType == TestType::found_always || comm.second.duration <= 0)
				{
					//incomplete
					if (result)
						log.buffer << comm.first << " " << " Success" << std::endl;
					log.LogToFile("result.txt");
					comm.second.state = CommandState::evaluated;
				}
			}
			if (comm.second.state == CommandState::evaluated)
			{
				comm.second.duration--;				
				if (comm.second.assetType == AssetType::image)
				{
					Mat testImageComm = imread(comm.second.testImage, IMREAD_ANYCOLOR);
					PixelMatch(testImageComm, frame, comm.second.label, TM_CCOEFF_NORMED, 0.85f, comm.second.RoI, comm.second.scale);
				}				
				else if (comm.second.assetType == AssetType::number)
				{
					std::string strResult = charRec.RecognitionArea(&frame, comm.second.RoI);
					putText(frame, comm.second.label + "_" + strResult, cv::Point(comm.second.RoI.x, comm.second.RoI.y - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
				}

				

				if(comm.second.visualType == VisualType::visual_transient || comm.second.duration <= 0)
					comm.second.state = CommandState::finished;
			}		
		}
	
		rectangle(frame, Point(area1.x - 2, area1.y - 2), Point(area1.width + area1.x + 2, area1.height + area1.y + 2), Scalar(255, 255, 255), 2, 8, 0);
		

		imshow(src_window, frame);
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}	

	waitKey();
	return 0;
}


void SaveImage(std::string fileName, Mat& image)
{
	struct stat buffer;
	int i = 0;
	std::ostringstream fileNameStream;
	bool exist = false;
	do
	{		
		fileNameStream.str(std::string());
		fileNameStream << fileName<< "_" << i << ".png";
		exist = (stat(fileNameStream.str().c_str(), &buffer) == 0);
		i++;
	} while (exist);

	Mat imagePortion = Mat(frame, area1);
	imwrite(fileNameStream.str().c_str(), image);
}

void SaveImage(Command command, Mat& test)
{
	std::ostringstream fileNameStream;
	std::time_t seconds = std::time(nullptr);
	fileNameStream << "result_" << seconds <<"_"<< command.label << "_index_" << command.index;	
	SaveImage(fileNameStream.str(), test);
}



double ShapeMatch(const Mat& test, const Mat& base)
{
	Mat i1, i2;
	i2 = base;
	cvtColor(test, i1, cv::COLOR_BGR2GRAY);
	cvtColor(base, i2, cv::COLOR_BGR2GRAY);

	threshold(i1, i1, 120, 255, THRESH_BINARY);
	threshold(i2, i2, 120, 255, THRESH_BINARY);	

	return matchShapes(i1, i2, CONTOURS_MATCH_I1, 0);
}

cv::Size Resize(cv::Size size, float factor)
{	
	size.width *= factor;
	size.height *= factor;

	return size;
}

bool PixelMatch(const Mat& test, const Mat& base, String label, int match_method, float threshold, Rect roi)
{
	Mat result;
	int result_cols = base.cols - test.cols + 1;
	int result_rows = base.rows - test.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);
	Mat MatRoI(base, roi);
	if(roi.size().area() > 0)
		matchTemplate(MatRoI, test, result, match_method);
	else
		matchTemplate(base, test, result, match_method);
	Mat threshold_im = result >= threshold;
	std::vector<cv::Point> locations;
	findNonZero(threshold_im, locations);
	if (!locations.empty())
	{

		Point matchLoc = locations[0];
		rectangle(base, Point(matchLoc.x+roi.x - 2, matchLoc.y+ roi.y - 2), Point(matchLoc.x + roi.x + test.cols + 2, matchLoc.y + roi.y + test.rows + 2), Scalar(0, 255, 0), 2, 8, 0);
		putText(base, label, Point(matchLoc.x+ roi.x, matchLoc.y+ roi.y - 10), FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 255, 0), 1);
		return true;
	}

return false;
}

bool PixelMatch(const Mat& test, const Mat& base, String label, int match_method, float threshold, Rect roi, unsigned int multiScale)
{	
	//no scale
	if (PixelMatch(test, base, label, match_method, threshold, roi))
		return true;


	for (unsigned int i = 1; i <= multiScale && i < 10 ; i++)
	{		
		
		Mat testResized1;
		cv::resize(test, testResized1, Resize(test.size(), 1.0 + (i * 0.1)), 0, 0, INTER_LINEAR);
		if (PixelMatch(testResized1, base, label, match_method, threshold, roi))
			return true;
		Mat testResized2;
		cv::resize(test, testResized2, Resize(test.size(), 1.0 - (i * 0.1)), 0, 0, INTER_LINEAR);			
		if (PixelMatch(testResized2, base, label, match_method, threshold, roi))
			return true;

	}

	return false;
}

#endif
