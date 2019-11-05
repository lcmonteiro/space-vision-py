#ifndef CHARACTER_RECOGNITION_H
#define CHARACTER_RECOGNITION_H

#include <string>
#include<iostream>
#include<sstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>


class SmartContour {
public:
	SmartContour(std::vector<cv::Point> contour)
	{
		rawContour = contour;
		boundingBox = cv::boundingRect(rawContour);
	}
	
	std::vector<cv::Point> rawContour;       
	cv::Rect boundingBox;                     	                    

	bool isValid(cv::Rect area) {
		float markArea = area.size().width*area.size().height;
		float contourArea = cv::contourArea(rawContour);
		if (contourArea / markArea < 0.02f) 
			return false;
		return true;                                  
	}
	
	static bool sortByX(const SmartContour& left, const SmartContour& right) 
	{      
		return(left.boundingBox.x < right.boundingBox.x);
	}

};

class CharacterRecognition
{
private:
	const int MIN_CONTOUR_AREA = 80;
	const int RESIZED_IMAGE_WIDTH = 20;
	const int RESIZED_IMAGE_HEIGHT = 30;
	cv::Ptr<cv::ml::KNearest>  KNN;

public:
	CharacterRecognition();
	~CharacterRecognition();

	bool Train(std::string imageXML, std::string classeXML);
	std::string RecognitionArea(cv::Mat* image, cv::Rect area);
};

#endif

