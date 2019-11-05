#include "CharacterRecognition.h"



CharacterRecognition::CharacterRecognition()
{
	
}

bool CharacterRecognition::Train(std::string imageXML, std::string classeXML)
{
	cv::Mat knnClasses;
	cv::FileStorage XmlClasses(classeXML, cv::FileStorage::READ);
	if (XmlClasses.isOpened() == false) {
		std::cout << "error opening class XML file";
		return false;
	}
	XmlClasses["classifications"] >> knnClasses;
	XmlClasses.release();

	cv::Mat knnImages;
	cv::FileStorage XmlImages(imageXML, cv::FileStorage::READ);
	if (XmlImages.isOpened() == false) {
		std::cout << "error opening image XML file";
		return false;
	}
	XmlImages["images"] >> knnImages;
	XmlImages.release();

	KNN = cv::Ptr<cv::ml::KNearest>(cv::ml::KNearest::create());
	return KNN->train(knnImages, cv::ml::ROW_SAMPLE, knnClasses);
}


CharacterRecognition::~CharacterRecognition()
{
}
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

std::string CharacterRecognition::RecognitionArea(cv::Mat* image, cv::Rect area)
{	
	std::vector<SmartContour> smartValidContoursVector;

	cv::Mat testImage = cv::Mat(*image, area);

	cv::Mat gray;          	      
	cv::Mat enhancedCopy;         

	cv::cvtColor(testImage, gray, cv::COLOR_BGR2GRAY);        
	//invert color
	cv::bitwise_not(gray, gray); 
	// blur
	cv::GaussianBlur(gray,gray, cv::Size(5, 5), 0);		
	// binarize
	cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
	enhancedCopy = gray.clone();	

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> contourHierarchy;

	cv::findContours(enhancedCopy, contours, contourHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		
	for (int i = 0; i < contours.size(); i++) 
	{
		SmartContour contourWithData(contours[i]);						
		if (contourWithData.isValid(area))
		{                   
			smartValidContoursVector.push_back(contourWithData);
			cv::rectangle(testImage, contourWithData.boundingBox, cv::Scalar(0, 255, 0), 2);			
		}
	}

	
	
	std::sort(smartValidContoursVector.begin(), smartValidContoursVector.end(), SmartContour::sortByX);

	std::string strFinalString;
	for (int i = 0; i < smartValidContoursVector.size(); i++) 
	{            
		cv::Mat matROI = enhancedCopy(smartValidContoursVector[i].boundingBox);		
		cv::Mat matROIResized;
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
		cv::Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1); 
		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);		
		cv::Mat matCurrentChar(0, 0, CV_32F);

		KNN->findNearest(matROIFlattenedFloat, 1, matCurrentChar);
		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		strFinalString = strFinalString + char(int(fltCurrentChar));
	}
	

	return strFinalString;
}
