/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_video_display.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VIDEO_DISPLAY_H
#define VIDEO_DISPLAY_H
/**
 * opencv
 */
#include <opencv2/core.hpp>
/**
 * local
 */
#include "vision_types.hpp"
/**
 * ------------------------------------------------------------------------------------------------
 * VideoDisplay 
 * ------------------------------------------------------------------------------------------------
 */
class VideoDisplay {
public:
	VideoDisplay(const std::string& id): __id(id) {
		cv::namedWindow(__id, 1);
	}
	void print(const VideoFrame& frame) {
		cv::imshow(__id, frame);
	}
private:
	std::string __id;
};
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
#endif /* VIDEO_DISPLAY_H */

