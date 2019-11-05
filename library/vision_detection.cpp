/**
 * ------------------------------------------------------------------------------------------------
 * File:   STime.cpp
 * Author: Luis Monteiro
 * 
 * Created on Mai 19, 2019, 22:29 PM
 * ------------------------------------------------------------------------------------------------
 **
 * std
 */
#include <iostream>
#include <fstream>
#include <string>  
#include <iomanip>
#include <sstream>
/**
 * opencv
 */
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
/**
 * local
 */
#include "vision_detection.hpp"
#include "vision_config.hpp"
#include "vision_video_display.hpp"
/**
 * namespaces
 */
using namespace std;
using namespace cv;
/**
 * ----------------------------------------------------------------------------
 * serve
 * ----------------------------------------------------------------------------
 */
int VisionDetection::serve(function<void(FilterID, VisionFilter::SharedResult)> callback) {				
	VideoDisplay display("1");
	VideoCapture capture(0);
	/**
	 * setup video capture
	 */
	capture.set(CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CAP_PROP_FRAME_HEIGHT, 960);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	if (!capture.isOpened()){
		cout << "Could not open reference " << endl;
		return -1;
	}
	/**
	 * image processing
	 */
	for(VideoFrame frame; capture.read(frame); display.print(frame)) {
		/**
		 * select filters
		 */
		for(auto& cmd : _read_commands()) {
			if(std::get<1>(cmd) > 0) {
				__selected.emplace(std::get<0>(cmd), std::get<1>(cmd));
			} else {
				__selected.erase(std::get<0>(cmd));
			}
		}
		/**
		 * process filters
		 */
		for(auto& s: __selected) {
			try {
				for(auto& result: __filters.at(std::get<0>(s))->process(frame)) {
					if(result->level() > std::get<1>(s)) {
						callback(std::get<0>(s), result);
					}
				}
			} catch(...) {

			}
		}
		if (waitKey(10) == 27) {
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			return 0;
		}
	}	
	cout << "Found the end of the video" << endl;
	return 0;
}
/**
 * ----------------------------------------------------------------------------
 * load filters
 * ----------------------------------------------------------------------------
 */
VisionDetection::Filters VisionDetection::_load_filters(std::istream& input) {
	auto conf = FilterConfig::Load(input);
	/**
	 * load filters
	 */
	auto filters = VisionDetection::Filters();
	for(auto filter: conf["filters"]) {
		try {
			filters.emplace(
				std::get<0>(filter).as<FilterID>(), 
				FilterBuilder::Build(
					std::get<1>(filter)["type"].as<string>(), 
					std::get<1>(filter)["conf"]
				)
			);
		} catch(std::exception&) {
			Log::Warning("fail to load {} filter", std::get<0>(filter).as<FilterID>());
		}
	}
	return filters;
}
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
