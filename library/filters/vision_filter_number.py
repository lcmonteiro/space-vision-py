/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_filter_numbers.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VIDEO_FILTER_NUMBERS_H
#define VIDEO_FILTER_NUMBERS_H
/**
 * local
 */
#include "vision_filter.hpp"
/**
 * ------------------------------------------------------------------------------------------------
 * VisionFilterNumbers 
 * ------------------------------------------------------------------------------------------------
 */
class VisionFilterNumber: public VisionFilter {
public:
    /**
	 * ----------------------------------------------------
	 * configuration
	 * ----------------------------------------------------
	 */
	VisionFilterNumber(FilterConfig conf);
    /**
	 * ----------------------------------------------------
	 * process
	 * ----------------------------------------------------
	 */
    SharedResults process(const VideoFrame& frame);
    /**
	 * ----------------------------------------------------
	 * result
	 * ----------------------------------------------------
	 */
	class ResultNumber: public Result {
	public:
		/**
		 * constructor
		 */
		ResultNumber(float_t number): __number(number) {
		}
		/**
		 * properties
		 */
		inline float_t number() {
			return __number;
		}
	private:
		float_t __number;
    };
};
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
#endif /* VIDEO_FILTER_NUMBERS_H */

