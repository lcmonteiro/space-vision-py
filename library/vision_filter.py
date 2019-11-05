/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_filter.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VISION_FILTER_H
#define VISION_FILTER_H
/**
 * local
 */
#include "vision_types.hpp"
/**
 * ------------------------------------------------------------------------------------------------
 * VisionFilter 
 * ------------------------------------------------------------------------------------------------
 */
class VisionFilter {
public:
	/**
	 * ----------------------------------------------------
	 * result
	 * ----------------------------------------------------
	 */
	class Result {
	public:
		/**
		 * properties 
		 */
		inline auto level() {
			return __level;
		}
		inline auto label() {
			return __label;
		}
	protected:
		Result() = default;
		/**
		 * setters
		 */
		Result& level(FilterLevel level) {
			__level = level;
			return *this;
		}
		Result& label(FilterLabel label) {
			__label = label;
			return *this;
		}
	private:
		FilterLevel	__level;
		FilterLabel __label;
	};
	/**
	 * ----------------------------------------------------
	 * shared result 
	 * ----------------------------------------------------
	 */
	using SharedResult  = std::shared_ptr<Result>;
	using SharedResults = std::list<SharedResult>;
	/**
	 * ----------------------------------------------------
	 * process
	 * ----------------------------------------------------
	 */
   virtual SharedResults process(const VideoFrame& frame) = 0;
};
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
#endif /* VISION_FILTER_H */

