/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_detection.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VISION_DETECTION_H
#define VISION_DETECTION_H
/**
 * std
 */
#include <istream>
#include <string>
#include <map>
/**
 * local
 */
#include "vision_filter.hpp"
/**
 * ------------------------------------------------------------------------------------------------
 * VisionDetection 
 * ------------------------------------------------------------------------------------------------
 */
class VisionDetection {
public:
    /**
     * ------------------------------------------------------------------------
     * constructor
     * ------------------------------------------------------------------------
     */
    VisionDetection(std::istream&  conf): __filters(_load_filters(conf)){}
    VisionDetection(std::istream&& conf): __filters(_load_filters(conf)){}
    /**
     * ------------------------------------------------------------------------
     * serve
     * ------------------------------------------------------------------------
     */
    int serve(std::function<
       void(FilterID, VisionFilter::SharedResult)> callback);
    /**
     * ------------------------------------------------------------------------
     * write commands
     * ------------------------------------------------------------------------
     */
    inline VisionDetection& set_filter(FilterID id, FilterLevel level=1.0) {
        std::lock_guard<std::mutex> lock(__locker);
        __select.emplace_back(id, level);
		return *this;
    }
    inline VisionDetection& clr_filter(FilterID id) {
        std::lock_guard<std::mutex> lock(__locker);
        __select.emplace_back(id, -1);
		return *this;
    }
	inline VisionDetection& set_filter_all(FilterLevel level=1.0) {
		std::lock_guard<std::mutex> lock(__locker);
		for(auto& f : __filters)
        	__select.emplace_back(std::get<0>(f), level);
		return *this;
    }
    inline VisionDetection& clr_filter_all() {
		std::lock_guard<std::mutex> lock(__locker);
		for(auto& f : __filters)
        	__select.emplace_back(std::get<0>(f), -1);
		return *this;
    }
protected:
    /**
     * ------------------------------------------------------------------------
     * read commands
     * ------------------------------------------------------------------------
     */
    using Commands = std::list<std::pair<FilterID, FilterLevel>>;
    /**
     * read commads selection 
     */
    inline Commands _read_commands() {
        std::lock_guard<std::mutex> lock(__locker);
        return std::move(__select);
    }
	/**
     * ------------------------------------------------------------------------
     * load filters
     * ------------------------------------------------------------------------
     */
	using Filters = std::map<FilterID, std::unique_ptr<VisionFilter>>;
	/**
     * load filters from a yaml stream
     */
	Filters _load_filters(std::istream& input);
private:
	/**
     * ------------------------------------------------------------------------
     * variables
     * ------------------------------------------------------------------------
     * locker
     */
    std::mutex __locker;
    /**
     * command containers  
     */
    Commands __select;
    /**
     * filter selected  
     */
    std::map<FilterID, FilterLevel> __selected;
    /**
     * filter data base 
     */
    const Filters __filters;
};
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
#endif /* VISION_DETECTION_H */

