/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_detection.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VISION_TYPES_H
#define VISION_TYPES_H
/**
 * std
 */
#include <istream>
#include <string>
/**
 * opencv
 */
#include <opencv2/core.hpp>
/**
 * yaml
 */
#include <yaml-cpp/yaml.h>
/**
 * log
 */
#include <spdlog/spdlog.h>
/**
 * ------------------------------------------------------------------------------------------------
 * Definitions 
 * ------------------------------------------------------------------------------------------------
 * Filter - Configuration
 * --------------------------------------------------------
 */
class FilterConfig: public YAML::Node {
    using Base = YAML::Node;
public:
    /**
     * constructor
     */
    FilterConfig(Base&& node): Base(node) {
    }
    /**
     * load
     */
    static FilterConfig Load(std::istream& input) {
        return YAML::Load(input);
    }
};
/**
 * --------------------------------------------------------
 * Filter - Types
 * --------------------------------------------------------
 */
using FilterID    = std::string; 
using FilterLabel = std::string; 
using FilterLevel = std::double_t; 
/* 
 * --------------------------------------------------------
 * video
 * --------------------------------------------------------
 */
using VideoFrame  = cv::Mat;
/**
 * --------------------------------------------------------
 * log
 * --------------------------------------------------------
 */
class Log {
public:
    template<typename... Args>
    static void Info(const Args &... args) {
        spdlog::info(args...);
    }
    template<typename... Args>
    static void Warning(const Args &... args) {
        spdlog::warn(args...);
    }
    template<typename... Args>
    static void Error(const Args &... args) {
        spdlog::error(args...);
    }
};
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
#endif /* VISION_TYPES_H */