/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_detection_lib.h
 * Author: Luis Monteiro
 * 
 * Created on October 3, 2019, 2:29 PM
 * ------------------------------------------------------------------------------------------------
 */
#ifndef VIDEO_DETECTION_LIB_H
#define VIDEO_DETECTION_LIB_H
/**
 * ----------------------------------------------------------------------------
 * imports/exports
 * ----------------------------------------------------------------------------
 */
#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif
/**
 * ----------------------------------------------------------------------------
 * definitions
 * ----------------------------------------------------------------------------
 * filter ID
 * --------------------------------------------------------
 */
typedef unsigned int FilterID;
/**
 * --------------------------------------------------------
 * filter type
 * -------------------------------------------------------- 
 */
typedef enum {
    GENERAL = 0
} FilterType;
/**
 * --------------------------------------------------------
 * filter configuration
 * -------------------------------------------------------- 
 */
typedef struct {
    FilterID   id;
    FilterType type;
    union {
        struct {

        } general;
    } filter;
} FilterConfig;
/**
 * --------------------------------------------------------
 * filter data
 * -------------------------------------------------------- 
 */
typedef struct {
    FilterID   id;
    FilterType type;
    union {
        struct {

        } general;
    } filter;
} FilterData;
/**
 * --------------------------------------------------------
 * filter callback
 * -------------------------------------------------------- 
 */
typedef int (*FilterCallback)(FilterID, FilterData*);
/**
 * ----------------------------------------------------------------------------
 * interfaces
 * ----------------------------------------------------------------------------
 */
#ifdef __cplusplus
extern "C" {
#endif
/**
 * 
 */
EXPORT void vision_detection_init(void);

EXPORT void vision_detection_config_filter(FilterConfig* filter);

EXPORT void vision_detection_enable_filter(FilterID id);

EXPORT void vision_detection_disable_filter(FilterID id);

EXPORT void vision_detection_disable_all_filters(void);

EXPORT void vision_detection_set_callback(FilterCallback func);

EXPORT void vision_detection_end(void);

#ifdef __cplusplus
}
#endif
/**
 * ------------------------------------------------------------------------------------------------
 * end
 * ------------------------------------------------------------------------------------------------
 */
#endif
