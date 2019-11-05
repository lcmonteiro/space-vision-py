/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_detection_lib.h
 * Author: Luis Monteiro
 * 
 * Created on October 3, 2019, 2:29 PM
 * ------------------------------------------------------------------------------------------------
 */
#include "vision_detection_lib.h"
/**
 * ----------------------------------------------------------------------------
 * interfaces
 * ----------------------------------------------------------------------------
 */
void vision_detection_init(void) {

}

void vision_detection_set_callback(FilterCallback func) {
    (void)func;
}

void vision_detection_config_filter(FilterConfig* filter) {
    (void)filter;
}

void vision_detection_enable_filter(FilterID id) {
    (void)id;
}

void vision_detection_disable_filter(FilterID id) {
    (void)id;
}

void vision_detection_disable_all_filters(void) {
}

void vision_detection_end(void) {

}
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
