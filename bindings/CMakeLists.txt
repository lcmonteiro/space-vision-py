cmake_minimum_required (VERSION 3.10)
# -------------------------------------------------------------------
# traget
# -------------------------------------------------------------------
# create
add_library(VisionDetectionBinding
    vision_detection_lib.cpp
)
# includes
target_include_directories(VisionDetectionBinding
PUBLIC
	.
)
# dependencies 
# target_link_libraries(SpaceConverter
# PRIVATE
#     ??
# )
# -------------------------------------------------------------------
# end
# -------------------------------------------------------------------