cmake_minimum_required (VERSION 3.10)
# -------------------------------------------------------------------
# traget VisionDetection
# -------------------------------------------------------------------
add_executable(VisionDetectionApp 
	vision_detection_app.cpp
)
# dependencies 
target_link_libraries(VisionDetectionApp
PRIVATE
	VisionDetection
	CONAN_PKG::args
)
# -------------------------------------------------------------------
# end
# -------------------------------------------------------------------