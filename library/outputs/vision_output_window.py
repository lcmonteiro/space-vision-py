# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_output.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
# external
from cv2                   import imshow                  
from cv2                   import destroyWindow                  
from cv2                   import namedWindow, WINDOW_NORMAL                   
from cv2                   import rectangle, putText, FONT_HERSHEY_SIMPLEX
from numpy                 import array, flip
from seaborn               import color_palette
# internal
from library.vision_output import VisionOutput
# #############################################################################
# -----------------------------------------------------------------------------
# VisionOutput 
# -----------------------------------------------------------------------------
# #############################################################################
class VisionOutputWindow(VisionOutput):
	#
	# -------------------------------------------------------------------------
	# initialization
	# -------------------------------------------------------------------------
	#
	def __init__(self, id):
		self.__id = id
		namedWindow(id, WINDOW_NORMAL)
	#
	# -------------------------------------------------------------------------
	# destroy
	# -------------------------------------------------------------------------
	#
	def __del__(self):
		destroyWindow(self.__id)
	# 
	# -------------------------------------------------------------------------
	# set frame
	# -------------------------------------------------------------------------
	#
	def set_frame(self, frame):
		from library.vision_filter import VisionFilter
		self.__frame  = frame
		self.__filter = {
			'test': (
				VisionFilter.Region((0.25,0.25), (0.25, 0.25)),[
					VisionFilter.Result(
						'123', 0.9, VisionFilter.Region((0.5,0.5), (0.5, 0.5)))
				]
			)
		}
	# 
	# -------------------------------------------------------------------------
	# add filter frame
	# -------------------------------------------------------------------------
	#
	def add_filter(self, name, region, results:list):
		self.__filter[name] = (region, results)
	# 
	# -------------------------------------------------------------------------
	# write
	# -------------------------------------------------------------------------
	#
	def write(self):
		# colors generation
		colors = array(color_palette(None, len(self.__filter))) * 255
		# print regions
		for (name, (region, results)), color in zip(self.__filter.items(), colors):
			self._write_region(name, region, flip(color))
			for result in results:
				self._write_detection(region, result, flip(color))
		imshow(self.__id, self.__frame)
	# 
	# -------------------------------------------------------------------------
	# write tools
	# -------------------------------------------------------------------------
	#
	def _write_region(self, name, region, color):
		f_size = array(self.__frame.shape[:2])
		begin  = tuple(region.begin(f_size))
		end    = tuple(region.end  (f_size))
		putText(self.__frame, name, begin, FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		rectangle(self.__frame, begin, end, color, 1)

	def _write_detection(self, region, result, color):
		f_size = array(self.__frame.shape[:2])
		r_size = region.size(f_size)
		begin  = tuple(result.region().begin(r_size) + region.begin(f_size))
		end    = tuple(result.region().end  (r_size) + region.begin(f_size))
		putText(self.__frame, result.label(), begin, FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		rectangle(self.__frame, begin, end, color, 2)

# #################################################################################################
# -------------------------------------------------------------------------------------------------
# End
# -------------------------------------------------------------------------------------------------
# #################################################################################################