# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_filter.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
#
# imports
from numpy import array
#
# -------------------------------------------------------------------------------------------------
# VisionFilter 
# -------------------------------------------------------------------------------------------------
#
class VisionFilter:
	#
	# -------------------------------------------------------------------------
	# region
	# -------------------------------------------------------------------------
	#
	class Region:
		#
		# initialization
		#
		def __init__(self, position, dimension):
			self.__position  = array(position)
			self.__dimension = array(dimension)
		#
		# begin region
		#
		def begin(self, scale=array([1.0, 1.0])):
			return (self.__position * scale).astype(int)
		#
		# end region
		#
		def end(self, scale=array([1.0, 1.0])):
			return ((self.__position + self.__dimension) * scale).astype(int)
		#
		# size 
		#
		def size(self, scale=array([1.0, 1.0])):
			return self.__dimension * scale
	#
	# -------------------------------------------------------------------------
	# result
	# -------------------------------------------------------------------------
	#
	class Result:
		#
		# initialization
		#
		def __init__(self, label, level, region):
			self.__label  = label
			self.__level  = level
			self.__region = region
		#
		# properties 
		#
		def level(self):
			return self.__level
		def label(self):
			return self.__label
		def region(self):
			return self.__region
	#
	# ----------------------------------------------------
	# initialization
	# ----------------------------------------------------
	#
	def __init__(self, config):
		self._region = VisionFilter.Region(
			(0.1, 0.1), (0.8, 0.8)
		)
	#
	# ----------------------------------------------------
	# update
	# ----------------------------------------------------
	#
	def update(self, **kargs):
		pass
	#
	# ----------------------------------------------------
	# region
	# ----------------------------------------------------
	#
	def region(self):
		return self._region
	#
	# ----------------------------------------------------
	# process
	# ----------------------------------------------------
	#
	def process(self, frame):
		raise RuntimeError("no process found")
#
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
#