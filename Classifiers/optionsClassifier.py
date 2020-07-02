
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ArffPrinter import ArffPrinter
import numpy as np

continue_writing = False
number_options = 5
number_parameters = 8

optionsMat = [[True,False,False,False,False,False,False,True], #ball_grasp_array_pre
			  [False,True,False,False,False,False,True,False], #ball_grasp_array_post

			  [False,False,False,True,False,False,False,True], #ring_grasp_pre
			  [False,False,False,False,True,False,True,False], #ring_grasp_post

			  [False,True,False,False,False,False,True,False], #ball_release_pre
			  [False,False,True,False,False,False,False,False], #ball_release_post

			  [False,False,False,False,True,False,True,False], #ring_release_pre
			  [False,False,False,False,False,True,False,False], #ring_release_post

			  [False,False,False,False,False,False,False,False], #get_initial_pos_pre
			  [False,False,False,False,False,False,False,True]] #get_initial_pos_post


class optionsClassifier:

	def __init__(self):
		
		self.continue_writing = continue_writing
		self.number_options = number_options
		self.arffPrinter = ArffPrinter()
		
		if self.continue_writing:
			self.arffPrinter.initFiles(number_options)

	
	def classifier(self, values_pre, values_post):
		i=0
		for i in range 5:
			esitoPre = scorri_array(values_pre, optionsMat[i*2])
			values_pre[number_parameters] = esitoPre
			esitoPost = scorri_array(values_post, optionsMat[(i*2)+1])
			values_post[number_parameters] = esitoPost
			arffPrinter.writeArffLine(i, values_pre, "preconditions")
			arffPrinter.writeArffLine(i, values_post, "effects")
			if esitoPre and esitoPost
				arffPrinter.writeMaskLine(i, values_pre, values_post)

	def scorri_array(self, arrayValues, arrayConfronto):
		
		esito = True
		i=0
		for i in range number_parameters:
			if arrayConfronto[i]
				esito = esito and arrayValues[i]

		return esito