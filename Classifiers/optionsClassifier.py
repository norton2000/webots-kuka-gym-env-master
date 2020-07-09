
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ArffPrinter import ArffPrinter
import numpy as np
from enum import Enum

class Flags(Enum):
    true = True
    false = False
    Jolly = None

continue_writing = False
number_options = 5
number_parameters = 8

############################## SIGNIFICATO SEMANTICO DELLE VARIABILI ###################################
# s = [PALLA_POS_PRESA, PINZE_VICINE_PALLA, PALLA_CARICATA, ANELLO_POS_PRESSA, PINZE_VICINO_ANELLO, ANELLO_CARICATO, TOUCH_SENSOR_ATTTIVI, BRCCIO_POS_DEFAULT]

optionsMat = [[Flags.true,Flags.false,Flags.false,Flags.Jolly,Flags.false,Flags.Jolly,Flags.false,Flags.true], #ball_grasp_array_pre
			  [Flags.Jolly,Flags.true,Flags.false,Flags.Jolly,Flags.false,Flags.Jolly,Flags.true,Flags.false], #ball_grasp_array_post

			  [Flags.Jolly,Flags.false,Flags.Jolly,Flags.true,Flags.false,Flags.false,Flags.false,Flags.true], #ring_grasp_pre
			  [Flags.Jolly,Flags.false,Flags.Jolly,Flags.Jolly,Flags.true,Flags.false,Flags.true,Flags.false], #ring_grasp_post

			  [Flags.Jolly,Flags.true,Flags.false,Flags.Jolly,Flags.false,Flags.Jolly,Flags.true,Flags.false], #ball_release_pre
			  [Flags.false,Flags.false,Flags.true,Flags.Jolly,Flags.false,Flags.Jolly,Flags.false,Flags.Jolly], #ball_release_post

			  [Flags.Jolly,Flags.false,Flags.Jolly,Flags.Jolly,Flags.true,Flags.false,Flags.true,Flags.false], #ring_release_pre
			  [Flags.Jolly,Flags.false,Flags.Jolly,Flags.false,Flags.false,Flags.true,Flags.false,Flags.Jolly], #ring_release_post

			  [Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.false,Flags.false], #get_initial_pos_pre
			  [Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.Jolly,Flags.false,Flags.true]] #get_initial_pos_post


class optionsClassifier:

	def __init__(self):
		self.continue_writing = continue_writing
		self.number_options = number_options
		self.arffPrinter = ArffPrinter()
		
		if not self.continue_writing:
			self.arffPrinter.initFiles(number_options)

	
	def classifier(self, values_pre, values_post):
		i=0
		#print("======================================================== ciao =================")
		for i in range(5):
			esitoPre = self.scorri_array(values_pre, optionsMat[i*2])
			values_pre[number_parameters] = esitoPre
			esitoPost = self.scorri_array(values_post, optionsMat[(i*2)+1])
			values_post[number_parameters] = esitoPost
			self.arffPrinter.writeArffLine(i, values_pre, "preconditions")
			self.arffPrinter.writeArffLine(i, values_post, "effects")
			if esitoPre and esitoPost:
				self.arffPrinter.writeMaskLine(i, values_pre, values_post)

	def scorri_array(self, arrayValues, arrayConfronto):
		esito = True
		i=0
		for i in range(number_parameters):
			if not arrayConfronto[i] == Flags.Jolly:
				if arrayConfronto[i] == Flags.true:
					esito = esito and arrayValues[i]
					#print("EsitoTrue: ", esito)
					#print("ValueTrue: ", arrayValues[i])
				else:
					esito = esito and not arrayValues[i]
					#print("EsitoFalse: ", esito)
					#print("ValueFalse: ", arrayValues[i])

		return esito