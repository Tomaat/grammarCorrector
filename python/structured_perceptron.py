#!/usr/bin/python

""" Python implementation structured perceptron 

Pseudo code

for all sentences:
	
	for word in sentence: (via dependency tree):

		construct all possible input vectors

		for all input vectors:

			calculate score vector: f(x)*w
			total score sentence += score vector

			set backpointer (where did you come from?)

		if total score > actual score:
			update weights """


import numpy as np

def train_perceptron(rows_feature_vector):
	""" Input:	Number of rows of the feature vector --> equals number of input nodes
		Output:

		Method to train the entire perceptron
	"""

	weight_matrix = init_weights(rows_feature_vector)
	scores_word = []

	# This might change, depending on the output of the dependency tree and the decision we make considering the feature vectors --> not finished, and in some kind of pseudo code
	# Now we're looping over all feature vectors for a word, calculating the score and adding it to an array with tuples for this word
	for feature_vectors in feature_vectors_sentence:

		ideal_score = viterbi_ideal_score(correct_feature_vectors, weight_matrix) # this should also change --> how do you get your correct feature vectors
		
		for feature_vector in feature_vectors

			# Viterbi --> new function as you also need this for computing your ideal score
			update_feature_vector#...
			score = calculate_score_word(feature_vector, weight_matrx)
			scores_word.append((feature_vector, score))

			# at some point you have found a total score for the sentence --> this is the best score for this sentence and based on that you know your tag sequence

		if (total_score > ideal_score):
			updated_weight_matrix = update_weights



def viterbi_ideal_score(feature_vectors, weight_matrix):
	""" Input:	All feature vectors of the words in the sentence
		Output:	Total score of the word

		Method to compute the ideal score

	"""		

	# Viterbi algorithm here --> Depending on how we decide to feed our input vectors

	return ideal_score


def init_weights(no_rows):
	"""	Input:	number of rows of the feature vector --> Construct weight matrix with this many rows
		Output:	initialized weight matrix, with random values for the weights

		Method to initalize the weights of the perceptron. 
	"""

	weight_matrix = np.random.random((no_rows, 1))
	return weight_matrix

def calc_score_word(feature_vector, weight_matrix):
	""" Input: 	Feature vector of the word you're looking at
				current weight matrix
		Output:	Viterbi score of the current word

		Method to calculate the Viterbi score of the current word: feature vector * weight matrix
	"""

	score = np.dot(feature_vector, weight_matrix)
	return score


def construct_feature_vect():
	""" Input:	
		Output:	

		Method to construct the feature vector of a word, also based on the word before

	"""

def update_feature_vect():
	""" Input:
		Output:

		Method to update a feature vector with the previous tag

	"""

def update_weights():
	""" Input:	Old weight matrix
				Ideal total score
				Actual total score
		Output:	Updated weight matrix

		Method to update the weight matrix of the perceptron
	"""

	return updated_weights





