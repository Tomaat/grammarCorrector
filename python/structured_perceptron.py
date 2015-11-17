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


def set_train(bool):
	train = bool

def get_train():
	return train


def get_ideal_dict():
	return ideal_dict ## how do we set the ideal dict?



def structured_perceptron(sentence_dict, no_input_nodes):
	""" Input:	Dictionary of the sentence you are correcting. Based on dependency tree.
				Number of input nodes. This corresponds to the rows of your weight matrix.
		Output:	Sequence of errors (???)

		Method that 'runs' the perceptron. Behaves differently depending on whether in 
		training phase or not.

	"""

	if (get_train() == True):
		ideal_dict = get_ideal_dict
		train_perceptron(sentence_dict, ideal_dict, no_input_nodes)

	else:
		give_sequence(sentence_dict)




def train_perceptron(sentence_dict, ideal_dict, no_input_nodes):
	""" Input:	Number of rows of the feature vector --> equals number of input nodes
				Number of input nodes. This corresponds to the rows of your weight matrix.
		Output:

		Method to train the entire perceptron

	"""

	weight_matrix = init_weights(no_input_nodes)

	# 1) Loop over the sentence dictionary, 2) get array with all errors, 3) get array with possible feature vectors per error
	for position in range(len(sentence_dict)):
		all_error_vectors = sentence_dict[position]
		best_score_end = 0

		for feature_vectors in all_error_vectors:  # [(feature_vector, no_previous), (feature_vector, no_previous), etc]

			best_vector_score = 0
			count = 0
			position_best_score = 0
			best_feature_vector_tuple = ()

			for feature_vector_tuple in feature_vectors:
				vector_score = calculate_score_word(feature_vector_tuple[0], weight_matrix)
				count += 1
				if (vector_score > best_vector_score):
					best_vector_score = vector_score
					position_best_score = count
					best_feature_vector_tuple = feature_vector_tuple

			# Only keep the feature vector for a certain error that got the highest score
			feature_vectors[position_best_score] = best_feature_vector_tuple

			# Only in the end you're interested in the best score and you want to know how to follow the best path
			if position == len(sentence_dict):
				if best_vector_score > best_score_end:
					best_score_end = best_vector_score
					best_feature_vector_end = best_feature_vector_tuple

				all_error_vectors = best_feature_vector_tuple

	# Now you have a dictionary with for every word a vector with the best feature vector tuples

	# Follow the backpointers to get the correct sequence
	# Once you know for sure what vector you're going to use, you can immediately update the weights with this
	for position in range(len(sentence_dict)), 0, -1):
		all_errors = sentence_dict[position]

		# HIER VERDER!!



	'''scores_word = []

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
			updated_weight_matrix = update_weights '''



def give_sequence(sentence_dict):
	""" Input:	Dictionary of the sentence you are correcting. Based on dependency tree.
		Output:	Sequence of error tags

		Method that runs the viterbi algorithm to get final error sequence of the sentence.
		Only called once the perceptron has been trained. (???)

	"""

	return error_sequence


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





