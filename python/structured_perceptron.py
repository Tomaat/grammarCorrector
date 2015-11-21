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





###########################################################################################################





def train_perceptron(all_tags, history):

	weight_matrix = init_weights(len(all_tags))

	# For loop around this, so that you loop through all sentences --> weights should be updated
	train_perceptron_once(sentence, target_dict, all_tags, history)



def train_perceptron_once(sentence, target_dict, all_tags, history):
	"""	Input:	Sentence that is fed into the perceptron
				Dictionary with feature vectors of the correct tagged sentence

	"""

	viterbi(sentence, all_tags, history, weight_matrix)


def viterbi(sentence, all_tags, history, weight_matrix):
	""" Input:	The sentence to be tagged
				A list of all possible tags (strings)
				History: how far you want to look back

	"""

	sentence_dict = {} # per word all possible tags
	no_tags = len(all_tags)

	# Viterbi forward path
	for i,wrd in enumerate(sentence): # now you know the position of the word in your sentence
		feature_vector_array = np.zeros((no_tags, 2)) # now we assume we have only two features per tag (n.b. so this is not only correct or false, it's features)
		tag_score_array = np.zeros((no_tags))
		
		for j,tag in enumerate(all_tags): 
			# here you're gonna add your history. First pretending we don't take history into account:
			
			#for z in range(history):
			#	sentence_dict.get(i-z):

			feature_vector_tag = construct_feature_vector(wrd, tag) # still add the history to this function
			feature_vector_array[j,:] = feature_vector_tag
			#print 'feature_vector_tag: ', feature_vector_tag
			
			tag_score = np.dot(feature_vector_tag, weight_matrix.transpose()) # with history you need to take the max here
			tag_score_array[j] = tag_score
			#print 'tag_score: ', tag_score
		
		sentence_dict[i] = (tag_score_array, feature_vector_array)

	# Viterbi backward path
	final_feature_vectors = []

	for entry in range(len(sentence_dict)-1, -1, -1):
		(score, vector) = sentence_dict[entry]
		high_score =  score.argmax()
		best_vector = vector[high_score]
		final_feature_vectors.append(best_vector) # but now we still need to implement the history
	
	print final_feature_vectors


	#print sentence_dict


def construct_feature_vector(word, tag):
	""" Input:	word
				Tag
		Output:	Feature vector --> Now this is a random vector. Maurits writes this method based on the data.

		Method to construct the feature vector of a word, also based on the word before

	"""
	return np.random.randint(2, size=2)


if __name__ == '__main__':
	weights = np.random.random((1, 2))
	print 'weights: ', weights, weights.shape[0], weights.shape[1]
	viterbi(['hello','world'],['g','f'],0, weights)


def init_weights(no_rows):
	"""	Input:	number of rows of the feature vector --> Construct weight matrix with this many rows
		Output:	initialized weight matrix, with random values for the weights

		Method to initalize the weights of the perceptron. 
	"""

	weight_matrix = np.random.random((no_rows, 1))
	return weight_matrix


###################################################################################################################3


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




def calc_score_word(feature_vector, weight_matrix):
	""" Input: 	Feature vector of the word you're looking at
				current weight matrix
		Output:	Viterbi score of the current word

		Method to calculate the Viterbi score of the current word: feature vector * weight matrix
	"""

	score = np.dot(feature_vector, weight_matrix)
	return score





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






