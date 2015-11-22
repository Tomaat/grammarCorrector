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

		Output:	A list with all feature vectors, in order per word
	"""

	sentence_dict = {} # per word all possible tags
	no_tags = len(all_tags)

	# Viterbi forward path
	for i,wrd in enumerate(sentence): # now you know the position of the word in your sentence
		feature_vector_array = np.zeros((no_tags, 2)) # now we assume we have only two features per tag (n.b. so this is not only correct or false, it's features)
		tag_score_array = np.zeros((no_tags))
		history_list = []
		
		for j,tag in enumerate(all_tags): 
			# here you're gonna add your history. 
			
			history_vectors = []
			for z in range(history):				
				history_tuple = sentence_dict.get(i-z)
				if history_tuple != None:
					history_vectors.append(history_tuple[1]) # you need to add this feature vector --> then you've got some sort of backpointer


			feature_vectors_tag = construct_feature_vector(wrd, tag, history_vectors) # now it should return a vector based on the history --> please return list with numpy arrays
			#[(history_vectors, feature_vector), (history_vectors, feature_vector), ...] --> Though I guess one history vector should be enough, as then you've got a backpointer for every feature vector
			# history vector should be an array with numbers --> numbers correspnding to tag positions

			print "feature vectors tag: ", feature_vectors_tag


			best_tag_score = 0 # init scores --> delete once more clever list implementation with max
			best_feature_vector = np.zeros(2) # number of features --> CHANGE
			history_word = -1 # what's the position of the tag the current tag is 'coming from'
			for tple in feature_vectors_tag:
				print "tuple: ", tple
				tag_score = np.dot(tple[1], weight_matrix.transpose()) # might want to this with this python list stuff, but like this for now
				print "tag_score: ", tag_score
				if tag_score > best_tag_score:
					best_tag_score = tag_score
					best_feature_vector = tple[1]
					history_word = tple[0]


			tag_score_array[j] = best_tag_score
			feature_vector_array[j,:] = best_feature_vector
			history_list.append(history)

		
		sentence_dict[i] = (tag_score_array, feature_vector_array, history_list)

	# Viterbi backward path
	final_feature_vectors = []

	dict_len = len(sentence_dict)
	for entry in range(dict_len-1, -1, -1):
		
		(score, vector, history) = sentence_dict[entry]
		history_best_vector = -1

		# if you're at the end of the sentence you have to make your decision slightly differently
		if entry == dict_len: 
			high_score =  score.argmax()
			best_vector = vector[high_score]
			history_best_vector = history[high_score] # is a number
		else:
			best_vector = vector[history_best_vector]
			history_best_vector = history[history_best_vector]
		
		final_feature_vectors.append(best_vector) ## might want to change the order of this, or not, depends a bit on how we decide to give the output for the sequence




	
	print "final feature vectors: ", final_feature_vectors


	#print sentence_dict


def construct_feature_vector(word, tag, history_vectors):
	""" Input:	word
				Tag
		Output:	Feature vector --> Now this is a random vector. Maurits writes this method based on the data.

		Method to construct the feature vector of a word, also based on the word before

	"""

	# for now this is just a dummy method, assuming history=1
	feature_vector = np.random.randint(2, size=2)
	history = [0,1] # just assuming the first vector is coming from the first mistake and the second vector is coming from the second mistake
	return_list = [(history, feature_vector)]
	return return_list


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






