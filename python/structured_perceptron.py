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
import dataparser as dp




###########################################################################################################

all_tags = {
			"-TAGSTART-":"root of tree",
			"Ne":"No Error",
			"Vt":"Verb tense",
			"Vm":"Verb modal",
			"V0":"Missing verb",
			"Vform":"Verb form",
			"SVA":"Subject-verb-agreement",
			"ArtOrDet":"Article or Determiner",
			"Nn":"Noun number",
			"Npos":"Noun possesive",
			"Pform":"Pronoun form",
			"Pref":"Pronoun reference",
			"Wcip":"Wrong collocation/idiom/preposition",
			"Wa":"Acronyms",
			"Wform":"Word form",
			"Wtone":"Tone",
			"Srun":"Runons, comma splice",
			"Smod":"Dangling modifier",
			"Spar":"Parallelism",
			"Sfrag":"Fragment",
			"Ssub":"Subordinate clause",
			"WOinc":"Incorrect sentence form",
			"WOadv":"Adverb/adjective position",
			"Trans":"Link word/phrases",
			"Mec":"Punctuation, capitalization, spelling, typos",
			"Rloc":"Local redundancy",
			"Cit":"Citation",
			"Others":"Other errors",
			"Um":"Unclear meaning (cannot be corrected)",						# hash with the full discription of every mistake in the dataset 
		}.keys()
tag_idxes = { tag:i for i,tag in enumerate(all_tags) }
SIZE = 1873
dt = None

def _init_(size,tb):
	global SIZE,dt
	SIZE,dt = size,tb


def train_perceptron(all_sentences, feature_dict, tbank, history):
	weight_matrix = init_weights(len(feature_dict))

	for sentence in all_sentences:
		if len(sentence.raw_sentence) < 1:
			continue
		parsed_tree = tbank.parse(sentence.raw_sentence)
		# For loop around this, so that you loop through all sentences --> weights should be updated
		sentence.words_tags
		context_words = [w.orth_ for w in dt.dfirst(parsed_tree) ]
		context_tags = [sentence.words_tags[dt.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in dt.dfirst(parsed_tree)]
		
		target_feature_vectors = []
		for i,wrd in enumerate(context_words):
			history_vectors = ('ph', [tuple(['-TAGSTART-']+context_tags[:i])] )
			if len(history_vectors[1][0]) > history:
				history_vectors = ('ph', [tuple(context_tags[i-history:i])] )
			#print history_vectors
			target_feature_vectors.append( dp.construct_feature_vector(wrd, context_tags[i], 
					feature_dict, context_words, i, history, history_vectors) )

		weight_matrix = train_perceptron_once(parsed_tree, target_feature_vectors, feature_dict, 
					history, weight_matrix, context_words)


	return weight_matrix


def init_weights(no_rows):
	"""	Input:	number of rows of the feature vector --> Construct weight matrix with this many rows
		Output:	initialized weight matrix, with random values for the weights

		Method to initalize the weights of the perceptron. 
	"""
	np.random.seed(43)
	weight_matrix = np.random.random((1,no_rows))
	return weight_matrix


def train_perceptron_once(parsed_tree, target_feature_vectors, feature_dict, history, weight_matrix, context_words):
	"""	Input:	Sentence that is fed into the perceptron
				Dictionary with feature vectors of the correct tagged sentence

	"""

	feature_vectors_sentence = viterbi(parsed_tree, feature_dict, history, weight_matrix, context_words)
	#print 'hello', target_feature_vectors
	new_weights = update_weights(weight_matrix, feature_vectors_sentence, target_feature_vectors)
	#print "old_weights: ", weight_matrix
	#print "new_weights: ", new_weights
	return new_weights

def get_tag_from_vector(feature_vector,feature_dict):
	idxes = [feature_dict.get('i tag+'+k,-1) for k in all_tags]
	tags = []
	for i,idx in enumerate(idxes):
		if not idx == -1:
			#print idx,feature_vector[idx]
			if feature_vector[idx] >= 1.0:
				tags.append(all_tags[i])
	return tags

def test_perceptron_once(E, parsed_tree, feature_dict, history, weight_matrix, context_words, context_tags=None):
	if context_tags is None:
		context_tags = ['Ne']*len(parsed_tree)
	feature_vectors_sentence = viterbi(parsed_tree, feature_dict, history, weight_matrix, context_words)

	for i,v in enumerate(feature_vectors_sentence):
		possible_tags = get_tag_from_vector(v,feature_dict)
		real_tag = context_tags[i]
		#print i,possible_tags,real_tag
		if not real_tag in possible_tags:
			E += 1
		#E += np.sum((target_feature_vectors[i][0][1]-v)**2)
	return E
		

def viterbi(parsed_tree, feature_dict, history, weight_matrix, context_words):
	""" Input:	The sentence to be tagged
				A list of all possible tags (strings)
				History: how far you want to look back

		Output:	A list with all feature vectors, in order per word
	"""

	sentence_dict = {} # per word all possible tags
	no_tags = len(all_tags)

	# --------------------------- Viterbi forward path --------------------------- #

	for i,wrd in enumerate(dt.dfirst(parsed_tree) ): # now you know the position of the word in your sentence
		#if i == 3:
		#	break
		feature_vector_array = np.zeros((no_tags, SIZE) ) # now we assume we have only two features per tag (n.b. so this is not only correct or false, it's features)
		tag_score_array = np.zeros((no_tags))
		history_list = []

		#####=====####==
		for j,tag in enumerate(all_tags): 
			# here you're gonna add your history. 
			
			history_vectors = sentence_dict.get(i-1,(0,0,[],0))[1:3]
			# for z in range(1,history+1):				
			# 	history_tuple = sentence_dict.get(i-z)
			# 	if history_tuple != None:
			# 		history_vectors.append((history_tuple[1:3])) # you need to add this feature vector --> then you've got some sort of backpointer
			
			#feature_vectors_tag = construct_feature_vector(wrd.orth_, tag, history_vectors) # now it should return a vector based on the history --> please return list with numpy arrays
			#print wrd.orth_,tag,history_vectors
			feature_vectors_tag = dp.construct_feature_vector(wrd.orth_, tag, 
					feature_dict, context_words, i , history, history_vectors)
			#[(history_vectors, feature_vector), (history_vectors, feature_vector), ...] --> Though I guess one history vector should be enough, as then you've got a backpointer for every feature vector
			
			best_tag_score = -1e1000 # init scores --> delete once more clever list implementation with max
			best_feature_vector = np.zeros(SIZE) # number of features --> CHANGE
			history_word = ('Um') # what's the position of the tag the current tag is 'coming from'
			#print len(feature_vectors_tag)
			#print feature_vectors_tag[0][0][0:28]
			for tple in feature_vectors_tag:
				#print "tuple: ", tple
				tag_score = np.dot(tple[0], weight_matrix.transpose()) # might want to this with this python list stuff, but like this for now
				#print "tag_score: ", tag_score
				if tag_score > best_tag_score:
					best_tag_score = tag_score
					best_feature_vector = tple[0]
					history_word = tple[1]


			tag_score_array[j] = best_tag_score
			#print best_feature_vector
			feature_vector_array[j,:] = best_feature_vector
			#print feature_vector_array[j,0:28]
			history_list.append(history_word)

		# print 'scores',tag_score_array
		# print 'fvec',feature_vector_array
		# print 'hislist',history_list
		#print feature_vector_array[:,0:28]
		sentence_dict[i] = (tag_score_array, feature_vector_array, history_list)


	# --------------------------- Viterbi backward path --------------------------- #

	final_feature_vectors = []

	dict_len = len(sentence_dict)
	for entry in range(dict_len-1, -1, -1):
		
		(score, vector, history_list) = sentence_dict[entry]
		history_best_vector = -1

		# if you're at the end of the sentence you have to make your decision slightly differently
		if entry == dict_len-1: 
			high_score =  score.argmax()
			best_vector = vector[high_score]
			if len(history_list[high_score]) < 2:
				history_best_vector = -1
			else:
				history_best_vector = tag_idxes[history_list[high_score][-2]] # is a number
		else:
			best_vector = vector[history_best_vector]
			if len(history_list[high_score]) < 2:
				history_best_vector = -1
			else:
				history_best_vector = tag_idxes[history_list[high_score][-2]] # is a number
		#print history_list[high_score]
		final_feature_vectors.append(best_vector) ## might want to change the order of this, or not, depends a bit on how we decide to give the output for the sequence


	#print "final feature vectors: ", final_feature_vectors
	#print [v[0:28] for v in final_feature_vectors]
	return final_feature_vectors


# def construct_feature_vector(word, tag, history_vectors):
# 	""" Input:	word
# 				Tag
# 		Output:	Feature vector --> Now this is a random vector. Maurits writes this method based on the data.

# 		Method to construct the feature vector of a word, also based on the word before --> Dummy method for now

# 	"""

# 	# for now this is just a dummy method, assuming history=1
# 	feature_vector = np.random.randint(2, size=2)
# 	history = [0,1] # just assuming the first vector is coming from the first mistake and the second vector is coming from the second mistake
# 	return_list = [(history, feature_vector)]
# 	return return_list


def update_weights(old_weights, feature_vectors_sentence, target_feature_vectors):
	""" Input:	Old weight matrix
				Feature vectors as predicted by viterbi
				Correct feature vectors
		Output:	Updated weight matrix

		Method to update the weight matrix of the perceptron
	"""

	for i in range(len(feature_vectors_sentence)):
		#print i,target_feature_vectors[i][0][0],feature_vectors_sentence[i]
		diff = target_feature_vectors[i][0][0] - feature_vectors_sentence[i]
		#print "diff: ", diff 
		updated_weights = np.add(old_weights, diff) 
		old_weights = updated_weights
		#print "updated weights: ", updated_weights

	return updated_weights

if __name__ == '__main__':


	weights = np.random.random((1, 2))
	target_feature_vectors = [np.array((1, 0)), np.array((1,1))]
	train_perceptron_once(['hello','world'], target_feature_vectors, ['g','f'],0, weights)

	#print 'weights: ', weights, weights.shape[0], weights.shape[1]
	#viterbi(['hello','world'],['g','f'],0, weights)


###################################################################################################################3


# def give_sequence(sentence_dict):
# 	""" Input:	Dictionary of the sentence you are correcting. Based on dependency tree.
# 		Output:	Sequence of error tags

# 		Method that runs the viterbi algorithm to get final error sequence of the sentence.
# 		Only called once the perceptron has been trained. (???)

# 	"""

# 	return error_sequence


# def viterbi_ideal_score(feature_vectors, weight_matrix):
# 	""" Input:	All feature vectors of the words in the sentence
# 		Output:	Total score of the word

# 		Method to compute the ideal score

# 	"""		

# 	# Viterbi algorithm here --> Depending on how we decide to feed our input vectors

# 	return ideal_score




# def calc_score_word(feature_vector, weight_matrix):
# 	""" Input: 	Feature vector of the word you're looking at
# 				current weight matrix
# 		Output:	Viterbi score of the current word

# 		Method to calculate the Viterbi score of the current word: feature vector * weight matrix
# 	"""

# 	score = np.dot(feature_vector, weight_matrix)
# 	return score





# def update_feature_vect():
# 	""" Input:
# 		Output:

# 		Method to update a feature vector with the previous tag

# 	"""

# def update_weights():
# 	""" Input:	Old weight matrix
# 				Ideal total score
# 				Actual total score
# 		Output:	Updated weight matrix

# 		Method to update the weight matrix of the perceptron
# 	"""

# 	return updated_weights






