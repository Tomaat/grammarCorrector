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
import pipeline
from time import time


all_tags = ["Ne",#:"No Error",
			"Vt",#:"Verb tense",
			"Vm",#:"Verb modal",
			"V0",#:"Missing verb",
			"Vform",#:"Verb form",
			"SVA",#:"Subject-verb-agreement",
			"ArtOrDet",#:"Article or Determiner",
			"Nn",#:"Noun number",
			"Npos",#:"Noun possesive",
#'na'];xxx=[
			"Pform",#:"Pronoun form",
			"Pref",#:"Pronoun reference",
			"Wcip",#:"Wrong collocation/idiom/preposition",
			"Wa",#:"Acronyms",
			"Wform",#:"Word form",
			"Wtone",#:"Tone",
			"Srun",#:"Runons, comma splice",
			"Smod",#:"Dangling modifier",
			"Spar",#:"Parallelism",
			"Sfrag",#:"Fragment",
			"Ssub",#:"Subordinate clause",
			"WOinc",#:"Incorrect sentence form",
			"WOadv",#:"Adverb/adjective position",
			"Trans",#:"Link word/phrases",
			"Mec",#:"Punctuation, capitalization, spelling, typos",
			"Rloc",#:"Local redundancy",
			"Cit",#:"Citation",
			"Others",#:"Other errors",
			"Um",#:"Unclear meaning (cannot be corrected)",						# hash with the full discription of every mistake in the dataset 
		]
tag_idxes = { tag:i for i,tag in enumerate(all_tags) }
tag_idxes["-TAGSTART-"] = -1

SIZE = 1873
dt = None
iterloop = None
golinear = True

iters = 10
it = 0


def _init_(size,tb,gotype):
	global SIZE,dt,iterloop,golinear
	SIZE,dt = size,tb
	golinear=gotype
	if golinear:
		iterloop = dt.linear
	else:
		iterloop = dt.dfirst

def v2d(vec):
	dic = {} 
	for i,v in enumerate(vec):
		if v != 0.:
			dic[i] = v
	return (vec.shape , dic)
	
def d2v(dic):
	shape,data = dic
	vec = np.zeros(shape)
	for i,v in data.iteritems():
		vec[i] = v
	return vec
		
def train_perceptron(all_sentences, feature_dict, tbank, history):
	weight_matrix = init_weights(len(feature_dict))
	pre_pros = []
	t1 = time()
	current_sen = 1
	for sentence in all_sentences:
		print "train sentence: "+str(current_sen)
		current_sen += 1
		try:
			parsed_tree = tbank.parse(sentence.raw_sentence)
			# For loop around this, so that you loop through all sentences --> weights should be updated
		
			histories = []
			target_feature_vectors = []
			if golinear:
				context_words = [w.orth_ for w in iterloop(parsed_tree) ]
				context_pos_tags = [w.tag_ for w in iterloop(parsed_tree) ]
				context_tags = [sentence.words_tags[dt.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in iterloop(parsed_tree)]
				for i,wrd in enumerate(context_words):
					if i < history:
						history_tags = tuple(['-TAGSTART-']+context_tags[0:i])
						history_words = ['-START-']+context_words[0:i]
						history_pos_tags = ['-POSTAGSTART-']+context_pos_tags[0:i]
					else:
						history_tags = context_tags[i-history:i]
						history_words = context_words[i-history:i]
						history_pos_tags = context_pos_tags[i-history:i]
					history_vectors = ('ph', [history_tags] )
					cur_idx = i
					prev_idx = cur_idx-1
					distance = 0
					if prev_idx >= 0:
						distance = parsed_tree[cur_idx].similarity(parsed_tree[prev_idx])
					target_feature_vectors.append( dp.construct_feature_vector(wrd, context_tags[i], 
							feature_dict, history_words, history, history_vectors, history_pos_tags, distance) )
					histories.append((prev_idx,history_words,history_pos_tags,distance))
			else:
				for i,wrd in enumerate(iterloop(parsed_tree)):
					cur = wrd
					history_words = []
					history_tags = []
					history_pos_tags = []
					for j in range(history):
						par = cur.head
						if cur == par:
							parw = '-START-'
							idx = -1
							tag = '-TAGSTART-'
							pos = '-POSTAGSTART-'
							history_tags.insert(0,tag)
							history_words.insert(0,parw)
							history_pos_tags.insert(0,pos)
							break
						else:
							parw = par.orth_
							idx = dt.sen_idx(sentence.raw_sentence,par)
							tag = sentence.words_tags[idx][1]
							pos = par.tag_
							cur = par
							history_tags.insert(0,tag)
							history_words.insert(0,parw)
							history_pos_tags.insert(0,pos)
					history_vectors = ('ph',[history_tags] )
					cur_idx = dt.sen_idx(sentence.raw_sentence,wrd)
					
					for prev_idx,w in enumerate(iterloop(parsed_tree)):
						if w == wrd.head:
							break
					if wrd.head == wrd:
						prev_idx = -1

					distance = 0
					if prev_idx >= 0:
						distance = parsed_tree[cur_idx].similarity(parsed_tree[prev_idx])
					
					cur_tag = sentence.words_tags[idx][1]
					target_feature_vectors.append( dp.construct_feature_vector(wrd.orth_, cur_tag, feature_dict,
									history_words, history, history_vectors, history_pos_tags, distance) )
					histories.append((prev_idx,history_words,history_pos_tags,distance))
			
			dict_target_feature_vectors = [v2d(target_feature_vector[0][0]) for target_feature_vector in target_feature_vectors]
			pre_pros.append((parsed_tree,dict_target_feature_vectors,histories))
		except Exception as ex:
			print "error"
			pipeline.log('train',sentence)
	
	print 'pre_pros',time()-t1
	t2 = time()
	print len(pre_pros)

	for i in range(iters):
		iter_time = time()
		print "at iter",i
		cum_weights = (i)*weight_matrix
		for parsed_tree,dict_target_feature_vectors,histories in pre_pros:
			target_feature_vectors = [d2v(dict_target_feature_vector) for dict_target_feature_vector in dict_target_feature_vectors]
			weight_matrix = train_perceptron_once(parsed_tree, target_feature_vectors, feature_dict, 
							history, weight_matrix, histories)
		weight_matrix = (cum_weights + weight_matrix)/(i+1)
		print "one iter: ", time() - iter_time
	print 'train',time()-t2
	return weight_matrix


def init_weights(no_rows):
	"""	Input:	number of rows of the feature vector --> Construct weight matrix with this many rows
		Output:	initialized weight matrix, with random values for the weights

		Method to initalize the weights of the perceptron. 
	"""
	np.random.seed(43)
	weight_matrix = np.zeros((1,no_rows))
	return weight_matrix


def train_perceptron_once(parsed_tree, target_feature_vectors, feature_dict, history, weight_matrix, histories):
	"""	Input:	Sentence that is fed into the perceptron
				Dictionary with feature vectors of the correct tagged sentence

	"""

	feature_vectors_sentence = viterbi(parsed_tree, feature_dict, history, weight_matrix, histories)
	new_weights = update_weights(weight_matrix, feature_vectors_sentence, target_feature_vectors)
	return new_weights

def get_tag_from_vector(feature_vector,feature_dict):
	idxes = [feature_dict.get('i tag+'+k,-1) for k in all_tags]
	tags = []
	for i,idx in enumerate(idxes):
		if not idx == -1:
			if feature_vector[idx] >= 1.0:
				tags.append(all_tags[i])
	return tags

def test_perceptron_once(E, parsed_tree, feature_dict, history, weight_matrix, histories, context_tags=None):
	if context_tags is None:
		context_tags = ['Ne']*len(parsed_tree)
	feature_vectors_sentence = viterbi(parsed_tree, feature_dict, history, weight_matrix, histories)

	for i,v in enumerate(feature_vectors_sentence):
		possible_tags = get_tag_from_vector(v,feature_dict)
		real_tag = context_tags[i]
		pipeline.out( (i,possible_tags,real_tag) )
		if not real_tag in possible_tags:
			E += 1
	return E
		

def viterbi(parsed_tree, feature_dict, history, weight_matrix, histories):
	""" Input:	The sentence to be tagged
				A list of all possible tags (strings)
				History: how far you want to look back

		Output:	A list with all feature vectors, in order per word
	"""

	sentence_dict = {} # per word all possible tags
	no_tags = len(all_tags)

	# --------------------------- Viterbi forward path --------------------------- #
	t1=time()
	for i,wrd in enumerate(iterloop(parsed_tree) ): # now you know the position of the word in your sentence
		
		feature_vector_array = np.zeros((no_tags, SIZE) ) # now we assume we have only two features per tag (n.b. so this is not only correct or false, it's features)
		tag_score_array = np.zeros((no_tags))
		history_list = []
		t2=time()
		history_vectors = sentence_dict.get(histories[i][0],(0,0,[('-TAGSTART-',)]))[1:3]
		calc_feat = [None]
		for j,tag in enumerate(all_tags): 
			
			t4=time()
			feature_vectors_tag = dp.construct_feature_vector2(wrd.orth_, tag, feature_dict, histories[i][1], history, history_vectors, histories[i][2], histories[i][3], calc_feat)
			
			t4=time()-t4
			t5=time()
			best_tag_score = -1e1000 # init scores --> delete once more clever list implementation with max
			best_feature_vector = np.zeros(SIZE) # number of features --> CHANGE
			history_word = ('Um') 
			for tple in feature_vectors_tag:
				tag_score = np.dot(tple[0], weight_matrix.transpose()) 
				if tag_score > best_tag_score:
					best_tag_score = tag_score
					best_feature_vector = tple[0]
					history_word = tple[1]
			t5=time()-t5

			tag_score_array[j] = best_tag_score
			feature_vector_array[j,:] = best_feature_vector
			history_list.append(history_word)
		t2=time()-t2
		sentence_dict[i] = (tag_score_array, feature_vector_array, history_list)
	t1=time()-t1

	# --------------------------- Viterbi backward path --------------------------- #
	t6=time()
	final_feature_vectors = []

	dict_len = len(sentence_dict)
	for entry in range(dict_len-1, -1, -1):
		(score, vector, history_list) = sentence_dict[entry]
		if entry == dict_len-1: 
			high_score =  score.argmax()
			best_vector = vector[high_score]
			history_best_vector = tag_idxes[history_list[high_score][-2]] 
		else:
			best_vector = vector[history_best_vector]
			history_best_vector = tag_idxes[history_list[high_score][-2]] 
		final_feature_vectors.append(best_vector) 
	t6=time()-t6
	return final_feature_vectors


def update_weights(old_weights, feature_vectors_sentence, target_feature_vectors):
	""" Input:	Old weight matrix
				Feature vectors as predicted by viterbi
				Correct feature vectors
		Output:	Updated weight matrix

		Method to update the weight matrix of the perceptron
	"""
	for i in range(len(feature_vectors_sentence)):
		diff = target_feature_vectors[i] - feature_vectors_sentence[i]
		updated_weights = np.add(old_weights, diff) 
		old_weights = updated_weights
		

	return updated_weights

if __name__ == '__main__':
	weights = np.random.random((1, 2))
	target_feature_vectors = [np.array((1, 0)), np.array((1,1))]
	train_perceptron_once(['hello','world'], target_feature_vectors, ['g','f'],0, weights)

