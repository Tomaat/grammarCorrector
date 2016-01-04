from nltk.tokenize import *
from Sentence import Sentence 
import sys
from nltk import word_tokenize, pos_tag 
import re
import os 
import numpy as np
import structured_perceptron as sp
from multiprocessing import Pool
import pipeline
from spacy.en import English, LOCAL_DATA_DIR
from time import time

# some default variables for testing 
nlp = None 
golinear = True

def _init_(tbank):
	global nlp
	nlp = tbank.nlp

'''
	Normalization used in pre-processing.
	- All words are lower case
	- Digits in the range 1800-2100 are represented as !YEAR;
	- Other digits are represented as !DIGITS
	:rtype: str
'''
def normalize(word):
    if '-' in word and word[0] != '-':
        return "!HYPHEN"
    elif re.match(r"[^@]+@[^@]+\.[^@]+", word):
    	return '!EMAIL'
    elif word.isdigit() and len(word) == 4:
        return '!YEAR'
    elif word.isdigit() and len(word) >= 6 and len(word) <= 12:
        return '!PHONENUMBER'
    elif len(word) > 0 and word[0].isdigit():
        return '!DIGITS'
    else:
        return word.lower()

def makeFeatures(word,history_words,history_tags, history_pos_tags, distance, tag=''):
	'''
	Make a list with strings (the names of all the feature for a word) in a sentence
	:rtype: list 
	'''
	feature_array = [] 
	
	def add(name, *args):
		feature_array.append('+'.join((name,) + tuple(args)))
	nword = normalize(word)
	
	# distane with previous word 
	if distance < 0.1:
		add('word embedding 0-0.1 i + i-1',tag)
	elif(distance < 0.2):
		add('word embedding 0.1-0.2 i + i-1',tag)
	elif(distance < 0.3):
		add('word embedding 0.2-0.3 i + i-1',tag)
	elif(distance < 0.4):
		add('word embedding 0.3-0.4 i + i-1',tag)
	elif(distance < 0.5):
		add('word embedding 0.4-0.5 i + i-1',tag)
	elif(distance < 0.6):
		add('word embedding 0.5-0.6 i + i-1',tag)
	elif(distance < 0.7):
		add('word embedding 0.6-0.7 i + i-1',tag)
	elif(distance < 0.8):
		add('word embedding 0.7-0.8 i + i-1',tag)
	elif(distance < 0.9):
		add('word embedding 0.8-0.9 i + i-1',tag)
	else:
		add('word embedding 0.9-1 i + i-1',tag)

	# all the suffixes of the current word
	add('i suffix-1', nword[-1:],tag)
	add('i suffix-2', nword[-2:],tag)
	add('i suffix-3', nword[-3:],tag)
	add('i suffix-4', nword[-4:],tag)

	# prefixes of the current word
	add('i pref1', nword[0],tag)
	add('i pref2', nword[0:2],tag)
	add('i pref3', nword[0:3],tag)

	# tag of the word features
	add('i tag',tag)
	add('i word', nword,tag)

	# features based on the hstroy length 
	hmax = len(history_words)-1
	for i in range(hmax+1):
		add('i-'+str(i+1)+' word',history_words[hmax-i],tag)
		add('i-'+str(i+1)+' tag',history_tags[hmax-i],tag)
		add('i-'+str(i+1)+' pos tag', history_pos_tags[hmax-i],tag)
		add('i tag + i-'+str(i+1)+' tag', tag, history_tags[hmax-i],tag)
		add('i-'+str(i+1)+' tag+i word', history_tags[hmax-i], nword,tag)
		add('i word i-1 word', nword, history_words[hmax-i],tag)
		add('i tag i-1 tag', tag, history_tags[hmax-i],tag)
		add('i-'+str(i+1)+' suffix', history_words[hmax-i][-3:],tag)
	
	# feature word structure of the word
	word_sturct = ""
	for char in word:
		if char.isupper():
			word_sturct += "X"
		else:
			word_sturct += "x"

	add('i structure', word_sturct,tag)
	return feature_array

def makeFeatureDict(processed_sentences,history):
	"""
		 make a dictionary with all the features found in the dataset
		 rtype: dictionary
	"""

	feature_dictionary = {} # this will be a dict with key the feature name as key 
	feature_dictionary['i tag+-TAGSTART-'] = 0
	index = 1
	# make a feature for every possible tag for a word
	for tag in sp.all_tags:
		feature_dictionary['i tag+'+ tag] = index
		index += 1
	# make feature of every posible history tag and his index 
	for p in range(history):
		for tag in sp.all_tags:
			feature_dictionary['i-'+str(p+1)+' tag+'+ tag] = index
			index += 1

	# make features for every word in the sentence. If lineair parsed, the make different features based on this tyoe of parsing. 
	for sentence in processed_sentences:
		try:
			if golinear:
				# make lineair features 
				context_words = [word_tag[0] for word_tag in sentence.words_tags]
				context_tags  = [word_tag[1] for word_tag in sentence.words_tags]
				context_pos_tags = [ pos_tag_tuple[1] for pos_tag_tuple in sentence.pos_tags_sentence]
				

				for i, tagTouple in enumerate(sentence.words_tags):
					history_words = ['-START-']+ context_words[:i]
					history_tags = ['-TAGSTART-']+ context_tags[:i]
					history_pos_tags = ['-POSTAGSTART-'] + context_pos_tags[:i]
					
					if len(history_words) > history:
						history_words = context_words[i-history:i]
						history_tags = context_tags[i-history:i]
						history_pos_tags = context_pos_tags[i-history:i]

					distance = nlp(unicode(normalize(history_words[-1:][0]))).similarity(nlp(unicode(normalize(context_words[i]))))
					features =  makeFeatures(context_words[i],history_words,history_tags, history_pos_tags, distance)
					for feature in features:
						#print feature
						for tag in sp.all_tags:
							feature = feature+'+'+tag
							if feature not in feature_dictionary:
								feature_dictionary[feature] = index	
								index += 1
			else:
				# depenceny wise parsing features
				parsed_tree = nlp(unicode(sentence.raw_sentence))
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
					features =  makeFeatures(wrd.orth_,history_words,history_tags, history_pos_tags, distance, cur_tag)
					for feature in features:
						if feature not in feature_dictionary:
							feature_dictionary[feature] = index	
							index += 1
		except:
			pipeline.log('feat',sentence)

	return feature_dictionary

def construct_feature_vector(word, tag, feature_dictionary, history_words, history, history_vectors, history_pos_tags, distance):#
	"""
		 make a list  with the tag of a word and the feature vector of this word
		 rtype: list with tuples
	"""
	ans = [] #answer
	
	for i,history_tags in enumerate(history_vectors[1]):
		# make a feature vector for all possible history tags
		feature_vector = np.zeros(len(feature_dictionary))
		features = makeFeatures(word,history_words,history_tags,history_pos_tags,distance,tag)
		
		for feature in features:
			if feature in feature_dictionary:
				feature_vector[feature_dictionary[feature]] =  1
		
		new_tags = ['']*min(history,len(history_tags)+1)
		new_tags[-1] = tag
		for i in range(1,len(new_tags)):
			new_tags[-(i+1)] = history_tags[-i]
		
		ans += [ (feature_vector, tuple(new_tags)) ]
	return ans

def construct_feature_vector2(word, tag, feature_dictionary, history_words, history, history_vectors, history_pos_tags, distance, calc_features=[None]):	
	"""
		 different version of the function above
		 rtype: list with tuples
	"""

	ans = []
	if calc_features[0] is None:
		calc_features[0] = [['']]*len(history_vectors[1])
		do_calc = True
	else:
		do_calc = False
	
	for i,history_tags in enumerate(history_vectors[1]):
		feature_vector = np.zeros(len(feature_dictionary))

		if do_calc:
		 	calc_features[0][i] = makeFeatures(word,history_words,history_tags, history_pos_tags,distance)
		
		for feature in calc_features[0][i]:
			feature_n = feature+tag
			if feature_n in feature_dictionary:
				feature_vector[feature_dictionary[feature_n]] =  1
		
		new_tags = ['']*min(history,len(history_tags)+1)
		new_tags[-1] = tag
		for i in range(1,len(new_tags)):
			new_tags[-(i+1)] = history_tags[-i]
		ans += [ (feature_vector, tuple(new_tags)) ]
	return ans

def process(filename,history):
	"""
		 make  objects  for every sentence in the dataset and a feature dict
		 rtype: list with sentence objects and feature dictionary
	"""
	reload(sys)  
	sys.setdefaultencoding('utf8') 
	processed_sentences = []
	with open (filename) as datafile: 
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).replace('\r','').split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"
	for sentence_tuple in sentence_tuples: 
		if len( sentence_tuple[0]) < 1:
			continue
		try:
			processed_sentences.append(Sentence(sentence_tuple))
		except Exception as ex:
			pipeline.log('init',sentence_tuple)
	print "make feature vectors"
	feature_dictionary = makeFeatureDict(processed_sentences,history)

	return processed_sentences,feature_dictionary




if __name__ == '__main__':
	process('../release3.2/data/conll14st-preprocessed.m2.small')
