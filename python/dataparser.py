from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *
from Sentence import Sentence 
import sys
from nltk import word_tokenize, pos_tag 
import re
import os 
import numpy as np
import structured_perceptron as sp
#from structured_perceptron import all_tags
from multiprocessing import Pool
import pipeline
from spacy.en import English, LOCAL_DATA_DIR


nlp = None 

def _init_(tbank):
	global nlp
	nlp = tbank.nlp

'''
	Normalization used in pre-processing.
	- All words are lower cased
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

def makeFeatures(word,tag,history_words,history_tags, history_pos_tags):
	feature_array = [] 
	
	def add(name, *args):
		feature_array.append('+'.join((name,) + tuple(args)))
	nword = normalize(word)
	distance = nlp(unicode(normalize(history_words[-1:][0]))).similarity(nlp(unicode(nword)))
	if distance < 0.1:
		add('word embedding 0-0.1 i + i-1')
	elif(distance < 0.2):
		add('word embedding 0.1-0.2 i + i-1')
	elif(distance < 0.3):
		add('word embedding 0.2-0.3 i + i-1')
	elif(distance < 0.4):
		add('word embedding 0.3-0.4 i + i-1')
	elif(distance < 0.5):
		add('word embedding 0.4-0.5 i + i-1')
	elif(distance < 0.6):
		add('word embedding 0.5-0.6 i + i-1')
	elif(distance < 0.7):
		add('word embedding 0.6-0.7 i + i-1')
	elif(distance < 0.8):
		add('word embedding 0.7-0.8 i + i-1')
	elif(distance < 0.9):
		add('word embedding 0.8-0.9 i + i-1')
	else:
		add('word embedding 0.9-1 i + i-1')

	add('i suffix-1', nword[-1:])
	add('i suffix-2', nword[-2:])
	add('i suffix-3', nword[-3:])
	add('i suffix-4', nword[-4:])

	add('i pref1', nword[0])
	add('i pref2', nword[0:3])
	add('i pref3', nword[0:2])

	add('i tag',tag)
	add('i word', nword)

	hmax = len(history_words)-1
	for i in range(hmax+1):
		add('i-'+str(i+1)+' word',history_words[hmax-i])
		add('i-'+str(i+1)+' tag',history_tags[hmax-i])
		add('i-'+str(i+1)+' pos tag', history_pos_tags[hmax-i])
		add('i tag + i-'+str(i+1)+' tag', tag, history_tags[hmax-i])
		add('i-'+str(i+1)+' tag+i word', history_tags[hmax-i], nword)
		add('i word i-1 word', nword, history_words[-1:][0])
		add('i tag i-1 tag', tag, history_tags[-1:][0])
	word_sturct = ""
	for char in word:
		if char.isupper():
			word_sturct += "X"
		else:
			word_sturct += "x"

	add('i structure', word_sturct)
	add('i-1 suffix', history_words[i-1][-3:])
	return feature_array

def makeFeatureDict(processed_sentences,history):
	feature_dictionary = {} # this will be a dict with key the feature name as key 
	feature_dictionary['i tag+-TAGSTART-'] = 1
	index = 1
	for tag in sp.all_tags:
		feature_dictionary['i tag+'+ tag] = index
		index += 1
	for p in range(history):
		for tag in sp.all_tags:
			feature_dictionary['i-'+str(p+1)+' tag+'+ tag] = index
			index += 1


	for sentence in processed_sentences:
		print sentence.raw_sentence
		if True:
			context_words = [word_tag[0] for word_tag in sentence.words_tags]
			context_tags  = [word_tag[1] for word_tag in sentence.words_tags]
			context_pos_tags = [ pos_tag_tuple[1] for pos_tag_tuple in sentence.pos_tags_sentence]
			
			print context_words
			print context_tags
			print context_pos_tags

			for i, tagTouple in enumerate(sentence.words_tags):
				history_words = ['-START-']+ context_words[:i]
				history_tags = ['-TAGSTART-']+ context_tags[:i]
				history_pos_tags = ['-POSTAGSTART-'] + context_pos_tags[:i]
				
				if len(history_words) > history:
					history_words = context_words[i-history:i]
					history_tags = context_tags[i-history:i]
					history_pos_tags = context_pos_tags[i-history:i]

				features =  makeFeatures(context_words[i], context_tags[i],history_words,history_tags, history_pos_tags)
				
				for feature in features:
					print feature
					if feature not in feature_dictionary:
						feature_dictionary[feature] = index	
						index += 1
		else:
			pipeline.log('feat',sentence)

	return feature_dictionary

def construct_feature_vector(word, tag, feature_dictionary, context_words, i, history, history_vectors, context_pos_tags):
	history_words = ['-START-'] + context_words[:i]
	history_pos_tags = ['-POSTAGSTART-'] + context_pos_tags[:i]
	if len(history_words) > history:
		history_words = context_words[i-history:i]
		history_pos_tags = context_pos_tags[i-history:i]
	#/#context_tags = ['-START2-', '-START-'] + context_tags + ['END-', 'END2']
	ans = []
	if history_vectors[1] == []:
		history_vectors = (history_vectors[0], [('-TAGSTART-',)] )
	
	
	for history_tags in history_vectors[1]:
		feature_vector = np.zeros(len(feature_dictionary))
		
		# if history_tags == -1:
		#	print 'din',history_words, history_tags, history_vectors
		# if history_tags == ('NaN',):
		# 	print 'nan'
		feature_array = makeFeatures(word,tag,history_words,history_tags, history_pos_tags)
		
		for feature in feature_array:
			if feature in feature_dictionary:
				feature_vector[feature_dictionary[feature]] =  1
		#print 'histtag', history_words, history_tags
		#print 'fear',feature_array
		history_tags = history_tags+(tag,)
		if len(history_tags) > history:
			history_tags = history_tags[1:]

		ans += [ (feature_vector, history_tags) ]
	return ans

def process(filename,history):
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		print data_raw
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"
	for sentence_tuple in sentence_tuples: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
		print sentence_tuple
		if len( sentence_tuple[0]) < 1:
			continue
		if True:#try:
			processed_sentences.append(Sentence(sentence_tuple))
		else:#except Exception as ex:
			pipeline.log('init',sentence_tuple)
	print "make feature vectors"
	feature_dictionary = makeFeatureDict(processed_sentences,history)

	return processed_sentences,feature_dictionary

def multi_once(sentence_tuple):
	ans = None
	try:
		ans = Sentence(sentence_tuple)
	except:
		pipeline.log('init_mul',sentence_tuple)
	return ans

def process_multi(filename,history,workers=8):
	reload(sys) 
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"
	pool = Pool(workers)
	processed_sentences = pool.map(multi_once,sentence_tuples)

	print "make feature vectors"
	feature_dictionary = makeFeatureDict(processed_sentences,history)

	return processed_sentences,feature_dictionary

if __name__ == '__main__':
	process('../release3.2/data/conll14st-preprocessed.m2.small')
