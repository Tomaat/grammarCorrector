from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *
from Sentence import Sentence 
import sys
from nltk import word_tokenize, pos_tag 
import re
import numpy as np
import structured_perceptron as sp
#from structured_perceptron import all_tags
from multiprocessing import Pool
import pipeline
#from structured_perceptron import all_tags

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

	add('i suffix', nword[-3:])
	add('i pref1', nword[0])
	add('i tag',tag)
	add('i word', nword)

	hmax = len(history_words)-1
	for i in range(hmax+1):
		add('i-'+str(i+1)+' word',history_words[hmax-i])
		add('i-'+str(i+1)+' tag',history_tags[hmax-i])
		add('i-'+str(i+1)+' pos tag', history_pos_tags[hmax-i])
		add('i tag + i-'+str(i+1)+' tag', tag, history_tags[hmax-i])
		add('i-'+str(i+1)+' tag+i word', history_tags[hmax-i], nword)
	
	word_sturct = ""
	for char in word:
		if char.isupper():
			word_sturct += "X"
		else:
			word_sturct += "x"

	add('i structure', word_sturct)
	#add('i-1 suffix', history_words[i-1][-3:])
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
		try:
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

				features =  makeFeatures(context_words[i], context_tags[i],history_words,history_tags, history_pos_tags)
				
				for feature in features:
					if feature not in feature_dictionary:
						feature_dictionary[feature] = index	
						index += 1
		except:
			pipeline.log('feat',sentence)

	return feature_dictionary

#def construct_feature_vector(word, tag, feature_dictionary, context_words, i, history, history_vectors, context_pos_tags):
def construct_feature_vector(word, tag, feature_dictionary, history_words, i, history, history_vectors, history_pos_tags):
	# #if i < history:
	# history_words = ['-START-'] + context_words[0:i]
	# history_pos_tags = ['-POSTAGSTART-'] + context_pos_tags[0:i]
	# if len(history_words) > history:
	# 	history_words = context_words[i-history:i]
	# 	history_pos_tags = context_pos_tags[i-history:i]
	# #/#context_tags = ['-START2-', '-START-'] + context_tags + ['END-', 'END2']
	
	# if history_vectors[1] == []:
	# 	history_vectors = (history_vectors[0], [('-TAGSTART-',)] )
	
	ans = []
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
		
		new_tags = ['']*min(history,len(history_tags)+1)
		new_tags[-1] = tag
		for i in range(1,len(new_tags)):
			new_tags[-(i+1)] = history_tags[-i]
		#new_tags = history_tags+(tag,)
		#if len(new_tags) > history:
		#	new_tags = new_tags[1:]
		ans += [ (feature_vector, tuple(new_tags)) ]
	return ans

	
		# add('i suffix', normalize(word)[-3:])
		# add('i pref1', normalize(word)[0])
		# add('i tag', tag)
		# add('i-1 tag', tag1)
		# #/#add('i-1 tag', context_tags[i-1])
		# #/#add('i-2 tag', context_tags[i-2])
		# #/#add('i tag+i-2 tag', context_tags[i], context_tags[i-2])
		# add('i word', context_words[i])
		# #/#add('i-1 tag+i word', context_tags[i-1], context_words[i])
		# add('i-1 word', context_words[i-1])
		# add('i-1 suffix', context_words[i-1][-3:])
		# add('i-2 word', context_words[i-2])
		# add('i+1 word', context_words[i+1])
		# add('i+1 suffix', context_words[i+1][-3:])
		# add('i+2 word', context_words[i+2])

def process(filename,history):
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"
	for sentence_tuple in sentence_tuples: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
		if len( sentence_tuple[0]) < 1:
			continue
		try:
			processed_sentences.append(Sentence(sentence_tuple))
		except Exception as ex:
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

def process_multi(filename,history,workers=7):
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
