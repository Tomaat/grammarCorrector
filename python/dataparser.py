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
from time import time


nlp = None 
golinear = True

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

def makeFeatures(word,history_words,history_tags, history_pos_tags, distance, tag=''):
	feature_array = [] 
	
	def add(name, *args):
		feature_array.append('+'.join((name,) + tuple(args)))
	nword = normalize(word)
	#distance = nlp(unicode(normalize(history_words[-1:][0]))).similarity(nlp(unicode(nword)))
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

	add('i suffix-1', nword[-1:],tag)
	add('i suffix-2', nword[-2:],tag)
	add('i suffix-3', nword[-3:],tag)
	add('i suffix-4', nword[-4:],tag)

	add('i pref1', nword[0],tag)
	add('i pref2', nword[0:3],tag)
	add('i pref3', nword[0:2],tag)

	add('i tag',tag)
	add('i word', nword,tag)

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
	word_sturct = ""
	for char in word:
		if char.isupper():
			word_sturct += "X"
		else:
			word_sturct += "x"

	add('i structure', word_sturct,tag)
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
		#print sentence.raw_sentence
		try:
			if golinear:
				# """
				# ==== comment 2
				# hier loopt de code nog op de oude manier door de zin, dit moet dus via de nieuwe manier (zie comment 0 in structured_perceptron)
				# """
				context_words = [word_tag[0] for word_tag in sentence.words_tags]
				context_tags  = [word_tag[1] for word_tag in sentence.words_tags]
				context_pos_tags = [ pos_tag_tuple[1] for pos_tag_tuple in sentence.pos_tags_sentence]
				
				#print context_words
				#print context_tags
				#print context_pos_tags

				for i, tagTouple in enumerate(sentence.words_tags):
					history_words = ['-START-']+ context_words[:i]
					history_tags = ['-TAGSTART-']+ context_tags[:i]
					history_pos_tags = ['-POSTAGSTART-'] + context_pos_tags[:i]
					
					if len(history_words) > history:
						history_words = context_words[i-history:i]
						history_tags = context_tags[i-history:i]
						history_pos_tags = context_pos_tags[i-history:i]

					distance = nlp(unicode(normalize(history_words[-1:][0]))).similarity(nlp(unicode(normalize(context_words[i]))))
					features =  makeFeatures(context_words[i],history_words,history_tags, history_pos_tags, distance, context_tags[i])
			
			else:
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
				# """
				# /==== end comment 2
				# """
					for feature in features:
						#print feature
						if feature not in feature_dictionary:
							feature_dictionary[feature] = index	
							index += 1
		except:
			pipeline.log('feat',sentence)

	return feature_dictionary

#def construct_feature_vector(word, tag, feature_dictionary, context_words, i, history, history_vectors, context_pos_tags):
def construct_feature_vector(word, tag, feature_dictionary, history_words, history, history_vectors, history_pos_tags, distance, calc_features=None):
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
	if calc_features is None:
		calc_features = [['']]*len(history_vectors[1])
		do_calc = True
	else:
		do_calc = False
	
	for i,history_tags in enumerate(history_vectors[1]):
		feature_vector = np.zeros(len(feature_dictionary))
		
		# if history_tags == -1:
		#	print 'din',history_words, history_tags, history_vectors
		# if history_tags == ('NaN',):
		# 	print 'nan'
		#t#t1 = time()
		if do_calc:
			calc_features[i] = makeFeatures(word,history_words,history_tags, history_pos_tags,distance)
		
		#t#t1 = time()-t1

		#t#t2 = time()
		for feature in calc_features[i]:
			feature = feature+tag
			if feature in feature_dictionary:
				feature_vector[feature_dictionary[feature]] =  1
		#t#t2 = time()-t2
		#print 'histtag', history_words, history_tags
		#print 'fear',feature_array
		
		#t#t3 = time()
		new_tags = ['']*min(history,len(history_tags)+1)
		new_tags[-1] = tag
		for i in range(1,len(new_tags)):
			new_tags[-(i+1)] = history_tags[-i]
		#t#t3 = time()-t3
		#new_tags = history_tags+(tag,)
		#if len(new_tags) > history:
		#	new_tags = new_tags[1:]
		#t#t4=time()
		ans += [ (feature_vector, tuple(new_tags)) ]
		#t#t4=time()-t4
	return ans, calc_features

def process(filename,history):
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		#print data_raw
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"
	for sentence_tuple in sentence_tuples: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
		#print sentence_tuple
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

def process_multi(filename,history,workers=7):
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	print "parsing sentences"

	for sentence_tuple in sentence_tuples: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
		#print sentence_tuple
		#print sentence_tuple
		if len( sentence_tuple[0]) < 1:
			print "true"
			continue
		#try:
		print Sentence(sentence_tuple)
		print "test"
		processed_sentences.append(Sentence(sentence_tuple))
		#except Exception as ex:
		#	print ex
			

	#pool = Pool(workers)
	#processed_sentences = pool.map(multi_once,sentence_tuples)

	print "make feature vectors"
	feature_dictionary = makeFeatureDict(processed_sentences,history)

	return processed_sentences,feature_dictionary

if __name__ == '__main__':
	process('../release3.2/data/conll14st-preprocessed.m2.small')
