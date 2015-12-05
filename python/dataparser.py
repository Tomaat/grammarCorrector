from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *
from Sentence import Sentence 
import sys
from nltk import word_tokenize, pos_tag 
import re
import numpy as np
import structured_perceptron as sp
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

def makeFeatures(word,tag,history_words,history_tags):
	feature_array = [] 
	
	def add(name, *args):
		feature_array.append('+'.join((name,) + tuple(args)))

	nword = normalize(word)

	add('i suffix', nword[-3:])
	add('i pref1', nword[0])
	add('i tag',tag)
	add('i word', nword)
	for i,w in enumerate(history_words):
		add('i-'+str(i+1)+' word',w)
		add('i-'+str(i+1)+' tag',history_tags[i])

	
	# add('i-1 tag', context_tags[i-1])
	# add('i-2 tag', context_tags[i-2])
	# add('i tag+i-2 tag', context_tags[i], context_tags[i-2])
	# add('i-1 tag+i word', context_tags[i-1], nword)
	# add('i-1 word', context_words[i-1])
	# add('i-1 suffix', context_words[i-1][-3:])
	# add('i-2 word', context_words[i-2])
	# add('i+1 word', context_words[i+1])
	# add('i+1 suffix', context_words[i+1][-3:])
	# add('i+2 word', context_words[i+2])

	# add feature die zegt hoe ver een woord van een ander woord is, 
	# check if word is in dict , wel/geen woord 
	#

	#word (string)
	# een tag 
	return feature_array

def makeFeatureDict(processed_sentences,history=1):
	feature_dictionary = {} # thiss willl be a dict with key the peature name, value the index in the 
	index = 0
	for tag in sp.all_tags:
		feature_dictionary['i tag+'+tag] = index
		index += 1

	for sentence in processed_sentences:
		context_words = [word_tag[0] for word_tag in sentence.words_tags]
		context_tags  = [word_tag[1] for word_tag in sentence.words_tags]
		for i, tagTouple in enumerate(sentence.words_tags):
			history_words = ['-START-']+context_words[:i]
			history_tags = ['-TAGSTART-']+context_tags[:i]
			if len(history_words) > history:
				history_words = context_words[i-history:i]
				history_tags = context_tags[i-history:i]

			features =  makeFeatures(context_words[i], context_tags[i],history_words,history_tags)
			
			for feature in features:
				if feature not in feature_dictionary:
					feature_dictionary[feature] = index	
					index += 1

	return feature_dictionary

def construct_feature_vector(word, tag, feature_dictionary, context_words, i, history, history_vectors):
	"""
	 - word: moet het woord zijn - als string - van het huidige woord
	 - tag: string met de huidige tag van het woord.
	 - feature_dictory is een dict die je aanmaakt met alle mogelijke features er in. 
	 
	 - word_context is een array met strings van de woorden, dit kunnen de woorden zijn in volgorde van de 
	 	tree, maar ook in de originele volgorde van de zin. ! VERGEET NIET DAT START = ['-START2-', '-START-']
	END = ['-END-', '-END2-'] toegvoegd moeten worden. en dat de woorden genormaliseerd moeten worden die in deze array staat

	context_words = START + [normalize(word_tag[0]) for word_tag in sentence_words] + END 
	 - dit moet 

	 -als het niet lukt om de vorige tags mee te geven dan kan je ze voor nu uitzetten.
	 
	 - we moeten op een manier wel zien te ontdekken wat de tags zijn van de woorden 
	 
	 voor het woord dat we nu gaan taggen, je volgt toch verschillende paden voor viterbi,
	 kunnen we daar niet iets mee?
	 """
	history_words = ['-START-'] + context_words[:i]
	if len(history_words) > history:
		history_words = context_words[i-history:i]
	#/#context_tags = ['-START2-', '-START-'] + context_tags + ['END-', 'END2']
	#print 'hisin',history_vectors
	ans = []
	if history_vectors[1] == []:
		history_vectors = (history_vectors[0], [('-TAGSTART-',)] )
	
	#print 'hisout',history_vectors
	for history_tags in history_vectors[1]:
		feature_vector = np.zeros(len(feature_dictionary))
		
		# if history_tags == -1:
		# 	print history_words, history_tags, history_vectors
		# if history_tags == ('NaN',):
		# 	print 'nan'
		feature_array = makeFeatures(word,tag,history_words,history_tags)
		
		# def add(name, *args):
		# 	feature_array.append('+'.join((name,) + tuple(args)))
		
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
		
		#TODO:
		# pos tags toevoegen
		# vorm van het woord (Horse: Xxxxx)
		# geschiedenis: de (goed/fout) tag van het vorige woord

		for feature in feature_array:
			if feature in feature_dictionary:
				feature_vector[feature_dictionary[feature]] =  1

		history_tags = history_tags+(tag,)
		if len(history_tags) > history:
			history_tags = history_tags[1:]

		ans += [ (feature_vector, history_tags) ]
	return ans


def process(filename):
	#print "start of program"
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	processed_sentences = []
	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
		#print "parsing sentences"
		for sentence_tuple in sentence_tuples: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
			if len( sentence_tuple[0]) < 1:
				continue
			processed_sentences.append(Sentence(sentence_tuple))
	#print "make feature vectors"
	feature_dictionary = makeFeatureDict(processed_sentences)
	#print feature_dictionary
	#print len(feature_dictionary)
	#print "end of program"
	return processed_sentences,feature_dictionary

if __name__ == '__main__':
	process('../release3.2/data/conll14st-preprocessed.m2.small')

 


"""
[(history_vectors, feature_vector), (history_vectors, feature_vector), ...]
"""

