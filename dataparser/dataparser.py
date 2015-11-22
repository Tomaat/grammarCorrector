from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *
from Mistake import Mistake 
from Sentence import Sentence 
import sys
from nltk import word_tokenize, pos_tag 



if __name__ == '__main__':
	print "start of program"
	reload(sys)  
	sys.setdefaultencoding('utf8') # hack for some encoding problems in the sentences 
	with open ('../release3.2/data/conll14st-preprocessed.m2') as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		processed_sentences = []
		
		print "parsing sentences"
		for sentence_tuple in sentence_tuples[1:1000]: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
			processed_sentences.append(Sentence(sentence_tuple))
			
			
	print "end of program" 

def construct_feature_vector(word, tag, history_vectors):

	return None   


"""
[(history_vectors, feature_vector), (history_vectors, feature_vector), ...]

"""

