
# make senences in  de goede volgorde van de history \

from spacy.en import English, LOCAL_DATA_DIR
import os
import nltk
from nltk.corpus import brown
import depTree as dts


if __name__ == '__main__':
	data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
	nlp = English(data_dir=data_dir)
	brown_sentenecs = brown.sents()
	tbank = dts.tbankparser()
	iterloop = dts.dfirst
	for sentence in brown_sentenecs[:10]:
			parsed_sentence = tbank.parse(sentence)
			histories = []
			for i,wrd in enumerate(iterloop(parsed_sentence)):
				cur = wrd
				history_words = ['']*len(parsed_sentence) 
				for j in range(1,len(parsed_sentence)+1):
					par = cur.head
					if cur == par:
						parw = '-START-'
					else:
						parw = par.orth_
					history_words[-j] = parw
			print history_words
		#except Exception as ex:
			#print "fout"


def test(tbank,sentence,history):
	parsed_tree = tbank.parse(sentence.raw_sentence)
	histories = []
	target_feature_vectors = []
	for i,wrd in enumerate(iterloop(parsed_tree)):
		cur = wrd
		history_words = ['']*history 
		for j in range(1,history+1):
			par = cur.head
			if cur == par:
				parw = '-START-'
			else:
				parw = par.orth_
			history_words[-j] = parw
		print history_words
			