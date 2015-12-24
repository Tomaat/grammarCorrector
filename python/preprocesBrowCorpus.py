from spacy.en import English, LOCAL_DATA_DIR
import os
import nltk
from nltk.corpus import brown
import depTree as dts

"""
	write sentences to a new file, not in lineair order but in parse tree order.
"""

def test(brown_sentenecs, tbank, iterloop):
	for sentence in brown_sentenecs[:1]:
		parsed_sentence = tbank.parse(sentence)
		for i in range(0,len(sentence)):
			cur = parsed_sentence[i]
			childern = False
			for child in enumerate(cur.children):
				childern = True  
			if not childern:
				sentence = []
				sentence.insert(0,cur.orth_)
				print recursive_tree_climb(cur, sentence)

def recursive_tree_climb(current_word,sentence):
	parent = current_word.head
	if current_word == parent:
		parw = '-START- -START- -START-'
		sentence.insert(0,parw)
		return sentence
	else:
		parw = parent.orth_
		current_word = parent
		sentence.insert(0,parw)
		return recursive_tree_climb(current_word, sentence)


if __name__ == '__main__':
	print "start"
	data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
	nlp = English(data_dir=data_dir)
	brown_sentenecs = brown.sents()
	tbank = dts.tbankparser()
	iterloop = dts.dfirst
	text_file = open("preprocessed-BrownCorpus.txt", "w")
	print "start looping"
	for sentence in brown_sentenecs:
		parsed_sentence = tbank.parse(sentence)
		for i in range(0,len(sentence)):
			cur = parsed_sentence[i]
			childern = False
			for child in enumerate(cur.children):
				childern = True  

			if not childern:
				sentence = []
				sentence.insert(0,cur.orth_)
				sentence_array = recursive_tree_climb(cur, sentence)	
				text_file.write(' '.join(sentence_array))
				text_file.write("\n")
				text_file.write("\n")
	print "end program"