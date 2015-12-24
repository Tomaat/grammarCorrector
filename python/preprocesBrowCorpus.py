from spacy.en import English, LOCAL_DATA_DIR
import os
import nltk
from nltk.corpus import brown
import depTree as dts

"""
	write sentences to a new file, not in lineair order but in parse tree order.
"""
"""
if parsed_sentence[i].child
			cur = wrd
			print wrd
			history_words = []
			for j in range(1,len(parsed_sentence)+1):
				par = cur.head
				if cur == par:
					parw = '-START-'
					cur = par
					history_words.insert(0,parw) # TEST DIT EVEN  
					break  
				else:
					parw = par.orth_
					cur = par
					history_words.insert(0,parw)
				#print history_words
		#print sentence
		print parsed_sentence 
		print len(parsed_sentence)
		print "----------------------------"
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