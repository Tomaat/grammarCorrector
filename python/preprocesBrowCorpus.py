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
		for i,wrd in enumerate(iterloop(parsed_sentence)):
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
			sentence_array = ['-START-'] 
			for i,wrd in enumerate(iterloop(parsed_sentence)):
				sentence_array.append(wrd.orth_)		
			text_file.write(' '.join(sentence_array))
			text_file.write("\n")
			text_file.write("\n")
	print "end program"