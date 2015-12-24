
from time import time
import dependencyTree as dto
import depTree as dts
import structured_perceptron as sp
import dataparser as dp
from multiprocessing import Pool
import datetime
import numpy as np



if __name__ == '__main__':
	TRAIN_FILE = '../release3.2/data/test.txt'
	all_sentences, feature_dict = dp.process(TRAIN_FILE,1)
	tbank = dts.tbankparser()
	text_file = open("preprocessed-4gram-sentences.txt", "w")
	print "start looping through sent"
	for sentence in all_sentences:
		try:
			parsed_tree = tbank.parse(sentence.raw_sentence)
			context_tags = [sentence.words_tags[dts.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in dts.dfirst(parsed_tree)]
			sentence_array = ['-START-','-START-','-START-']
			seen_mistakes = []
			for i,wrd in enumerate(dts.dfirst(parsed_tree)):
				i += 1
				sentence_array.append(wrd.orth_)
				if context_tags[i-1] != "Ne":
						sentence_len =  len(sentence_array)
						n_gram_words = sentence_array[i-3:i+1]
						for error in sentence.error_list:
							if error.error_type == context_tags[i-1] and error.error_start_index + 1 == error.error_end_index:
								if not error.error_type in seen_mistakes:
									if error.correction != "" and error.correction != None:
										text_file.write("4-gram: " + ' '.join(n_gram_words))
										text_file.write("\n")
										text_file.write("Correction: " + error.correction)
										text_file.write("\n")
										text_file.write("\n")
										seen_mistakes.append(error.error_type)
		except Exception as ex:
			print "fout"