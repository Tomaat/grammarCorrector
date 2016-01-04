
from time import time
import dependencyTree as dto
import depTree as dts
import structured_perceptron as sp
import dataparser as dp
from multiprocessing import Pool
import datetime
import numpy as np


def test(all_sentences, tbank):
	text_file = open("preprocessed-4gram-sentences.txt", "w")
	for sentence in all_sentences:
		seen_mistakes = []
		parsed_sentence = tbank.parse(sentence.raw_sentence)
		context_tags = [word_tag[1] for word_tag in sentence.words_tags]
		for i in range(0,len(sentence.raw_sentence.split(' '))):
			if context_tags[i] != "Ne":
				cur = parsed_sentence[i]
				sentence_array = []
				sentence_array.insert(0,cur.orth_)
				result = recursive_tree_climb(cur, sentence_array)
				four_gram = result[len(result)-4:]
				for error in sentence.error_list:
					if error.error_type == context_tags[i] and error.error_start_index + 1 == error.error_end_index:
						if not error.error_type in seen_mistakes:
							if error.correction != "" and error.correction != None:
								text_file.write("4-gram: " + ' '.join(four_gram))
								text_file.write("\n")
								text_file.write("Correction: " + error.correction)
								text_file.write("\n")
								text_file.write("\n")
								seen_mistakes.append(error.error_type)


def recursive_tree_climb(current_word,sentence):
	parent = current_word.head
	if current_word == parent:
		parw = '-START-'
		sentence.insert(0,parw)
		sentence.insert(0,parw)
		sentence.insert(0,parw)
		return sentence
	else:
		parw = parent.orth_
		current_word = parent
		sentence.insert(0,parw)
		return recursive_tree_climb(current_word, sentence)

if __name__ == '__main__':
	
	print 'start'
	TRAIN_FILE = 'test_data/test_linear.txt' #'../release3.2/data/test.txt'
	all_sentences, feature_dict = dp.process(TRAIN_FILE,1)
	tbank = dts.tbankparser()
	text_file = open("preprocessed-4gram-sentences2.txt", "w")
	print "start looping through sentece"
	
	for sentence in all_sentences:
		try:
			seen_mistakes = []
			parsed_sentence = tbank.parse(sentence.raw_sentence)
			context_tags = [word_tag[1] for word_tag in sentence.words_tags]
			for i in range(0,len(sentence.raw_sentence.split(' '))):
				if context_tags[i] != "Ne":
					cur = parsed_sentence[i]
					sentence_array = []
					sentence_array.insert(0,cur.orth_)
					result = recursive_tree_climb(cur, sentence_array)
					four_gram = result[len(result)-4:]
					for error in sentence.error_list:
						if error.error_type == context_tags[i] and error.error_start_index + 1 == error.error_end_index:
							if not error.error_type in seen_mistakes:
								if error.correction != "" and error.correction != None:
									text_file.write("4-gram: " + ' '.join(four_gram))
									text_file.write("\n")
									text_file.write("Correction: " + error.correction)
									text_file.write("\n")
									text_file.write("\n")
									seen_mistakes.append(error.error_type)
									seen_mistakes.append(error.error_type)
		except Exception as ex:
			print "fout"