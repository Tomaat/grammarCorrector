import nltk
from nltk.corpus import gutenberg, brown
from nltk import ConditionalFreqDist
from nltk import ngrams
from math import log10
import spacy
import depTree as dt

from random import choice

class NGrams:

	def __init__(self):
		self.ngrams_gutenberg = []
		self.ngrams_brown = []
		self.ngram_freq_dict = {}
		self.ngram_prob_dict = {}
		self.total_ngram_count = 0

	def find_ngrams(self, n):
		""" Input: the 'n' of 'n-grams'

			Find all the n-grams in the brown corpus. Store in frequency dictionary.
			Optionally it can be decided to use more corpora in order to have more data.

			Note: these are of course n-grams based on going through the sentence from left to right
			If we want to give the correction back based on the dependency tree, we need to
			parse the brown corpus (or any other data set) with the dependency parser, so that
			we can use this data. 			

		"""

		#self.ngrams_gutenberg = ngrams(gutenberg.words(), n)
		self.ngrams_brown = ngrams(brown.words(), n)		

		for i in self.ngrams_brown:
			self.total_ngram_count += 1

			if self.ngram_freq_dict.has_key(i):
				self.ngram_freq_dict[i] += 1
				#print i
			else:
				self.ngram_freq_dict[i] = 1
				#print i

		#print self.ngram_freq_dict

		"""for i in self.ngrams_gutenberg:
			self.total_ngram_count += 1

			if self.ngram_freq_dict.has_key(i):
				self.ngram_freq_dict[i] += 1
				#print i
			else:
				self.ngram_freq_dict[i] = 1
				#print i 

		print self.total_ngram_count """

	def ngram_log_likelihood(self):
		""" Computes the log likelood for all n-grams. The log likelihood has been used
			to avoid numeric overflow.

		"""

		for key,value in self.ngram_freq_dict.iteritems():
			prob = float(value) / self.total_ngram_count
			log_prob = log10(float(prob))
			self.ngram_prob_dict[key] = log_prob

		return self.ngram_prob_dict



class Correction(NGrams):

	def __init__(self):
		self.pot_corrections = []
		self.ngram_corrections = []
		self.final_corrections = []
		self.final_sorted_list = []

	def find_correction(self, ngram_dict, error_sequence, error_tag, nlp):
		""" Input:	the n-gram that ends in the error
					the error tag predicted by the structured perceptron

			Output:	corrected word --> in the code around this, this needs to be added to the new sentence

			Function to find a correction for the mistake
			1) What words do we expect based on our n-gram dictionary?
			2) What do we expect based on (1) and the word distance (spacy)? -->
			   This only works if the error was actually a well written word, otherwise we need
			   to use something else
			   Plus: spacy uses word similarity based on semantics. We rather want it on word form

			3) Potentially: take the error_tag into account --> 
		"""

		len_sequence = len(error_sequence)-1

		# 1) Find suitable n-grams
		for key,value in ngram_dict.iteritems():
			if key[0:len_sequence] == error_sequence[0:len_sequence]:

				# You want to give a higher score to words that have many letters in common --> might want to take word length into account
				score = 0
				no_letters = 0
				error = error_sequence[len_sequence]
				for char in key[len_sequence]:
					no_letters += 1
					if char in error:
						score += 1

				score = float(score)/no_letters

				final_score = -1*value*(2**score) # this is something you can refine
				print 'Value: ', value
				print 'Score: ', score
				print 'Final score: ', final_score

				self.pot_corrections.append((key[len_sequence], final_score))

		# For testing
		self.pot_corrections.append((u'jus', 2))
		self.pot_corrections.append((u'test', 8))

		sortedlist = sorted(self.pot_corrections, key=lambda x:x[1])

		# 1b) Find the best n-grams. 
		no_best_ngrams = 2
		self.ngram_corrections = sortedlist[no_best_ngrams-1:] # note this is from small to big
		print self.ngram_corrections

		# 2) Select n-grams based on word distance (spacy) --> you might not want to do this --> lot of effort
		print "Getting all the English spacy stuff..."
		
		print "Done with that!"

		spacy_error = nlp(unicode(error_sequence[len_sequence], encoding="utf-8"))

		print "Finding similarity measures"
		for correction in self.ngram_corrections:
			spacy_correction = nlp(correction[0]) # assuming that this is already written in unicode
			similarity_score = spacy_correction.similarity(spacy_error)
			word_score = correction[1]*(2**similarity_score)
			self.final_corrections.append((correction[0], word_score))

		self.final_sorted_list = sorted(self.final_corrections, key=lambda x:x[1])
		print self.final_sorted_list

		return self.final_sorted_list # now you can try the proposed corrections



if __name__ == '__main__':
	ng = NGrams()
	print "Finding ngrams..."
	ng.find_ngrams(3)
	print "Compute log likelihood..."
	ngram_dict = ng.ngram_log_likelihood()

	#print ngram_dict

	c = Correction()
	from spacy.en import LOCAL_DATA_DIR, English
	tbank = dt.tbankparser()
	print "Finding corrections..."
	c.find_correction(ngram_dict, ("will", "behave", "jist"), 'true', tbank.nlp)