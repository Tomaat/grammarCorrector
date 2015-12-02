import nltk
from nltk.corpus import gutenberg, brown
from nltk import ConditionalFreqDist
from nltk import ngrams
from math import log10

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

	def find_correction(self, ngram_dict, error_sequence, error_tag):
		""" Input:	the n-gram that ends in the error
					the error tag predicted by the structured perceptron

			Output:	corrected word

			Function to find a correction for the mistake
			1) What words do we expect based on our n-gram dictionary?
			2) What do we expected based on (1) and the word distance (spacy)?
		"""

		len_sequence = len(error_sequence)-1

		# 1) Find suitable n-grams
		for key,value in ngram_dict.iteritems():
			if key[0:len_sequence] == error_sequence[0:len_sequence]:
				self.pot_corrections.append((key[len_sequence], value))

		self.pot_corrections.append(("u'testword", -2))
		self.pot_corrections.append(("u'testword2", -8))

		sortedlist = sorted(self.pot_corrections, key=lambda x:x[1])

		# 1b) Find the best n-grams. 
		no_best_ngrams = 2
		self.ngram_corrections = sortedlist[-no_best_ngrams:]


		# 2) Select n-grams based on word distance (spacy)


			










if __name__ == '__main__':
	ng = NGrams()
	print "Finding ngrams..."
	ng.find_ngrams(3)
	print "Compute log likelihood..."
	ngram_dict = ng.ngram_log_likelihood()

	#print ngram_dict

	c = Correction()
	print "Finding corrections..."
	c.find_correction(ngram_dict, ("will", "behave", "just"), 'true')