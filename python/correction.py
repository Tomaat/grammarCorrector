import nltk
from nltk.corpus import gutenberg, brown
from nltk import ngrams
from math import log10

class NGrams:

	def __init__(self):
		self.ngrams_gutenberg = []
		self.ngrams_brown = []
		self.ngram_freq_dict = {}
		self.ngram_prob_dict = {}
		self.total_ngram_count = 0

	def find_ngrams(self, n):
		""" Input: the 'n' of 'n-grams'

			Find all the n-grams in the brown corpus. Store in frequency dictionary

		"""

		#self.ngrams_gutenberg = ngrams(gutenberg.words(), n)
		self.ngrams_brown = ngrams(brown.words(), n)

		"""for i in self.ngrams_gutenberg:
			self.total_ngram_count += 1

			if self.ngram_freq_dict.has_key(i):
				self.ngram_freq_dict[i] += 1
				#print i
			else:
				self.ngram_freq_dict[i] = 1
				#print i"""

		for i in self.ngrams_brown:
			self.total_ngram_count += 1

			if self.ngram_freq_dict.has_key(i):
				self.ngram_freq_dict[i] += 1
				#print i
			else:
				self.ngram_freq_dict[i] = 1
				#print i

		print self.total_ngram_count

	def ngram_log_likelihood(self):
		""" Computes the log likelood for all n-grams. The log likelihood has been used
		to avoid numeric overflow.

		"""

		for key,value in self.ngram_freq_dict.iteritems():
			prob = float(value) / self.total_ngram_count
			log_prob = log10(float(prob))
			self.ngram_prob_dict[key] = log_prob

if __name__ == '__main__':
	ng = NGrams()
	print "Finding ngrams..."
	ng.find_ngrams(2)
	print "Compute log likelihood..."
	ng.ngram_log_likelihood()