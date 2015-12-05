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
		#self.ngram_freq_dict = {}
		#self.ngram_prob_dict = {}
		#self.total_ngram_count = 0


	def find_ngrams(self, n):
		""" Input: the 'n' of 'n-grams'

			Find all the n-grams in the brown corpus. Store in frequency dictionary.
			Optionally it can be decided to use more corpora in order to have more data.

			Note: these are of course n-grams based on going through the sentence from left to right
			If we want to give the correction back based on the dependency tree, we need to
			parse the brown corpus (or any other data set) with the dependency parser, so that
			we can use this data. 			

		"""
		
		ngrams_brown = ngrams(brown.words(), n)
		total_ngram_count = 0
		ngram_freq_dict = {}

		for i in ngrams_brown:
			total_ngram_count += 1

			if ngram_freq_dict.has_key(i):
				ngram_freq_dict[i] += 1
				#print i
			else:
				ngram_freq_dict[i] = 1
				#print i

		return ngram_freq_dict, total_ngram_count


	def ngram_log_likelihood(self, ngram_dict, total_ngram_count):
		""" Computes the log likelood for all n-grams. The log likelihood has been used
			to avoid numeric overflow.

		"""

		ngram_prob_dict = {}
		for key,value in ngram_dict.iteritems():
			prob = float(value) / total_ngram_count
			log_prob = log10(float(prob))
			ngram_prob_dict[key] = log_prob

		return ngram_prob_dict



class Correction(NGrams):

	def __init__(self):
		self.pot_corrections = []
		self.ngram_corrections = []
		self.final_corrections = []
		self.final_sorted_list = []

	def find_correction(self, quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, error_sequence, error_tag):
		""" Input:	all constructed ngram dictionaries
					the 4-gram that ends in the error (!! really needs to be a 4-gram !!)
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

		

		# 1) Find suitable n-grams
		self.pot_corrections = self.find_suitable_ngrams(quatrogram_dict, error_sequence)
		#self.pot_corrections = self.find_suitable_ngrams(trigram_dict, error_sequence)
		print "Corrections quatro: ", self.pot_corrections

		# If you can't find suitable n-grams --> backoff
		if self.pot_corrections == []:
			self.pot_corrections = self.find_suitable_ngrams(trigram_dict, error_sequence[1:])
			print "Corrections tri: ", self.pot_corrections

			if self.pot_corrections == []:
				self.pot_corrections = self.find_suitable_ngrams(bigram_dict, error_sequence[2:])
				print "Corrections bi: ", self.pot_corrections

				if self.pot_corrections == []:
					self.pot_corrections == self.find_suitable_ngrams(unigram_dict, error_sequence[3:])
					print "Corrections uni: ", self.pot_corrections

		# For testing
		self.pot_corrections.append((u'jus', 2))
		self.pot_corrections.append((u'test', 8))

		sortedlist = sorted(self.pot_corrections, key=lambda x:x[1])

		# 1b) Find the best n-grams. 
		no_best_ngrams = 2
		self.ngram_corrections = sortedlist[no_best_ngrams-1:] # note this is from small to big
		print self.ngram_corrections

		"""# 2) Select n-grams based on word distance (spacy) --> you might not want to do this --> lot of effort
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
		print self.final_sorted_list"""

		self.final_sorted_list = self.ngram_corrections

		return self.final_sorted_list # now you can try the proposed corrections


	def find_suitable_ngrams(self, ngram_dict, error_sequence):

		len_sequence = len(error_sequence)-1
		potential_corrections = []

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

				potential_corrections.append((key[len_sequence], final_score))

		return potential_corrections


if __name__ == '__main__':
	ng = NGrams()
	
	print "Finding ngrams..."
	quatrograms, quatrogram_count = ng.find_ngrams(4)
	trigrams, trigram_count = ng.find_ngrams(3)
	bigrams, bigram_count = ng.find_ngrams(2)
	unigrams, unigram_count = ng.find_ngrams(1)

	print "Compute log likelihood..."
	quatrogram_dict = ng.ngram_log_likelihood(quatrograms, quatrogram_count)
	trigram_dict = ng.ngram_log_likelihood(trigrams, trigram_count)
	bigram_dict = ng.ngram_log_likelihood(bigrams, bigram_count)
	unigram_dict = ng.ngram_log_likelihood(unigrams, unigram_count)
	#print ngram_dict

	c = Correction()
	#print "Starting with the spacy stuff.."
	#from spacy.en import LOCAL_DATA_DIR, English
	#tbank = dt.tbankparser()
	print "Finding corrections..."
	c.find_correction(quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, ("I", "see", "you", "smike"), 'true')
	#c.find_correction({}, trigram_dict, {}, {}, ("will", "behave", "jist"), 'true', tbank.nlp)
	#c.find_correction({}, trigram_dict, {}, {}, ("will", "behave", "jist"), 'true')