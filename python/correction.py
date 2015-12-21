import nltk
from nltk.corpus import gutenberg, brown
from nltk import ConditionalFreqDist
from nltk import ngrams
from math import log10
import spacy
import depTree as dt
import pprint

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
		
		total_ngram_count = 0
		ngram_freq_dict = {}

		sents = brown.sents()
		for sent in sents:
			sent = ['-START-']*(n-1)+sent
			ngrams_brown = ngrams(sent, n)
			
			for i in ngrams_brown:
				total_ngram_count += 1
				old = ngram_freq_dict.get(i,0)
				old += 1
				ngram_freq_dict[i] = old
				#print i,old

		return ngram_freq_dict, total_ngram_count

	def find_ngrams_dep(self, filename, n):
		""" Finds ngrams based on a dependency parsed corpus

			Input:	dependency parsed corpus
					the 'n' of n-grams

			Output:	frequency dictionary for the ngrams
					total ngram count
		"""

		total_ngram_count = 0
		ngram_freq_dict = {}

		f = open(filename)
		for line in f.readlines():
			if line != '\n':
				split = line.split()
				ngrams_dep = ngrams(split, n)
				
				for i in ngrams_dep:
					total_ngram_count += 1
					old = ngram_freq_dict.get(i,0)
					old += 1
					ngram_freq_dict[i] = old

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
		self.no_best_ngrams = 5
		self.no_ngrams = 5

	def find_correction(self, quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, error_sequence, error_tag, nlp=None,nlp2=None):
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
		self.pot_corrections = self.find_suitable_ngrams(quatrogram_dict, error_sequence,True,nlp2)
		#self.pot_corrections = self.find_suitable_ngrams(trigram_dict, error_sequence)
		#print "Corrections quatro: ", self.pot_corrections

		# If you can't find suitable n-grams --> backoff
		if self.pot_corrections == []:
			self.pot_corrections = self.find_suitable_ngrams(trigram_dict, error_sequence[1:],True,nlp2)
			#print "Corrections tri: ", self.pot_corrections

			if self.pot_corrections == []:
				self.pot_corrections = self.find_suitable_ngrams(bigram_dict, error_sequence[2:],True,nlp2)
				#print "Corrections bi: ", self.pot_corrections

				#if self.pot_corrections == []:
				#	self.pot_corrections == self.find_suitable_ngrams(unigram_dict, error_sequence[3:])
				#	print "Corrections uni: ", self.pot_corrections

		# For testing
		#self.pot_corrections.append((u'jus', 2))
		#sself.pot_corrections.append((u'test', 8))

		sortedlist = sorted(self.pot_corrections, key=lambda x:x[1])
		#print 'lensl',len(sortedlist)
		# 1b) Find the best n-grams. 
		
		self.ngram_corrections = sortedlist[-self.no_ngrams:] # note this is from small to big
		#print 'ngram',len(self.ngram_corrections)
		#print self.ngram_corrections

		# 2) Select n-grams based on word distance (spacy) --> you might not want to do this --> lot of effort
		#print "Getting all the English spacy stuff..."
		
		#print "Done with that!"
		#nlp = 1
		if not nlp is None:
			spacy_error = nlp(unicode(error_sequence[-1], encoding="utf-8"))

			#print "Finding similarity measures"
			for correction in self.ngram_corrections:
				spacy_correction = nlp(unicode(correction[0], encoding="utf-8")) # assuming that this is already written in unicode
				similarity_score = spacy_correction.similarity(spacy_error)
				word_score = correction[1]*(2**similarity_score)
				self.final_corrections.append((correction[0], word_score))

			self.final_sorted_list = sorted(self.final_corrections, key=lambda x:x[1])
		else:
			self.final_sorted_list = self.ngram_corrections
			
		#print self.final_sorted_list


		return self.final_sorted_list[-self.no_best_ngrams:] # now you can try the proposed corrections


	def find_suitable_ngrams(self, ngram_dict, error_sequence,letters=True,nlp=None):

		len_sequence = len(error_sequence)-1
		potential_corrections = []

		for key,value in ngram_dict.iteritems():
			if key[0:len_sequence] == error_sequence[0:len_sequence]:

				# You want to give a higher score to words that have many letters in common --> might want to take word length into account
				if letters:
					error = error_sequence[len_sequence]
					score = calc_common_letters(error, key[len_sequence])
					
				if not nlp is None:
					spacy_error = nlp(unicode(error_sequence[len_sequence], encoding="utf-8"))
					#print "Finding similarity measures"
					spacy_correction = nlp(unicode(key[len_sequence], encoding="utf-8"))
					similarity_score = spacy_correction.similarity(spacy_error)
					print "similarity_score: ", similarity_score
				
				if nlp is None:
					similarity_score = 0

				print "value: ", value
				print "score: ", score
				final_score = 4**(-1*value)*(2**similarity_score)*(3**score)
				#print 'Value: ', value
				#print 'Score: ', score
				print 'Final score: ', final_score

				potential_corrections.append((key[len_sequence], final_score))

		return potential_corrections

def calc_common_letters(error, pred):

	sim_dict = {}
	for char in error:
		if char in pred:
			sim_dict[char]=sim_dict.get(char,0)+1
	
	score = len(sim_dict.keys()) / len(pred)
	return score

def correct(quatrogram_dict, trigram_dict, bigram_dict, unigram_dict,filename='../release3.2/data/train.data.tiny',nlp=None):
	""" Finds corrections and compares with target corrections

		Input: 	all ngram dictionaries
				error file
				nlp
	"""

	with open (filename) as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		print data_lines
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(sentence[0][1:],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw]
		
	N = 4-1
	#N = 3-1
	cor1 = 0
	cor5 = 0
	it = 0
	for sentence,errors in sentence_tuples:
		tokens = sentence.split()
		#print "tokens: ", tokens
		tokens = ['-START-']*N+tokens
		#print "new tokens: ", tokens
		
		# Find the errors so that you can compare with the predicted correction
		for error in errors:
			#print "error: ", error
			begin = int(error[0].split()[1])+N
			#print "begin: ", begin
			end = int(error[0].split()[2])+N
			#print "end: ", end
			er_type = error[1]
			#print "error type: ", er_type
			er_correct = error[2]
			#print "error correction: ", er_correct

			if ( begin+1 != end ) or (len(er_correct.split()) != 1):
				#print 'Continue'
				continue

			it += 1
			error_ngram = tuple(tokens[end-N-1:end]) # find error ngrams that are fed to the correction (always four tokens)
			#print "error ngram: ", error_ngram
			c = Correction()
			c.no_best_ngrams = 50
			c.no_best_ngrams = 10

			# find the correction predicted by the corrector
			correct = c.find_correction(quatrogram_dict,trigram_dict,bigram_dict,unigram_dict,error_ngram,er_type,nlp2=nlp)
			print "correct: ", correct 

			if correct:		
				if correct[-1][0] == er_correct:
					cor1 += 1
				if er_correct in [k for k,s in correct]:
					cor5 += 1
				print error_ngram, er_type, er_correct, correct[-3:]
			else:
				print "no correction found" # sometimes the list is empty

	print it,cor1,cor5

def prepare(parse_type, filename):
	""" Preperation for the correction: find ngrams and compute likelihood

		Input:	linear or dependency wise
				filename, in case you want to parse dep wise

		Output:	all ngram likelihood dictionaries

	"""

	ng = NGrams()
	
	if parse_type == "linear":		
		print "Finding ngrams linear..."
		quatrograms, quatrogram_count = ng.find_ngrams(4)
		trigrams, trigram_count = ng.find_ngrams(3)
		bigrams, bigram_count = ng.find_ngrams(2)
		unigrams, unigram_count = ng.find_ngrams(1)

	elif parse_type == "dep":
		print "Finding ngrams dependency..."
		quatrograms, quatrogram_count = ng.find_ngrams_dep(filename, 4)
		trigrams, trigram_count = ng.find_ngrams_dep(filename, 3)
		bigrams, bigram_count = ng.find_ngrams_dep(filename, 2)
		unigrams, unigram_count = ng.find_ngrams_dep(filename, 1)

	print "Compute log likelihood..."
	quatrogram_dict = ng.ngram_log_likelihood(quatrograms, quatrogram_count)
	trigram_dict = ng.ngram_log_likelihood(trigrams, trigram_count)
	bigram_dict = ng.ngram_log_likelihood(bigrams, bigram_count)
	unigram_dict = ng.ngram_log_likelihood(unigrams, unigram_count)
	
	return quatrogram_dict,trigram_dict,bigram_dict,unigram_dict


if __name__ == '__main__':

	#parse_type = "linear"
	parse_type = "dep"
	#filename_prep = "preprocessed-BrownCorpus.txt"
	filename_prep = "test.txt"
	quatrogram_dict,trigram_dict,bigram_dict,unigram_dict = prepare(parse_type, filename_prep)

	"""ng = NGrams()
	
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
	#print ngram_dict"""

	c = Correction()
	# comment these three lines out if you don't want spacy
	print "Starting with the spacy stuff.."
	from spacy.en import LOCAL_DATA_DIR, English
	tbank = dt.tbankparser()
	
	print "Finding corrections..."

	filename='../release3.2/data/train.data.tiny'
	# with spacy
	correct(quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, filename, tbank.nlp)
	
	# without spacy
	#correct(quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, filename)
	

	#c.find_correction(quatrogram_dict, trigram_dict, bigram_dict, unigram_dict, ("I", "see", "you", "smike"), 'true')
	#c.find_correction({}, trigram_dict, {}, {}, ("will", "behave", "jist"), 'true', tbank.nlp)
	#c.find_correction({}, trigram_dict, {}, {}, ("will", "behave", "jist"), 'true')
