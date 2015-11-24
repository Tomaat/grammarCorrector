from nltk.tokenize import *
from Mistake import Mistake 
from nltk import word_tokenize, pos_tag

class Sentence: 
	def __init__(self, sentance_tuple):
		self.raw_sentence = sentance_tuple[0][2:]
		self.sentence_words = self.removePunctuation()
		self.error_list = self.setSentenceErros(sentance_tuple) 
		self.pos_tags_sentence = self.posTagSentece(self.raw_sentence) 
		self.words_tags = self.makeTagTouples() 

	def makeTagTouples(self):
		splited_sentences = self.raw_sentence.split(' ') #Do we need a better way to split a sentence? Little bit tricky to split in spaces
		pointer_one = 0
		pointer_two = 1
		error_index = 0
		if not self.error_list:
			return 	[(word, "Ne") for word in splited_sentences]
		else:
			word_tags = []
			for word in splited_sentences:
				if (error_index + 1) <= len(self.error_list):
					if ((pointer_one == int(self.error_list[error_index].error_start_index) and pointer_one == int(self.error_list[error_index].error_end_index)) 
				    or  (pointer_one == int(self.error_list[error_index].error_start_index) and pointer_two == int(self.error_list[error_index].error_end_index))):
						word_tags.append((word, self.error_list[error_index].error_type))
						error_index += 1 
					else:
						word_tags.append((word, "Ne"))
				else:
					word_tags.append((word, "Ne"))
				
				pointer_one += 1
				pointer_two += 1
			return word_tags
				
	def posTagSentece(self, raw_sentence):
		# First char is always S this you can remove  
		return pos_tag(word_tokenize(raw_sentence[1:])) 
	
	def removePunctuation(self):
		# remove punctuation from a sentance, First char is always S this you can remove  
		return word_tokenize(self.raw_sentence[1:])
	
	def getPosTaggedSentence(self):
		return self.pos_tags_sentence

	def getSentencesTagCouples(self):
		return self.words_tags
	
	def getRawSentence():
		#return string with the raw sentce 
		return self.raw_sentance

	def getSentenceWords():
		# return words without the punctuatuin 
		return self.sentence_words

	def getErrorList():
		# return a list with array objects for that sentence 
		return self.error_list

	def setSentenceErros(self, sentance_tuple):
		error_list = []
		raw_sentence = sentance_tuple[0] 
		for error_tuple in sentance_tuple[1]:
			error_list.append(Mistake(raw_sentence ,error_tuple))
		return error_list	