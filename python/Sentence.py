from nltk.tokenize import *
from nltk import word_tokenize, pos_tag

class Sentence: 
	def __init__(self, sentance_tuple):
		self.raw_sentence = sentance_tuple[0][2:] #skip first 2 indexes becuase they are indexs 
		self.sentence_words = self.removePunctuation()
		self.error_list = self.setSentenceErrors(sentance_tuple) 
		self.pos_tags_sentence = self.posTagSentece(self.raw_sentence) 
		self.words_tags = self.makeTagTouples() 

	def makeTagTouples(self):
		splited_sentences = self.raw_sentence.split(' ') #Do we need a better way to split a sentence? Little bit tricky to split in spaces
		if not self.error_list:
			return 	[(word, "Ne") for word in splited_sentences]
		else:
			word_tags = [(word, "Ne") for word in splited_sentences]
			for error in self.error_list:
				word_tags[error.error_start_index:error.error_end_index] = [(splited_sentences[error.error_start_index],error.error_type)] * (error.error_end_index - error.error_start_index)
			return word_tags
				
	def posTagSentece(self, raw_sentence):
		# First char is always S this you can remove  
		return pos_tag(word_tokenize(raw_sentence)) 
	
	def removePunctuation(self):
		# remove punctuation from a sentance, First char is always S this you can remove  
		return word_tokenize(self.raw_sentence[1:])
	
	def setSentenceErrors(self, sentance_tuple):
		error_list = []
		raw_sentence = sentance_tuple[0] 
		for error_tuple in sentance_tuple[1]:
			error_list.append(Mistake(raw_sentence ,error_tuple))
		return error_list	

class Mistake:
	def __init__(self,raw_sentence ,error_tuple):
		self.error_type = error_tuple[1] # type of the grammer mistake 
		self.correction = error_tuple[2]	# the correction of the mistake
		self.error_start_index = int(error_tuple[0].split(' ')[1])  	# start index in the sentence for where the mistake has been made
		self.error_end_index = int(error_tuple[0].split(' ')[2])	# end index of the mistake 
		self.splited_sentane = raw_sentence[2:].split(' ')
		self.error_word =  self.setErrorWord(error_tuple, self.splited_sentane) 		# the word, of the sentence part that is wrong
		self.error_hash = error_has = {
			"Ne":"No Error",
			"Vt":"Verb tense",
			"Vm":"Verb modal",
			"V0":"Missing verb",
			"Vform":"Verb form",
			"SVA":"Subject-verb-agreement",
			"ArtOrDet":"Article or Determiner",
			"Nn":"Noun number",
			"Npos":"Noun possesive",
			"Pform":"Pronoun form",
			"Pref":"Pronoun reference",
			"Wcip":"Wrong collocation/idiom/preposition",
			"Wa":"Acronyms",
			"Wform":"Word form",
			"Wtone":"Tone",
			"Srun":"Runons, comma splice",
			"Smod":"Dangling modifier",
			"Spar":"Parallelism",
			"Sfrag":"Fragment",
			"Ssub":"Subordinate clause",
			"WOinc":"Incorrect sentence form",
			"WOadv":"Adverb/adjective position",
			"Trans":"Link word/phrases",
			"Mec":"Punctuation, capitalization, spelling, typos",
			"Rloc":"Local redundancy",
			"Cit":"Citation",
			"Others":"Other errors",
			"Um":"Unclear meaning (cannot be corrected)",						# hash with the full discription of every mistake in the dataset 
		}

	def giveFullMistakeDeclaration(self):
		# returns the the full description of the mistake 			
		return self.error_has[self.error_type]

	def setErrorWord(self,error_tuple, splited_sentane):
		# returns the word where the mistake has been made, is the word is empty, a word has been forgotten or another  mistake has been made
		try:
			return splited_sentane[int(self.error_start_index):int (self.error_end_index)]
		except ValueError:
			print "Error word can not be set, error tuple:"+error_tuple[0]
