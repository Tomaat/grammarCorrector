from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *


class Sentence: 
	def __init__(self, sentance_tuple):
		self.raw_sentance = sentance_tuple[0]
		self.sentence_words = self.removePunctuation()
		self.error_list = self.setSentenceErros(sentance_tuple) 

	def removePunctuation(self):
		# remove punctuation from a sentance, First char is always S this you can remove  
		return word_tokenize(self.raw_sentance[1:])
	
	def getRawSentence():
		#return string with the raw sentce 
		return self.raw_sentance

	def getSentenceWords():
		# return words without the punctuatuin 
		return self.sentence_words

	def setSentenceErros(self, sentance_tuple):
		error_list = []
		for error_tuple in sentance_tuple[1]:
			error_list.append(Mistake(error_tuple))
		return error_list	


class Mistake:
	def __init__(self, error_tuple):
		self.error_type = error_tuple[1]
		self.error_word = None  
		self.correction = error_tuple[2]
		self.error_start_index = error_tuple[0][2] 
		self.error_end_index = error_tuple[0][4] 
		self.error_has = error_has = {
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
								"Um":"Unclear meaning (cannot be corrected)",
							}

	def giveFullMistakeDeclaration(self):
		return self.error_has[self.error_type]




if __name__ == '__main__':
	with open ('../release3.2/data/conll14st-preprocessed.m2') as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(x[0],[tuple(y.split('|||')) for y in x[1:]]) for x in data_raw]
		processed_sentences = []
		for sentence_tuple in sentence_tuples[1:]: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
			processed_sentences.append(Sentence(sentence_tuple))
			
	print "end of program"



"""
Some experimental code

sentance, errors_tuple = sentance_tuples[6]
splited_sentanec = sentance.split(' ')
a,b,c = errors_tuple[1][0].split(' ')
print splited_sentanec[int(b):int(c)] 

"""
	
	     


