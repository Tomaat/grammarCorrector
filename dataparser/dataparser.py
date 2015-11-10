from sgmllib import SGMLParser # maybe we need this after a while 
from nltk.tokenize import *


class Sentence: 
	def __init__(self, sentance):
		self.raw_sentance = sentance
		self.sentence_words = self.removePunctuation()
		self.erros_list = None 


		#self. 

	def removePunctuation(self):
		# remove punctuation from a sentance, First char is always S this you can remove  
		return word_tokenize(self.raw_sentance[1:])
	
	def getRawSentence():
		return self.raw_sentance

	def getSentenceWords():
		return self.sentence_words

	def setSentenceErros(self):
		return


class Mistake:
	def __init__(self):
		self.error_type = None
		self.correcy_form = None
		self.error_start_index = None 
		self.error_end_index = None 
		sekf.error_has = error_has = {
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
		



if __name__ == '__main__':
	with open ('../release3.2/data/conll14st-preprocessed.m2') as datafile: # import sgml data-file
		data_lines = datafile.readlines()
		data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]
		sentence_tuples = [(x[0],[tuple(y.split('|||')) for y in x[1:]]) for x in data_raw]
		
		for sentence_tuple in sentence_tuples[1:]: # er gaat nog iets mis met de eerste zin kijken of dat vaker gebeurt?
			
			sentence = Sentence(sentence_tuple[0],sentence_tuple[1])
			




"""
Some experimental code

sentance, errors_tuple = sentance_tuples[6]
splited_sentanec = sentance.split(' ')
a,b,c = errors_tuple[1][0].split(' ')
print splited_sentanec[int(b):int(c)] 

"""
	
	     


