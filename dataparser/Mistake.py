class Mistake:
	def __init__(self,raw_sentence ,error_tuple):
		self.error_type = error_tuple[1] # type of the grammer mistake 
		self.correction = error_tuple[2]	# the correction of the mistake
		self.error_start_index = error_tuple[0].split(' ')[1]  	# start index in the sentence for where the mistake has been made
		self.error_end_index = error_tuple[0].split(' ')[2] 	# end index of the mistake 
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
