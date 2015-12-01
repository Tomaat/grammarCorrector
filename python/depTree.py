"""
	Usage:
		import spacy
		import depTree as dt

		tbank = dt.tbankparser()

		# for each sentence:

		sentence = "The quick brown fox jumps over the lazy dog"
		# or: sentence = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

		parsed = tbank.parse(sentence)
		
		for i,w in enumerate(dt.dfirst(parsed)): # loop breadth-first through sentence
			word = w.orth_ # string of word
			tag = w.tag_ # string of tag
			# do things with words and tags

		list_of_words_only = [w.orth_ for w in dt.dfirst(parsed)]
		list_of_tags_only = [w.tag_ for w in dt.dfirst(parsed)]
"""

import os
from time import time
import spacy
from spacy.en import LOCAL_DATA_DIR, English

def pprint(tree, indent=0):
	if type(tree) == spacy.tokens.doc.Doc:
		tree = tree[:].root
	elif type(tree) == spacy.tokens.span.Span:
		tree = tree.root
	assert type(tree) == spacy.tokens.token.Token
	if list(tree.children) == []:
		print " "*indent, "(",tree,")"
	else:
		print  " "*indent, "(",tree
		for children in tree.children:
			pprint(children, indent+1)
		print " "*indent,")"

def sen_idx(sentence,word):
	"""black magic to get the index of the word in the tokenised string
	"""
	return len(sentence[:word.idx].split())


class dfirst(object):
	def __init__(self,tree):
		self.ttype = spacy.tokens.doc.Doc
		msg = 'wrong input of type %s for type %s!'%(str(type(tree) ), str(self.ttype))
		assert type(tree) == self.ttype, msg
		self.root = tree[:].root
		self.current = -1
		self.todo = []
	def __iter__(self):
		return self
	def __next__(self):
		return self.next()
	def next(self):
		if self.current == -1:
			self.current += 1
			self.tree = list(self.root.children)
			return self.root
		elif self.current < len(self.tree):
			ans = self.tree[self.current]
			child = list(ans.children)
			if not child == []:
				self.todo.append(child)
			self.current += 1
			return ans
		elif len(self.todo) > 0:
			self.tree = self.todo.pop(0)
			self.current = 0
			return self.next()
		else:
			raise StopIteration()

class all_sents(object):
	def __init__(self,filename='../release3.2/data/conll14st-preprocessed.m2'):
		self.filename = filename
		f = open(filename,'r')
		data_raw = [p.split('\n') for p in ''.join(f.readlines() ).split('\n\n')]
		self._sentence_tuples = ((sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw)
		f.close()

	def __iter__(self):
		return self
	def __next__(self):
		return self.next()
	def next(self):
		s,e = self._sentence_tuples.next()
		sp = s.split()[1:]
		fe = ['NE']*len(sp)
		for er in e:
			x = er[0].split()
			be = int(x[1])
			en = int(x[2])
			fe[be:en] = [er[1]]*(en-be)
		return sp,fe

class tbankparser:
	def __init__(self):
		DIR = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
		self.nlp = English(data_dir=DIR)
	
	def parse(self,sentence):
		if isinstance(sentence,str):
			sentence = unicode(sentence)
		elif isinstance(sentence,list):
			sentence = u' '.join(sentence)
		parsed = self.nlp(sentence)

		#pprint(parsed)
		return parsed

def demo():
	sent = "The big man gives a red present to the pretty girl"
	sentf = "The bgi man give an red prsent to the pretty grl"
	t1 = time()
	tbank = tbankparser()
	print "loaded data in %f sec."%(time()-t1)
	t2 = time()
	parsed_sent = tbank.parse(sent)
	t2 = time()-t2
	t3 = time()
	parsed_sentf = tbank.parse(sentf)
	t3 = time()-t3
	print "parsed sentences in %f and %f sec."%(t2,t3)
	print sent
	pprint(parsed_sent[:].root)
	print list(dfirst(parsed_sent))
	print sentf
	pprint(parsed_sentf[:].root)
	print list(dfirst(parsed_sentf))

if __name__ == '__main__':
	demo()
