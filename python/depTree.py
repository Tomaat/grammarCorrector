import os
import spacy

def pprint(tree, indent=0):
	assert type(tree) == spacy.tokens.token.Token
	print  " "*indent, "(",tree
	for children in tree.children:
		pprint(children, indent+1)
	print " "*indent,")"


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

class tbankparser:
	def __init__(self):
		DIR = os.environ.get('SPACY_DATA', spacy.en.LOCAL_DATA_DIR)
		self.nlp = spacy.en.English(data_dir=DIR)
	
	def parse(self,sentence):
		if isinstance(sentence,str):
			sentence = unicode(sentence)
		elif isinstance(sentence,list):
			sentence = u' '.join(sentence)
		parsed = self.nlp(sentence)
		return parsed