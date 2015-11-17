import nltk
import parseHack
import pickle

class dfirst(object):
	def __init__(self,tree):
		self.ctype = nltk.tree.Tree
		if not isinstance(tree,self.ctype):
			msg = 'wrong input of type %s for type %s!'%(str(type(tree) ), str(self.ctype)) 
			raise TypeError(msg)
		self.currentTree = tree
		self.current = -1
		self.todo = []
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.next()
	
	def next(self):
		if self.current == -1:
			self.current += 1
			return self.currentTree.label()
		elif self.current < len(self.currentTree):
			ans = self.currentTree[self.current]
			self.current += 1
			if not isinstance(ans,self.ctype):
				return ans
			else:
				self.todo += [ans]
				return ans.label()
		elif len(self.todo) > 0:
			self.currentTree = self.todo.pop(0)
			self.current = 0
			return self.next()
		else:
			raise StopIteration()

class tbankparser:
	def __init__(self):
		self._tbank = nltk.corpus.dependency_treebank
		self._parsed = self._tbank.parsed_sents()
		self._sents = self._tbank.sents()
		self._n = len(self._sents)
	
	def sen(self,i=0):
		return self._sents[i]
	
	# @staticmethod
	# def _strip(word):
		# word = str(word).replace("'","")
		# word = word.replace('"','')
		# if word == "":
			# word = "-"
		# return "'"+word+"'"
	
	# def _par(self,i=0):
		# tr = self._parsed[i].tree()
		# todo = [tr]
		# p = ""
		# while len(todo) > 0:
			# ctr = todo.pop(0)
			# p += tbankparser._strip(ctr.label())+" --> "
			# for i in range(0,len(ctr)):
				# ans = ctr[i]
				
				# if not isinstance(ans,nltk.tree.Tree):
					# p += tbankparser._strip(ans)+" "
				# else:
					# todo.append(ans)
					# p += tbankparser._strip(ans.label())+" "
			# p += "\n"
		# return p
	
	# def getParser(self,max=None):
		# if max == None:
			# max = self._n
		# p = ""
		# for i in range(0,max):
			# p += self._par(i)
		# gram = nltk.DependencyGrammar.fromstring(p)
		# self._parser = nltk.ProjectiveDependencyParser(gram)
	
	def getParser(self,max=None,load=False,save=False,filename='parser.pkl'):
		if load:
			parser = pickle.load(open(filename,'r') )
		else:
			if max == None:
				max = self._n
			
			parser = parseHack.ProbabilisticProjectiveDependencyParser()
			parser.train(self._parsed[:max])
			if save:
				pickle.dump(parser,open(filename,'w') )
		self._parser = parser
	
	def pprint(self,sen=0,max=None, ptype=1,):
		if not hasattr(self,'_parser'):
			self.getParser(max)
		ps = self._parser.parse(self.sen(sen))
		print self.sen(sen)
		for t in ps:
			if ptype==1:
				t.pprint()
			elif ptype==2:
				t.pretty_print()
	
	def parse(self,sentence):
		if isinstance(sentence,str):
			sentence = sentence.split()
		if not hasattr(self,'_parser'):
			self.getParser(self._n)
		
		(score,tree), = self._parser.parse(sentence)
		
		print score
		tree.pprint()
		
		return dfirst(tree)