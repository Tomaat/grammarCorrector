import nltk
import parseHack

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
		self.tbank = nltk.corpus.dependency_treebank
		self.pars = self.tbank.parsed_sents()
		self.sens = self.tbank.sents()
		self.n = len(self.sens)
	
	def sen(self,i=0):
		return self.sens[i]
	
	def strip(self,s):
		s = str(s).replace("'","")
		s = s.replace('"','')
		if s == "":
			s = "-"
		return "'"+s+"'"
	
	def par(self,i=0):
		tr = self.pars[i].tree()
		todo = [tr]
		p = ""
		while len(todo) > 0:
			ctr = todo.pop(0)
			p += self.strip(ctr.label())+" --> "
			for i in range(0,len(ctr)):
				ans = ctr[i]
				
				if not isinstance(ans,nltk.tree.Tree):
					p += self.strip(ans)+" "
				else:
					todo.append(ans)
					p += self.strip(ans.label())+" "
			p += "\n"
		return p
	
	def getParser(self,max=None):
		if max == None:
			max = self.n
		p = ""
		for i in range(0,max):
			p += self.par(i)
		gram = nltk.DependencyGrammar.fromstring(p)
		return nltk.ProjectiveDependencyParser(gram)
	
	def getPParser(self,max=None):
		if max == None:
			max = self.n
		
		pars = parseHack.ProbabilisticProjectiveDependencyParser()
		pars.train(self.pars[:max])
		
		return pars
	
	def pprint(self,sen=0,max=None, ptype=1, pars=None):
		if pars == None:
			pars = self.getParser(max)
		ps = pars.parse(self.sen(sen))
		print self.sen(sen)
		for t in ps:
			if ptype==1:
				t.pprint()
			elif ptype==2:
				t.pretty_print()
		