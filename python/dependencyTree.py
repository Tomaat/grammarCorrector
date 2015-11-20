import nltk
import parseHack
import pickle
import time
import random
import copy

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
	
	def _chose_word(self,i):
		keys = self._parsed[i].nodes.keys()[1:-1]
		w = random.choice(keys)
		return w
	
	def _change_word(self,node):
		word = node['word']
		tag = node['tag']
		if len(word) > 0:
			i = random.randint(1,len(word))
			c = random.choice([c for c in 'abcdefghijklmnopqrstuvwxyz']+[''])
			new_word = word[:i-1]+c+word[i:]
		else:
			new_word = word
		return new_word
	
	def _add_noise(self,n=1,keep=True):
		for p in range(n):
			i = random.randint(0,self._n-1)
			if len(self._parsed[i].nodes) < 4:
				continue
			wi = self._chose_word(i)
			new_word = self._change_word(self._parsed[i].nodes[wi])
			if keep:
				new_graph = copy.deepcopy(self._parsed[i])
				new_graph.nodes[wi]['word'] = new_word
				self._parsed.append(new_graph)
			else:
				self._parsed[i].nodes[wi]['word'] = new_word
			#print i,wi,new_word
		
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
		
		x = self._parser.parse(sentence)
		(score,tree) = x.next()
		print score
		tree.pprint()
		
		return dfirst(tree)
	
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
	
	# def getNPParser(self,max=None):
		# if max == None:
			# max = self._n
		# p = ""
		# for i in range(0,max):
			# p += self._par(i)
		# gram = nltk.DependencyGrammar.fromstring(p)
		# self._parser = nltk.ProjectiveDependencyParser(gram)
	
def demo():
	t1 = time.time()
	tbank = tbankparser()
	tbank.getParser()
	t1 = time.time()-t1
	print "build tbank in %f sec"%(t1)
	sentence = tbank.sen(44) 
	print sentence
	t1 = time.time()
	tree = tbank.parse(sentence)
	t1 = time.time()-t1
	print "parsed sentence in %f sec"%(t1)
	for i,w in enumerate(tree):
		print i,w

def demo2():
	t1 = time.time()
	tbank = nltk.corpus.dependency_treebank
	parsed = tbank.parsed_sents()
	sents = tbank.sents()
	n = len(sents)
	parser = nltk.parse.ProbabilisticProjectiveDependencyParser()
	parser.train(parsed[:n])
	t1 = time.time()-t1
	print "build tbank in %f sec"%(t1)
	sentence = sents[44] 
	print sentence
	t1 = time.time()
	tree = parser.parse(sentence)
	t1 = time.time()-t1
	print "parsed sentence in %f sec"%(t1)
	
if __name__ == '__main__':
	demo()