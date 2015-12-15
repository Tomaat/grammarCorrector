"""
	Usage:
		import nltk
		import dependencyTree as dt

		tbank = dt.tbankparser()
		# for noise: tbank.add_noise(1000,True,True)

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

import nltk
import parseHack
import pickle
import time
import random
import copy

class wordHack(object):
	def __init__(self,word,tag):
		self.orth_ = word
		self.tags_ = tag
		self.tag_ = list(tag)[0]

	def __repr__(self):
		return self.orth_+':'+self.tag_
	def __str__(self):
		return self.__repr__()

def sen_idx(sentence,word):
	"""black magic to get the index of the word in the tokenised string
	"""
	return len(sentence[sentence.find(word.orth_)].split())

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
	def __init__(self,filename='../release3.2/data/conll14st-preprocessed.m2'):
		f = open(filename,'r')
		data_raw = [p.split('\n') for p in ''.join(f.readlines() ).split('\n\n')]
		self._sentence_tuples = ((sentence[0],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw)
		f.close()
		self._tbank = nltk.corpus.dependency_treebank
		self._parsed = self._tbank.parsed_sents()
		self._sents = self._tbank.sents()
		self._n = len(self._sents)

	
	def sen(self,i=0):
		return self._sents[i]
	
	def truncate(self,n):
		self._parsed = self._parsed[:n]
		self._n = self._m = n
		
	
	def getParser(self,max=None,load=False,save=False,filename='parser.pkl'):
		if load:
			parser = pickle.load(open(filename,'r') )
		else:
			if max != None:
				self.truncate(max)
			parser = parseHack.ProbabilisticProjectiveDependencyParser()
			parser.train(self._parsed)
			if save:
				pickle.dump(parser,open(filename,'w') )
		self._parser = parser
	
	def _change_real_word(self,i):
		cursen = self._parsed[i]
		keys = cursen.nodes.keys()[1:-1]
		poss = []
		for w in keys:
			if cursen.nodes[w]['word'] in self._flaws:
				poss.append(w)
		if poss == []:
			wi = 1
			new_word = cursen.nodes[1]['word']
		else:
			wi = random.choice(poss)
			node = cursen.nodes[wi]
			word = node['word']

			possto = self._flaws[word]
			new_word = random.choice(possto)
			new_word = u' '.join(new_word)
		return new_word, wi
	
	def _change_word(self,i):
		keys = self._parsed[i].nodes.keys()[1:-1]
		wi = random.choice(keys)
		node = self._parsed[i].nodes[wi]

		word = node['word']
		#tag = node['tag']
		if len(word) > 0:
			i = random.randint(1,len(word))
			c = random.choice([c for c in 'abcdefghijklmnopqrstuvwxyz']+[''])
			new_word = word[:i-1]+c+word[i:]
		else:
			new_word = word
		return new_word,wi
	
	def add_noise(self,n=1,keep=True,real=True):
		#if not hasattr(self,'_parser'):
		#	self.getParser(self._n)
		if not hasattr(self,'_m'):
			self._m = self._n
		if real:
			change = self._change_real_word
			if not hasattr(self,'_flaws'):
				self._get_flaws()
		else:
			change = self._change_word
		for p in range(n):
			i = random.randint(0,self._m-1)
			if len(self._parsed[i].nodes) < 4:
				continue
			new_word,wi = change(i)
			if keep:
				new_graph = copy.deepcopy(self._parsed[i])
				new_graph.nodes[wi]['word'] = new_word
				self._parsed.append(new_graph)
				self._n += 1
			else:
				self._parsed[i].nodes[wi]['word'] = new_word
			#print i,wi,new_word


	def _get_flaws(self):
		flaws = {}
		for s,f in self._sentence_tuples:
			sp = s.split()[1:]
			for er in f:
				x = er[0].split()
				be = int(x[1])
				en = int(x[2])
				good = unicode(er[2])
				bad = sp[be:en]
				old = flaws.get(good,())
				if not bad in old:
					old = old + ( sp[be:en], )
				flaws[good] = old
		self._flaws = flaws
	
	def __str__(self):
		ans = ''
		if not hasattr(self,'_parser'):
			self.getParser(self._n)
		for i in range(0,max(self._m,30)):
			 ans += ' '.join(self._parsed[i].nodes[z]['word'] for z in self._parsed[i].nodes.keys()[1:-1])
			 ans += '\r\n'
		return ans

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
		#print score
		#tree.pprint()
		#self._possify(tree)

		return tree

	def _possify(self,tree):
		assert type(tree) == nltk.tree.Tree
		word = tree.label()
		tag = self._parser._grammar._tags.get(word,{'Null'})
		tree.set_label(wordHack(word,tag))
		for i in range(0,len(tree)):
			if type(tree[i]) == nltk.tree.Tree:
				self._possify(tree[i])
			else:
				word = tree[i]
				tag = self._parser._grammar._tags.get(word,{'Null'})
				tree[i] = wordHack(word,tag)



	
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