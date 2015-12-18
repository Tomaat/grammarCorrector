try:
	import spacy
	import depTree as dts
except:
	print 'loaded without spacy'

import dependencyTree as dto
import nltk
import numpy as np
import matplotlib.pyplot as plt

from time import time
#import pathos.multiprocessing as mp
from multiprocessing import Pool

def dictify(tree):
	"""Given a tree, return a dictionary with the arcs.
		calls dictifies according to tree-type
		dictionary should consist of u'head->child':1 pairs
	"""
	if type(tree) == nltk.tree.Tree:
		return _dictify_nltk(tree)
	elif type(tree) == spacy.tokens.doc.Doc:
		return _dictify_spacy(tree)
	return {}

def _dictify_spacy(tree):
	"""Given a spacy parsed sentence object, return a dictionary of the arcs
	"""
	assert type(tree) == spacy.tokens.doc.Doc
	ans = {}
	for i in range(len(tree)):
		curr = tree[i]
		prev = curr.head
		if prev != curr:
			ans[prev.orth_+u'->'+curr.orth_] = 1
	return ans

def _dictify_nltk(tree,ansi=None):
	"""Given a nltk Tree, return a dictionary of arcs
	"""
	assert type(tree) == nltk.tree.Tree
	head = tree.label()
	if ansi == None:
		ans = {}
	else:
		ans = ansi
	assert type(ans) == dict
	
	for c in tree:
		if type(c) == nltk.tree.Tree:
			ans[head+u'->'+c.label()] = 1
			_dictify_nltk(c,ans)
		else:
			ans[head+u'->'+c] = 1
	#if ansi == None:
	return ans

def compare(tree1,tree2):
	"""given two trees, dictify them and calculate likeness.
		likeness is calculated by inverting normalised F1 score.
		precision is calculated by checking how many of the arcs of the first
		tree are also in the second.
		recall is calculated by checking how many arcs of the second tree
		are also in the first.
		F1 = 2*p*r/(p+r)
		score = 1-F1/tree_size
	"""
	d1 = dictify(tree1)
	d2 = dictify(tree2)
	precision = 0
	recall = 0
	for k in d1.keys():
		if k in d2:
			precision += 1
	for k in d2.keys():
		if k in d1:
			recall += 1
	if precision==recall==0:
		score = 0
	else:
		score = 2.*precision*recall/(precision+recall)
	return 1-score/max(len(d1),len(d2),1)

def score(tbank,inputs,targets):
	"""given a treebank, calculate total score from inputs and targets
	"""
	assert len(inputs) == len(targets)
	score = np.zeros((len(inputs)))
	for i,input in enumerate(inputs):
		#print input
		t1 = tbank.parse(input)
		t2 = targets[i]
		score[i] = compare(t1,t2)
	return score

def out(*args):
	"""custom print function
	"""
	ans = ''
	for ar in args:
		ans += str(ar)+' '
	print ans
	
def main(xin=0, tbank=None):
	"""load a given treebank, score it's accuracy and time runtime
	"""
	# x is the type of treebank
	if tbank is None:
		x = xin
	else:
		x = -1
		name = "user"
	# X is the amount of train-trees for nltk-based tbank
	# Y is the amount of added flaws to the nltk-based tbank
	# slice X:Z are the sentences tested on
	X,Y,Z = 3750,15000,3900
	
	out( 'making targets')
	tt = time()
	data = nltk.corpus.dependency_treebank
	testing_targets = [t.tree() for t in data.parsed_sents()[X:Z]]
	testing_inputs = data.sents()[X:Z]
	tt = time()-tt
	out( 'in',tt,'sec')
	
	out( "loading tbank")
	
	tl = time()

	if x == 0:
		name = "spacy"
		tbank = dts.tbankparser()
	elif x == 1:
		name = "ntlk no noise"
		tbank = dto.tbankparser()
		tbank.getParser(X)
	elif x == 2:
		name = "nltk random noise"
		tbank = dto.tbankparser()
		tbank.truncate(X)
		tbank.add_noise(Y,True,False)
		tbank.getParser()
	elif x == 3:
		name = "ntlk flaws noise"
		tbank = dto.tbankparser()
		tbank.truncate(X)
		tbank.add_noise(Y,True,True)
		tbank.getParser()
	elif x == 4:
		name = "nltk only random noise"
		tbank = dto.tbankparser()
		tbank.truncate(X)
		tbank.add_noise(Y,False,False)
		tbank.getParser()
	elif x == 5:
		name = "ntlk only flaws noise"
		tbank = dto.tbankparser()
		tbank.truncate(X)
		tbank.add_noise(Y,False,True)
		tbank.getParser()
	tl = time()-tl
	
	out( "scoring...")
	ts = time()
	s = score(tbank,testing_inputs,testing_targets)
	ts = time()-ts
	
	out("%s loaded in %f sec. Scored %f on %d targets in %f sec."%(name,tl,s.sum(),len(testing_targets),ts))
	np.save(name+'data.npy',s)
	return s 

def main2():
	

if __name__ == '__main__':
	main(1)
	main(4)
	main(5)
	