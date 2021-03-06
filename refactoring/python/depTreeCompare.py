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
from multiprocessing import Pool
import random
import sys

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
	
def main(xin=0, tbank=None, train_test=None):
	"""load a given treebank, score it's accuracy and time runtime
	"""
	# x is the type of treebank
	if tbank is None:
		x = xin
	else:
		x = -1
		name = "user"
	X,Y,Z = 3750,15000,3900
	
	out( 'making targets')
	tt = time()
	if train_test is None:
		data = nltk.corpus.dependency_treebank
		testing_targets = [t.tree() for t in data.parsed_sents()[X:Z]]
		testing_inputs = data.sents()[X:Z]
	else:
		testing_targets = train_test[1]
		testing_inputs = train_test[0]
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
	np.save(name+str(time())+'data.npy',s)
	return s 

def main2(): 
	# preproces the data
	filename= '../release3.2/data/conll14st-preprocessed.m2'
	print "Load data from", filename
	f = open(filename,'r')
	data_raw = [p.split('\n') for p in ''.join(f.readlines() ).split('\n\n')]
	sentence_tuples = [(sentence[0][2:],[tuple(errors.split('|||')) for errors in sentence[1:]]) for sentence in data_raw[:len(data_raw)-1]]
	f.close()

	random.shuffle(sentence_tuples)

	sents = sentence_tuples[:150] # select 150 sentences for testing 
	tbank_s = dts.tbankparser()
	targets = [tbank_s.parse(t[0]) for t in sents]
	inputs = [t[0] for t in sents]

	main(0,None,(inputs,targets))

	main(1,None,(inputs,targets))
	
	main(4,None,(inputs,targets))
	reload(sys)  
	sys.setdefaultencoding('utf8')

	main(5,None,(inputs,targets))
	
	

if __name__ == '__main__':
	main(0)
	main(1)
	main(4)
	main(5)
	