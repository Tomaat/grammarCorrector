try:
	import spacy
	import depTree as dts
except:
	print 'loaded without spacy'

import dependencyTree as dto
import nltk

from time import time
#import pathos.multiprocessing as mp
from multiprocessing import Pool

def dictify(tree):
	if type(tree) == nltk.tree.Tree:
		return dictify_nltk(tree)
	elif type(tree) == spacy.tokens.doc.Doc:
		return dictify_spacy(tree)
	return {}

def dictify_spacy(tree):
	assert type(tree) == spacy.tokens.doc.Doc
	ans = {}
	for i in range(len(tree)):
		curr = tree[i]
		prev = curr.head
		if prev != curr:
			ans[prev+u'->'+curr] = 1
	return ans

def dictify_nltk(tree,ansi=None):
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
			dictify_nltk(c,ans)
		else:
			ans[head+u'->'+c] = 1
	#if ansi == None:
	return ans

def compare(tree1,tree2):
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

def score(tbank,inputs,outputs):
	score = 0.
	for i,input in enumerate(inputs):
		#print input
		t1 = tbank.parse(input)
		t2 = outputs[i]
		score += compare(t1,t2)
	return score

import dill

def run_dill_encoded(what):
    fun, args = dill.loads(what)
    return fun(*args)

def apply_async(pool, fun, args):
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),))

	
#glo_tbank = None
def compare_multi(arg):
	parse,input,t2 = arg
	t1 = parse(input)
	s = compare(t1,t2)
	return s
	
def score_multi(tbank,inputs,outputs):
	#global glo_tbank
	#glo_tbank = tbank
	io = zip(inputs,outputs)
	io = [(tbank.parse,i,o) for i,o in io]
	workers = Pool(3)
	jobs = []
	for i in io:
		print i
		job = apply_async(workers,compare_multi,i)
		jobs.append(job)
	
	print jobs
	ans = []
	for job in jobs:
		ans.append( job.get() )
	#ans = workers.map(compare_multi,io)
	return sum(ans)
	
def out(*args):
	ans = ''
	for ar in args:
		ans += str(ar)+' '
	print ans
	
def main2():
	x = 5
	X,Y,Z = 3000,3000,3100
	
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
		tbank.getParser(X)
		tbank.add_noise(Y,True,False)
	elif x == 3:
		name = "ntlk flaws noise"
		tbank = dto.tbankparser()
		tbank.getParser(X)
		tbank.add_noise(Y,True,True)
	elif x == 4:
		name = "nltk only random noise"
		tbank = dto.tbankparser()
		tbank.getParser(X)
		tbank.add_noise(Y,False,False)
	elif x == 5:
		name = "ntlk only flaws noise"
		tbank = dto.tbankparser()
		tbank.getParser(X)
		tbank.add_noise(Y,True,True)
	tl = time()-tl
	
	out( "scoring...")
	ts = time()
	s = score(tbank,testing_inputs,testing_targets)
	ts = time()-ts
	
	out("%s loaded in %f sec. Scored %f on %d targets in %f sec."%(name,tl,s,len(testing_targets),ts))
	
	
def main():
	print 'loading tbanks'
	#tbanks = dts.tbankparser()


	#tbanko._parsed = tbanko._parsed[:3000]
	#tbankr._parsed = tbankr._parsed[:3000]
	#tbankn._parsed = tbankn._parsed[:3000]
	X = 3000
	#print 'loading parser'
	#print 'adding noise'
	to = time()
	tbanko = dto.tbankparser()
	tbanko.getParser(X)
	to = time()-to
	tr = time()
	tbankr = dto.tbankparser()
	tbankr.getParser(X)
	tbankr.add_noise(3000,True,False)
	tr = time()-tr
	tn = time()
	tbankn = dto.tbankparser()
	tbankn.getParser(X)
	tbankn.add_noise(3000,True,True)
	tn = time()-tn
	ts = [to,tr,tn]
	
	print 'making targets'
	data = nltk.corpus.dependency_treebank
	testing_targets = [t.tree() for t in data.parsed_sents()[3000:]]
	testing_inputs = data.sents()[3000:]
	
	ars = [ #(tbanks,testing_inputs,testing_targets),
			(tbanko,testing_inputs,testing_targets),
			(tbankr,testing_inputs,testing_targets),
			(tbankn,testing_inputs,testing_targets)]
	scores = []
	for ar in ars:
		print 'scoring...'
		t = time()
		s = score(*ar)
		t = time()-t
		scores.append((s,t))
	
	print 'results\n','\n'.join([str((len(testing_targets),)+s+(ts[i],)) for i,s in enumerate(scores)])

if __name__ == '__main__':
	main2()
	