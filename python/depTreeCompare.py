import spacy
import depTree as dts

import dependencyTree as dto
import nltk

from time import time

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
	return 1-score/max(len(d1),len(d2))

def score(tbank,inputs,outputs):
	score = 0.
	for i,input in enumerate(inputs):
		#print input
		t1 = tbank.parse(input)
		t2 = outputs[i]
		score += compare(t1,t2)
	return score

def main():
	print 'loading tbanks'
	tbanks = dts.tbankparser()
	tbanko = dto.tbankparser()
	tbankr = dto.tbankparser()
	tbankn = dto.tbankparser()

	print 'making targets'
	testing_targets = [t.tree() for t in tbanko._parsed[3000:]]
	testing_inputs = tbanko._sents[3000:]

	tbanko._parsed = tbanko._parsed[:3000]
	tbankr._parsed = tbankr._parsed[:3000]
	tbankn._parsed = tbankn._parsed[:3000]
	
	print 'loading parser'
	tbanko.getParser()
	tbankr.getParser()
	tbankn.getParser()
	
	print 'adding noise'
	tbankr.add_noise(3000,True,False)
	tbankn.add_noise(3000,True,True)
	
	ars = [ (tbanks,testing_inputs,testing_targets),
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
	
	print 'results\n','\n'.join([str(s) for s in scores])

if __name__ == '__main__':
	main()
	