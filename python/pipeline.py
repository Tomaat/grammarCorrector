from time import time
import dependencyTree as dto
import depTree as dts
import structured_perceptron as sp
import dataparser as dp
from multiprocessing import Pool

DUMMY = """
ans = 'hello {0} {1}'
"""

MAIN_CODE = """
t2 = time()-t1
print 'loading tree bank {0}'
tbank = {0}.tbankparser()
{1}
t3 = time()-t1-t2
it = 10	
sp._init_(len(feature_dict),{0} )
ans = 'SSE random weights: %d'%(flaws({0},all_sentences[:4],feature_dict,tbank) )
t4 = time()
weights = sp.train_perceptron(all_sentences, feature_dict, tbank, it, history=0)
t4 = time()-t4
t1=time()-t1
ans += '\nafter %d iterations over %d sentences: %d'%(it,len(all_sentences), flaws({0},all_sentences[:4],feature_dict,tbank,weights) )
ans += '\ntotal %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4)
"""

OPTS = [('dto',''), ('dto','tbank.add_noise(1000,True,False)'), ('dto','tbank.add_noise(1000,True,True)'), ('dts','')]

all_sentences,feature_dict = None,None

def run_once(i):
	t1 = time()
	comm = MAIN_CODE.format(OPTS[i][0],OPTS[i][1])
	exec comm
	return ans

def run_all():
	global all_sentences, feature_dict
	t1 = time()
	FILE = '../release3.2/data/conll14st-preprocessed.m2.small'
	print 'loading sentences'
	#all_sentences, feature_dict = dp.process(FILE)
	pool = Pool(4)
	anses = pool.map(run_once,[0,1,2,3])
	print '\n\n'.join(anses)

def main():
	t1 = time()
	TRAIN_FILE = '../release3.2/data/train.dat'
	VAL_FILE = '../release3.2/data/val.dat'
	print 'loading sentences'
	all_sentences, feature_dict = dp.process(TRAIN_FILE)
	val_sentences, 
	t2 = time()-t1
	print 'loading tree bank'
	tbank = dts.tbankparser()
	
	t3 = time()-t1-t2
	sp._init_(len(feature_dict),dts )
	print 'SSE random weights',flaws(dts,all_sentences[:4],feature_dict,tbank)	
	t4 = time()
	weights = sp.train_perceptron(all_sentences, feature_dict, tbank, history=0)
	t4 = time()-t4
	t1=time()-t1
	print 'after %d iterations over %d sentences'%(it,len(all_sentences)), flaws(dts, all_sentences[:4],feature_dict,tbank,weights)
	print 'total %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4)



def flaws(dt,all_sentences,feature_dict,tbank,weight_matrix=None,history=0):
	if weight_matrix is None:
		weight_matrix = sp.init_weights(len(feature_dict))
	E = 0.0
	for sentence in all_sentences:
		parsed_tree = tbank.parse(sentence.raw_sentence)
		# For loop around this, so that you loop through all sentences --> weights should be updated
		sentence.words_tags
		context_words = [w.orth_ for w in dt.dfirst(parsed_tree) ]
		context_tags = [sentence.words_tags[dt.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in dt.dfirst(parsed_tree)]
		
		target_feature_vectors = []
		for i,wrd in enumerate(context_words):
			target_feature_vectors.append( dp.construct_feature_vector(wrd, context_tags[i], 
					feature_dict, context_words, i, history_vectors=None, true_tags=context_tags ) )

		E = sp.test_perceptron_once(E, parsed_tree, target_feature_vectors, feature_dict, 
					history, weight_matrix, context_words, context_tags)
	return E

if __name__ == '__main__':
	run_all()