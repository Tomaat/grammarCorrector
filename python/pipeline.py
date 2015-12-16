from time import time
import dependencyTree as dto
import depTree as dts
import structured_perceptron as sp
import dataparser as dp
from multiprocessing import Pool
import datetime
import numpy as np

DUMMY = """
ans = 'hello {0} {1}'
"""

MAIN_CODE = """
import structured_perceptron as spr
t2 = time()-t1
print 'loading tree bank {0}'
tbank = {0}.tbankparser()
{1}
t3 = time()-t1-t2
it = 10	
spr._init_(len(feature_dict),{0} )
ans = 'SSE random weights, only Ne-tags: %d'%(flaws({0},val_sentences,feature_dict,tbank,history,with_tags=False) )
ans = 'SSE random weights: %d'%(flaws({0},val_sentences,feature_dict,tbank,history) )
t4 = time()
weights = spr.train_perceptron(all_sentences, feature_dict, tbank, history)
t4 = time()-t4
t1=time()-t1
ans += '\\nafter %d sentences: %d'%(len(all_sentences), flaws({0},val_sentences,feature_dict,tbank,history,weights) )
ans += '\\ntotal %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4)
"""

#OPTS = [('dto',''), ('dto','tbank.add_noise(1000,True,False)'), ('dto','tbank.add_noise(1000,True,True)'), ('dts','')]
#OPTS = [('dts',''),('dts','print "hello"'),('dts','t1=5'),('dts','print "wehee"')]

#all_sentences,val_sentences,feature_dict,history = None,None,None,None

def run_once(inp):
	all_sentences, val_sentences, feature_dict, history,i = inp
	t1 = time()
	#comm = MAIN_CODE.format(OPTS[i][0],OPTS[i][1])
	#exec comm
	if i == 0:
		dt = dto
	elif i == 1:
		dt = dto
	elif i == 2:
		dt = dto
	elif i == 3:
		dt = dts
	else:
		return "placeholder"


	import structured_perceptron as spr
	t2 = time()-t1
	print 'loading tree bank dts'
	tbank = dt.tbankparser()
	if i == 1:
		tbank.add_noise(1000,True,False)
	elif i == 2:
		tbank.add_noise(1000,True,True)
	
	t3 = time()-t1-t2

	spr._init_(len(feature_dict),dt )
	ans = 'SSE random weights, only Ne-tags: %d'%(flaws(dt,val_sentences,feature_dict,tbank,history,with_tags=False) )
	ans += '\nSSE random weights: %d'%(flaws(dt,val_sentences,feature_dict,tbank,history) )
	t4 = time()
	weights = spr.train_perceptron(all_sentences, feature_dict, tbank, history)
	t4 = time()-t4
	t1=time()-t1
	ans += '\nafter %d sentences: %d'%(len(all_sentences), flaws(dt,val_sentences,feature_dict,tbank,history,weights) )
	ans += '\nafter %d sentences (Ne-only): %d'%(len(all_sentences), flaws(dt,val_sentences,feature_dict,tbank,history,weights,with_tags=False) )
	ans += '\ntotal %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4)

	return ans

def run_all(hist=1,tiny='.tiny'):
	#global all_sentences, val_sentences, feature_dict,history
	history = hist
	t1 = time()
	TRAIN_FILE = '../release3.2/data/train.data'+tiny
	VAL_FILE = '../release3.2/data/validate.data'+tiny
	print 'loading sentences'
	all_sentences, feature_dict = dp.process(TRAIN_FILE)
	val_sentences, _val_feat = dp.process(VAL_FILE)
	#pool = Pool(4)
	commands = [(all_sentences, val_sentences, feature_dict, history,i) for i in range(0,8)]
	#anses = pool.map(run_once,commands)
	anses = [run_once(com) for com in commands]
	print '\n\n'.join(anses)

def main(history=1,tiny='.tiny'):
	assert history >= 1, "use at least some history"
	t1 = time()
	TRAIN_FILE = '../release3.2/data/train.data.pre.small'
	VAL_FILE = '../release3.2/data/validate.data.pre.small'
	print 'loading tree bank'
	t2 = time()-t1
	tbank = dts.tbankparser()
	print 'loading sentences'
	dp._init_(tbank)
	all_sentences, feature_dict = dp.process_multi(TRAIN_FILE,history)
	val_sentences, _val_feat = dp.process_multi(VAL_FILE,history)
	t3 = time()-t1-t2
	print "features has been made"
	
	sp._init_(len(feature_dict),dts, False )
	out( ('SSE random weights, only Ne-tags',flaws(dts,val_sentences,feature_dict,tbank,history,with_tags=False)) )
	out( ( 'SSE random weights',flaws(dts,val_sentences,feature_dict,tbank,history) ) )
	t4 = time()
	weights = sp.train_perceptron(all_sentences, feature_dict, tbank, history)
	np.save('weights'+str(history)+tiny+'.npy',weights)
	t4 = time()-t4
	t1=time()-t1
	out( ( 'after %d sentences, only Ne-tags'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights,False) ) )
	out( ( 'after %d sentences'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights) ) )
	out( ( 'total %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4) ) )


#flaws pipe line 
#training struct 
#featre dict 
def flaws(dt,all_sentences,feature_dict,tbank,history,weight_matrix=None,with_tags=True):
	if weight_matrix is None:
		weight_matrix = sp.init_weights(len(feature_dict))
	E = 0.0
	for sentence in all_sentences:
		try:
			parsed_tree = tbank.parse(sentence.raw_sentence)

		#try:
			# """
			# ==== comment 1
			# hier loopt de code nog op de oude manier door de zin, dit moet dus via de nieuwe manier (zie comment 0 in structured_perceptron)
			# """

			histories = []
			target_feature_vectors = []

			if sp.golinear:
				#print parsed_tree
				context_words = [w.orth_ for w in dt.linear(parsed_tree) ]
				context_pos_tags = [w.tag_ for w in dt.linear(parsed_tree) ]
				context_tags = [sentence.words_tags[dt.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in dt.linear(parsed_tree)]
				target_feature_vectors = []
				for i,wrd in enumerate(context_words):
					if i < history:
						history_tags = tuple(['-TAGSTART-']+context_tags[0:i])
						history_words = ['-START-']+context_words[0:i]
						history_pos_tags = ['-POSTAGSTART-']+context_pos_tags[0:i]
					else:
						history_tags = context_tags[i-history:i]
						history_words = context_words[i-history:i]
						history_pos_tags = context_pos_tags[i-history:i]
					history_vectors = ('ph', [history_tags] )

					cur_idx = i
					prev_idx = cur_idx-1
					distance = 0
					if prev_idx >= 0:
						distance = parsed_tree[cur_idx].similarity(parsed_tree[prev_idx])


					target_feature_vectors.append( dp.construct_feature_vector(wrd, context_tags[i], 
							feature_dict, history_words, history, history_vectors, history_pos_tags, distance) )
					histories.append((history_words,history_pos_tags,distance))
			else:
				for i,wrd in enumerate(dt.dfirst(parsed_tree)):
					cur = wrd
					history_words = []
					history_tags = []
					history_pos_tags = []
					for j in range(history):
						par = cur.head
						if cur == par:
							parw = '-START-'
							idx = -1
							tag = '-TAGSTART-'
							pos = '-POSTAGSTART-'
							history_tags.insert(0,tag)
							history_words.insert(0,parw)
							history_pos_tags.insert(0,pos)
							break
						else:
							parw = par.orth_
							idx = dt.sen_idx(sentence.raw_sentence,par)
							tag = sentence.words_tags[idx][1]
							pos = par.tag_
							cur = par
							history_tags.insert(0,tag)
							history_words.insert(0,parw)
							history_pos_tags.insert(0,pos)
					history_vectors = ('ph',[history_tags] )
					cur_idx = dt.sen_idx(sentence.raw_sentence,wrd)
					
					for prev_idx,w in enumerate(iterloop(parsed_tree)):
						if w == wrd.head:
							break
					if wrd.head == wrd:
						prev_idx = -1

					distance = 0
					if prev_idx >= 0:
						distance = parsed_tree[cur_idx].similarity(parsed_tree[prev_idx])
					#else:
					#	prev_idx = dt.sen_idx(sentence.raw_sentence,wrd.head)
					cur_tag = sentence.words_tags[idx][1]
					target_feature_vectors.append( dp.construct_feature_vector(wrd.orth_, cur_tag, feature_dict,
									history_words, history, history_vectors, history_pos_tags, distance) )
					# hist_hist = []
					# for tag in all_tags:
					# 	hist_hist.append(
					# 		dp.construct_feature_vector(wrd.orth_,tag,feature_dict,history_words,history, history_vectors, history_pos_tags, distance)
					# 	)
					# histories.append(hist_hist)
					histories.append((prev_idx,history_words,history_pos_tags,distance))

				# """
				# /==== end comment 1
				# """
			if not with_tags:
				context_tags = None
			E = sp.test_perceptron_once(E, parsed_tree, feature_dict, 
						history, weight_matrix, histories, context_tags)
		
		except Exception as ex:
			log('flaw',sentence)
	return E

def log(f,m):
	logfile = open('stdlog.log','a')
	ts = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S -- ')
	logfile.write(ts+f+' -- '+str(m))
	logfile.write('\n')
	logfile.close()
def out(m):
	logfile = open('stdout.out','a')
	ts = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S -- ')
	logfile.write(ts+str(m))
	logfile.write('\n')
	logfile.close()


def test():
	TRAIN_FILE = '../release3.2/data/train.data.tiny'
	print 'loading sentences'
	t1=time()
	all_sentences, feature_dict = dp.process(TRAIN_FILE)
	t1=time()-t1
	t2=time()
	all_sentences, feature_dict = dp.process_multi(TRAIN_FILE,6)
	t2=time()-t2
	print t1,t2

if __name__ == '__main__':
	#log('test',(1,2,3))
	#test()
	main(2,'.pre.small')


# history=2;tiny='.tiny'
# t1 = time()
# TRAIN_FILE = '../release3.2/data/train.data'+ tiny 
# VAL_FILE = '../release3.2/data/validate.data'+tiny
# print 'loading sentences'
# all_sentences, feature_dict = dp.process(TRAIN_FILE,history)
# val_sentences, _val_feat = dp.process(VAL_FILE,history)
# t2 = time()-t1
# print 'loading tree bank'
# tbank = dts.tbankparser()

# t3 = time()-t1-t2
# sp._init_(len(feature_dict),dts )
# print 'SSE random weights, only Ne-tags',flaws(dts,val_sentences,feature_dict,tbank,history,with_tags=False)
# print 'SSE random weights',flaws(dts,val_sentences,feature_dict,tbank,history)
# t4 = time()
# weights = sp.train_perceptron(all_sentences, feature_dict, tbank, history)
# t4 = time()-t4
# t1=time()-t1
# print 'after %d sentences, only Ne-tags'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights,False)
# print 'after %d sentences'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights)
# print 'total %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4)