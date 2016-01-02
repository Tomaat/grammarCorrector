from time import time
import dependencyTree as dto
import depTree as dts
import structured_perceptron as sp
import dataparser as dp
from multiprocessing import Pool
import datetime
import numpy as np

def main(history=1,tiny='.tiny',tbank = None):
	assert history >= 1, "use at least some history"
	t1 = time()
	TRAIN_FILE = '../release3.2/final_data/train-data.pre'
	VAL_FILE =   '../release3.2/final_data/validate-data.pre'
	print 'loading tree bank'
	t2 = time()-t1
	if tbank is None:
		tbank = dts.tbankparser()
	print 'loading sentences'
	dp._init_(tbank)
	all_sentences, feature_dict = dp.process(TRAIN_FILE,history)
	val_sentences, _val_feat = dp.process(VAL_FILE,history)
	t3 = time()-t1-t2
	print "features has been made"
	print "init perceptron"
	sp._init_(len(feature_dict),dts, False)
	print "end init"
	out( ('SSE random weights, only Ne-tags',flaws(dts,val_sentences,feature_dict,tbank,history,with_tags=False)) )
	print "SSE random weights, only Ne-tags"
	out( ( 'SSE random weights',flaws(dts,val_sentences,feature_dict,tbank,history) ) )
	print "SSE random weight"
	t4 = time()
	print "learning"
	weights = sp.train_perceptron(all_sentences, feature_dict, tbank, history)
	np.save('weights'+str(history)+tiny+'.npy',weights)
	t4 = time()-t4
	print weights.shape
	t1=time()-t1
	print "validating"
	out( ( 'after %d sentences, only Ne-tags'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights,False) ) )
	out( ( 'after %d sentences'%(len(all_sentences)), flaws(dts, val_sentences,feature_dict,tbank,history,weights) ) )
	out( ( 'total %f sec (loading: %f, %f; training: %f'%(t1,t2,t3,t4) ) )
	return feature_dict,weights

#flaws pipe line 
#training struct 
#featre dict 
def flaws(dt,all_sentences,feature_dict,tbank,history,weight_matrix=None,with_tags=True):
	if weight_matrix is None:
		weight_matrix = sp.init_weights(len(feature_dict))
	E = 0.0
	counter_flaw = 1
	for sentence in all_sentences:
		current_time = time()
		print "at flaws, sentence ",counter_flaw
		counter_flaw += 1
		try:
			parsed_tree = tbank.parse(sentence.raw_sentence)

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
					histories.append((prev_idx,history_words,history_pos_tags,distance))
			else:
				context_tags = [sentence.words_tags[dt.sen_idx(sentence.raw_sentence, wrd)][1] for wrd in dt.dfirst(parsed_tree)]
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
					
					for prev_idx,w in enumerate(dt.dfirst(parsed_tree)):
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
			#print histories

			E = sp.test_perceptron_once(E, parsed_tree, feature_dict, 
						history, weight_matrix, histories, context_tags)
		
		except Exception as ex:
			log('flaw',sentence)

		print time() - current_time
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