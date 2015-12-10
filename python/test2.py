import depTree as dt

iterloop = dt.dfirst

def test(tbank,sentence,history):
	parsed_tree = tbank.parse(sentence.raw_sentence)
	histories = []
	target_feature_vectors = []
	for i,wrd in enumerate(iterloop(parsed_tree)):
		cur = wrd
		history_words = ['']*history
		history_tags = ['']*history
		history_pos_tags = ['']*history
		for j in range(1,history+1):
			par = cur.head
			if cur == par:
				parw = '-START-'
				idx = -1
				tag = '-TAGSTART-'
				pos = '-POSTAGSTART-'
			else:
				parw = par.orth_
				tag = sentence.words_tags[idx][1]
				pos = par.tag_
				cur = par
			if j == 1:
				prev_idx = idx
			history_tags[-j] = tag
			history_words[-j] = parw
			history_pos_tags[-j] = pos
		history_vectors = ('ph',[history_tags] )
		cur_idx = dt.sen_idx(sentence.raw_sentence,par)
		cur_tag = sentence.words_tags[idx][1]
		target_feature_vectors.append( dp.construct_feature_vector(wrd.orth_, cur_tag, feature_dict,
						history_words, history, history_vectors, history_pos_tags) )
		histories.append((prev_idx,history_words,history_pos_tags))