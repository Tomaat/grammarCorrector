class FeaturesVector:
	def __init__(self, word, history, ):
		self.word = _normalize(word)
		self.vector = # maak nparray van lengte alle vectoren
		

	
		

	def _make_tagdict(self, sentences):
		'''Make a tag dictionary for single-tag words.'''
		counts = defaultdict(lambda: defaultdict(int))
		for words, tags in sentences:
		    for word, tag in zip(words, tags):
		        counts[word][tag] += 1
		        self.classes.add(tag)
		freq_thresh = 20
		ambiguity_thresh = 0.97
		for word, tag_freqs in counts.items():
		    tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
		    n = sum(tag_freqs.values())
		    # Don't add rare words to the tag dictionary
		    # Only add quite unambiguous words
		    if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
		        self.tagdict[word] = tag