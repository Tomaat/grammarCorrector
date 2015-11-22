class Features:
	def __init__(self,):
	







	def _normalize(self, word):
		'''Normalization used in pre-processing.
		- All words are lower cased
		- Digits in the range 1800-2100 are represented as !YEAR;
		- Other digits are represented as !DIGITS
		:rtype: str
		'''
		if '-' in word and word[0] != '-':
	        return '!HYPHEN'
	    elif re.match(r"[^@]+@[^@]+\.[^@]+", email):
	    	return '!EMAIL'
	    elif word.isdigit() and len(word) == 4:
	        return '!YEAR'
	    elif word.isdigit() and len(word) >= 6 and len(word) <= 12:
	        return '!PHONENUMBER'
	    elif word[0].isdigit():
	        return '!DIGITS'
	    else:
	        return word.lower()

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