import nltk
from nltk.util import ngrams
import pprint

def find_ngrams_dep(filename, n):

	total_ngram_count = 0
	ngram_freq_dict = {}

	f = open(filename)
	for line in f.readlines():
		if line != '\n':
			split = line.split()
			ngrams_dep = ngrams(split, n)
			
			for i in ngrams_dep:
				total_ngram_count += 1
				old = ngram_freq_dict.get(i,0)
				old += 1
				ngram_freq_dict[i] = old

	return ngram_freq_dict, total_ngram_count


if __name__ == '__main__':
	filename = "test.txt"
	ngram_freq_dict, ngram_count = find_ngrams_dep(filename, 4)
	pprint.pprint(ngram_freq_dict)