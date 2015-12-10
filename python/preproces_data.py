import nltk
#from nltk.corpus import words
from nltk.corpus import brown

word_dict = {}
english_words = []
english_words_raw = brown.words() #words.words('en')
print 'LOWER WORDS'
for word_upper in english_words_raw:
	word_lower = word_upper.lower()
	english_words.append(word_lower)

for word in english_words:
	word_dict[word] = 1

punctuation = ['.', ",", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "=", "?", ";", "]", "[", "{", "}"]

for punct in punctuation:
	word_dict[punct] = 1

#conll14st-preprocessed.m2
with open ('../release3.2/data/validate.data') as datafile: # import sgml data-file
	data_lines = datafile.readlines()
	data_raw = [p.split('\n') for p in ''.join(data_lines).split('\n\n')]

new_data = []
for block in data_raw:
	sentence = block[0][2:]
	words = sentence.split()
	add = False
	for split_word in words:
		if split_word.lower() not in word_dict:
			add = True
			break
	if add:
		new_data.append(block)
			

filename = "../release3.2/data/validate.data.pre"
f = open(filename,'w')
for block in new_data:
	for ding in block:
		f.write(ding+'\n')
	f.write('\n')
