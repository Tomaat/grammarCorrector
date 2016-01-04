import nltk
#from nltk.corpus import words
from nltk.corpus import brown
from random import shuffle 
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
with open ('../release3.2/data/full-dataset.txt') as datafile: # import sgml data-file
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
			
shuffle(new_data)

train_data = new_data[:int(len(new_data)*0.9)-1]
validate_data = new_data[int(len(new_data)*0.9):]


train_path = "train-data.data.pre"
validate_path = "validate-data.data.pre"

train_file = open(train_path,'w')
for block in train_data[0:500]:
	for ding in block:
		train_file.write(ding+'\n')
	train_file.write('\n')
train_file.close()


validate_file = open(validate_path,'w')
for block in validate_data[:50]:
	for ding in block:
		validate_file.write(ding+'\n')
	validate_file.write('\n')
validate_file.close()
