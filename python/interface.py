
def flag_error(sentence):
	return 'errors'

def correct(sentence,errors):
	return 'correct sentence'

def train(data):
	return 'model'

def test(data,model):
	return 'complexity/accuracy'


"""
For the structure perceptron:

Input: dictionary with all feature vectors for every sentence

e.g. dict = {0: [feature vectors tuple error 1], [feature vectors tuple error 2],
			1:	[feature vectors tuple error 1], [feature vectors tuple error 2],
			etc. }

So, what's 'feature vectors tuple'?
Feature vector with different previous tags for every error, combined with the number of the previous error, so:
(feature vector, number previous error)

This means that we need to number the errors we're looking at. But this way it's easier to proceed through all
feature vectors at the end of viterbi.


"""