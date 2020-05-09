import os
import sys

f = open('./../test_sents.txt','r')

M = {}

for i,line in enumerate(f):
	line = line.strip('\n')
	words = line.split(' ')
	leng = int(min(8, len(words)))
	print (' '.join(words[:leng]))
'''
for i,line in enumerate(f):
	leng = int(min(75, len(line)-1))
	print (line[:leng])
'''
'''
test_sents = open('./DUC2004/input.txt').read().split('\n')[:-1]
test_sents = [preprocess.filter_words(sent.split()) for sent in test_sents]

for line in test_sents:
	#print ' '.join(line)
	print (M[' '.join(line)])
'''
