"""
Implementation of KNN in tensorflow for the purpose of speeding up python and taking advantage of the gpgpu

KNN is between a clip data item and a DB of word items - each of the same size (80 properties)

The goal is to find words likely to be present in a clip

Program tries to maximize gpupu utilization by keeping all daqta including randomization in tf vars

"""
import numpy as np
import tensorflow as tf
import csv
import random
import math
import time
import os

whwordsknn_path = '/devlink2/data/stt/whwordsknn.txt'
whclipsknn_path = '/devlink2/data/stt/whclipsknn.txt'
whwordsknn_fh = open(whwordsknn_path, 'rt')
whwordsknn_reader = csv.reader(whwordsknn_fh, delimiter=',')

allwords = []
words = []

data_start_col = 1

for row in whwordsknn_reader:
	words.append(row[0])
	allwords.append([float(d) for d in row[data_start_col:]])

whwordsknn_fh.close()

whclipsknn_fh = open(whclipsknn_path, 'rt')
whclipsknn_reader = csv.reader(whclipsknn_fh, delimiter=',')

allclips = []
clip_words = []

for row in whclipsknn_reader:
	numwords = int(row[0])
	clip_words.append(row[1:numwords+1])
	allclips.append([float(d) for d in row[numwords+1:]])

whclipsknn_fh.close()

labels_unique = set(words)
num_labels = len(labels_unique)

def norm_input(allrows, allvals):
	allcols = zip(*allrows)
	mins = [min(col) for col in allcols]
	maxs = [max(col) for col in allcols]
	for row in allrows:
		vals = []
		for icol, col in enumerate(row):
			if maxs[icol] == mins[icol]:
				vals.append(0.0)
			else:
				vals.append((col - mins[icol]) / (maxs[icol] - mins[icol]))
		sqrtsum = math.sqrt(sum([x ** 2 for x in vals]))
		allvals.append([x / sqrtsum for x in vals])


wordvals = []
norm_input(allwords, wordvals)
clipvals = []
norm_input(allclips, clipvals)

reclen = len(wordvals[0])
numwordrecs = len(wordvals)
numcliprecs = len(clipvals)

t_wordvals = tf.placeholder(shape=[numwordrecs, reclen], dtype=tf.float32)
t_clipvals = tf.placeholder(shape=[numcliprecs, reclen], dtype=tf.float32)

num_ks = 100
AllCDs = tf.matmul(t_clipvals, t_wordvals, transpose_b=True)
t_bestCDs, t_bestCDIDxs = tf.nn.top_k(AllCDs, num_ks)

# tf.assign(t_wordvals, wordvals)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

fd = {t_clipvals: clipvals, t_wordvals: wordvals}
v_bestCDIDxs = sess.run(t_bestCDIDxs, feed_dict=fd)

sess.close()

score = 0.0
num_unique_found = 0.0
for iclip, clipIDs in enumerate(v_bestCDIDxs):
	wordsfound = set([words[clipID] for clipID in clipIDs])
	num_unique_found += float(len(wordsfound))
	num_found = 0.0
	for clipword in clip_words[iclip]:
		if clipword in wordsfound:
			num_found += 1.0
	score += num_found / len(clip_words[iclip])
score /= len(v_bestCDIDxs)
num_unique_found /= len(v_bestCDIDxs)





print 'score:', score, 'avg unique found:',  num_unique_found