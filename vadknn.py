"""
Basic implementation of KNN in tensorflow for the purpose of speedingup python and taking advantage of the gpgpu

Clusters code present but only activated if b_use_clusters is True
However, for now, all candidates and not just the clusters are passed in

This version is a regression (not classification) on the start and end values of full audio clips

This program illustrates a move to feeding very little and keeping as much as possible in tensors
"""
import numpy as np
import tensorflow as tf
import csv
import random
import math
import time
import os

num_centroids = 2
num_cluster_iters = 2
num_clusters_to_get = 2
num_ks = 20


knn_in_fname = '/devlink2/data/stt/vadknn_small.txt'
csvfile = open(knn_in_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')
b_use_clusters = False

allrows = []
starts = []
ends = []

data_start_col = 2

for row in reader:
	starts.append(float(row[0]))
	ends.append(float(row[1]))
	vals = [float(d) for d in row[data_start_col:]]
	allrows.append(vals)

csvfile.close()

labels_unique = set(starts)
num_labels = len(labels_unique)

allcols = zip(*allrows)
mins = [min(col) for col in allcols]
maxs = [max(col) for col in allcols]
allvals = []
for row in allrows:
	vals = []
	for icol, col in enumerate(row):
		vals.append((col - mins[icol]) / (maxs[icol] - mins[icol]))
	sqrtsum = math.sqrt(sum([x ** 2 for x in vals]))
	allvals.append([x / sqrtsum for x in vals])

reclen = len(allvals[0])
numrecs = len(allvals)

random.seed(0)
shuffle_stick = [i for i in range(numrecs)]
random.shuffle(shuffle_stick)
allvals = [allvals[i] for i in shuffle_stick]
starts = [starts[i] for i in shuffle_stick]
ends = [ends[i] for i in shuffle_stick]

lasttrain = 4 * numrecs / 5
trainvals = allvals[:lasttrain]
testvals = allvals[lasttrain:]

if b_use_clusters:
	centroids = []
	for icent in range(num_centroids):
		centroids.append(trainvals[random.randint(0, len(trainvals))])

	for iiter in range(num_cluster_iters):
		new_centroids = [[0.0 for val in centroid] for centroid in centroids]
		num_in_cluster = [0 for i in range(num_centroids)]

		for dbitem in trainvals:
			CDs = [sum([val * dbitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids ]
			i_best_centroid = np.argmax(np.asarray(CDs))
			new_centroids[i_best_centroid] = [new_centroids[i_best_centroid][ival] + val for ival, val in enumerate(dbitem)]
			num_in_cluster[i_best_centroid] += 1

		centroids = [[val / num_in_cluster[icentroid] if num_in_cluster[icentroid] > 0  else 0.0 for val in centroid] for icentroid, centroid in enumerate(new_centroids)]

	clusters = [[] for i in range(num_centroids)]
	# labels_for_clusters = [[] for i in range(num_centroids)]

	for iitem, dbitem in enumerate(trainvals):
		CDs = [sum([val * dbitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids]
		i_best_centroid = np.argmax(np.asarray(CDs))
		clusters[i_best_centroid].append(iitem)
		# labels_for_clusters[i_best_centroid].append(labels[iitem])

train_size = lasttrain
test_size = numrecs - train_size

rank_sum = sum([1.0/float(r) for r in range(2, num_ks+2)])


trainx = tf.placeholder(tf.float32, [train_size, reclen])
testx = tf.placeholder(tf.float32, [test_size, reclen])
trains = tf.placeholder(tf.float32, [train_size])
traine = tf.placeholder(tf.float32, [train_size])
tests = tf.placeholder(tf.float32, [test_size])
teste = tf.placeholder(tf.float32, [test_size])
AllCDs = tf.matmul(testx, trainx, transpose_b=True)
bestCDs, bestCDIDxs = tf.nn.top_k(AllCDs, num_ks)
rank = tf.range(2, num_ks+2, dtype=tf.float32)

ranked_starts = tf.gather(trains, bestCDIDxs)
w_ranked_starts = tf.div(ranked_starts, rank)
sum_starts = tf.reduce_sum(w_ranked_starts, axis=1)
predicted_starts = tf.div(sum_starts, rank_sum)
# predicted_ends = tf.div(tf.reduce_sum(tf.div(tf.gather(traine, bestCDIDxs), rank), axis=1), rank_sum)

ranked_ends = tf.gather(traine, bestCDIDxs)
w_ranked_ends = tf.div(ranked_ends, rank)
sum_ends = tf.reduce_sum(w_ranked_ends, axis=1)
predicted_ends = tf.div(sum_ends, rank_sum)

errs = tf.sqrt(tf.reduce_mean((tests - predicted_starts)**2))
erre = tf.sqrt(tf.reduce_mean((teste - predicted_ends)**2))
err = (errs + erre)/2

ranked_ends = tf.gather(traine, bestCDIDxs)
# CDs = tf.reduce_sum(tf.mul(testx, trainx), axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fd = {trainx: trainvals, testx: testvals, trains: starts[:lasttrain], tests: starts[lasttrain:], traine: ends[:lasttrain], teste: ends[lasttrain:]}
verr = sess.run(err, feed_dict=fd)
print('err:', verr)
# print('predicted:', sess.run(predicted_starts, feed_dict=fd))
# print('truth starts:', sess.run(tests, feed_dict=fd))
# print('predicted:', sess.run(predicted_ends, feed_dict=fd))
# print('truth ends:', sess.run(teste, feed_dict=fd))


"""
num_errors = 0
num_steps = 1
for step in range(num_steps):
	for itestitem, testitem in enumerate(testvals):
		cand_ids = []
		if b_use_clusters:
			CDs = [sum([val * testitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids]
			npCDs = np.asarray(CDs)
			for icluster in range(num_clusters_to_get):
				i_best_centroid = np.argmax(npCDs)
				cand_ids = cand_ids + [i for i in clusters[i_best_centroid]]
				npCDs = np.delete(npCDs, i_best_centroid)
		else:
			cand_ids = [i for i in range(len(trainvals))] # Note! Not actually used


		fd = {trainx: trainvals, testx: testitem}
		bestCDIDxs_vals, = sess.run([bestCDIDxs], feed_dict=fd)

		vote_board = { lval : 0.0 for lval in labels_unique}
		for iidx, idx in enumerate(bestCDIDxs_vals):
			vote_board[starts[idx]] += 1.0 / float(iidx + 1)

		top_label = 'unfounded'
		best_score = -1.0
		for lval in vote_board:
			if (vote_board[lval] > best_score) :
				best_score = vote_board[lval]
				top_label = lval
		if (top_label != starts[itestitem + lasttrain]):
			num_errors += 1

	print num_errors, ' errors out of ', len(testvals), 'accuracy = ', (len(testvals) - num_errors) * 100.0 / len(testvals), '%'
"""
sess.close()






print 'done'