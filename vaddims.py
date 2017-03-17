"""
This program shrinks the vadvec.py output from 1000+ properties per audio file to 40-80

The matrix that transforms the large to small vector is learned such that the distance between
two arbitrary audio files is in accordance with the overlap of their silent and speech activated
component.

This program does not use a feed dict at all. It keeps all the data in tensors across all steps
The different selection of data is controlled by tf generation of random numbers

"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
import random
import math
import matplotlib.pyplot as plt

knn_large_fname = '/devlink2/data/stt/vadknn.txt'
knn_small_fname = '/devlink2/data/stt/vadknn_small.txt'

b_fname_included = True
num_steps = 100000
batch_size = 64
output_size = 40
data_start_col = 2
hop_length = 512
sr = 16000
hops_per_sec = float(sr) / float(hop_length)
rsize = batch_size ** 2
vars_per_hop = 8 # N.B. Must maintain in manual sync with vadvec.py, the number of outputs before reshape

#the following is for illustration of the algorithm only
def calc_distance(start1, end1, start2, end2, max_hops):
	max_time = float(max_hops) / hops_per_sec
	agree = 0
	agree += min(start1, start2)
	overlap_vad = min(end1, end2) - max(start1, start2)
	if overlap_vad > 0:
		agree += int(overlap_vad)
	agree += max_time - max(end1, end2)
	return agree / max_time

def tensor_distance(rstarts1, rends1, rstarts2, rends2, max_time):
	starts_gt = tf.greater(rstarts1, rstarts2)
	ends_gt = tf.greater(rends1, rends2)
	first_start = tf.select(starts_gt, rstarts2, rstarts1)
	# last_start = tf.select(tf.logical_not(starts_gt), rstarts1, rstarts2)
	last_start = tf.select(starts_gt, rstarts1, rstarts2)
	first_end = tf.select(ends_gt, rends2, rends1)
	last_end = tf.select(ends_gt, rends1, rends2)
	boverlap = tf.greater(first_end, last_start)
	overlap = tf.select(boverlap, tf.subtract(first_end, last_start), tf.zeros([rsize]))
	t_agree = tf.add_n([first_start, overlap, tf.subtract(max_time, last_end)])
	return tf.div(t_agree, max_time)
	# return tf.div(t_agree, max_time), starts_gt, ends_gt, first_start, last_start, first_end, last_end, boverlap, overlap, t_agree

def create_knn_file(sess, W, allvals, starts, ends):
	TheMatrix = sess.run(W)
	# TheMatrix = np.clip(TheMatrix, 0, 10.0)
	print('The Matrix: ', TheMatrix)
	output = np.dot(allvals, TheMatrix)
	csvfile = open(knn_small_fname, 'wb')
	writer = csv.writer(csvfile, delimiter=',')
	for irow, row in enumerate(output):
		data = [str(d) for d in row]
		label_prefix = [str(starts[irow])]
		label_prefix.append(str(ends[irow]))
		writer.writerow(label_prefix + data)
	csvfile.close()


csvfile = open(knn_large_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')

allrows = []
starts = []
ends = []



for row in reader:
	starts.append(float(row[0]))
	ends.append(float(row[1]))
	vals = [float(d) for d in row[data_start_col:]]
	allrows.append(vals)

csvfile.close()

allcols = zip(*allrows)
mins = [min(col) for col in allcols]
maxs = [max(col) for col in allcols]
allvals = []
for row in allrows:
	vals = [(col - mins[icol]) / (maxs[icol] - mins[icol]) if (maxs[icol] - mins[icol]) > 0.0 else 0.0 for icol, col in enumerate(row)]
	sqrt = sum([x ** 2 for x in vals])
	allvals.append([x / sqrt for x in vals])
# print mins, maxs

reclen = len(allrows[0])
numrecs = len(allrows)
max_time = float(reclen / vars_per_hop) / float(hops_per_sec)

callvals = tf.Variable(tf.constant(allvals))
cstarts = tf.Variable(tf.constant(starts))
cends = tf.Variable(tf.constant(ends))

r1arr = tf.Variable(tf.random_uniform([rsize], minval=0, maxval=batch_size, dtype=tf.int32))
r2arr = tf.Variable(tf.random_uniform([rsize], minval=0, maxval=batch_size, dtype=tf.int32))

t_batch_begin = tf.Variable(tf.random_uniform([], minval=0, maxval=numrecs - batch_size, dtype=tf.int32))
# t_batch_begin = tf.constant([0])
x = tf.slice(input_=callvals, begin=[t_batch_begin,0], size=[batch_size, reclen])
tstarts = tf.slice(cstarts, [t_batch_begin], [batch_size])
tends = tf.slice(cends, [t_batch_begin], [batch_size])
# starts = tf.Variable(tf.constant(starts))
W = tf.Variable(tf.random_normal([reclen, output_size], 0.01/float(reclen*output_size)))
b = tf.Variable(tf.random_uniform([output_size], 0.3))
x1 = tf.gather(x, r1arr)
x2 = tf.gather(x, r2arr)
y = tf.matmul(x, tf.clip_by_value(W, 0.0, 10.0)) # + b
y1 = tf.gather(y, r1arr)
y2 = tf.gather(y, r2arr)
rstarts1 = tf.gather(tstarts, r1arr)
rstarts2 = tf.gather(tstarts, r2arr)
rends1 = tf.gather(tends, r1arr)
rends2 = tf.gather(tends, r2arr)

# dist1 = tf.reduce_sum(tf.squared_difference(x1, x2), axis=1)
# dist2 = tf.reduce_sum(tf.squared_difference(y1, y2), axis=1)
# dist1 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
# dist1, starts_gt, ends_gt, first_start, last_start, first_end, last_end, boverlap, overlap, t_agree = tensor_distance(rstarts1, rends1, rstarts2, rends2, max_time)
dist1 = tensor_distance(rstarts1, rends1, rstarts2, rends2, max_time)
dist2 = tf.reduce_sum(tf.multiply(y1, y2), axis=1)
err = tf.reduce_mean((dist1 - dist2) ** 2)
train_step = tf.train.AdagradOptimizer(0.0005).minimize(err)
train_step2 = tf.train.AdagradOptimizer(0.00005).minimize(err)
# dist = tf.reduce_sum(tf.squared_difference(x1, x2))

TheMatrix = np.empty([reclen, output_size], dtype=np.float64)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# plotvec = []
# plt.figure('vaddims - shortening the distance of vad!')
# plt.yscale('log')
# plt.ion()
learn_phase = 0
for step in range(num_steps+1):
	sess.run(tf.variables_initializer([r1arr, r2arr, t_batch_begin]))
	# print(sess.run([r1arr, r2arr, t_batch_begin]))
	# print(sess.run([dist1, rstarts1, rends1, rstarts2, rends2]))
	# print(sess.run([starts_gt, ends_gt, first_start, last_start, first_end, last_end, boverlap, overlap, t_agree]))
	# if step is 0:
	# 	print 'W\n', sess.run([W], feed_dict=fd)
	if step % (num_steps / 100) == 0:
		errval1 = math.sqrt(sess.run(err))
		print('step:', step, ', err : ', errval1)
		if learn_phase == 0 and errval1 < 0.1:
			print('reduce learn rate to', 0.00005 )
			learn_phase = 1

	if step % 10000 == 0:
		create_knn_file(sess, W, allvals, starts, ends)
		# plotvec.append(errval1)
		# plt.plot(plotvec)
		# plt.pause(0.05)

	if learn_phase == 0:
		sess.run([train_step])
	else:
		sess.run([train_step2])


sess.close()

print('done')

# while True: plt.pause(0.5)




