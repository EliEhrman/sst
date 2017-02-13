"""
This program takes the output from makefullmfccs.py and trains a cnn to determine whether a frame
has vad (voice activation detection) active or not.

As the cnn trains it takes the activations before a fully-connected and creates a knn file where
each row has a vad start and end followed by some 1000+ activation values

"""
from __future__ import print_function
import csv
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import os
import math


signalsfile_path = '/devlink2/data/stt/svadsigs.txt'
knn_path = '/devlink2/data/stt/vadknn.txt'
chkpoint_path = '/devlink2/data/stt/vadchk/chkpoint.dat'
b_skip_learning = False

batch_size = 256
num_steps = 100000
# num_samps = 100 # number of times DURING the run we reinitialize the sample and its random
max_time_sample = 5 # seconds
hop_length = 512
sr = 16000

max_num_spectrums = max_time_sample * sr / hop_length
spectrum_size = 13
num_input_channels = 3 # mfcc, delta and delta2
hops_per_sec = sr / hop_length
num_outputs = max_num_spectrums / 10

# matplotlib.use('TkAgg')

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable

	Returns:
	Variable Tensor
	"""
	# with tf.device('/cpu:0'):
	# 	var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	# return var
	return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
	name: name of the variable
	shape: list of ints
	stddev: standard deviation of a truncated Gaussian
	wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.

	Returns:
	Variable Tensor
	"""
	dtype = tf.float32
	var = _variable_on_cpu(
						name,
						shape,
						tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def data_init():
	sigfile = open(signalsfile_path, 'rb')
	images = []
	labels = []
	bordertimes = []
	num_rows = 0
	reader = csv.reader(sigfile, delimiter=',')
	for irow, row in enumerate(reader):
		rowpos = 2
		num_gold_words = int(row[rowpos])
		rowpos += num_gold_words + 1
		num_spectrums = int(row[rowpos])
		if num_spectrums >= max_num_spectrums:
			continue;
		# num_words_read += 1
		spectrum_size = int(row[rowpos+1])
		num_input_channels = int(row[rowpos+2])
		# num_spectrums_arr.extend([num_spectrums])
		rowpos += 3
		spectrums = []
		for ispec in range(num_spectrums):
			comps = []
			for icomp in range(spectrum_size):
				chans = []
				for ichan in range(num_input_channels):
					chans.append(float(row[rowpos]))
					rowpos += 1
				comps.append(chans)
			# spectrum = [float(sval) for sval in row[6+(ispec* data_size_per_spectrum):6+((ispec+1) * data_size_per_spectrum)]]
			spectrums.append(comps)
		labels.append([1.0 if act > (float(row[0]) * hops_per_sec) and act < (float(row[1]) * hops_per_sec) else 0.0
					   for act in range(max_num_spectrums)])
		bordertimes.append(row[:2])
		images.append(spectrums)
	sigfile.close()

	# max_num_spectrums = max(num_spectrums_arr)
	return images, labels, bordertimes

def create_graph(t_sigs_input):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 2],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(t_sigs_input, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 5, 1], strides=[1, 1, 3, 1],
						   padding='SAME', name='pool1')

	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 2, 4],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [4], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	# pool2
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 5, 1],
						   strides=[1, 1, 3, 1], padding='SAME', name='pool2')


	LASTSIZE = max_num_spectrums

	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		flatguy = tf.reshape(pool2, [batch_size, -1])
		dim = flatguy.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, LASTSIZE],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [LASTSIZE], tf.constant_initializer(0.1))
		unnorm = tf.nn.sigmoid(tf.matmul(flatguy, weights) + biases, name=scope.name)

	# activations = tf.nn.l2_normalize(unnorm, dim=1)

	return unnorm, flatguy
#

def create_feed(images, labels, startrow, t_sigs_input, t_labels):

	sigsize = 0
	full_images = []
	batch_labels = []
	for irow in range(startrow, startrow + batch_size):
		# start_pad_size = (max_num_spectrums - len(spectrums)) / 2
		# image = [[[0.0] * num_input_channels] * spectrum_size] * start_pad_size
		# image.extend(spectrums)
		end_pad_size = max_num_spectrums - len(images[irow])
		full_image = images[irow] + [[[0.0] * num_input_channels] * spectrum_size] * end_pad_size
		# images.append([[[val] for val in row] for row in image])
		full_images.append(full_image)
		batch_labels.append(labels[irow])

	# sigfile.close()
	return {t_sigs_input: full_images, t_labels: batch_labels}

def create_knn(sess, images, labels, bordertimes, t_sigs_input, t_truthlabels, t_flatguy):
	knn_file = open(knn_path, 'wb')
	knn_writer = csv.writer(knn_file, delimiter=',')
	startrow = 0
	while True:
		fd = create_feed(images, labels, startrow, t_sigs_input, t_truthlabels)
		flatguy = sess.run(t_flatguy, feed_dict=fd)
		for ivals, vals in enumerate(flatguy):
			knn_writer.writerow(bordertimes[startrow + ivals]+[str(val) for val in vals])
		startrow += batch_size
		if len(labels) <= (startrow + batch_size):
			break
	knn_file.close()


images, labels, bordertimes = data_init()
num_src_images = len(labels)

t_sigs_input = tf.placeholder(tf.float32, [batch_size, max_num_spectrums, spectrum_size, num_input_channels])
t_truthtimes = tf.placeholder(tf.float32, [batch_size, 2])
t_truthlabels = tf.placeholder(tf.float32, [batch_size, max_num_spectrums])

t_labels, t_flatguy = create_graph(t_sigs_input)

err = tf.reduce_mean((t_truthlabels - t_labels) ** 2)
train_step = tf.train.AdagradOptimizer(0.05).minimize(err)
train_step2 = tf.train.AdagradOptimizer(0.005).minimize(err)

# testnorm = tf.reduce_sum(tf.square(lastones), axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if os.path.exists(chkpoint_path+'.index'):
	saver = tf.train.Saver()
	saver.restore(sess, chkpoint_path)

learn_phase = 0
if not b_skip_learning:
	plotvec = []
	# plt.figure('Find the start and end of speech!')
	# plt.yscale('log')
	# plt.ion()
	# -sys.float_info.max
	for step in range(num_steps):
		startrow = random.randint(0, num_src_images-batch_size)
		fd = create_feed(images, labels, startrow, t_sigs_input, t_truthlabels)
		if step % (num_steps/1000) == 0:
			errval = math.sqrt(sess.run(err, feed_dict=fd))
			print('step: ', step, 'err: ', errval)
			vlabels, vinput = sess.run([t_labels, t_sigs_input], feed_dict=fd)
			if learn_phase == 0 and  errval < 0.2:
				print('Reducing error val')
				learn_phase = 1
			# print(labels[startrow])
			# print(vinput[0])
			# print(vlabels[0])
			# plotvec.append(errval)
			# plt.plot(plotvec)
			# plt.pause(0.05)
			if step != 0 and step % 1000 == 0:
				saver = tf.train.Saver()
				saver.save(sess, chkpoint_path)
				create_knn(sess, images, labels, bordertimes, t_sigs_input, t_truthlabels, t_flatguy)
		if learn_phase == 0:
			sess.run([train_step], feed_dict=fd)
		else:
			sess.run([train_step2], feed_dict=fd)

sess.close()

print('bye')
