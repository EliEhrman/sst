from __future__ import print_function
import csv
import numpy as np
import tensorflow as tf
import random
import os
import math
import librosa

wordsigs_path = '/devlink2/data/stt/ssignalsdb.txt'
wordfiles_path = '/devlink2/data/stt/testdb.txt'
chkpoint_path = '/devlink2/data/stt/chk/chkwordwhen.dat'

batch_size = 128 # 8 #
num_steps = 100000 # 100000 #
c_num_spectra = 13
c_num_channels = 3 # mfcc, delta and delta2
max_word_examples = 5 # 3 # how many words to find for each example
max_clips_to_read = 10000 # 10000 # not really how many clips but how many rows from word clip file
max_words_to_read = 100000 # 100000 #

max_word_time = 1.2 # seconds
max_clip_time = 3.0 # seconds
samples_per_frame = 512
sr = 16000 #sr is sampling rate i.e. samples per second
max_word_frames = int(max_word_time * sr / samples_per_frame)
max_clip_frames = int(max_clip_time * sr / samples_per_frame)

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
	var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def sigsread(row, rowpos, num_spectrums, max_frames):
	if c_num_spectra != int(row[rowpos + 1]):
		print('Error: incorrect number of input channels.Program aborting.')
		exit()
	if int(row[rowpos + 2]) != c_num_channels:
		print('Error: incorrect number of input channels.Program aborting.')
		exit()
	# num_spectrums_arr.extend([num_spectrums])
	rowpos += 3
	spectrums = []
	for ispec in range(num_spectrums):
		comps = []
		for icomp in range(c_num_spectra):
			chans = []
			for ichan in range(c_num_channels):
				chans.append(float(row[rowpos]))
				rowpos += 1
			chans = np.asarray(chans)
			comps.append(chans)
		# spectrum = [float(sval) for sval in row[6+(ispec* data_size_per_spectrum):6+((ispec+1) * data_size_per_spectrum)]]
		comps = np.asarray(comps)
		spectrums.append(comps)
	spectrums = np.asarray(spectrums)
	data_size = len(spectrums)
	start_pad_size = (max_frames - data_size) / 2
	image = np.zeros([max_frames, c_num_spectra, c_num_channels], dtype=np.float32)
	image[start_pad_size:start_pad_size + data_size] = spectrums
	return image




def data_input():
	l_clip_sigs = []
	wordsigs = []
	starts = []
	ends = []
	clip_start_pad_sizes = []
	clip_words_on = []
	clipwords = []
	clip_sig_ids = []
	clipfile_dict = {}
	wordsigs_dict = dict()
	wordfiles_fh = open(wordfiles_path, 'rt')
	wordfiles_reader = csv.reader(wordfiles_fh, delimiter=',')

	for irow, row in enumerate(wordfiles_reader):
		if irow > max_clips_to_read:
			break
		fname = row[0]
		id = clipfile_dict.get(fname, -2)
		if id != -2:
			continue;

		clipfile_dict[fname] = -1
		y, sr = librosa.load(row[0], sr=None)
		if y.size == 0:
			print('row %d produced no data.', irow)
			continue

		S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
		if np.shape(S)[1] > max_clip_frames:
			continue
		log_S = librosa.logamplitude(S, ref_power=np.max)
		mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
		delta_mfcc = librosa.feature.delta(mfcc)
		delta2_mfcc = librosa.feature.delta(mfcc, order=2)
		spectrums = np.dstack((mfcc.T, delta_mfcc.T, delta2_mfcc.T))
		data_size = len(spectrums)
		start_pad_size = (max_clip_frames - data_size) / 2
		image = np.zeros([max_clip_frames,  c_num_spectra, c_num_channels], dtype=np.float32)
		image[start_pad_size:start_pad_size+data_size] = spectrums
		l_clip_sigs.append(image)
		clip_start_pad_sizes.append(start_pad_size)

		clipfile_dict[fname] = len(l_clip_sigs) - 1

	clip_sigs = np.asarray(l_clip_sigs)
	wordfiles_fh.seek(0)
	wordfiles_reader = csv.reader(wordfiles_fh, delimiter=',')

	irow = 0
	for irow, row in enumerate(wordfiles_reader):
		if irow > max_clips_to_read:
			break
		fname = row[0]
		id = clipfile_dict.get(fname, -2)
		if id == -2:
			print('strange error: File', fname, 'is not in file names dictionary.')
			exit()
		if id == -1:
			continue
		clip_sig_ids.append(id)
		clipwords.append(row[2]) # the middle word is the recorded word
		word_frame_start = clip_start_pad_sizes[id] + int(float(row[4]) * sr / samples_per_frame)
		word_frame_end = clip_start_pad_sizes[id] + int(float(row[5]) * sr / samples_per_frame)
		# if word_frame_end == word_frame_start:
		# 	word_frame_end += 1
		clip_word_on = np.zeros(max_clip_frames, dtype=np.float32)
		clip_word_on[word_frame_start:word_frame_end+1] = 1.0 # np.ones(1+word_frame_end-word_frame_start)
		clip_words_on.append(clip_word_on)
		# print(irow, id, word_frame_start, word_frame_end)

	wordfiles_fh.close()

	# ws = []
	wordsigs_fh = open(wordsigs_path, 'rb')
	wordsigs_reader = csv.reader(wordsigs_fh, delimiter=',')
	rows_so_far = 0
	for irow, row in enumerate(wordsigs_reader):
		if irow > max_words_to_read:
			break
		the_word = row[1]
		rowpos = 3
		# ws.append(row)
		num_spectrums = int(row[rowpos])
		if num_spectrums >= max_word_frames:
			continue;
		spectrums = sigsread(row, rowpos, num_spectrums, int(max_word_frames))
		wordsigs.append(spectrums)
		allrefs = wordsigs_dict.get(the_word, [])
		allrefs.append(rows_so_far)
		wordsigs_dict[the_word] = allrefs
		rows_so_far += 1
	wordsigs_fh.close()
	num_words = len(wordsigs)
	wordsigs = np.asarray(wordsigs)

	pair_clip_ids = []
	pair_word_ids = []
	pair_frames_on = []
	for iclip, word in enumerate(clipwords):
		reflist = wordsigs_dict.get(word, [])
		if len(reflist) == 0: continue
		random.shuffle(reflist)
		numtrue = 0
		for iref, aref in enumerate(reflist):
			if iref > max_word_examples:
				break
			pair_clip_ids.append(clip_sig_ids[iclip])
			pair_word_ids.append(aref)
			pair_frames_on.append(clip_words_on[iclip])

	shuffle_stick = [i for i in range(len(pair_clip_ids))]
	random.shuffle(shuffle_stick)
	pair_clip_ids = [pair_clip_ids[i] for i in shuffle_stick]
	pair_word_ids = [pair_word_ids[i] for i in shuffle_stick]
	pair_frames_on = [pair_frames_on[i] for i in shuffle_stick]

	return clip_sigs, wordsigs, pair_clip_ids, pair_word_ids, pair_frames_on

def create_graph(wordsigs, clipsigs, drop_keep_rate):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 8],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(wordsigs, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm1')

	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 8, 16],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
						   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	MERGESIZE = max_clip_frames

	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, MERGESIZE],
											  stddev=0.04, wd=0.004)
		biases = tf.get_variable('biases', [MERGESIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		words_final = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 8],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(clipsigs, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)

	# pool1
	pool4 = tf.nn.max_pool(conv4, ksize=[1, 1, 5, 1], strides=[1, 1, 3, 1],
						   padding='SAME', name='pool4')
	# norm1
	norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm4')

	with tf.variable_scope('conv5') as scope:
		kernel = _variable_with_weight_decay('weights',
											shape=[5, 5, 8, 16],
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(norm4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)

	# norm2
	norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool5 = tf.nn.max_pool(norm5, ksize=[1, 1, 5, 1],
						   strides=[1, 1, 3, 1], padding='SAME', name='pool2')

	with tf.variable_scope('local6') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		clips_flat = tf.reshape(pool5, [batch_size, -1])
		dim = clips_flat.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, MERGESIZE],
											  stddev=0.04, wd=0.004)
		biases = tf.get_variable('biases', [MERGESIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		clips_final = tf.nn.relu(tf.matmul(clips_flat, weights) + biases, name=scope.name)

	clip_weights = _variable_with_weight_decay('clip_weights', shape=[MERGESIZE],
											  stddev=0.04, wd=0.004)
	word_weights = _variable_with_weight_decay('word_weights', shape=[MERGESIZE],
											  stddev=0.04, wd=0.004)
	combined = tf.add(tf.mul(clip_weights, tf.nn.dropout(clips_final, drop_keep_rate)),
					  tf.mul(word_weights, tf.nn.dropout(words_final, drop_keep_rate)))

	chokepoint = 40

	Wfc1 = _variable_with_weight_decay('wfc1', shape=[MERGESIZE, chokepoint],
										  stddev=0.04, wd=0.004)
	Bfc1 = tf.get_variable('bfc1', [chokepoint], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

	fc1 = tf.nn.l2_normalize(tf.nn.dropout(tf.nn.relu(tf.matmul(combined, Wfc1) + Bfc1, name='fc1'), drop_keep_rate), dim=1)


	Wfc2 = _variable_with_weight_decay('wfc2', shape=[chokepoint, MERGESIZE],
										  stddev=0.04, wd=0.004)
	Bfc2 = tf.get_variable('bfc2', [MERGESIZE], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

	final = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(fc1, Wfc2) + Bfc2, name='fc2'), dim=1)

	return final

def do_test(sess, err, max_train, num_pairs, ph_batch_begin, op_batch_begin_input, batch_size):
	errsum = 0.0
	num_runs = 0
	for batch_begin in range(max_train, num_pairs-batch_size, batch_size):
		sess.run(op_batch_begin_input, feed_dict={ph_batch_begin:batch_begin})
		errsum += math.sqrt(sess.run(err))
		num_runs += 1.0
	print ('err on validation set:', errsum / num_runs)


# clip_sigs = np.empty((0, max_clip_frames, c_num_spectra, c_num_channels), dtype=np.float32)
clip_sigs, word_sigs, pair_clip_ids, pair_word_ids, pair_frames_on = data_input()

t_all_clipsigs = tf.constant(clip_sigs)
t_all_wordsigs = tf.constant(word_sigs)

num_pairs = len(pair_clip_ids)
print('created ', num_pairs, 'pairs for train and test')
max_train = num_pairs * 4 / 5

print('test set zero-compare error', np.sqrt(np.mean(np.power(pair_frames_on[max_train:-1],2))))
var_clip_ids = tf.Variable(pair_clip_ids, trainable=False)
var_word_ids = tf.Variable(pair_word_ids, trainable=False)
var_frames_on = tf.Variable(np.asarray(pair_frames_on), trainable=False)

# t_batch_begin = tf.Variable()
ph_batch_begin = tf.placeholder(dtype=tf.int32, shape=())
t_batch_begin = tf.Variable(0, dtype=tf.int32, trainable=False)
op_batch_begin_rand = tf.assign(t_batch_begin, tf.random_uniform([], minval=0, maxval=max_train - batch_size, dtype=tf.int32))
op_batch_begin_input = tf.assign(t_batch_begin, ph_batch_begin)

ph_drop_keep_rate = tf.placeholder(tf.float32)
t_drop_keep_rate = tf.Variable(0, dtype=tf.float32, trainable=False)
op_set_drop_rate =  tf.assign(t_drop_keep_rate, ph_drop_keep_rate)

t_clip_ids = tf.slice(input_=var_clip_ids, begin=[t_batch_begin], size=[batch_size])
t_word_ids = tf.slice(input_=var_word_ids, begin=[t_batch_begin], size=[batch_size])
t_frames_on = tf.slice(input_=var_frames_on, begin=[t_batch_begin, 0], size=[batch_size, -1])

t_clipsigs = tf.gather(t_all_clipsigs, t_clip_ids)
t_wordsigs = tf.gather(t_all_wordsigs, t_word_ids)

t_final = create_graph(t_wordsigs, t_clipsigs, t_drop_keep_rate)

t_frames_on_count = tf.reduce_sum(t_frames_on)
t_frames_on_mean = tf.reduce_mean(t_frames_on ** 2)
t_final_count = tf.reduce_sum(t_final)

err = tf.reduce_mean((t_final - t_frames_on) ** 2)
train_step1 = tf.train.AdagradOptimizer(15.0).minimize(err)
train_step2 = tf.train.AdagradOptimizer(5.0).minimize(err)
train_step3 = tf.train.AdagradOptimizer(0.05).minimize(err)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

c_drop_keep_rate = 0.8
sess.run(op_set_drop_rate, feed_dict={ph_drop_keep_rate:c_drop_keep_rate})
print('drop keep rate = ', sess.run(t_drop_keep_rate))

learn_phase = 0
for step in range(num_steps+1):
	sess.run(op_batch_begin_rand)
	if step % (num_steps / 1000) == 0:
		# vcomb_knn, vknn_distance, vinword = sess.run([t_final, t_knn_distance, t_inword], feed_dict=fd)
		# print(vcomb_knn, '\ndist:', vknn_distance, '\ninword', vinword)
		errval1 = math.sqrt(sess.run(err))
		# print(sess.run([t_final_count, t_frames_on_count]), math.sqrt(sess.run(t_frames_on_mean)))
		print('step:', step, ', err : ', errval1)
		if learn_phase == 0 and errval1 < 0.15:
			print('reduce learn rate to', 5.0 )
			learn_phase = 1

	if step % (num_steps / 100) == 0:
		sess.run(op_set_drop_rate, feed_dict={ph_drop_keep_rate: 1.0})
		print('drop keep rate = ', sess.run(t_drop_keep_rate))
		do_test(sess, err, max_train, num_pairs, ph_batch_begin, op_batch_begin_input, batch_size)
		sess.run(op_set_drop_rate, feed_dict={ph_drop_keep_rate:c_drop_keep_rate})
		print('drop keep rate = ', sess.run(t_drop_keep_rate))


	if step != 0 and step % 1000 == 0:
		saver = tf.train.Saver()
		saver.save(sess, chkpoint_path)

	if learn_phase == 0:
		sess.run([train_step1])
	else:
		sess.run([train_step2])

print('done')

