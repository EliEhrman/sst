from __future__ import print_function
import csv
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import os
import math

wordsigs_path = '/devlink2/data/stt/ssignalsdb.txt'
# wordsigs_small_path = '/devlink2/data/stt/ssignalsdb.txt'
clipsigs_path = '/devlink2/data/stt/svadsigs.txt'
whwordsknn_path = '/devlink2/data/stt/whwordsknn.txt'
whclipsknn_path = '/devlink2/data/stt/whclipsknn.txt'
chkpoint_path = '/devlink2/data/stt/chk/chkwordhere.dat'
b_skip_learning = True

batch_size = 512 # 8 #
rsize = batch_size * batch_size
num_steps = 100000 # 10000 #
# num_samps = 100 # number of times DURING the run we reinitialize the sample and its random
max_word_time = 1.2 # seconds
max_clip_time = 3.0 # seconds
samples_per_frame = 512
sr = 16000 #sr is sampling rate i.e. samples per second
c_num_spectra = 13
c_num_channels = 3 # mfcc, delta and delta2
max_word_examples = 3 # 5 # how many words to find for each example
max_clips_to_read = 10000 # 100 #
max_words_to_read = 100000 # 1000 #
max_clip_ids = 2**14 # 2**18 # sadly, it seems tensorflow cannot handle dynamic length for Variables

max_word_frames = max_word_time * sr / samples_per_frame
max_clip_frames = max_clip_time * sr / samples_per_frame

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
	var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return var


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
			comps.append(chans)
		# spectrum = [float(sval) for sval in row[6+(ispec* data_size_per_spectrum):6+((ispec+1) * data_size_per_spectrum)]]
		spectrums.append(comps)
	start_pad_size = (max_frames - len(spectrums)) / 2
	image = [[[0.0] * c_num_channels] * c_num_spectra] * start_pad_size
	image.extend(spectrums)
	end_pad_size = max_frames - len(image)
	image += [[[0.0] * c_num_channels] * c_num_spectra] * end_pad_size
	return image

def data_init():
	wordsigs_dict = dict()
	wordsigs = []
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

	clipsigs_fh = open(clipsigs_path, 'rb')
	clipsigs_reader = csv.reader(clipsigs_fh, delimiter=',')

	clip_words = []
	clipsigs = []
	for irow, row in enumerate(clipsigs_reader):
		if irow > max_clips_to_read:
			break
		rowpos = 2
		num_gold_words = int(row[rowpos])
		words_in_clip = row[rowpos+1:rowpos+num_gold_words+1]
		rowpos += num_gold_words + 1
		num_spectrums = int(row[rowpos])
		if num_spectrums >= max_clip_frames:
			continue;
		spectrums = sigsread(row, rowpos, num_spectrums, int(max_clip_frames))
		clipsigs.append(spectrums)
		clip_words.append(words_in_clip)

	clipsigs_fh.close()

	clip_ids = []
	word_ids = []
	inword = []
	words = []
	for iclip, words_in_clip in enumerate(clip_words):
		for word in words_in_clip:
			reflist = wordsigs_dict.get(word, [])
			if len(reflist) == 0: continue
			random.shuffle(reflist)
			numtrue = 0
			for iref, aref in enumerate(reflist):
				if iref > max_word_examples:
					break
				clip_ids.append(iclip)
				word_ids.append(aref)
				inword.append(1.0)
				words.append(word)
				numtrue += 1
			for irand in range(numtrue):
				clip_ids.append(iclip)
				word_ids.append(random.randint(0, num_words-1))
				inword.append(0.0)
				words.append('<error>')



	# ws_fh = open(wordsigs_small_path, 'wb')
	# ws_writer = csv.writer(ws_fh, delimiter=',')
	# for iword, word in enumerate(wordsigs_dict):
	# 	allrefs = wordsigs_dict[word]
	# 	for irow in range(max_word_examples):
	# 		if len(allrefs) > irow:
	# 			ws_writer.writerow(ws[allrefs[irow]])
	# ws_fh.close()
	# print('wrote a small version of', wordsigs_path, 'to', wordsigs_small_path)

	shuffle_stick = [i for i in range(len(clip_ids))]
	random.shuffle(shuffle_stick)
	clip_ids = [clip_ids[i] for i in shuffle_stick]
	word_ids = [word_ids[i] for i in shuffle_stick]
	inword = [inword[i] for i in shuffle_stick]
	words = [words[i] for i in shuffle_stick]

	return wordsigs, clipsigs, clip_ids, word_ids, inword, clip_words, words, wordsigs_dict

def create_graph(wordsigs, clipsigs):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 8],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(wordsigs, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.0))
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
		biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
						   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	MERGESIZE = 2048

	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, MERGESIZE],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [MERGESIZE], tf.constant_initializer(0.1))
		words_final = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 8],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(clipsigs, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)

	# pool1
	pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
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
		biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)

	# norm2
	norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool5 = tf.nn.max_pool(norm5, ksize=[1, 3, 3, 1],
						   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	MERGESIZE = 2048

	with tf.variable_scope('local6') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		clips_flat = tf.reshape(pool5, [batch_size, -1])
		dim = clips_flat.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, MERGESIZE],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [MERGESIZE], tf.constant_initializer(0.1))
		clips_final = tf.nn.relu(tf.matmul(clips_flat, weights) + biases, name=scope.name)

	clip_weights = _variable_with_weight_decay('clip_weights', shape=[MERGESIZE],
											  stddev=0.04, wd=0.004)
	word_weights = _variable_with_weight_decay('word_weights', shape=[MERGESIZE],
											  stddev=0.04, wd=0.004)
	combined1 = tf.add(tf.mul(clip_weights, clips_final), tf.mul(word_weights, words_final))

	KNNSIZE = 80

	clips_to_knn = _variable_with_weight_decay('cweights', shape=[MERGESIZE, KNNSIZE],
										  stddev=0.04, wd=0.004)
	clips_to_knn_biases = _variable_on_cpu('cbiases', [KNNSIZE], tf.constant_initializer(0.1))

	clips_knn = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(clips_final, clips_to_knn) + clips_to_knn_biases, name='clips_knn'), dim=1)

	words_to_knn = _variable_with_weight_decay('wweights', shape=[MERGESIZE, KNNSIZE],
										  stddev=0.04, wd=0.004)
	words_to_knn_biases = _variable_on_cpu('wbiases', [KNNSIZE], tf.constant_initializer(0.1))

	words_knn = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(words_final, words_to_knn) + words_to_knn_biases, name='words_knn'), dim=1)

	# combined_to_knn = _variable_with_weight_decay('combweights', shape=[MERGESIZE, KNNSIZE],
	# 									  stddev=0.04, wd=0.004)
	# combined_to_knn_biases = _variable_on_cpu('combbiases', [KNNSIZE], tf.constant_initializer(0.1))
	#
	# combined_knn = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(combined1, combined_to_knn) + combined_to_knn_biases, name='combined_knn'), dim=1)

	# nlizer = tf.sqrt(tf.reduce_sum(tf.square(onedee), axis=1))
	# spreader = tf.tile(nlizer, MERGESIZE)
	# spreader2 = tf.reshape(spreader, [batch_size, MERGESIZE])
	# nn = tf.gather(nlizer, spreader)
	# lastones = tf.mul(onedee, nlizer)

	# lastones = tf.nn.l2_normalize(combined1, dim=1)
	# lastones = tf.nn.l2_normalize(tf.square(combined_knn - clips_knn), dim=1) # learned well!

	# lastones = tf.nn.l2_normalize(tf.mul(words_knn, clips_knn), dim=1)

	wsize = KNNSIZE
	W = _variable_with_weight_decay('W', shape=[wsize],
									stddev=0.04, wd=0.004)
	b = _variable_on_cpu('b', [1], tf.constant_initializer(0.1))
	words_knn = tf.mul(words_knn, W)

	final = tf.sigmoid(tf.reduce_sum(tf.mul(words_knn, clips_knn) + 0, axis=1))

	return tf.reshape(final, [batch_size]), clips_knn, words_knn

"""
	lastones = tf.nn.l2_normalize(tf.mul(words_knn, clips_knn), dim=1)

	# wsize = MERGESIZE
	wsize = KNNSIZE
	W = _variable_with_weight_decay('W', shape=[wsize, 1],
										  stddev=0.04, wd=0.004)
	b = _variable_on_cpu('b', [1], tf.constant_initializer(0.1))
	final = tf.sigmoid(tf.matmul(lastones, W) + b)
"""

def create_knn_file(sess, clip_ids, words, clip_words, inword, clips_knn, words_knn, ph_batch_begin, op_batch_begin_input):
	knn_file1 = open(whwordsknn_path, 'wb')
	knn_writer1 = csv.writer(knn_file1, delimiter=',')
	knn_file2 = open(whclipsknn_path, 'wb')
	knn_writer2 = csv.writer(knn_file2, delimiter=',')
	clip_id_set = []
	for batch in range(0, len(clip_ids) - batch_size, batch_size):
		fd = {ph_batch_begin: batch}
		sess.run(op_batch_begin_input, feed_dict=fd)
		vclips, vwords = sess.run([clips_knn, words_knn])
		for i in range(batch_size):
			if inword[batch + i] == 1.0:
				knn_writer1.writerow([words[batch + i]] + [str(val) for val in vwords[i]])
			clip_id = clip_ids[batch + i]
			if clip_id not in clip_id_set:
				clip_id_set.append(clip_id)
				knn_writer2.writerow([str(len(clip_words[clip_id]))] + clip_words[clip_id] + [str(val) for val in vclips[i]])
			# if inword[batch + i] == 1.0:
			# 	if words[batch + i] not in clip_words[clip_id]:
			# 		print('Error: item', batch+i, ',', words[batch + i], 'not in',  clip_words[clip_id])

	knn_file1.close()
	knn_file2.close()

def test_results(sess, num_clips, num_words,  clip_words, words, t_final):
	word_ids = range(num_words)
	for iclip in range(num_clips):
		clip_ids = [iclip] * num_words
		for batch in range(0, len(clip_ids) - batch_size, batch_size):
			fd = {ph_batch_begin: batch}


wordsigs, clipsigs, clip_ids, word_ids, inword, clip_words, words, wordsigs_dict = data_init()

num_words = len(wordsigs)
ids_len = len(clip_ids)
num_clips = len(clipsigs)

if num_words > max_clip_ids or num_clips > max_clip_ids:
	print ('Sorry, the fized length of tensorflow variables, means that there are too many clips ids. You can always inrease the limit')
	exit()


# for iid, id in enumerate(word_ids):
# 	if id >= num_words:
# 		print('word id error!!! Element ', iid, '=', id, 'out of', num_words)
#
# for iid, id in enumerate(clip_ids):
# 	if id >= num_clips:
# 		print('clip id error!!! Element ', iid, '=', id, 'out of', num_clips)

ph_clip_ids = tf.placeholder(dtype=tf.int32)
var_clip_ids = tf.Variable([0]*max_clip_ids, dtype=tf.int32)
op_clip_ids_assign = tf.assign(var_clip_ids, ph_clip_ids)
ph_word_ids = tf.placeholder(dtype=tf.int32)
var_word_ids = tf.Variable([0]*max_clip_ids, dtype=tf.int32)
op_word_ids_assign = tf.assign(var_word_ids, ph_word_ids)
var_inword = tf.Variable(inword)
fd_data = {ph_clip_ids : clip_ids + [0] * (max_clip_ids - ids_len),
		   ph_word_ids: word_ids + [0] * (max_clip_ids - ids_len)}
# t_batch_begin = tf.Variable()
ph_batch_begin = tf.placeholder(dtype=tf.int32, shape=())
t_batch_begin = tf.Variable(0, dtype=tf.int32)
op_batch_begin_rand = tf.assign(t_batch_begin, tf.random_uniform([], minval=0, maxval=ids_len - batch_size, dtype=tf.int32))
op_batch_begin_input = tf.assign(t_batch_begin, ph_batch_begin)
t_clip_ids = tf.slice(input_=var_clip_ids, begin=[t_batch_begin], size=[batch_size])
t_word_ids = tf.slice(input_=var_word_ids, begin=[t_batch_begin], size=[batch_size])
t_inword = tf.slice(input_=var_inword, begin=[t_batch_begin], size=[batch_size])

# num_derivs = len(clipsigs[0][0][0])
# num_comps = len(clipsigs[0][0])
# num_frames = len(clipsigs[0])
fclipsigs = [chan for oneclip in clipsigs for frame in oneclip for spectrum in frame for chan in spectrum]
nclipsigs = np.array(fclipsigs, dtype=np.float32).reshape(num_clips, int(max_clip_frames), c_num_spectra, c_num_channels)
fwordsigs = [chan for oneword in wordsigs for frame in oneword for spectrum in frame for chan in spectrum]
nwordsigs = np.array(fwordsigs, dtype=np.float32).reshape(num_words, int(max_word_frames), c_num_spectra, c_num_channels)
onesigs = np.ones(dtype=np.float32, shape=[batch_size, int(max_word_frames), c_num_spectra, c_num_channels])
zerosigs = np.zeros(dtype=np.float32, shape=[batch_size, int(max_word_frames), c_num_spectra, c_num_channels])

t_all_clipsigs = tf.constant(nclipsigs)
t_all_wordsigs = tf.constant(nwordsigs)
t_clipsigs = tf.gather(t_all_clipsigs, t_clip_ids)
t_wordsigs = tf.gather(t_all_wordsigs, t_word_ids)
# t_wordsigs = tf.select(tf.equal(t_inword, 1.0), onesigs, zerosigs)

t_final, t_clips_knn, t_words_knn = create_graph(t_wordsigs, t_clipsigs)

# t_knn_distance = tf.sqrt(tf.reduce_mean(tf.square(t_combined_knn - t_clips_knn), axis=1))

# err = tf.reduce_mean((t_knn_distance - t_inword) ** 2)
# t_reduced_comb = tf.reduce_mean(t_combined_knn, axis=1)
# t_reduced_last = tf.reduce_mean(t_lastones, axis=1)

err = tf.reduce_mean((t_final - t_inword) ** 2)
# err = tf.reduce_mean(t_knn_distance)
train_step1 = tf.train.AdagradOptimizer(0.05).minimize(err)
train_step2 = tf.train.AdagradOptimizer(0.005).minimize(err)

# fd = {ph_batch_begin: 256}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run([op_clip_ids_assign, op_word_ids_assign], feed_dict=fd_data)

# print(sess.run(t_batch_begin))
# #
# print(sess.run([t_batch_begin, op_batch_begin_rand]))
# print(sess.run([t_batch_begin, op_batch_begin_rand]))
# sess.run(tf.variables_initializer([t_batch_begin]))
# print(sess.run(t_batch_begin))
# print(sess.run([t_batch_begin, op_batch_begin_input], feed_dict=fd))
# print(sess.run(t_batch_begin))

# if os.path.exists(chkpoint_path+'.index'):
# 	saver = tf.train.Saver()
# 	saver.restore(sess, chkpoint_path)



learn_phase = 0
for step in range(num_steps+1):
	sess.run(op_batch_begin_rand)
	# fd = {t_batch_begin: random.randint(0, ids_len - batch_size)}
	# sess.run(tf.variables_initializer([t_batch_begin]))
	if step % (num_steps / 100) == 0:
		# vcomb_knn, vknn_distance, vinword = sess.run([t_final, t_knn_distance, t_inword], feed_dict=fd)
		# print(vcomb_knn, '\ndist:', vknn_distance, '\ninword', vinword)
		errval1 = math.sqrt(sess.run(err))
		print('step:', step, ', err : ', errval1)
		if learn_phase == 0 and errval1 < 0.01:
			print('reduce learn rate to', 0.00005 )
			learn_phase = 1

	if step != 0 and step % 1000 == 0:
		saver = tf.train.Saver()
		saver.save(sess, chkpoint_path)

	if step % 10000 == 0:
		create_knn_file(sess, clip_ids, words, clip_words, inword, t_clips_knn, t_words_knn, ph_batch_begin, op_batch_begin_input)

	# errval1 = math.sqrt(sess.run(err, feed_dict=fd))
	if learn_phase == 0:
		sess.run([train_step1])
	else:
		sess.run([train_step2])
	# errval2 = math.sqrt(sess.run(err, feed_dict=fd))
	# print('step:', step, ', err : ', errval1, 'to', errval2)

sess.close()

print('bye')