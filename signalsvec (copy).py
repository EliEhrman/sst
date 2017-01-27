from __future__ import print_function
import csv
import numpy as np
import tensorflow as tf
import random


signalsfile_path = '/devlink2/data/stt/rsignalsdb.txt'
phoneme_dict_path = '/devlink2/data/stt/cmudict-en-us.dict'
groupspath = '/devlink2/data/stt/wordgroups.txt'

batch_size = 256
rsize = batch_size * batch_size
num_steps = 100000
# num_samps = 100 # number of times DURING the run we reinitialize the sample and its random
max_num_spectrums = 64
spectrum_size = 13
num_input_channels = 3 # mfcc, delta and delta2

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable

	Returns:
	Variable Tensor
	"""
	with tf.device('/cpu:0'):
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


def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[
							 j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1  # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]


def data_init():
	sigfile = open(signalsfile_path, 'rb')

	phoneme_dict_file = open(phoneme_dict_path, 'rb')
	phoneme_reader = csv.reader(phoneme_dict_file, delimiter=' ')
	phoneme_dict = {row[0]: row[1:] for row in phoneme_reader}

	groups_file = open(groupspath, 'rb')
	groups_reader = csv.reader(groups_file, delimiter=',')
	word_groups = [row for row in groups_reader]


	# max_num_spectrums = max(num_spectrums_arr)
	return sigfile, phoneme_dict, word_groups

def create_graph(t_sigs_input):
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 1, 8],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(t_sigs_input, kernel, [1, 1, 1, 1], padding='SAME')
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

	LASTSIZE = 80

	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, LASTSIZE],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [LASTSIZE], tf.constant_initializer(0.1))
		unnorm = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	# nlizer = tf.sqrt(tf.reduce_sum(tf.square(onedee), axis=1))
	# spreader = tf.tile(nlizer, LASTSIZE)
	# spreader2 = tf.reshape(spreader, [batch_size, LASTSIZE])
	# nn = tf.gather(nlizer, spreader)
	# lastones = tf.mul(onedee, nlizer)

	lastones = tf.nn.l2_normalize(unnorm, dim=1)

	return lastones
#

def create_feed(sigfile, phoneme_dict, word_groups, t_sigs_input, t_r1, t_r2, t_dists):

	sigsize = 0
	sigs = []
	while sigsize < batch_size:
		group_data = random.choice(word_groups)
		num_words_in_group = int(group_data[1])
		num_words_left = min(num_words_in_group, batch_size - sigsize)
		sigstart = sigsize
		sigsize += num_words_left
		sigend = sigsize
		sigfile.seek(int(group_data[2]))
		reader = csv.reader(sigfile, delimiter=',')

		for irow, row in enumerate(reader):
			if irow >= num_words_left:
				break
			num_spectrums = int(row[3])
			spectrum_size = int(row[4])
			num_input_channels = int(row[5])
			# num_spectrums_arr.extend([num_spectrums])
			rowpos = 6
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
				# print(spectrum)
			sigs.append([row[1], row[0], row[2], spectrums])

	# sigfile.close()

	batch_start = random.randint(0, len(sigs) - batch_size)
	images = []
	labels = []
	for isig, sig in enumerate(sigs[batch_start:batch_start+batch_size]):
		if isig >= batch_size:
			break;
		spectrums = sig[3]
		start_pad_size = (max_num_spectrums - len(spectrums)) / 2
		image = [[0.0] * spectrum_size] * start_pad_size
		image.extend(spectrums)
		end_pad_size = max_num_spectrums - len(image)
		image += [[0.0] * spectrum_size] * end_pad_size
		images.append([[[val] for val in row] for row in image])
		labels.append(sig[0])

	r1 = [random.randint(0, batch_size-1) for i in range(rsize)]
	r2 = [random.randint(0, batch_size-1) for i in range(rsize)]
	edit_distances = []
	remove_list = []
	for ir in range(rsize):
		s1 = phoneme_dict.get(labels[r1[ir]], '<error>')
		if s1 == '<error>':
			remove_list.append(ir)
			continue;
		s2 = phoneme_dict.get(labels[r2[ir]], '<error>')
		if s2 == '<error>':
			remove_list.append(ir)
			continue;
		edit_distances.append(float(levenshtein(s1, s2)) / float(max(len(s1), len(s2))))

	for iremove in reversed(remove_list):
		r1.pop(iremove)
		r2.pop(iremove)

	maxdist = max(edit_distances)
	edit_distances = [1 - val/maxdist for val in edit_distances]


	return {t_sigs_input: images, t_r1: r1, t_r2: r2, t_dists: edit_distances}

sigfile, phoneme_dict, word_groups = data_init()

t_sigs_input = tf.placeholder(tf.float32, [batch_size, max_num_spectrums, spectrum_size, 1])

lastones = create_graph(t_sigs_input)

t_r1 = tf.placeholder(tf.int32, [rsize])
t_r2 = tf.placeholder(tf.int32, [rsize])
t_dist_truth = tf.placeholder(tf.float32, [rsize])
t_lasts1 = tf.gather(lastones, t_r1)
t_lasts2 = tf.gather(lastones, t_r2)
t_dist_predict = tf.reduce_sum(tf.multiply(t_lasts1, t_lasts2), axis=1)
err = tf.reduce_mean((t_dist_truth - t_dist_predict) ** 2)
train_step = tf.train.AdagradOptimizer(0.05).minimize(err)

# testnorm = tf.reduce_sum(tf.square(lastones), axis=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# -sys.float_info.max
for step in range(num_steps):
	fd = create_feed(sigfile, phoneme_dict, word_groups, t_sigs_input, t_r1, t_r2, t_dist_truth)
	if step % (num_steps/100) == 0:
		print('err: ', sess.run([err], feed_dict=fd))
	v1 = sess.run([train_step], feed_dict=fd)


sess.close()

print('bye')