import numpy as np
import tensorflow as tf
import random
import sttinit
import math

from sttinit import max_word_frames
from sttinit import c_num_spectra
from sttinit import c_num_channels

batch_size = 8
num_steps = 10000

def create_wordsigs(l_worddata, clipsigs):
	wordsigs = []
	for worddata in l_worddata:
		clipid = worddata.clipid
		start = int(worddata.start)
		end = int(worddata.end) + 1
		data_size = end - start
		start_pad_size = int((max_word_frames - data_size) / 2)
		image = np.zeros([max_word_frames, c_num_spectra, c_num_channels], dtype=np.float32)
		image[start_pad_size:start_pad_size + data_size, : , :] = clipsigs[clipid, start:end, :, :]
		wordsigs.append(image)

	wordsigs = np.asarray(wordsigs)
	return wordsigs


def create_pairs(l_worddata, worddata_dict):
	r1 = []
	r2 = []
	distances =[]
	word_labels = []
	one_sided_r1 = []
	for word, worddata_ids in worddata_dict.iteritems():
		num_equal_pairs = 0
		for worddata_idx_outer in worddata_ids:
			for worddata_idx_inner in worddata_ids:
				# r1.append(l_worddata[worddata_idx_outer].clipid)
				r1.append(worddata_idx_inner)
				r2.append(worddata_idx_outer)
				num_equal_pairs += 1
			r1 += [random.randint(0, len(l_worddata)-1) for i in range(num_equal_pairs)]
			r2 += [random.randint(0, len(l_worddata)-1) for i in range(num_equal_pairs)]
			one_sided_r1.append(worddata_idx_outer)
			word_labels.append(l_worddata[worddata_idx_outer].word)

	for ir, _ in enumerate(r1):
		if l_worddata[r1[ir]].word == l_worddata[r2[ir]].word:
			distances.append(1.0)
		else:
			distances.append(0.0)

	r1.extend(one_sided_r1)
	return r1, r2, distances, num_equal_pairs * 2, word_labels

def create_conv_layer(prev_out, kernel_shape, conv_shape, out_depth):
	kernel = tf.Variable(	tf.truncated_normal(kernel_shape, stddev=0.04, dtype=tf.float32),
							name ='weights')
	conv = tf.nn.conv2d(prev_out, kernel, conv_shape, padding='SAME')
	biases = tf.Variable( tf.zeros([out_depth]))
	pre_activation = tf.nn.bias_add(conv, biases)
	conv1 = tf.nn.relu(pre_activation, name='conv1')
	return conv1

def create_net(sigs):
	conv1 = create_conv_layer(sigs, [5, 5, 3, 8], [1, 1, 1, 1], 8)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm1')

	conv2 = create_conv_layer(norm1, [5, 5, 8, 16], [1, 1, 1, 1], 16)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
						   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	LASTSIZE = 80

	# Move everything into depth so we can perform a single matrix multiply.
	flatty = tf.reshape(pool2, [batch_size, -1])
	dim = flatty.get_shape()[1].value
	wfc1 = tf.Variable(	tf.truncated_normal([dim, LASTSIZE], stddev=0.04, dtype=tf.float32),
							name ='wfc1')
	bfc1 = tf.Variable( tf.zeros([LASTSIZE]))
	unnorm = tf.nn.relu(tf.matmul(flatty, wfc1) + bfc1, name='unnorm')

	lastones = tf.nn.l2_normalize(unnorm, dim=1)
	return lastones


def create_graph(r1sigs, r2sigs):
	s1l1 = create_net(r1sigs)
	s2l1 = create_net(r2sigs)
	l2 = tf.reduce_sum(tf.multiply(s1l1, s2l1),  axis=1)
	return l2

def do_test(sess, err, max_train, num_pairs, ph_batch_begin, op_batch_begin_input, batch_size):
	errsum = 0.0
	num_runs = 0
	for batch_begin in range(max_train, num_pairs-batch_size, batch_size):
		sess.run(op_batch_begin_input, feed_dict={ph_batch_begin:batch_begin})
		errsum += math.sqrt(sess.run(err))
		num_runs += 1.0
	print ('err on validation set:', errsum / num_runs)
	return

clipsigs, _, _, l_worddata, worddata_dict = sttinit.data_init()

wordsigs = create_wordsigs(l_worddata, clipsigs)
r1, r2, distances = create_pairs(l_worddata, worddata_dict)


num_pairs = len(distances)
print('created ', num_pairs, 'pairs for train and test')
max_train = num_pairs * 4 / 5

shuffle_stick = [i for i in range(len(r1))]
random.shuffle(shuffle_stick)
r1 = [r1[ish] for ish in shuffle_stick]
r2 = [r2[ish] for ish in shuffle_stick]
distances = [distances[ish] for ish in shuffle_stick]

t_wordsigs = tf.constant(wordsigs)

ph_batch_begin = tf.placeholder(dtype=tf.int32, shape=())
t_batch_begin = tf.Variable(0, dtype=tf.int32, trainable=False)
op_batch_begin_rand = tf.assign(t_batch_begin, tf.random_uniform([], minval=0, maxval=max_train - batch_size, dtype=tf.int32))
op_batch_begin_input = tf.assign(t_batch_begin, ph_batch_begin)
t_r1 = tf.slice(input_=r1, begin=[t_batch_begin], size=[batch_size])
t_r2 = tf.slice(input_=r2, begin=[t_batch_begin], size=[batch_size])
t_distances = tf.slice(input_=distances, begin=[t_batch_begin], size=[batch_size])

t_r1sigs = tf.gather(t_wordsigs, t_r1)
t_r2sigs = tf.gather(t_wordsigs, t_r2)

t_final = create_graph(t_r1sigs, t_r2sigs)
err = tf.reduce_mean((t_final - t_distances) ** 2)
train_step1 = tf.train.AdamOptimizer(0.5).minimize(err)
train_step2 = tf.train.AdamOptimizer(0.00000005).minimize(err)

train_vars = tf.trainable_variables()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

learn_phase = 0
for step in range(num_steps+1):
	sess.run(op_batch_begin_rand)
	if step % (num_steps / 1000) == 0:
		errval1 = math.sqrt(sess.run(err))
		print('step:', step, ', err : ', errval1)
		# if learn_phase == 0 and errval1 < 0.15:
		# 	print('reduce learn rate to', 5.0 )
		# 	learn_phase = 1

	if step % (num_steps / 100) == 0:
		do_test(sess, err, max_train, num_pairs, ph_batch_begin, op_batch_begin_input, batch_size)

	if learn_phase == 0:
		# errval1 = math.sqrt(sess.run(err))
		sess.run([train_step1])
		# errval2 = math.sqrt(sess.run(err))
		# print(errval1, '->', errval2)

sess.close()

print('done')