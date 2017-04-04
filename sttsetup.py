from __future__ import print_function
import csv
import numpy as np
# import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import librosa
import os


wordfiles_path = '/devlink2/data/stt/testdb.txt'
vadfiles_path = '/devlink2/data/stt/vaddb.txt'
clipsigs_path = '/devlink2/data/stt/clipsigs.txt'
clipdata_path = '/devlink2/data/stt/clipdata.txt'

c_num_spectra = 13
c_num_channels = 3 # mfcc, delta and delta2
max_word_time = 1.2 # seconds
max_clip_time = 3.0 # seconds
samples_per_frame = 512
sr = 16000 #sr is sampling rate i.e. samples per second
max_word_frames = int(max_word_time * sr / samples_per_frame)
max_clip_frames = int(max_clip_time * sr / samples_per_frame)


max_clips_to_read = 1000
max_words_to_read = 10000

vadfiles_fh = open(vadfiles_path, 'rb')
vadfiles_reader = csv.reader(vadfiles_fh, delimiter=',')

l_clip_sigs = []
vadstarts = []
vadends = []
goldwords = []

clipfile_dict = {}
words_dict = {}

for irow, row in enumerate(vadfiles_reader):
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
	# if np.shape(S)[1] > max_clip_frames:
	# 	continue
	log_S = librosa.logamplitude(S, ref_power=np.max)
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
	delta_mfcc = librosa.feature.delta(mfcc)
	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	ems = zip(mfcc, delta_mfcc, delta2_mfcc)
	signals = np.transpose(ems, axes=(2,0,1))
	data_size = len(signals)
	l_clip_sigs.append(signals)
	vadstarts.append(float(row[1]))
	vadends.append(float(row[2]))
	num_words_in_clip = int(row[3])
	clip_goldwords = []
	for iword in range(num_words_in_clip):
		clip_goldwords.append(row[iword+4])
	goldwords.append(clip_goldwords)
	clipfile_dict[fname] = len(l_clip_sigs) - 1

vadfiles_fh.close()
num_clips = len(l_clip_sigs)

wordfiles_fh = open(wordfiles_path, 'rb')
wordfiles_reader = csv.reader(wordfiles_fh, delimiter=',')
words_in_clip_dict = {}
word_data = []

for irow, row in enumerate(wordfiles_reader):
	if irow > max_words_to_read:
		break
	fname = row[0]
	id = clipfile_dict.get(fname, -2)
	if id < 0:
		continue;
	word = row[2]
	word_data = (word, float(row[4]), float(row[5]))
	clip_word_data = words_in_clip_dict.get(id, [])
	clip_word_data.append(word_data)
	words_in_clip_dict[id] = clip_word_data

wordfiles_fh.close()

clipdata_fh = open(clipdata_path, 'wb')
clipdata_writer = csv.writer(clipdata_fh, delimiter=',')

for iclip in range(num_clips):
	row_data = [iclip, vadstarts[iclip], vadends[iclip], len(goldwords[iclip])]
	row_data.extend(goldwords[iclip])
	clip_word_data = words_in_clip_dict.get(iclip, [])
	if len(clip_word_data) == 0:
		continue
	b_data_good = True
	for word_data in clip_word_data:
		if word_data[2] > vadends[iclip]:
			b_data_good = False
			break
		if not b_data_good:
			continue
	row_data.append(len(clip_word_data))
	row_data.extend([d for word_data in clip_word_data for d in word_data])
	clipdata_writer.writerow(row_data)

clipdata_fh.close()

clipsigs_fh = open(clipsigs_path, 'wb')
clipsigs_writer = csv.writer(clipsigs_fh, delimiter=',')

for isigs, signals in enumerate(l_clip_sigs):
	clipsigs_writer.writerow([isigs] + list(signals.shape) + [str(sval) for sigrow in signals for sig in sigrow for sval in sig])

clipsigs_fh.close()


