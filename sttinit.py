from __future__ import print_function
import csv
import numpy as np
# import tensorflow as tf
import random
import collections
# import matplotlib.pyplot as plt
import os


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


max_clips_to_read = 100
max_words_to_read = 1000

struct_worddata = collections.namedtuple('worddata', ['word', 'clipid', 'start', 'end'])


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

def data_init():
	clipsigs_fh = open(clipsigs_path, 'rb')
	clipsigs_reader = csv.reader(clipsigs_fh, delimiter=',')

	clipsigs = []
	clipids_dict = {}

	for irow, row in enumerate(clipsigs_reader):
		if irow > max_words_to_read:
			break
		rowpos = 1
		# ws.append(row)
		num_spectrums = int(row[rowpos])
		if num_spectrums >= max_clip_frames:
			continue;
		spectrums = sigsread(row, rowpos, num_spectrums, int(max_clip_frames))
		clipids_dict[row[0]] = len(clipsigs)
		clipsigs.append(spectrums)

	clipsigs_fh.close()
	num_clips = len(clipsigs)
	clipsigs = np.asarray(clipsigs)

	clipsigs_fh.close()

	clipdata_fh = open(clipdata_path, 'rb')
	clipdata_reader = csv.reader(clipdata_fh, delimiter=',')

	clip_vadstarts = []
	clip_vadends = []

	l_worddata = []
	worddata_dict = {}
	for irow, row in enumerate(clipdata_reader):
		clip_iid = row[0]
		clip_id = clipids_dict.get(clip_iid, -1)
		if clip_id < 0:
			continue
		if float(row[2]) > max_clip_time:
			continue # invalidate entire clip if vadends is later that we'll accept for the entire clip
		clip_vadstarts.append(row[1])
		clip_vadends.append(row[2])
		num_goldwords = int(row[3])
		rowpos = 4
		clip_goldwords= [row[rowpos + igold] for igold in range(num_goldwords)]
		rowpos += num_goldwords
		clip_numwords = int(row[rowpos])
		for iword in range(clip_numwords):
			clipword = row[rowpos+1]
			data_for_word = worddata_dict.get(clipword, [])
			data_for_word.append(len(l_worddata))
			worddata_dict[clipword] = data_for_word
			l_worddata.append(struct_worddata(word=clipword, clipid=clip_id,
											  start=float(row[rowpos+2]) * sr / samples_per_frame,
											  end=float(row[rowpos+3]) * sr / samples_per_frame))
			rowpos+=3

	clipdata_fh.close()

	return clipsigs, clip_vadstarts, clip_vadends, l_worddata, worddata_dict


