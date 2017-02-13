from __future__ import print_function


# We'll need numpy for some mathematical operations
import numpy as np
import csv


dbfile = '/devlink2/data/stt/vaddb.txt'
signalsfile_path = '/devlink2/data/stt/vadsigs.txt'
# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

csvfile = open(dbfile, 'rt')
reader = csv.reader(csvfile, delimiter=',')

signalsfile = open(signalsfile_path, 'wb')
signals_writer = csv.writer(signalsfile, delimiter=',')

time_fudge_factor = 1.0
# labels = ['prespeech' , 'start', 'speech', 'end', 'postspeech']
labels = ['start', 'end']

for irow, row in enumerate(reader):
	starttime =  row[1]
	endtime = row[2]
	numgoldwords = row[3]
	goldwords = row[4:]
	y, sr = librosa.load(row[0], sr=None)
	if y.size < 2:
		continue

	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.logamplitude(S, ref_power=np.max)
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
	delta_mfcc = librosa.feature.delta(mfcc)
	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	ems = zip(mfcc, delta_mfcc, delta2_mfcc)
# ems = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=1)
	signals = np.transpose(ems, axes=(2,0,1))
	signals_writer.writerow([starttime, endtime] + row[3:-1] + list(signals.shape) + [str(sval) for sigrow in signals for sig in sigrow for sval in sig])
	# print ('flush here only for debugging. dont leave')
	# signalsfile.flush()
	print(irow)

csvfile.close()
signalsfile.close()

print('bye')


