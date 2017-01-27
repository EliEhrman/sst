from __future__ import print_function


# We'll need numpy for some mathematical operations
import numpy as np
import csv


dbfile = '/devlink2/data/stt/testdb.txt'
signalsfile_path = '/devlink2/data/stt/signalsdb.txt'
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

for irow, row in enumerate(reader):
	stime = float(row[4])
	dur = float(row[5]) - stime
	stime -= (stime * 0.11)
	y, sr = librosa.load(row[0], sr=None, offset=stime, duration=dur)
	if y.size == 0:
		print ('row %d produced no data.', irow)
		continue
	# if irow > 100 and irow < 200:
	# 	librosa.output.write_wav('testdata/test'+str(irow)+row[2]+'.wav', y, sr)
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.logamplitude(S, ref_power=np.max)
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
	delta_mfcc = librosa.feature.delta(mfcc)
	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	ems = zip(mfcc, delta_mfcc, delta2_mfcc)
	# ems = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=1)
	signals = np.transpose(ems, axes=(2,0,1))
	signals_writer.writerow(row[1:4] + list(signals.shape) + [str(sval) for sigrow in signals for sig in sigrow for sval in sig])
	# print ('flush here only for debugging. dont leave')
	# signalsfile.flush()
	print(irow)

csvfile.close()
signalsfile.close()

print('bye')


