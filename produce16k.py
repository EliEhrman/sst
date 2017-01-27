import os
import subprocess

srcwavs = 'wav48'
destwavs = 'wav16k'
os.chdir('/devlink2/sttw/speech-to-text-wavenet/asset/data')
dirs = os.listdir(srcwavs)
for dir in dirs:
	if not os.path.exists(os.path.join(destwavs, dir)):
		os.mkdir(os.path.join(destwavs, dir))
files = [os.path.join(dir, file) for dir in dirs for file in os.listdir(os.path.join(srcwavs, dir)) ]
for file in files:
	cmd = 'ffmpeg -i ' + os.path.join(srcwavs, file) + ' -acodec pcm_s16le -ac 1 -ar 16000 ' + os.path.join(destwavs, file)
	os.system(cmd)
print files
# retval = subprocess.call('ls -l', shell=True)