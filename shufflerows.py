import csv
from random import shuffle

inpath = '/devlink2/data/stt/signalsdb.txt'
outpath = '/devlink2/data/stt/rsignalsdb.txt'

with open(inpath, 'rt') as infile:
	content = infile.readlines()

shuffle(content)

with open(outpath, 'wt') as outfile:
	outfile.writelines(content)


# reader = csv.reader(csvfile, delimiter=',')
