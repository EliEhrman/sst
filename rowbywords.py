import csv
from random import shuffle

inpath = '/devlink2/data/stt/signalsdb.txt'
outpath = '/devlink2/data/stt/rsignalsdb.txt'
groupspath = '/devlink2/data/stt/wordgroups.txt'

with open(inpath, 'rb') as infile:
	reader = csv.reader(infile, delimiter=',')
	content = [row for row in reader]

shuffle(content)
content.sort(key=lambda row: row[1])

with open(outpath, 'wb') as outfile:
	outwriter = csv.writer(outfile, delimiter=',')
	for row in content:
		outwriter.writerow(row)

numrowswritten = len(content)

# infile.tell()
	# content = infile.readlines()
groupsfile = open(groupspath, 'wb')
with open(outpath, 'rb') as outfile:
	groupswriter = csv.writer(groupsfile, delimiter=',')
	# outreader = csv.reader(outfile, delimiter=',')
	wordlist = []
	word = '<error>'
	numofword = 0
	writestartpos = outfile.tell()
	for irow in range(numrowswritten):
		if word != content[irow][1] or numofword >= 64:
			if numofword > 0:
				groupswriter.writerow([word, numofword, writestartpos])
				writestartpos = outfile.tell()
			word = content[irow][1]
			numofword = 0
		numofword += 1
		row = outfile.readline()

groupsfile.close()



# reader = csv.reader(csvfile, delimiter=',')
