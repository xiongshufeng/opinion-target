import gzip
import simplejson
import sys

def parse(filename):
  f = gzip.open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry

dataFileName = sys.argv[1]
outFileName = sys.argv[2]

outFile = open(outFileName, 'w')

for e in parse(dataFileName):
  try:
    outFile.write(e["review/text"] + "\n")
  except KeyError:
    print simplejson.dumps(e)
    continue

outFile.close()
