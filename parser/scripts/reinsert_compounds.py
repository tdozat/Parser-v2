from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 
 
import os 
import sys 
import codecs 
 
input_file = sys.argv[2] 
output_file = sys.argv[1] 
 
lines = [] 
 
with codecs.open(output_file, encoding='utf-8') as f: 
  for line in f: 
    lines.append(line) 
 
with codecs.open(input_file, encoding='utf-8') as f: 
  with codecs.open(output_file, 'w', encoding='utf-8') as fout: 
    i = 0 
    for line in f: 
      line = line.strip() 
 
      if len(line) == 0: 
        fout.write(lines[i]) 
        i += 1 
        continue 
 
      if line[0] == '#': 
        continue 
 
      line = line.split('\t') 
      if '.' in line[0]: 
        continue 
 
      if '-' in line[0]: 
        fout.write('%s\n' % ('\t'.join(line))) 
        continue 

      fout.write(lines[i])
      i += 1
