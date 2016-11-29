import re
from data_pipline import Sentences


sentences = Sentences(dirname='./Affection analysis database/test/', split_line=True, parser='\t')


labels = []
for line in sentences:
    # _, str, raw_label = line.strip('\n').split('\t')
    print(line[2].strip(':: '))


