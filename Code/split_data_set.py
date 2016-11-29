import re
from data_pipline import Sentences


sentences = Sentences(dirname='./Affection analysis database/test/', split_line=False)

line_count = 0
for _ in sentences:
    line_count += 1

count = 0
for line in sentences:
    if count / line_count < 0.6:
        with open('data_set/train/train.txt', 'a') as f:
            f.write(line)
    elif count / line_count < 0.8:
        with open('data_set/eval/eval.txt', 'a') as f:
            f.write(line)
    else:
        with open('data_set/test/test.txt', 'a') as f:
            f.write(line)

    count += 1
