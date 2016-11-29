import os
# import nltk
from nltk.tokenize import TweetTokenizer
# import re


class Sentences(object):
    def __init__(self, dirname, parser=None, split_line=False, split_method='', label=False, matlabel=False):
        self.dirname = dirname
        self.parser = parser
        self.split_line = split_line
        self.split_method = split_method
        self.label = label
        self.matlabel = matlabel
        self.label_dict = label_dict = {'surprise': [1,0,0,0,0,0],
                                        'sadness': [0,1,0,0,0,0],
                                        'joy': [0,0,1,0,0,0],
                                        'disgust': [0,0,0,1,0,0],
                                        'fear': [0,0,0,0,1,0],
                                        'anger': [0,0,0,0,0,1]}


    def __iter__(self):
        for fname in os.listdir(self.dirname):
            # print(self.dirname)
            if fname[0] == '.':
                continue
            if fname == 'labels.txt':
                continue
            # print(path)
            for line in open(os.path.join(self.dirname, fname)):
                ori_line = line
                line = line.strip('\n').split('\t')
                if len(line) != 3:
                    print(line)
                text = line[1]
                raw_label = line[2].strip(':: ')
                # print(text)
                # print(label)

                if self.split_line:
                    if self.split_method == 'Twitter':
                        tknz = TweetTokenizer()
                        text = tknz.tokenize(text)
                    elif self.split_method == 'space':
                        text = text.split(' ')
                    else:
                        text = text.split(self.parser)


                if self.label:
                    # print(data)
                    if not self.split_line:
                        text = ori_line

                    if self.matlabel:
                        raw_label = self.label_dict[raw_label]

                    yield text, raw_label
                else:
                    if not self.split_line:
                        text = ori_line
                    # print(data)

                    yield text

if __name__ == "__main__":
    sentences = Sentences(dirname="./data_set/raw/", split_line=True, label=True, matlabel=True)
    for line, label in sentences:
        print(label)
