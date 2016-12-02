import os
# import nltk
from nltk.tokenize import TweetTokenizer
import re
import gensim
import numpy as np


class Sentences(object):
    def __init__(self, dirname, parser=None, split_line=False, split_method='', label=False, matlabel=False, w2v=False, tensor_out=False, max_length=140, stop_word=False, caseless=False, vec_length=100):
        self.dirname = dirname
        self.parser = parser
        self.split_line = split_line
        self.split_method = split_method
        self.label = label
        self.matlabel = matlabel
        self.w2v = w2v
        self.tensor_out = tensor_out
        self.max_length = max_length
        self.stop_word = stop_word
        self.caseless = caseless
        self.vec_length = vec_length

        self.label_dict = {'surprise': [1, 0, 0, 0, 0, 0],
                           'sadness': [0, 1, 0, 0, 0, 0],
                           'joy': [0, 0, 1, 0, 0, 0],
                           'disgust': [0, 0, 0, 1, 0, 0],
                           'fear': [0, 0, 0, 0, 1, 0],
                           'anger': [0, 0, 0, 0, 0, 1]}

        if self.w2v:
           self.model = gensim.models.Word2Vec.load('models/model')


    def __w2v__(self, text):
        return self.model[text]


    def __iter__(self):
        for fname in os.listdir(self.dirname):
            # print(self.dirname)
            if fname[0] == '.':
                continue
            if fname == 'labels.txt':
                continue
            # print(path)
            for line in open(os.path.join(self.dirname, fname), errors="ignore"):
                # print('line1')

                if self.caseless:
                    line = line[0].lower() + line[1:]

                ori_line = line
                line = line.strip('\n').split('\t')
                # print(len(line))
                if len(line) < 3:
                    # print(line)
                    continue
                if len(line) > 3:
                    continue
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
                    elif not self.parser is None:
                        text = text.split(self.parser)

                if not self.split_line:
                    text = ori_line

                if self.stop_word:
                    # text = re.sub(r'\bI\b', '', text)
                    # text = re.sub(r'\bi\b', '', text)
                    # text = re.sub(r'\ba\b', '', text)
                    # text = re.sub(r'#\w*', '', text)
                    # text = re.sub(r'\d*', '', text)
                    pass

                if self.w2v:
                    text = list(map(self.__w2v__, text))

                if self.matlabel:
                    raw_label = self.label_dict[raw_label]

                if self.tensor_out:
                    temp = np.array(text)
                    text = np.zeros((1, self.max_length, self.vec_length))
                    text[0, :len(temp), :] = temp
                    raw_label = np.array(raw_label).reshape((1, 6))

                if self.label:
                    yield text, raw_label
                else:
                    yield text

if __name__ == "__main__":
    sentences = Sentences(dirname="./data_set/train", split_line=False, split_method='Twitter', w2v=False, label=True, matlabel=True, stop_word=True)
    for line, label in sentences:
        print(line)
        # pass
