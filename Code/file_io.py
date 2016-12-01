from data_pipeline import Sentences
import numpy as np


class file_io(object):
    def __init__(self, path):
        self.sentences = Sentences(path, split_line=True, split_method="Twitter", w2v=True, label=True, matlabel=True)


    def read_data(self, path='./data_set/train/'):
        text = np.zeros((1, 140, 100))
        label = np.zeros((1, 6))

        i = 0
        for r_text, r_label in self.sentences:
            text_temp = np.zeros((1, 140, 100))
            text_temp[0, :len(r_text), :] = r_text
            text = np.append(text, text_temp, axis=0)

            label = np.append(label, np.array(r_label).reshape((1, 6)), axis=0)

            i += 1
            if i % 100 == 0:
                print(i)

        return text, label


    def next_batch(self, batch_size):
        text = np.zeros((1, 140, 100))
        label = np.zeros((1, 6))

        i = 0
        for r_text, r_label in self.sentences:
            text_temp = np.zeros((1, 140, 100))
            text_temp[0, :len(r_text), :] = r_text
            text = np.append(text, text_temp, axis=0)

            label = np.append(label, np.array(r_label).reshape((1, 6)), axis=0)

            i += 1
            if i % 100 == 0:
                print(i)

        return text, label




    text, label = read_data()

