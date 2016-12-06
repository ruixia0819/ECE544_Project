from data_pipeline import Sentences
import numpy as np
import tensorflow as tf
import random


class DataSet(object):
    def __init__(self, path='./data_set/train', max_length=140):
        self.max_length = max_length
        self.sentences = Sentences(path, max_length=max_length, split_line=True, tensor_out=True, split_method="Twitter", w2v=True, label=True, matlabel=True)
        self.gen = self.sentences.__iter__()

    def next_batch(self, batch_size=100):
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size

        try:
            text, label = next(self.gen)
        except StopIteration as e:
            self.gen = self.sentences.__iter__()
            text, label = next(self.gen)

        # shuffle batch
        #
        batch_text, batch_label = tf.train.shuffle_batch([text, label], batch_size=batch_size, capacity = capacity, min_after_dequeue=min_after_dequeue)

        # batch
        #
        # batch_text, batch_label = tf.train.batch(list(a), batch_size)

        return batch_text.eval(), batch_label.eval()
        # return batch_text, batch_label

    def all_data(self):
        text = np.zeros((1, self.max_length, 100))
        label = np.zeros((1, 6))

        for r_text, r_label in self.sentences:
            text = np.append(text, r_text)
            label = np.append(label, r_label)

        return np.delete(text, 0, axis=0), np.delete(label, 0, axis=0)

    def next_batch_stupid(self, batch_size):
        try:
            batch_x, batch_y = next(self.gen)
        except StopIteration as e:
            self.gen = self.sentences.__iter__()
            batch_x, batch_y = next(self.gen)

        for i in range(batch_size - 1):
            try:
                new_x, new_y = next(self.gen)
            except StopIteration as e:
                self.gen = self.sentences.__iter__()
                new_x, new_y = next(self.gen)

            batch_x = np.append(batch_x, new_x, axis=0)
            batch_y = np.append(batch_y, new_y, axis=0)

        return batch_x, batch_y

    def next_batch_stupid_shuffle(self, batch_size):
        try:
            batch_x, batch_y = next(self.gen)
        except StopIteration as e:
            self.gen = self.sentences.__iter__()
            batch_x, batch_y = next(self.gen)
        for i in range(batch_size - 1):
            try:
                new_x, new_y = next(self.gen)
            except StopIteration as e:
                self.gen = self.sentences.__iter__()
                new_x, new_y = next(self.gen)
            batch_x = np.append(batch_x, new_x, axis=0)
            batch_y = np.append(batch_y, new_y, axis=0)

        z = list(zip(batch_x, batch_y))
        # print(z)
        random.shuffle(z)

        batch_x[:], batch_y[:] = zip(*z)

        return batch_x, batch_y


if __name__ == '__main__':
    test = DataSet()

    i = 0
    while i < 100000:
        with open('./temp.txt', 'a') as f:
            result = test.next_batch(1000)

            for temp in result:
                f.write(str(temp))
                f.write('\n')

            f.write(str(i) + '~~~~~~')
            f.write('\n')

            print('shit')

        i += 1
