from data_pipeline import Sentences
import numpy as np
import tensorflow as tf


# sentences = Sentences('./data_set/train', split_line=True, tensor_out=True, max_length=140, split_method="Twitter", w2v=True, label=True, matlabel=True)
sentences = Sentences('./data_set/train', split_line=True, tensor_out=False, split_method="Twitter", w2v=False, label=True, matlabel=True)

a = sentences.__iter__()

def input_pipeline(batch_size):
    global a
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    try:
        text, label = next(a)
    except StopIteration as e:
        a = sentences.__iter__()
        text, label = next(a)

    # print(text)

    # shuffle batch
    #
    batch_text, batch_label = tf.train.shuffle_batch([text, label], batch_size=batch_size, capacity = capacity, min_after_dequeue=min_after_dequeue)
    # batch_text, batch_label = tf.train.shuffle_batch(list(a), batch_size=batch_size, capacity = capacity, min_after_dequeue=min_after_dequeue)

    # batch
    #
    # batch_text, batch_label = tf.train.batch(list(a), batch_size)

    return batch_text, batch_label


i = 0
while i < 100:
    with open('./temp.txt', 'a') as f:
        result = input_pipeline(3)

        for temp in result:
            f.write(str(temp))
            f.write('\n')

        f.write(str(i) + '~~~~~~')
        f.write('\n')

    i += 1

print('done')
