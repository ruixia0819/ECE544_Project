from data_pipeline import Sentences
import numpy as np
import tensorflow as tf
import gensim
from nltk.tokenize import TweetTokenizer


# sentences = Sentences('./data_set/train', split_line=True, tensor_out=True, split_method="Twitter", w2v=True, label=True, matlabel=True)

model = gensim.models.Word2Vec.load('models/model')

def w2v(text):
    return model[text]

label_dict = label_dict = {'surprise': [1,0,0,0,0,0],
                                        'sadness': [0,1,0,0,0,0],
                                        'joy': [0,0,1,0,0,0],
                                        'disgust': [0,0,0,1,0,0],
                                        'fear': [0,0,0,0,1,0],
                                        'anger': [0,0,0,0,0,1]}

tknz = TweetTokenizer()

def read_file_preprocess(filename):
    reader = tf.TextLineReader()
    key, value = reader.read(filename)
    print(str(value))
    text, label = tf.decode_csv(value, record_defaults=[[3], [3]])

    # line = line.strip('\n').split('\t')

    # text = line[1]
    print(tf.Print(text, [text]))
    text = tknz.tokenize(str(text))
    text = list(map(w2v, text))

    # raw_label = line[2].strip(':: ')
    label = self.label_dict[str(label)]

    return text, label

def input_pipeline(batch_size, filename='./data_set/train/train_.csv'):
    filename_queue = tf.train.string_input_producer(
        [filename], shuffle=True)
    text, label = read_file_preprocess(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size


    batch_text, batch_label = tf.train.shuffle_batch([text, label], batch_size=batch_size, capacity = capacity, min_after_dequeue=min_after_dequeue)

    return batch_text, batch_label


i = 0
while i < 10:
    with open('./temp.txt', 'a') as f:
        result = input_pipeline(batch_size=3)

        for temp in result:
            # with tf.Session() as sess:
            f.write(str(temp))
            f.write('\n')
        f.write('~~~~~~')
        f.write('\n')

    i += 1
