from data_pipeline import Sentences
import numpy as np
import tensorflow as tf


sentences = Sentences('./data_set/train', split_line=True, tensor_out=False, split_method="Twitter", w2v=False, label=False, matlabel=True)


def gen():
    for a in sentences:
        yield a

i = 0
while True:
    print(str(next(gen())))
    if i  % 1000 == 0:
        print(i)

    i += 1
