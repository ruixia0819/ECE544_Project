import re
from data_pipeline import Sentences
import os



def __pre_process__(text):
    text = re.sub(r'@\w* ', '', text)
    text = re.sub(r'&\w*;', '', text)
    text = re.sub(r'&#\w*;', '', text)
    # text = re.sub(r'@', '', text)

    return text


def data_preprocess(sentences):
    path = './data_set/full/'
    if not os.path.exists(path):
        os.mkdir(path)

    for line in sentences:
        with open('./data_set/full/full.txt', 'a') as wf:
            wf.write(__pre_process__(line))


def data_split(sentences):
    path = './data_set/train/'
    if not os.path.exists(path):
        os.mkdir(path)

    path = './data_set/eval/'
    if not os.path.exists(path):
        os.mkdir(path)

    path = './data_set/test/'
    if not os.path.exists(path):
        os.mkdir(path)

    line_count = 0
    for _ in sentences:
        line_count += 1

    count = 0
    for line, label in sentences:
        if count / line_count < 0.6:
            if not os.path.exists(path):
                os.mkdir(path)
            with open('data_set/train/train.txt', 'a') as f:
                f.write(line)
                f.write('\n')
        elif count / line_count < 0.8:
            with open('data_set/eval/eval.txt', 'a') as f:
                f.write(line)
                f.write('\n')
        else:
            with open('data_set/test/test.txt', 'a') as f:
                f.write(line)
                f.write('\n')

        count += 1


def label_count(sentences):
    labels = []
    for _, label in sentences:
        # print(label)
        if label not in labels:
            labels.append(label)

    with open('./data_set/full/labels.txt', 'w') as f:
        f.write(str(labels))


def check_abuse(sentences):
    label_dict = {'surprise': 0, 'sadness': 1, 'joy': 2, 'disgust': 3, 'fear': 4, 'anger': 5}
    for data, label in sentences:
        if label not in label_dict:
            print(data)


def find_max_length(sentences):
    max_length = 0
    for line in sentences:
        if len(line) > max_length:
            max_length = len(line)

    return max_length


if __name__ == "__main__":
    # preprocess data
    #
    sentences = Sentences(dirname='./data_set/raw/')
    data_preprocess(sentences)

    # generate label list
    #
    sentences = Sentences(dirname='./data_set/full/', label=True)
    label_count(sentences)

    # check there are labels that are not expected to have
    #
    check_abuse(sentences)

    # split data into train, eval and test
    #
    data_split(sentences)

    sentences = Sentences(dirname='./data_set/full/', split_line=True, split_method = 'Twitter', label=False)
    print(find_max_length(sentences))
