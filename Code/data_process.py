import re
from data_pipeline import Sentences
import os
import random


label_dict = {'surprise': 0,
              'sadness': 0,
              'joy': 0,
              'disgust': 0,
              'fear': 0,
              'anger': 0}

sw_list = [" in "," getting "," out "," the "," to "," a "," I'm "," It's "," come "," im "," from "," n "," over "," and "," ! "," is "," its "," myself "," very "," be "," way "," you "," has "," this "," I "," get "," me "," I've "," give "," so "," after "," your "," would "," one "," into "," it "," see "," little "," my "," as "," on "," My "," 1 "," other "," up "," much "," time "," with "," it's "," The "," ya "," at "," / "," an "," everyone "," possible "," for "," go "," again "," more "," One "," watching "," Just "," do "," And "," am "," all "," turned "," - "," just "," What "," are "," got "," think "," post "," under "," He "," change "," been "," having "," that "," better "," Its "," good "," even "," about "," always "," last "," So "," kind "," say "," called "," i "," It "," killing "," back "," know "," u "," what "," All "]


def __pre_process__(text):
    text = re.sub(r'@\w* ', '', text)
    text = re.sub(r'&\w*;', '', text)
    text = re.sub(r'&#\w*;', '', text)
    text = re.sub(r'#\w*', '', text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'[,.;"]+', '', text)

    # for word in sw_list:
    #     text = text.replace(word, ' ')


    return text


def data_preprocess(sentences):
    path = './data_set/full/'
    if not os.path.exists(path):
        os.mkdir(path)

    path = './data_set/labeled/'
    if not os.path.exists(path):
        os.mkdir(path)

    for line, label in sentences:
        # with open('./data_set/full/full.txt', 'a') as wf:
        #     wf.write(__pre_process__(line))

        if label not in label_dict:
            continue

        if label_dict[label] >= 3800:
            if label == 'joy':
                continue
        with open('./data_set/labeled/'+ label +'.txt', 'a') as wf:
            wf.write('\t')
            wf.write(__pre_process__(line))
            wf.write('\t')
            wf.write(label)
            wf.write('\n')
            label_dict[label] += 1

        with open('./data_set/full/full.txt', 'a') as wf:
            wf.write('\t')
            wf.write(__pre_process__(line))
            wf.write('\t')
            wf.write(label)
            wf.write('\n')

    with open('./data_set/full/full.txt', 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open('./data_set/full/full.txt', 'w') as target:
        for _, line in data:
            target.write(line)





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
                # f.write('\n')
        elif count / line_count < 0.8:
            with open('data_set/eval/eval.txt', 'a') as f:
                f.write(line)
                # f.write('\n')
        else:
            with open('data_set/test/test.txt', 'a') as f:
                f.write(line)
                # f.write('\n')

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
    sentences = Sentences(dirname='./data_set/raw/', split_line=True, stop_word=False, label=True)
    data_preprocess(sentences)
    print(label_dict)

    # generate label list
    #
    sentences = Sentences(dirname='./data_set/full/', label=True)
    # label_count(sentences)

    # check there are labels that are not expected to have
    #
    # check_abuse(sentences)


    # split data into train, eval and test
    #
    # data_split(sentences)

    sentences = Sentences(dirname='./data_set/full/', split_line=True, split_method = 'Twitter', label=False, caseless=False)
    print(find_max_length(sentences))
