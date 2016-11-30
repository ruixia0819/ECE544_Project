import gensim, logging
from data_pipeline import Sentences


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train word2vec on the sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)


# load sentences to varibables
sentences = Sentences(dirname='./data_set/full', split_line=True, split_method='Twitter')
print(next(sentences.__iter__()))

# train the model
model = gensim.models.Word2Vec(sentences, size=100, min_count=1, workers=10)

# evaluate model
# model.accuracy('./Affection analysis database/test/Jan9-2012-tweets-clean.txt')

# save the model
model.save('models/model')

# load model
new_model = gensim.models.Word2Vec.load('models/model')

# Online training
# model.train(new_train_set)

print(new_model['good'].shape)
print(new_model.most_similar(positive=['birthday'], topn=4))
