import gensim, logging
from data_pipeline import Sentences

# load model
new_model = gensim.models.Word2Vec.load('models/model')

word = 'shit'
print('the word most similar to [%s] is:' % word)
print(new_model.most_similar(positive=[word], topn=5))
