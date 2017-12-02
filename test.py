import logging
import gensim
import time
import spacy

# start timing
start = time.time()

# Login
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# loading model
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('./model/freebase-vectors-skipgram1000-en.bin', binary=True)


# sample gensim KeyValue function usage:
print(model.word_vec("office"))
print(model.doesnt_match("man woman child kitchen".split()))

print(model.most_similar("man"))
print(model.most_similar(positive =['girl', 'father'], negative = ['boy'], topn=3))
print(model.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse']))

print(model.similarity("breakfast", "lunch"))
print(model.similar_by_vector(model.word_vec("office")))

print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
print(model.n_similarity(['tree', 'brick'], ['forest', 'house']))
print(model.n_similarity(['tree', 'forest'], ['tree', 'brick']))
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

# # test line for freebase model
# print(model.most_similar(positive=['/en/forest', '/en/brick'], negative=['/en/tree'], topn=3))

# TODO: understand func evaluate_word_pairs
# model.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
# expected output: [('mother', 0.61849487), ('wife', 0.57972813), ('daughter', 0.56296098)]

# sample spacy token manipulation
sentence = "This is an open field west of a white house, with a boarded front door. There is a small mailbox here."
nlp = spacy.load('en')
doc = nlp(sentence)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)



more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    # print(predicted)
    print ('{} : {} :: {} : [{}]'.format(a, b, x, predicted))


# end timing and print out needed time
end = time.time()
print("time spent: ", end - start)
