import logging
import gensim
from gensim.models import word2vec
import time

# start timing
start = time.time()

# GoogleNews-vectors-negative300.bin.gz filter=lfs diff=lfs merge=lfs -text

# Login
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# loading model
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('./model/freebase-vectors-skipgram1000-en.bin', binary=True)

#
# print(model.word_vec("office"))
#
# print(model.doesnt_match("man woman child kitchen".split()))
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))
#
# print(model.most_similar("man"))
# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
# print(model.most_similar(['girl', 'father'], ['boy'], topn=3))
# print(model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'], topn=15))
# print(model.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse']))
#
# print(model.similarity("breakfast", "lunch"))
print(model.similar_by_vector(model.word_vec("office")))
# print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
#
# print(model.n_similarity(['tree', 'brick'], ['forest', 'house']))
# print(model.n_similarity(['tree', 'forest'], ['tree', 'brick']))
# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

# model.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))

# expected output [('mother', 0.61849487), ('wife', 0.57972813), ('daughter', 0.56296098)]

print(model.most_similar(positive=['/en/forest', '/en/brick'], negative=['/en/tree'], topn=3))


# vocab = model.vocab.keys()
#
# fileNum = 1
#
# wordsInVocab = len(vocab)
# wordsPerFile = int(100E3)
#
# # Write out the words in 100k chunks.
# for wordIndex in range(0, wordsInVocab, wordsPerFile):
#     # Write out the chunk to a numbered text file.
#     with open("vocabulary/vocabulary_%.2d.txt" % fileNum, 'w') as f:
#         # For each word in the current chunk...
#         for i in range(wordIndex, wordIndex + wordsPerFile):
#             # Write it out and escape any unicode characters.
#             f.write(vocab[i].encode('UTF-8') + '\n')
#
#     fileNum += 1

# more_examples = ["he his she", "big bigger bad", "going went being"]
# for example in more_examples:
#     a, b, x = example.split()
#     predicted = model.most_similar([x, b], [a])[0][0]

# print"'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
# 'he' is to 'his' as 'she' is to 'her'
# 'big' is to 'bigger' as 'bad' is to'worse'
# 'going' is to'went' as 'being' is to'was'

#
# sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
# vectors = [model[w] for w in sentence]
# print(vectors)

# end timing and print out needed time
end = time.time()
print("time spent: ", end - start)
