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


# canons should be a list of pair of "noun verb"
# this function take in a list of canonical word pair, and a noun to predict the top 10 related word
# TODO: run spacy word test on output to guarantee verb output
def demo_more_verb (conons, n2):
    m = 0
    sigma = 0
    for pair in conons:
        v1, n1 = pair.split()
        sigma += model.word_vec(v1) - model.word_vec(n1)
        m += 1
        predicted = model.most_similar([n2, v1], [n1])[0][0]
        print('{} : {} :: [{}] : {}'.format(v1, n1, predicted, n2))
    a = (1 / m) * sigma
    predicted = model.most_similar([a, n2], [])
    print("--------------------------------------------")
    for i in range(10):
        print('canon :: [{}] : {})'.format(predicted[i][0], n2))


# isverb is a boolean to indicate whether to predict verb or not
def predict_words (conons, word, pre_verb):
    m = 0
    sigma = 0
    for pair in conons:
        v1, n1 = pair.split()
        sigma += model.word_vec(v1) - model.word_vec(n1)
        m += 1
    a = (1/m)*sigma
    if pre_verb == 1:
        predicted = model.most_similar([a, word], [], topn = 10)
    elif pre_verb == 0:
        predicted = model.most_similar([word], [a], topn = 10)
    return predicted



def main():
    word_list = ["sing song", "drink water", "read book", "eat food", "wear coat", "drive car", "ride horse", "give gift",
                 "attack enemy", "say word", "open door", "climb tree", "heal wound", "cure disease", "paint picture"]
    demo_more_verb(word_list, "book")
    print(predict_words(word_list, "book", pre_verb = 0))
    print(predict_words(word_list, "book", pre_verb = 1))

if __name__ == "__main__": main()