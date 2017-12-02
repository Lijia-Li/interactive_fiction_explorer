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



# TODO: Output predicted word, but not necessarily verb.
# TODO: Also, it's hard to check they are verb or not if out of context.
    # ??? isverb is a boolean to indicate whether to predict verb or not
def predict_words (conons, word, word_position):
    m = 0
    sigma = 0
    for pair in conons:
        w1, w2 = pair.split()
        sigma += model.word_vec(w1) - model.word_vec(w2)
        m += 1
    a = (1/m)*sigma
    if word_position == 0:
        predicted = model.most_similar([word], [a], topn = 10)
        return predicted
    elif word_position == 1:
        predicted = model.most_similar([a, word], [], topn = 10)
        return predicted
    else:
        print("please put in correct cannon position")


# TODO check the graspability of the object (Fulda)
# TODO the output is not yet guarentee verbs
def pre_possible_action(sentence):
    nlp = spacy.load('en')
    doc = nlp(sentence)
    canons = ["sing song", "drink water", "read book", "eat food", "wear coat", "drive car", "ride horse",
              "give gift", "attack enemy", "say word", "open door", "climb tree", "heal wound", "cure disease",
              "paint picture"]
    dictionary = {}
    for chunk in doc.noun_chunks:
        word = chunk.root.text
        if word not in dictionary:
            dictionary[word] = predict_words(canons, word, 1)
    return dictionary


def pretty_print_dict(dictionary):
    print("")
    for key, val in dictionary.items():
        print(key, ":", val)


def main():
    tic = time.time()
    verb_noun = ["sing song", "drink water", "read book", "eat food", "wear coat", "drive car", "ride horse", "give gift",
                 "attack enemy", "say word", "open door", "climb tree", "heal wound", "cure disease", "paint picture"]
    noun_adj = ["knife sharp", "light bright", "ice cold", "fire burning", "desert dry", "sky blue", "night dark", "rope long"]
    words = ["book", "sword", "horse", "key"]
    for word in words:
        print(predict_words(verb_noun, word, 1))

    s = "Soon you’ll be able to send and receive money from friends and family right in Messages."
    s1 = "This is an open field west of a white house, with a boarded front door. There is a small mailbox here."
    s2 = "This is a forest, with trees in all directions around you."
    s3 = "This is a dimly lit forest, with large trees all around.  One particularly large tree with some low branches stands here."
    sentences = [s, s1, s2, s3]
    for sentence in sentences:
        print("-" * 3, sentence, "-" * 3)
        pretty_print_dict(pre_possible_action(sentence))
    toc = time.time()
    # print("total time spend:", toc - tic, "s")

if __name__ == "__main__": main()