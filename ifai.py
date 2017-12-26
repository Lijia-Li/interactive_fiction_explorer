import time
from os.path import exists as file_exists, splitext as split_ext

import gensim
import nltk.stem
import spacy

# download wordnet
nltk.download('wordnet')

DEFAULT_MODEL_PATH = 'models/GoogleNews-vectors-negative300.bin'


def load_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    cache_path = split_ext(model_path)[0] + '.cache'
    if file_exists(cache_path):
        # use the cached version if it exists
        model = gensim.models.KeyedVectors.load(cache_path)
    else:
        # otherwise, load from word2vec binary, but cache the result
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        model.init_sims()
        # ignore=[] means ignore nothing (ie. save all pre-computations)
        model.save(cache_path, ignore=[])
    return model


# helper function that compute the average sigma (vector difference of word pair) of a list of given canon pairs
def get_ave_sigma(model, canons):
    sigma = 0
    for pair in canons:
        v, n = pair.split()
        sigma += model.word_vec(v) - model.word_vec(n)
    ave_sigma = (1 / len(canons)) * sigma
    return ave_sigma


# return a list of lemmatized verbs that the noun can afford
def get_verbs_for_noun(model, noun):
    # load word lists
    canons = []
    with open('word_lists/verb_noun_pair.txt') as fd:
        canons.extend(line.strip() for line in fd.readlines())
    verb_list = []
    with open('./word_lists/top_1000_verbs.txt') as fd:
        verb_list.extend(line.strip() for line in fd.readlines())

    # prepare tools
    wnl = nltk.stem.WordNetLemmatizer()
    sigma = get_ave_sigma(model, canons)

    # list of common used verbs
    navigation_verbs = [
        "north", "south", "east", "west", "northeast", "southeast", "southwest", "northwest", "up", "down", "enter",
        "exit"
    ]
    essential_manipulation_verbs = ["get", "drop", "push", "pull", "open", "close"]

    # extract words from word2vec model & append lemmatized word to list
    model_verb = model.most_similar([sigma, noun], [], topn=10)
    word2vec_words = []
    for verb in model_verb:
        word2vec_words.append(wnl.lemmatize(str(verb[0].lower())))

    # set operations
    affordant_verbs = list(set(verb_list) & set(word2vec_words))
    final_verbs = list(set(navigation_verbs) | set(essential_manipulation_verbs) | set(affordant_verbs))

    # -----------test lines (uncomment below four lines to view different set of verbs)-------------
    #     print("-"*10, noun, "-"*10)
    #     print("word2vec words: ", word2vec_words)
    #     print("affordant verbs: ", affordant_verbs)
    #     print("final verbs: ", final_verbs)

    return affordant_verbs


# return a list of adjectives that describe the given noun
def get_adjectives_for_noun(model, noun):
    canons = list(filter(None, [line.rstrip() for line in open('./word_lists/noun_adj_pair.txt')]))
    #     canons = ["knife sharp", "light bright", "ice cold", "fire burning", "desert dry", "sky blue", "night dark",
    #                 "rope long"]
    sigma = get_ave_sigma(model, canons)
    model_adj = model.most_similar([sigma, noun], [], topn=10)
    word2vec_adj = []
    for adj in model_adj:
        word2vec_adj.append(adj[0])
    return word2vec_adj


# return a list of possible actions by compute affordable actions on nouns in the given sentence
def possible_actions(model, sentence):
    # prepare tools
    nlp = spacy.load('en')
    doc = nlp(sentence)
    wnl = nltk.stem.WordNetLemmatizer()

    # create dictionary in the form [noun: verbs]
    dictionary = {}
    for chunk in doc.noun_chunks:
        word = wnl.lemmatize(chunk.root.text)
        if word not in dictionary:
            dictionary[word] = get_verbs_for_noun(model, word)

    # loop through dictionary to create action list
    action_pair = []
    for key, values in dictionary.items():
        for value in values:
            action_pair.append(value + " " + key)
    return action_pair


def get_tools_for_verb(model, verb):
    canons = []
    with open('./word_lists/verb_noun_pair.txt') as fd:
        canons.extend(line.strip() for line in fd.readlines())
    sigma = get_ave_sigma(model, canons)

    model_tools = model.most_similar([verb], [sigma], topn=10)
    word2vec_tools = []
    for tool in model_tools:
        word2vec_tools.append(tool[0])
    return word2vec_tools


def main():
    model = load_model(DEFAULT_MODEL_PATH)

    # start timing
    tic = time.time()

    # prepare samples
    test_nouns = ["book", "sword", "horse", "key"]
    test_verbs = ["climb", "use", "open", "lift", "kill", "murder", "drive", "ride", "cure", "type", "sing"]
    s = "Soon you’ll be able to send and receive money from friends and family right in Messages."
    s1 = "This is an open field west of a white house, with a boarded front door. There is a small mailbox here."
    s2 = "This is a forest, with trees in all directions around you."
    s3 = "This is a dimly lit forest, with large trees all around.  One particularly large tree with some low branches stands here."
    sentences = [s, s1, s2, s3]

    # run samples

    # get_verbs_for_noun tests
    print("-" * 5, "get_verbs_for_noun function tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":", get_verbs_for_noun(model, noun))
    print()

    # get_adjectives_for_noun tests
    print("-" * 5, "get_adjectives_for_noun function tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":", get_adjectives_for_noun(model, noun))

    # possible_actions tests
    for sentence in sentences:
        print()
        print(sentence)
        print(possible_actions(model, sentence))

    # get_tools_for_verb tests
    for verb in test_verbs:
        print(verb, ":", get_tools_for_verb(model, verb))

    toc = time.time()
    print("total time spend:", toc - tic, "s")


if __name__ == "__main__":
    main()
