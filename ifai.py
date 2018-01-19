import time
from os.path import exists as file_exists, splitext as split_ext

import gensim
import nltk.stem
from nltk.corpus import wordnet as wn
import numpy as np
import spacy
import requests

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
        a, b = pair.split()
        sigma += model.word_vec(a) - model.word_vec(b)
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
    wnl = nltk.stem.WordNetLemmatizer()
    for verb in model_verb:
        verb = wnl.lemmatize(str(verb[0].lower()))

        # use wordnet to assert verb (can be a verb)
        if wn.morphy(verb, wn.VERB):
            word2vec_words.append(verb)

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
    canons = list(filter(None, [line.rstrip() for line in open('./word_lists/adj_noun_pair.txt')]))
    sigma = get_ave_sigma(model, canons)
    model_adj = model.most_similar([sigma, noun], [], topn=10)
    word2vec_adj = []
    for adj in model_adj:
        word2vec_adj.append(adj[0])
    return word2vec_adj


def get_verbs_with_adjective(model, adj):
    canons = []
    with open('word_lists/verb_adj_pair.txt') as fd:
        canons.extend(line.strip() for line in fd.readlines())
    sigma = get_ave_sigma(model, canons)
    model_verbs = model.most_similar([sigma, adj], [], topn = 10)
    word2vec_verb = []
    for verb in model_verbs:
        word2vec_verb.append(verb[0])
    return word2vec_verb

# return a list of possible actions by compute affordable actions on nouns in the given sentence
def possible_actions(model, sentence):
    # prepare tools
    nlp = spacy.load('en')
    doc = nlp(sentence)
    wnl = nltk.stem.WordNetLemmatizer()
    l = []
    action_pair = []

    for chunk in doc.noun_chunks:
        word = wnl.lemmatize(chunk.root.text)
        l.append(word)
    sorted_list = rank_manipulability(model, l)
    for word in sorted_list:
        verbs = get_verbs_for_noun(model, word[0])
        for verb in verbs:
            action_pair.append(verb + " " + word[0])
    return action_pair

# get possible tools to realize the intended action
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

# rank inputs nouns from most manipulative to least manipulative
def rank_manipulability(model, nouns):
    x_axis = model.word_vec("forest") - model.word_vec("tree")
    dic = {}
    for noun in nouns:
        if noun not in dic:
            vec = model.word_vec(noun)
            dic[noun] = np.dot(vec, x_axis)
    sorted_list = sorted(dic.items(), key=(lambda kv: kv[1]))
    return sorted_list

# get "capable of" & "used for" relations from ConceptNet, return a list of possible verbs with weight
def get_verbs_cn(noun):
    v_dic = {}
    wnl = nltk.stem.WordNetLemmatizer()

    # query conceptNet
    rel_list = ["CapableOf", "UsedFor"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + noun + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:

            # get the possible verb
            verb = edge["end"]["label"].split()[0]
            verb = wnl.lemmatize(verb, 'v')

            # use wordnet to assert verb (can be a verb)
            if wn.morphy(verb, wn.VERB):

                # add to dic with weight
                if verb not in v_dic:
                    v_dic[verb] = edge["weight"]
                if verb in v_dic:
                    v_dic[verb] += edge["weight"]


    sorted_list = sorted(v_dic.items(), key=(lambda kv: kv[1]), reverse=True)
    return sorted_list[:10]

# return a list of adj best describe the noun from ConceptNet with "has property" Relation
def get_adjectives_cn(noun):
    v_dic = {}
    wnl = nltk.stem.WordNetLemmatizer()

    rel_list = ["HasProperty"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + noun + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:

            # get the possible verb
            word = edge["end"]["label"].split()[0]
            word = wnl.lemmatize(word, 'v')

            # add to dic with weight
            if word not in v_dic:
                v_dic[word] = edge["weight"]
            if word in v_dic:
                v_dic[word] += edge["weight"]

    sorted_list = sorted(v_dic.items(), key=(lambda kv: kv[1]), reverse=True)
    return sorted_list

# return a list of synonym of the noun from ConceptNet and wordnet (not ideal)
def get_synonyms(word, pos):
    # initialize the list with wordnet synonyms
    syn_list = []
    for lemma in wn.synset(word + "." + pos + ".01").lemmas():
        syn = lemma.name()
        if syn != word:
            syn_list.append(syn)

    # add conceptNet's Synonyms
    rel_list = ["Synonym", "IsA"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + word.replace(" ", "_") + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:
            if edge["end"]["language"] == 'en':
                syn = edge["end"]["label"]
                if syn not in syn_list and syn != word:
                    syn_list.append(syn)
    return syn_list

# return a list of locations that the noun possibly located according to ConceptNet's relations ("at location", "locate near", "part of"
def get_locations_cn(noun):
    loca_list = []
    rel_list = ["AtLocation", "LocatedNear", "PartOf"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + noun.replace(" ", "_") + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:
            if edge["end"]["language"] == 'en':
                syn = edge["end"]["label"]
                if syn not in loca_list and syn != noun:
                    loca_list.append(syn)
    return loca_list

def combine_verbs_ls(model, noun):
    # initialize both list
    w2v_ls = get_verbs_for_noun(model, noun)
    cn_ls = get_verbs_cn(noun)

    # pass word2vec list to the combine list
    combine_ls = w2v_ls
    for element in cn_ls:
        verb = element[0]
        if verb not in combine_ls:
            combine_ls.append(verb)

    return combine_ls


def main():
    model = load_model(DEFAULT_MODEL_PATH)

    # start timing
    tic = time.time()

    # prepare samples
    test_nouns = ["book", "sword", "horse", "key"]
    test_verbs = ["climb", "use", "open", "lift", "kill", "murder", "drive", "ride", "cure", "type", "sing"]
    test_adjectives = ["sharp", "heavy", "hot", "iced", "clean", "long"]
    s = "Soon you’ll be able to send and receive money from friends and family right in Messages."
    s1 = "This is an open field west of a white house, with a boarded front door. There is a small mailbox here."
    s2 = "This is a forest, with trees in all directions around you."
    s3 = "This is a dimly lit forest, with large trees all around.  One particularly large tree with some low branches stands here."
    s4 = "You open the mailbox, revealing a small leaflet."
    sentences = [s, s1, s2, s3, s4]

    # run samples

    # get_verb_cn tests compare with get_verbs_for_noun tests
    print("-" * 5, "obtain verbs tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":")
        print("ConceptNet:", get_verbs_cn(noun))
        print("word2vec result:", get_verbs_for_noun(model, noun))
        print("combine version", combine_verbs_ls(model, noun))
        print()

    # get_adjectives_cn compare with get_adjectives_for_noun tests
    print("-" * 5, "obtain adjetcives tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":")
        print("ConceptNet:", get_adjectives_cn(noun))
        print("word2vec result:", get_adjectives_for_noun(model, noun))
        print()

    # get_verbs_with_adjective tests
    print("-" * 5, "get_verbs_with_adjective", "-" * 5)
    for adj in test_adjectives:
        print(adj, ":", get_verbs_with_adjective(model, adj))

    # possible_actions tests
    for sentence in sentences:
        print()
        print(sentence)
        print(possible_actions(model, sentence))

    # get_tools_for_verb tests
    for verb in test_verbs:
        print()
        print("-" * 5, "get_tools_for_verb function test", "-" * 5)
        print(verb, ":", get_tools_for_verb(model, verb))

    # get_used_for tests
    print("-" * 5, "get_used_for test", "-" * 5)
    for noun in test_nouns:
        print(get_verbs_cn(noun))

    # get_synonyms tests
    print("-" * 5, "obtain synonyms tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":", get_synonyms(noun, 'n')[:10])

    # get_locations_cn tests
    print("-" * 5, "obtain locations tests", "-" * 5)
    for noun in test_nouns:
        print(noun, ":", get_locations_cn(noun)[:10])

    toc = time.time()
    print("total time spend:", toc - tic, "s")


if __name__ == "__main__":
    main()
