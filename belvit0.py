# Importing the required libraries
from typing import List, Any

import nltk
import numpy as np
import itertools
from nltk.corpus import sidama

# Getting the tagged sentences
sidama_tagged_words = sidama.tagged_words()
sidama_tagged_sents = sidama.tagged_sents()
tags = [tag for (tag, word) in sidama_tagged_words]

# Splitting the data for train and test
size = int(len(sidama_tagged_sents) * 0.9)
train_data = sidama_tagged_sents[:size]
test_data = sidama_tagged_sents[size:]

# train_tagged_words=list[word for (tag, word) in train_data]
# ttest_tagged_words = list[word for (tag, word) in test_data]

##train_tagged_words = [word for (tag, word) in train_tagged_words]

sid_sent_tag = []
for s in sidama_tagged_sents:
    s.insert(0, ('START', 'START'))
    s.append(('END', 'END'))
    sid_sent_tag.append(s)

##train_tagged_words = [ tup for sent in train_data for tup in sent ]
##test_tagged_words = [ tup for sent in test_data for tup in sent ]
# =================================================The Emission Probabilities ================================
print('=======================================================================')
print('The Emission Probabilities')


# Creating a dictionary whose keys are tags and values contain words which were assigned the correspoding tag
def emission_prob():
    train_word_tag = {}
    for s in train_data:
        for (w, t) in s:
            w = w.lower()
            try:
                try:
                    train_word_tag[t][w] += 1
                except:
                    train_word_tag[t][w] = 1
            except:
                train_word_tag[t] = {w: 1}

    # Calculating the emission probabilities
    train_emission_prob = {}
    for k in train_word_tag.keys():
        train_emission_prob[k] = {}
        count = sum(train_word_tag[k].values())
        for k2 in train_word_tag[k].keys():
            p_emiss = train_word_tag[k][k2] / count
            ##    print(str(k2)  +' / '  +str(k) + " :  ", p_emiss)
            return p_emiss


# ========================================== The Transition Probabilities =======================
print('=======================================================================')
print('The Transition Probabilities')


# Estimating the bigram of tags to be used for transition probability
def transition_prob():
    bigram_tag_data = {}
    for s in train_data:
        bi = list(nltk.bigrams(s))
        for b1, b2 in bi:
            try:
                try:
                    bigram_tag_data[b1[1]][b2[1]] += 1
                except:
                    bigram_tag_data[b1[1]][b2[1]] = 1
            except:
                bigram_tag_data[b1[1]] = {b2[1]: 1}

    # Calculating the probabilities of tag bigrams for transition probability
    bigram_tag_prob = {}
    for i in bigram_tag_data.keys():
        bigram_tag_prob[i] = {}
        count = sum(bigram_tag_data[i].values())
        for j in bigram_tag_data[i].keys():
            p_trans = bigram_tag_data[i][j] / count
            ##    print(str(i) + '/' +  str(j) +  ":   ",  p_trans)
            return p_trans
            # Laplace Smoothing
            p_trans_lsm = (bigram_tag_data[i][j] + 1) / (count + 31)
            ##    print(str(i) + ' / ' +  str(j) +  ":   ",  p_trans_lsm)
            return p_trans_lsm


##def viterbi(sentence, tag_list, p_trans_lsm,  p_emiss, tag_count, word_set):
def viterbi():
    ##  inptext = input("Enter a sentence:   ")
    ##  sentence = inptext.strip("\n")
    ##  word_list = sentence.split(" ")
    # Calculating the possible tags for each word
    tags_of_tokens = {}
    count = 0
    for s in train_data:
        for (w, t) in s:
            w = w.lower()
            try:
                if t not in tags_of_tokens[w]:
                    tags_of_tokens[w].append(t)
            except:
                l = []
                tags_of_tokens[w] = l

    for s in test_data:
        for (w, t) in s:
            w = w.lower()
            try:
                if t not in tags_of_tokens[w]:
                    tags_of_tokens[w].append(t)
            except:
                l = []
                l.append(t)
                tags_of_tokens[w] = l

    # Dividing the test data into test words and test tags
    test_words = []
    test_tags = []
    for s in test_data:
        temp_word = []
        temp_tag = []
        for (w, t) in s:
            temp_word.append(w.lower())
            temp_tag.append(t)
        test_words.append(temp_word)
        test_tags.append(temp_tag)

    # Executing the Viterbi Algorithm
    predicted_tags = []  # intializing the predicted tags
    for x in range(len(test_words)):  # for each tokenized sentence in the test data
        s = test_words[x]
        # storing_values is a dictionary which stores the required values
        # ex: storing_values = {step_no.:{state1:[previous_best_state,value_of_the_state]}}
        storing_values = {}
        for q in range(len(s)):
            step = s[q]
            # for the starting word of the sentence
            if q == 1:
                storing_values[q] = {}
                tags = tags_of_tokens[step]
                for t in tags:
                    # this is applied since we do not know whether the word in the test data is present in train data or not
                    try:
                        storing_values[q][t] = ['START', p_trans['START'][t] * p_emiss[t][step]]
                    # if word is not present in the train data but present in test data we assign a very low probability of 0.0001
                    except:
                        storing_values[q][t] = ['START', 0.0001]  # *train_emission_prob[t][step]]

            # if the word is not at the start of the sentence
            if q > 1:
                storing_values[q] = {}
                previous_states = list(storing_values[q - 1].keys())  # loading the previous states
                current_states = tags_of_tokens[step]  # loading the current states
                # calculation of the best previous state for each current state and then storing
                # it in storing_values
                for t in current_states:
                    temp = []
                    for pt in previous_states:
                        try:
                            temp.append(storing_values[q - 1][pt][1] * p_trans[pt][t] * p_emiss[t][step])
                        except:
                            temp.append(storing_values[q - 1][pt][1] * 0.0001)
                    max_temp_index = temp.index(max(temp))
                    best_pt = previous_states[max_temp_index]
                    storing_values[q][t] = [best_pt, max(temp)]

        # Backtracing to extract the best possible tags for the sentence


##
##    pred_tags = []
##    total_steps_size = storing_values.keys()
##    last_step_size = max(total_steps_size)
##    for bs in range(len(total_steps_size)):
##      step_size = last_step_size - bs
##      if step_size == last_step_size:
##        pred_tags.append('END')
##        pred_tags.append(storing_values[step_size]['END'][0])
##      if step_size<last_step_size and step_size>0:
##        pred_tags.append(storing_values[step_size][pred_tags[len(pred_tags)-1]][0])
##        predicted_tags.append(list(reversed(pred_tags)))

# Check how a sentence is tagged by the two POS taggers and compare them
test_sent = "daraaro onte ooso iltino."
##print(viterbi(test_sent.split()))
vit_pred_tags = viterbi(test_sent.split())
# print(words, train_words)
print(vit_pred_tags)
##        viterbi(vit_pred_tags)
##viterbi()
