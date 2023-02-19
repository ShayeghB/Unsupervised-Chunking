CPCFG_PATH = "comppcfg_test_out.txt"
TOKEN_PATH = "../conll/data_test_tokens.pkl"
TARGET_PATH = "test_psuedo_data_hrnn_model2.pkl"

import pickle as pkl

tokens = pkl.load(open(TOKEN_PATH, 'rb'))
cpcfg = open(CPCFG_PATH).readlines()
cpcfg_line_no = 0
tags = []
for sentence in tokens:
    sentence_tags = []
    cpcfg_line_no += 1
    for token in sentence[:-1]:
        sentence_tags.append('2' if cpcfg[cpcfg_line_no].split()[-1]=='B' else '1')
        cpcfg_line_no += 1
    sentence_tags.append('1')
    tags.append(sentence_tags)

targets = [
    [sentence_tokens, sentence_tags]
    for sentence_tags, sentence_tokens in zip(tags, tokens)
]

pkl.dump(open(TARGET_PATH, 'wb'))