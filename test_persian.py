import argparse
import pickle
import gensim
import math
import time
import random
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import transformers
from transformers import *
from extract_features import compute_feat
from torchtext.data import Field, Iterator, BucketIterator
from persian_test_model4_hrnn import conll_eval
from subprocess import run, PIPE
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

from data_conll.conll_utils import epoch_time, valid_conll_eval, data_padding
from data_conll.conll_utils import build_w2v, build_vocab, convert_to_word
#torch.autograd.set_detect_anomaly(True)


# EMBEDDING_DIM = 1024 # 768 base, 1024 large
EMBEDDING_DIM = 300 # for bilingual
HIDDEN_DIM = 100
NUM_LAYERS = 1
is_training = 0

BATCH_SIZE = 1
NUM_ITER = 50
L_RATE = 0.001
warmup = 50

##### load psuedo labels from compound pcfg model ####
test_data = pickle.load(open("persian_data/persian_psuedo_data_hrnn_model_test.pickle", "rb"))

# ##### initialize BERT model ####
bert_model = [BertModel, BertTokenizer, BertConfig, 'bert-large-cased']

from model4_hrnn import HRNNtagger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-training', default=1, type=int) 
    parser.add_argument('--dire', default="save_files/" , type=str) 	
    args = parser.parse_args()
    best_model_path = args.dire + 'persian-started-best-model-hrnn-finetuned.pt'
    # best_model_path = args.dire + 'best-model-hrnn.pt'
    pred_path_test_out = 'persian_best_test.txt'
    opt_path = args.dire + 'sgd.opt'

    print("------------->")
    print('[INFO] Build vocabulary.....')
    word_to_ix, ix_to_word, tag_to_ix = build_vocab(test_data)
    tag_to_ix = {"<pad>": 0, "1": 1, "2": 2, "0": 3}

    print("device is:", device)
    hrnn_model = HRNNtagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 3, BATCH_SIZE, device).to(device)

    loss_function = nn.NLLLoss().to(device)
    #optimizer = optim.SGD(hrnn_model.parameters(), lr=L_RATE)
    optimizer = optim.Adam(hrnn_model.parameters(), lr=L_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
        num_warmup_steps = warmup, num_training_steps = NUM_ITER, 
        num_cycles = 0.5, last_epoch = - 1)

    ############ test data ############
    test_tokens, test_tags, test_msl = data_padding(test_data, word_to_ix, tag_to_ix)
    get_sent = {v: k for k, v in word_to_ix.items()}

    print("------------->")
    print('[INFO] Create test embeddings matrix.....')

    test_matrix = compute_feat(bert_model, test_tokens, word_to_ix, ix_to_word, device)
    with open(args.dire+'persian_test_matrix.pkl', 'wb') as f:
        pickle.dump(test_matrix, f) 

    #test_matrix = pickle.load(open("hrnn_pcfg/hrnn5/" + "test_matrix.pkl", "rb"))

    test_final_data = [(test_tokens[i], test_tags[i]) for i in range(0, len(test_tokens))] 	
    test_iterator = BucketIterator(test_final_data, 
        batch_size=BATCH_SIZE, sort_key=lambda x: np.count_nonzero(x[0]), sort=False, 
        shuffle=False, sort_within_batch=False, device = device)

    loss_nimp = conll_eval(hrnn_model, pred_path_test_out, test_iterator, loss_function, test_matrix, test_msl, best_model_path)

    print(f'| Test Loss: {loss_nimp:.3f}')


if __name__ == "__main__":
	main()
