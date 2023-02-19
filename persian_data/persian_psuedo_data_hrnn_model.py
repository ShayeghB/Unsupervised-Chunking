import pickle

tokens = pickle.load(open("persian_data/persian_sentences.pickle", "rb"))
tags = pickle.load(open("persian_data/persian_labels.pickle", "rb"))
tags = [['2' if t=='O' else '1' if t=='I-NP' else '2' for t in tag] for tag in tags]
pickle.dump(tags, open("persian_data/persian_data_test_tags.pickle", "wb"))
data = list(zip(tokens, tags))
print(data[0])
pickle.dump(data[-300:], open("persian_data/persian_psuedo_data_hrnn_model_test.pickle", "wb"))
pickle.dump(data[-500:-300], open("persian_data/persian_psuedo_data_hrnn_model_val.pickle", "wb"))
pickle.dump(data[:-500], open("persian_data/persian_psuedo_data_hrnn_model_train.pickle", "wb"))
pickle.dump([['B' if a=='2' else 'I' for a in d[1]] for d in data[-300:]], open("persian_data/persian_test_tags.pickle", "wb"))
pickle.dump([['B' if a=='2' else 'I' for a in d[1]] for d in data[-500:-300]], open("persian_data/persian_val_tags.pickle", "wb"))
