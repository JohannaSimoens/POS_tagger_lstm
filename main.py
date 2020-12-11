from corpus import read_corpus
from lstm_pos_tagger import LSTMTagger
from lstm_pos_tagger_pretrained_emb import LSTMTagger_w2vec
import embeddings_builder
import torch
import torch.nn as nn
import torch.optim as optim


def create_dict_str_to_idx(datas):
    """
    create a dictionary to get a numerical encoding of symbolic features
    :param datas: list of list _ ex: [ [D, N, V, D, N], [D, N, V] ]
    :return: dictionary which maps a string (tag or word) to an index (unique integer) ex: {"D":0, "N":1}
    """
    str_to_idx = {}
    for data_example in datas:
        for str in data_example:
            if str not in str_to_idx:
                str_to_idx[str] = len(str_to_idx)
    return str_to_idx


def sequence_encoding(sequence, str_to_idx):
    """
    Transform list of strings into a tensor of integers to be processed by the pytorch model
    :param sequence: list of strings
    :param str_to_idx: dictionary that maps a string to a unique integer
    :return: pytorch tensor (vector) of long values
    """
    sequence_of_indexes = [str_to_idx[element] for element in sequence]
    return torch.tensor(sequence_of_indexes, dtype=torch.long)


# READ CORPUS, PREPARE DATA:

liste_X_train, liste_Y_train, liste_X_test, liste_Y_test, liste_X_dev, liste_Y_dev = read_corpus("sequoia-7.0/sequoia.deep.conll", 0.8, 0.2, 0)
liste_X_whole_corpus, liste_Y_whole_corpus, _, _, _, _ = read_corpus("sequoia-7.0/sequoia.deep.conll", 1, 0, 0)
print("len(liste_X_train): ", len(liste_X_train))
print("len(liste_X_test): ", len(liste_X_test), " len(liste_X_dev): ", len(liste_X_dev))
print("first element (x,y) train:  x = ", liste_X_train[1], ", y = ", liste_Y_train[1])

"""
import json
with open("pos_data_sequoia.txt", 'w', encoding="utf-8") as pos_data_sequoia_file:
    sequoia_pos_json = json.dumps({"train_data":{"X":liste_X_train,"Y":liste_Y_train},"test_data":{"X":liste_X_test,"Y":liste_Y_test},"dev_data":{"X":liste_X_dev,"Y":liste_Y_dev}})
    pos_data_sequoia_file.write(sequoia_pos_json)
"""

# Create dictionaries
tag_to_idx = create_dict_str_to_idx(liste_Y_whole_corpus)
word_to_idx = create_dict_str_to_idx(liste_X_whole_corpus)
print(tag_to_idx)

# Load word embeddings and create weight matrix
embedding_index = embeddings_builder.read_embeddings("vecs100-linear-frwiki")
weight_matrix_we, word_to_idx_w2vec = embeddings_builder.create_word_embeddings_matrix_from_corpus(liste_X_train+liste_X_test, embedding_index)
weight_matrix_we = torch.tensor(weight_matrix_we)
"""
# INSTANCIATION OF AN LSTM MODEL using only one-hot vectors to encode words
torch.manual_seed(1)
EMBEDDING_DIM = 68
HIDDEN_DIM = 32
NUMBER_EPOCHS = 10
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
"""

# INSTANCIATION OF AN LSTM MODEL with Pretrained Word Embeddings
torch.manual_seed(1)
PRETRAINED_EMBEDDING_DIM = len(embedding_index["le"])
HIDDEN_DIM = 32
NUMBER_EPOCHS = 10
model_pre_trained_we = LSTMTagger_w2vec(PRETRAINED_EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), weight_matrix_we)


# TRAINING STEP
loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.SGD(model_pre_trained_we.parameters(), lr=0.1)

for epoch in range(NUMBER_EPOCHS):
    print("epoch n° ", epoch)
    total_loss_one_epoch = 0
    for id_data, x_data in enumerate(liste_X_train):
        #model.zero_grad()
        model_pre_trained_we.zero_grad()
        # encode x and y
        #input_tensor = sequence_encoding(x_data, word_to_idx)
        input_tensor = sequence_encoding(x_data, word_to_idx_w2vec)
        expected_output = sequence_encoding(liste_Y_train[id_data], tag_to_idx)
        # forward pass
        # predicted_output = model(input_tensor)
        predicted_output = model_pre_trained_we(input_tensor)
        # calculate loss
        loss = loss_function(predicted_output, expected_output)
        # update weights
        loss.backward()
        total_loss_one_epoch += loss.item()
        optimizer.step()
    print("loss: ", total_loss_one_epoch/len(liste_X_train))


# TEST STEP ON 1 EXAMPLE
success_counts = 0
total_token_in_test = 0
with torch.no_grad():
    for id_data, x_data in enumerate(liste_X_test):
        #input_tensor = sequence_encoding(x_data, word_to_idx)
        input_tensor = sequence_encoding(x_data, word_to_idx_w2vec)
        #tag_scores = model(input_tensor)
        tag_scores = model_pre_trained_we(input_tensor)
        list_of_predicted_tag_idx = []
        for token in tag_scores:
            values, idx = token.max(0)
            list_of_predicted_tag_idx.append(idx.item())
        if id_data == 0:
            print("len(list_of_predicted_tag_idx) ", len(list_of_predicted_tag_idx))
            print("liste_Y_test[id_data] ", len(liste_Y_test[id_data]))
            print("x_data: ", x_data)
            print("list_of_predicted_tag_idx: ", list_of_predicted_tag_idx)
            print("liste_Y_test[id_data]:     ", [tag_to_idx[e] for e in liste_Y_test[id_data]])
        for id_y_data, y_data in enumerate(list_of_predicted_tag_idx):
            if int(y_data) == int(tag_to_idx[liste_Y_test[id_data][id_y_data]]):
                success_counts += 1
            total_token_in_test += 1
    print("accuracy:", (success_counts/total_token_in_test)*100)



"""
    print("X: ", liste_X_test[3])
    print("tag to Idx: ", tag_to_idx)
    input_tensor = sequence_encoding(liste_X_test[3], word_to_idx)
    output_tensor = [tag_to_idx[element] for element in liste_Y_test[3]]
    print("Y: ")
    print(output_tensor)
    tag_scores = model(input_tensor)
    print("predicted_output:")
    list_of_predicted_tag_idx = []
    for token in tag_scores:
        values, idx = token.max(0)
        list_of_predicted_tag_idx.append(idx.item())
    print(list_of_predicted_tag_idx)
"""





