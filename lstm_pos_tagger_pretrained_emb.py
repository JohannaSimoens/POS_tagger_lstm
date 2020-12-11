# Author: Johanna Simoens

import torch.nn as nn
import torch.nn.functional as functional


class LSTMTagger_w2vec(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, weight_tensor):
        super(LSTMTagger_w2vec, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(weight_tensor)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = functional.log_softmax(tag_space, dim=1)
        return tag_scores
