#!/user/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import GHGAT, TextEncoder, EntityEncoder, Pooling, MatchingTransform

class Classifier(nn.Module):
    def __init__(self, params, vocab_size, pte=None):
        super(Classifier, self).__init__()
        self.params = params
        self.vocab_size = vocab_size
        self.pte = False if pte is None else True
        self.model = GHGAT(params)
        self.text_encoder = TextEncoder(params)
        self.enti_encoder = EntityEncoder(params)
        self.topi_encoder = nn.Embedding(100, 100)
        self.topi_encoder.from_pretrained(torch.eye(100))
        self.match_encoder = MatchingTransform(params)
        # self.match_encoder = ConcatTransform(params)
        self.word_embeddings = nn.Embedding(vocab_size, params.emb_dim)
        if pte is None:
            nn.init.xavier_uniform_(self.word_embeddings.weight)
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pte))

        self.pooling = Pooling(params)
        self.encoder = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=512, dropout=0)
        self.classifier_sen = nn.Linear(params.node_emb_dim, params.ntags)
        self.classifier_ent = nn.Linear(params.node_emb_dim, params.ntags)
        self.dropout = nn.Dropout(params.dropout, )

    def forward(self, documents, ent_desc, doc_lens, ent_lens, adj_lists, feature_lists, sentPerDoc, entiPerDoc=None,
                all_distance=None):
        x_list = []
        embeds_docu = self.word_embeddings(documents)
        d = self.text_encoder(embeds_docu, doc_lens)
        d = self.dropout(F.relu_(d))
        x_list.append(d)
        if self.params.node_type == 3 or self.params.node_type == 2:
            embeds_enti = self.word_embeddings(ent_desc)
            e = self.enti_encoder(embeds_enti, ent_lens, feature_lists[1])
            e = self.dropout(F.relu_(e))
            x_list.append(e)
        if self.params.node_type == 3 or self.params.node_type == 1:
            t = self.topi_encoder(feature_lists[-1])
            x_list.append(t)

        X = self.model(x_list, adj_lists, all_distance)
        X_s = self.pooling(X[0], sentPerDoc)
        output = self.classifier_sen(X_s)

        if entiPerDoc is not None:
            E_GCN = X[1]
            # E_KB = self.gating(x_list[1], feature_lists[1])
            E_KB = x_list[1]
            X_e = self.match_encoder(E_GCN, E_KB)
            X_e = self.pooling(X_e, entiPerDoc)
            X_e = self.classifier_ent(X_e)
            output += X_e
        output = F.softmax(output, dim=1)
        # output = torch.sigmoid(output)
        return output
