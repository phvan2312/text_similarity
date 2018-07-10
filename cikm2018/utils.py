import pandas as pd
import numpy as np
import nltk
from gensim.models import KeyedVectors
import os

class TextUtils:
    def __inni__(self):
        pass

    def clean_str(self, text):

        pass

    def __mapping(self, vocab):
        assert type(vocab) is dict

        sorted_items = sorted(vocab.items(), key=lambda elem: -elem[1])
        id2x = {i: v[0] for i, v in enumerate(sorted_items)}
        x2id = {v[0]:i for i,v in enumerate(sorted_items)}

        return id2x, x2id

    def tokenize_word(self, sentence, language):
        tokens = nltk.word_tokenize(sentence, language=language)
        return tokens

    """
    sentences must be tokenized
    """
    def create_word_vocab(self, sentences, fasttext_path = None, word_dim = None):
        assert type(sentences) is list

        word_vocab = {}
        for sentence in sentences:
            for token in sentence:
                if token in word_vocab: word_vocab[token] += 1
                else: word_vocab[token] = 1

        "Now we have to load word in the pretrained file one"
        pretrained_tokens = []
        pretrained_model = None
        if fasttext_path is not None:
            _, ext = os.path.splitext(fasttext_path)

            assert ext in ['.vec']   # check valid extension

            # load file
            pretrained_model = KeyedVectors.load_word2vec_format(fasttext_path)

            # adding the new one to vocabulary
            for token in pretrained_model.vocab:
                if token in word_vocab: word_vocab[token] += 1
                else:
                    word_vocab[token] = 1
                    pretrained_tokens += [token]

        word_vocab['<unk>'] = 10000001
        word_vocab['<pad>'] = 10000000

        # handle if user choose option loading word from pretrained
        id2word, word2id = self.__mapping(vocab=word_vocab)

        # xavier initializer for embedding
        E_by_id = np.random.uniform(low=-1, high=1, size=[len(id2word), word_dim]) * np.sqrt(6./(len(id2word) + word_dim))
        for pretrain_token in pretrained_tokens:
            E_by_id[word2id[pretrain_token]] = pretrained_model[pretrain_token]

        return (id2word, word2id), E_by_id

if __name__ == '__main__':
    pass