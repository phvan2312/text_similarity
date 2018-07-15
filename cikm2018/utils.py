import pandas as pd
import numpy as np
import nltk
from gensim.models import KeyedVectors
import os
from preprocessing.eng import EngPreprocessing
from preprocessing.spa import SpaPreprocessing

class TextUtils:
    def __inni__(self):
        self.eng_preprocessing = EngPreprocessing()
        self.spa_preprocessing = SpaPreprocessing()

        self.english = 'english'
        self.spanish = 'spanish'

        self.word_unk = '<unk>'
        self.word_pad = '<pad>'

    def clean(self, sentences, language):
        assert language in [self.english, self.english]

        if language == self.english: return self.eng_preprocessing.process(sentences)
        elif language == self.spanish: return self.spa_preprocessing.process(sentences)
        else:
            raise Exception("language must be in ['spa', 'eng']")

    def __mapping(self, vocab):
        assert type(vocab) is dict

        sorted_items = sorted(vocab.items(), key=lambda elem: -elem[1])
        id2x = {i: v[0] for i, v in enumerate(sorted_items)}
        x2id = {v[0]:i for i,v in enumerate(sorted_items)}

        return id2x, x2id

    def tokenize(self, sentences, language):
        clean_sentences = self.clean(sentences,language)

        result = []
        for sentence in clean_sentences:
            tokens = nltk.word_tokenize(text=sentence,language=language)
            result.append(tokens)

        return result

    def create_label_vocab(self, labels):
        assert type(labels) is list

        label_vocab = {}
        for label in labels:
            if label in label_vocab: label_vocab[label] += 1
            else:
                label_vocab[label] = 1

        id2label, label2id = self.__mapping(vocab=label_vocab)

        return (id2label, label2id)

    """
    sentences must be tokenized before feeding
    """
    def create_word_vocab(self, lst_tokens, fasttext_path = None, word_dim = None):
        assert type(lst_tokens) is list

        word_vocab = {}
        for tokens in lst_tokens:
            for token in tokens:
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

        word_vocab[self.word_unk] = 10000001 # hard code here
        word_vocab[self.word_pad] = 10000000

        # handle if user choose option loading word from pretrained
        id2word, word2id = self.__mapping(vocab=word_vocab)

        # xavier initializer for embedding
        E_by_id = np.random.uniform(low=-1, high=1, size=[len(id2word), word_dim]) * np.sqrt(6./(len(id2word) + word_dim))
        for pretrain_token in pretrained_tokens:
            E_by_id[word2id[pretrain_token]] = pretrained_model[pretrain_token]

        return (id2word, word2id), E_by_id

    """
    create dataset
    """
    def create_dataset(self, lst_tokens_1, lst_tokens_2, labels, word2id, label2id):
        dataset = []
        len_label = len(label2id)

        for tokens_1, tokens_2, label in zip(lst_tokens_1, lst_tokens_2, labels):
            word_ids_1 = [word2id[token if token in word2id else self.word_unk] for token in tokens_1]
            word_ids_2 = [word2id[token if token in word2id else self.word_unk] for token in tokens_2]

            label_id = [0] * len_label
            label_id[label2id[label]] = 1

            data = {
                'word_ids_1': word_ids_1,
                'word_ids_2': word_ids_2,
                'label_id': label_id
            }

            dataset.append(data)

        return dataset


    def create_batch(self, dataset, batch_size):
        batch_datas = []

        previous_batch_ids = range(0, len(dataset), batch_size)
        next_batch_ids = [(i + batch_size) if (i + batch_size) < len(dataset) else len(dataset) for i in previous_batch_ids]

        for s_i, e_i in zip(previous_batch_ids, next_batch_ids):
            if e_i >= s_i:
                batch_datas.append(dataset[s_i:e_i])

        return batch_datas

    """
    code from : https://github.com/guillaumegenthial/sequence_tagging original method name : _pad_sequence
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    def pad_common(sequences, pad_tok, max_length):

        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

if __name__ == '__main__':
    pass