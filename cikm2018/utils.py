import pandas as pd
import numpy as np
import nltk
from gensim.models import KeyedVectors
import os
from preprocessing.eng import EngPreprocessing
from preprocessing.spa import SpaPreprocessing
from gensim.utils import to_utf8, smart_open
from sklearn.model_selection import StratifiedShuffleSplit
import math
from sklearn.utils import shuffle

class TextUtils:

    def __init__(self):
        self.eng_preprocessing = EngPreprocessing()
        self.spa_preprocessing = SpaPreprocessing()

        self.english = 'english'
        self.spanish = 'spanish'

        self.word_unk = '<unk>'
        self.word_pad = '<pad>'

        self.unk_id, self.pad_id = None, None

    def clean(self, sentences, language):
        assert language in [self.english, self.spanish]

        if language == self.english: return self.eng_preprocessing.process(sentences)
        elif language == self.spanish: return self.spa_preprocessing.process(sentences)
        else:
            raise Exception("language must be in ['spanish', 'english']")

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
            #print ("tokenizing sentence: ", sentence)

            tokens = nltk.word_tokenize(text=sentence,language=language) #sentence.split(" ")
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

        num_words = len(word_vocab)

        "Now we have to load word in the pretrained file one"
        pretrained_tokens = []
        pretrained_model = None
        if fasttext_path is not None:
            _, ext = os.path.splitext(fasttext_path)

            assert ext in ['.vec']   # check valid extension

            # load file
            pretrained_model = KeyedVectors.load_word2vec_format(fasttext_path)

            word_found = 0

            # USE IT WHEN YOU WANT TO CREATE SUB WORD EMBEDDINGS
            # in this snippet, to save memory, we just get the embedding of the word
            # which appear in our own vocabulary
            # for token in word_vocab:
            #     if token in pretrained_model.vocab:
            #         word_found += 1
            #         pretrained_tokens += [token]

            # USE IT WHEN YOU WANT TO USE WORD EMBEDDINGS
            for token in pretrained_model.vocab:
                pretrained_tokens += [token]
                if token in word_vocab:
                    word_found += 1
                    word_vocab[token] += 1
                else:
                    word_vocab[token] = 1

            print ("number of word which comes from pre-trained file: ", word_found, '/', num_words)

        word_vocab[self.word_unk] = 10000001 # hard code here
        word_vocab[self.word_pad] = 10000000

        # handle if user choose option loading word from pretrained
        id2word, word2id = self.__mapping(vocab=word_vocab)

        # xavier initializer for embedding
        E_by_id = np.random.uniform(low=-1, high=1, size=[len(id2word), word_dim]) * np.sqrt(6./(len(id2word) + word_dim))
        for pretrain_token in pretrained_tokens:
            E_by_id[word2id[pretrain_token]] = pretrained_model[pretrain_token]

        print ("Vocabulary shape: ", E_by_id.shape)

        self.unk_id, self.pad_id = word2id[self.word_unk], word2id[self.word_pad]
        return (id2word, word2id), E_by_id

    """
    create dataset
    """
    def create_dataset(self, lst_tokens_1, lst_tokens_2, labels, word2id_1, word2id_2, label2id):
        dataset = []
        len_label = len(label2id)

        for tokens_1, tokens_2, label in zip(lst_tokens_1, lst_tokens_2, labels):
            word_ids_1 = [word2id_1[token.lower().strip() if token.lower().strip() in word2id_1 else self.word_unk] for token in tokens_1]
            word_ids_2 = [word2id_2[token.lower().strip() if token.lower().strip() in word2id_2 else self.word_unk] for token in tokens_2]

            label_id = [0] * len_label
            label_id[label2id[label]] = 1

            data = {
                'word_ids_1': word_ids_1,
                'word_ids_2': word_ids_2,
                'label': label_id
            }

            dataset.append(data)

        return dataset


    def __split_by_buckets(self, data_len, n_buckets):
        batch_size = int (math.floor(data_len / n_buckets))

        buckets = np.array([batch_size] * n_buckets)
        data_remains = data_len - n_buckets * batch_size

        buckets[np.random.permutation(range(n_buckets))[:data_remains]] += 1

        indexes = [0]
        batches = []
        for bucket in buckets.tolist():
            indexes += [bucket + indexes[-1]]
            batches += [(indexes[-2], indexes[-1])]

        return batches

    def test_split_by_buckets(self, data_len, n_buckets):
        return self.__split_by_buckets(data_len,n_buckets)

    def create_batch(self, dataset, batch_size):
        np_dataset = np.array(dataset)

        #np_indexes = np.arange(0,np_dataset.shape[0],1)
        np_labels  = np.array([ e['label'][1] for e in dataset ])

        np_1_indexes = np.where(np_labels == 1)[0]
        np_0_indexes = np.where(np_labels == 0)[0]
        n_splits = int(math.ceil(np_dataset.shape[0]/batch_size))

        batches_1 = self.__split_by_buckets(np_1_indexes.shape[0], n_splits)
        batches_0 = self.__split_by_buckets(np_0_indexes.shape[0], n_splits)

        batch_datas = []
        for n_split in range(n_splits):
            batch_data = np_dataset[ np_1_indexes[ batches_1[n_split][0]:batches_1[n_split][1] ] ].tolist() + \
                         np_dataset[ np_0_indexes[ batches_0[n_split][0]:batches_0[n_split][1] ] ].tolist()
            batch_datas.append(batch_data)

        return batch_datas

    """
    code from : https://github.com/guillaumegenthial/sequence_tagging original method name : _pad_sequence
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    def pad_common(self, sequences, pad_tok, max_length):

        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    """
    save word embedding to (.vec) file  
    """
    def save_word_embedding(self, words, embeddings, out_path):
        with smart_open(out_path, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % embeddings.shape))

            for index, word in enumerate(words):
                row = embeddings[index]
                fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

    """
    
    """


if __name__ == '__main__':
    #
    # # """
    # # Test split by buckets
    # # """
    # # TextUtils().test_split_by_buckets(data_len=4223, n_buckets=535)
    # # exit()
    #

    ##################################################################################################################

    # """
    # Spanish
    # """
    # file_path_1 = './data/new/cikm_all_spanish.txt'
    # headers_1 = ['spa']
    #
    # df_1 = pd.read_csv(file_path_1, header=None, names=headers_1, sep='\t', encoding='utf-8')
    # spa_sents = df_1.iloc[:,0].tolist()
    #
    # text_util = TextUtils()
    # spa_lst_of_tokens = text_util.tokenize(sentences=spa_sents, language=text_util.spanish)
    #
    # (spa_id2word, spa_word2id), spa_E_by_id = text_util.create_word_vocab(lst_tokens=spa_lst_of_tokens, word_dim=300,
    #                                                                       fasttext_path='./data/pretrained/wiki.es.vec')
    #
    # # text_util.save_word_embedding(words=list(spa_id2word.values()), embeddings=spa_E_by_id,
    # #                               out_path='./data/new/pretrained/mine.wiki.es.vec')

    ##################################################################################################################

    # """
    # English
    # """
    #
    # file_path_2 = './data/new/cikm_all_english.txt'
    # headers_2 = ['eng']
    #
    # df_2 = pd.read_csv(file_path_2, header=None, names=headers_2, sep='\t', encoding='utf-8')
    # eng_sents = df_2.iloc[:, 0].tolist()
    #
    # text_util = TextUtils()
    #
    # eng_lst_of_tokens = text_util.tokenize(sentences=eng_sents, language=text_util.english)
    #
    # (eng_id2word, eng_word2id), eng_E_by_id = text_util.create_word_vocab(lst_tokens=eng_lst_of_tokens, word_dim=300,
    #                                                                       fasttext_path='./data/pretrained/wiki.en.vec')
    #
    # text_util.save_word_embedding(words=list(eng_id2word.values()), embeddings=eng_E_by_id,
    #                               out_path='./data/new/pretrained/mine.wiki.en.vec')

    ##################################################################################################################

    print('a')
    file_path_1 = './data/cikm_english_train_20180516.txt'
    headers_1 = ['eng_1', 'spa_1', 'eng_2', 'spa_2', 'label']
    df_1 = pd.read_csv(file_path_1, sep='\t', header=None, names=headers_1, encoding='utf-8')

    file_path_2 = './data/cikm_spanish_train_20180516.txt'
    headers_2 = ['spa_1', 'eng_1', 'spa_2', 'eng_2', 'label']
    df_2 = pd.read_csv(file_path_2, sep='\t', header=None, names=headers_2, encoding='utf-8')

    file_path_3 = "./data/cikm_unlabel_spanish_train_20180516.txt"
    headers_3 = ['spa_1', 'eng_1']
    df_3 = pd.read_csv(file_path_3, sep='\t', header=None, names=headers_3, encoding='utf-8')

    df = pd.DataFrame()
    # create positive samples
    print('c')
    df['spa'] = df_1['spa_1'].tolist() + df_1['spa_2'].tolist() + df_2['spa_1'].tolist() + df_2['spa_2'].tolist() + \
                df_3['spa_1'].tolist()
    df['eng'] = df_1['eng_1'].tolist() + df_1['eng_2'].tolist() + df_2['eng_1'].tolist() + df_2['eng_2'].tolist() + \
                df_3['eng_1'].tolist()
    df['label'] = (df_1.shape[0] * 2 + df_2.shape[0] * 2 + df_3.shape[0]) * [1.0]   # all is positive samples

    # create negative samples
    random_indexs_1 = np.random.permutation(range(df.shape[0]))
    random_indexs_2 = np.random.permutation(range(df.shape[0]))

    df = df.append(pd.DataFrame({
        "spa":df['spa'].iloc[random_indexs_1].tolist(),
        "eng":df['eng'].iloc[random_indexs_1].tolist(),
        "label":[0.0 if index_1 != index_2 else 1.0 for (index_1, index_2) in zip(random_indexs_1, random_indexs_2)]
    }),ignore_index=True)

    df.reindex(np.random.permutation(df.index)).to_csv("./data/new/pair.txt", sep='\t', columns=['spa','eng','label'], index=None, encoding='utf-8')







