import pandas as pd
import numpy as np
from utils import TextUtils
from nn.similarNN_pair import Model
import _pickle as cPickle
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.model_selection import train_test_split

"""
Some global parameters
"""
data_path = './data/new/pair.txt'
n_epoch = 50
init_lr = 0.002
init_keep_prob = 0.9
save_path = "./saved/pair/24072018/"
if not os.path.exists(save_path): os.makedirs(save_path) # create save_path
n_splits = 2
batch_size = 2048

"""
Adding seed
"""
np.random.seed(2312)
tf.set_random_seed(2312)


def train(train_batchs, test_batchs, n_epoch, init_lr, init_keep_prob, init_word_emb_1, init_word_emb_2, text_util, save_model_class,
          save_model_deep, word2id_1, word2id_2, label2id):
    """
    model parameters
    """
    word_emb_dim = init_word_emb_1.shape[1]
    rnn_hid_dim = 50
    rnn_n_layers = 1
    max_sen_length = 60
    learning_rate = init_lr
    keep_prob = init_keep_prob
    vocab_size_1 = init_word_emb_1.shape[0]
    vocab_size_2 = init_word_emb_2.shape[0]
    n_class = 2

    """
    build model
    """
    model = Model(word_emb_dim=word_emb_dim, rnn_hid_dim=rnn_hid_dim, rnn_n_layers=rnn_n_layers, max_sen_length=max_sen_length,
                  learning_rate=learning_rate, keep_prob=keep_prob, vocab_size_1=vocab_size_1, vocab_size_2=vocab_size_2, n_class=n_class,
                  class_weights=[1.0,1.0], word2id_1=word2id_1, word2id_2=word2id_2, label2id=label2id)

    if save_model_class is not None:
        cPickle.dump(model,open(save_model_class,'wb'))

    model.build(build_session=True,init_word_embedding_1=init_word_emb_1,init_word_embedding_2=init_word_emb_2)

    """
    training parameters
    """
    best_test = -np.inf

    decay_lr_every = 500
    lr_decay = 0.98

    eval_every = 500

    len_train_batches = len(train_batchs)
    count = 1

    for epoch in range(n_epoch):
        train_losses = []
        train_scores = []

        for i, batch_id in enumerate(np.random.permutation(len_train_batches)):
            score, loss = model.batch_run(batch_input=train_batchs[batch_id], text_util=text_util, init_lr=init_lr,
                                          init_keep_prob=init_keep_prob, mode=model.training)

            train_losses += [loss]
            train_scores += [score]

            print ("Epoch: ", epoch + 1, ", Score: ", "%.3f" % score, ", Loss: ", "%.5f" % loss)

            if count % decay_lr_every == 0:
                init_lr = init_lr * lr_decay

            if count % eval_every == 0:
                pass

            count += 1

        model.save(save_model_deep)

        """
        Inference
        """
        inf_losses = []
        inf_scores = []
        for inf_batch in test_batchs:
            inf_score, inf_loss = model.batch_run(batch_input=inf_batch, text_util=text_util,
                                                  mode=model.inference)

            inf_scores += [inf_score]
            inf_losses += [inf_loss]

        print('+' * 20)
        print('Inference, Epoch: ', epoch + 1, ' Score: ', "%.3f" % np.mean(inf_scores), ", Loss: ",
              "%.5f" % np.mean(inf_losses))
        print('+' * 20)

        if np.mean(inf_scores) >= best_test:
            print("The best was found")

            best_test = np.mean(inf_scores)
        print('Training, Epoch: ', epoch + 1, ' Score: ', "%.3f" % np.mean(train_scores), ", Loss: ",
              "%.5f" % np.mean(train_losses))

    return best_test

def make_fold(train_df, test_df, save_model_class, save_model_deep):
    text_util = TextUtils()

    # preprocessing and tokenize
    train_spa_sent_1_df = train_df['spa_sent_1'].tolist()
    train_eng_sent_2_df = train_df['eng_sent_1'].tolist()

    test_spa_sent_1_df = test_df['spa_sent_1'].tolist()
    test_eng_sent_2_df = test_df['eng_sent_1'].tolist()

    train_spa_tokens_1 = text_util.tokenize(sentences=train_spa_sent_1_df, language=text_util.spanish)
    train_eng_tokens_1 = text_util.tokenize(sentences=train_eng_sent_2_df, language=text_util.english)

    test_spa_tokens_1 = text_util.tokenize(sentences=test_spa_sent_1_df, language=text_util.spanish)
    test_eng_tokens_1 = text_util.tokenize(sentences=test_eng_sent_2_df, language=text_util.english)

    # building vocabulary (#using only training dataset)
    train_label_df = train_df['label'].tolist()

    (spa_id2word, spa_word2id), spa_E_by_id = text_util.create_word_vocab(lst_tokens=train_spa_tokens_1, word_dim=300,
                                                                          fasttext_path='./data/new/pretrained/mine.wiki.es.vec')
    (eng_id2word, eng_word2id), eng_E_by_id = text_util.create_word_vocab(lst_tokens=train_eng_tokens_1, word_dim=300,
                                                                          fasttext_path='./data/new/pretrained/mine.wiki.en.vec')

    (id2label, label2id) = text_util.create_label_vocab(labels=train_label_df)

    # builing dataset (mean convert token, label to its corressponding id)
    train_dataset = text_util.create_dataset(lst_tokens_1=train_spa_tokens_1, lst_tokens_2=train_eng_tokens_1,
                                             labels=train_label_df, label2id=label2id, word2id_1=spa_word2id, word2id_2=eng_word2id)

    test_dataset = text_util.create_dataset(lst_tokens_1=test_spa_tokens_1, lst_tokens_2=test_eng_tokens_1,
                                            labels=test_df['label'].tolist(), label2id=label2id, word2id_1=spa_word2id, word2id_2=eng_word2id)

    # create batch
    train_batches = text_util.create_batch(dataset=train_dataset, batch_size=batch_size)
    test_batches = text_util.create_batch(dataset=test_dataset, batch_size=batch_size)

    # training
    """
    train_batchs, test_batchs, n_epoch, init_lr, init_keep_prob, init_word_emb_1, init_word_emb_2, text_util, save_model_class,
          save_model_deep, word2id_1, word2id_2, label2id):
    """
    train_score = train(train_batchs=train_batches, test_batchs=test_batches, n_epoch=n_epoch, init_lr=init_lr, init_keep_prob=init_keep_prob,
                        init_word_emb_1=spa_E_by_id, init_word_emb_2=eng_E_by_id, text_util=text_util, save_model_class=save_model_class,
                        save_model_deep=save_model_deep, word2id_1=spa_word2id, word2id_2=eng_word2id, label2id=label2id)

    return train_score

def main():

    data_df_1 = pd.read_csv(data_path,sep='\t',header=None,names=['spa_sent_1', 'eng_sent_1','label'], encoding='utf-8')

    data_df = data_df_1 #data_df_1.append(data_df_2)
    data_df = data_df.iloc[:90000,:]

    print ("Number of pos/neg samples")
    print (data_df['label'].value_counts())

    # data_df_indexes = np.arange(0, data_df.shape[0])
    # data_df_labels  = np.array(data_df['label'].tolist())
    #
    # sfk = StratifiedKFold(n_splits=n_splits, random_state=2312)
    # scores = []

    train_df, test_df = data_df.iloc[:-10000,:], data_df.iloc[-10000:,:]

    print(train_df['label'].value_counts())
    print(test_df['label'].value_counts())

    save_model_class = os.path.join(save_path, "model.pkl")
    save_model_deep = os.path.join(save_path, "model.ckpt")
    score = make_fold(train_df=train_df, test_df=test_df, save_model_class=save_model_class,
                      save_model_deep=save_model_deep)

    print ("score", score)

    # for id, (train_indexes, test_indexes) in enumerate(sfk.split(data_df_indexes,data_df_labels)):
    #     print (train_indexes)
    #     print (test_indexes)
    #
    #     print ("+" * 20)
    #     print ("K-Fold-%d" % id)
    #
    #     train_df, test_df = data_df.iloc[train_indexes], data_df.iloc[test_indexes]
    #     print ("K-Fold-%d has number of pos/neg samples in train" % id)
    #     print (train_df['label'].value_counts())
    #     print ("K-Fold-%d has number of pos/neg samples in test" % id)
    #     print (test_df['label'].value_counts())
    #
    #     save_fold = os.path.join(save_path, "kfold_%d" % id)
    #     if not os.path.exists(save_fold): os.makedirs(save_fold)
    #
    #     save_model_class = os.path.join(save_fold, "model.pkl")
    #     save_model_deep  = os.path.join(save_fold, "model.ckpt")
    #
    #     score = make_fold(train_df=train_df, test_df=test_df, save_model_class=save_model_class,
    #                       save_model_deep=save_model_deep)
    #     scores += [score]
    #
    #     print ("K-Fold-%d returns score: " % id, "%.3f" % score)
    #
    # print ("%d-folds score: " % n_splits, "%.3f" % np.mean(scores))

if __name__ == '__main__':
    main()