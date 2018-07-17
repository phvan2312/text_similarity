import pandas as pd
import numpy as np
from utils import TextUtils
from nn.model2 import Model

data_path = './data/cikm_spanish_train_20180516.txt'

def train(train_batchs, test_batchs, n_epoch, init_lr, init_keep_prob, init_word_emb, text_util):
    """
    model parameters
    """
    word_emb_dim = init_word_emb.shape[1]
    rnn_hid_dim = 50
    rnn_n_layers = 2
    max_sen_length = 100
    learning_rate = init_lr
    keep_prob = init_keep_prob
    vocab_size = init_word_emb.shape[0]
    n_class = 2

    """
    build model
    """
    model = Model(word_emb_dim=word_emb_dim, rnn_hid_dim=rnn_hid_dim, rnn_n_layers=rnn_n_layers,max_sen_length=max_sen_length,
                  learning_rate=learning_rate,keep_prob=keep_prob,vocab_size=vocab_size,n_class=n_class)

    model.build(build_session=True,init_word_embedding=init_word_emb)


    """
    training parameters
    """
    best_test = -np.inf

    decay_lr_every = 500
    lr_decay = 0.95

    eval_every = 500

    len_train_batches = len(train_batchs)
    count = 1

    for epoch in range(n_epoch):
        for i, batch_id in enumerate(np.random.permutation(len_train_batches)):
            score, loss = model.batch_run(batch_input=train_batchs[batch_id], text_util=text_util, init_lr=init_lr,
                                          init_keep_prob=init_keep_prob, mode=model.training)

            print ("Epoch: ", epoch + 1, ", Score: ", score, ", Loss: ", loss)

            if count % decay_lr_every == 0:
                init_lr = init_lr * lr_decay

            if count % eval_every == 0:

                """
                Inference
                """
                inf_losses = []
                inf_scores = []
                for inf_batch in test_batchs:
                    inf_score, inf_loss = model.batch_run(batch_input=train_batchs[batch_id], text_util=text_util,
                                                          mode=model.inference)

                    inf_scores += [inf_score]
                    inf_losses += [inf_loss]


                print ('+' * 20)
                print ('Inference, Score: ', np.mean(inf_scores), ", Loss: ", np.mean(inf_losses))
                print ('+' * 20)


def main():
    text_util = TextUtils()
    data_df = pd.read_csv(data_path,sep='\t',header=None,names=['spa_sent_1', 'eng_sent_1','spa_sent_2','eng_sent_2','label'], encoding='utf-8')

    print ("Count None/Nan=N")
    print (data_df.isnull().sum())

    # split training, testing dataset
    data_legnth = data_df.shape[0]
    split_factor = 0.8

    train_df = data_df.iloc[:int(data_legnth*split_factor),:]
    test_df  = data_df.iloc[int(data_legnth*split_factor):,:]

    # preprocessing and tokenize

    train_spa_sent_1_df = train_df['spa_sent_1'].tolist()
    train_spa_sent_2_df = train_df['spa_sent_2'].tolist()
    train_label_df  = train_df['label'].tolist()

    train_spa_tokens_1 = text_util.tokenize(sentences=train_spa_sent_1_df, language=text_util.spanish)
    train_spa_tokens_2 = text_util.tokenize(sentences=train_spa_sent_2_df, language=text_util.spanish)

    train_spa_tokens = train_spa_tokens_1 + train_spa_tokens_2

    # building vocabulary
    (spa_id2word, spa_word2id), spa_E_by_id = text_util.create_word_vocab(lst_tokens=train_spa_tokens)
    (id2label, label2id) = text_util.create_label_vocab(labels=train_label_df)

    # builing dataset (mean convert token, label to its corressponding id)
    train_dataset = text_util.create_dataset(lst_tokens_1=train_spa_sent_1_df, lst_tokens_2=train_spa_sent_2_df,
                                             labels=train_label_df, label2id=label2id, word2id=spa_word2id)

    test_dataset = text_util.create_dataset(lst_tokens_1=test_df['spa_sent_1'].tolist(), lst_tokens_2=test_df['spa_sent_2'].tolist(),
                                             labels=test_df['label'].tolist(), label2id=label2id, word2id=spa_word2id)

    # create batch
    batch_size = 16
    train_batches = text_util.create_batch(dataset=train_dataset, batch_size=batch_size)
    test_batches  = text_util.create_batch(dataset=test_dataset , batch_size=batch_size)

    # training
    #train(train_batchs, test_batchs, n_epoch, init_lr, init_keep_prob, init_word_emb, text_util):
    train(train_batchs=train_batches,test_batchs=test_batches,n_epoch=20,init_lr=0.001,init_keep_prob=0.9,
          init_word_emb=spa_E_by_id, text_util=text_util)


if __name__ == '__main__':
    main()