import pandas as pd
import tensorflow as tf
from nn.model2 import Model
import _pickle as cPickle
from utils import TextUtils

save_model_class = './saved/model2/model.pkl'
save_model_deep  = './saved/model2/model.ckpt'

data_file_path   = './data/cikm_test_a_20180516.txt'
data_file_headers= ['spa_sent_1', 'spa_sent_2']

if __name__ == '__main__':
    text_util = TextUtils()
    text_util.pad_id = 1
    text_util.unk_id = 0

    """
    Restore model
    """
    model = cPickle.load(open(save_model_class,'rb'))
    model.build(build_session = True, init_word_embedding=None)

    model.restore(save_model_deep)

    """
    Load data
    """
    data_df = pd.read_csv(data_file_path,sep='\t',header=None,names=data_file_headers)

    """
    Processing (tokenize)
    """
    eval_spa_tokens_1 = text_util.tokenize(sentences=data_df['spa_sent_1'].tolist(), language=text_util.spanish)
    eval_spa_tokens_2 = text_util.tokenize(sentences=data_df['spa_sent_2'].tolist(), language=text_util.spanish)

    dataset = text_util.create_dataset(lst_tokens_1=eval_spa_tokens_1,lst_tokens_2=eval_spa_tokens_2,
                                       labels=[0] * data_df.shape[0], word2id=model.word2id, label2id=model.label2id)


    """
    Create batches
    """
    batch_size = 32
    eval_batch_ids = [(s,min(s + batch_size, len(dataset))) for s in range(0,len(dataset),batch_size)]

    """
    Get scores
    """
    results = []
    for batch in eval_batch_ids:
        #batch_run(self, batch_input, text_util, mode, init_lr = None, init_keep_prob=None, metric=f1_score):
        scores = model.batch_run(batch_input=dataset[batch[0]:batch[1]],text_util=text_util,mode=model.get_score_only)
        results += scores[:,1].tolist()

    with open('cikm_submission.txt','wb') as f:
        f.write("\n".join([str(result) for result in results]).encode('utf-8'))

