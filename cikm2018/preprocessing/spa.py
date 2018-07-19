"""
five steps:
1. Steaming
2. Remove number
3. Normalising
"""

import re
from string import digits
from cucco import Cucco
import pattern.es as lemEsp

# List of number terms
nums = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'once',
        'doce', 'trece', 'catorce', 'quince', 'dieciseis', 'diecisiete', 'dieciocho', 'diecinueve',
        'veinte', 'veintiuno', 'veintidos', 'veinticinco', 'veintiséis', 'veintisiete', 'veintiocho', 'veintinueve', 'cien',
        'mil', 'millón', 'billón', 'trillón', 'quadrillion', 'quintillion', 'sextillion',
        'septillion', 'octillion', 'nonillion', 'decillion']

class SpaPreprocessing:
    def __init__(self):
        self.norm_spa = Cucco(language='es')
        self.norm_ops = ['replace_punctuation', 'remove_extra_whitespaces']


    def process(self, sentences):
        result = []

        for sentence in sentences:
            print ('preprocessing sentence: ', sentence)

            expand_contraction = self.__expand_contraction(sentence.lower())
            steamming = self.__steaming(expand_contraction)
            remove_number = self.__remove_number(steamming)
            normalising = self.__normalise(remove_number)

            result.append(normalising)

        return result

    def __expand_contraction(self, sentence):
        return sentence

    def __steaming(self, sentence):
        return sentence
        #return ' '.join(lemEsp.Sentence(lemEsp.parse(sentence, lemmata=True)).lemmata)

    def __remove_number(self, sentence):
        """
        Removes all numbers from strings, both alphabetic (in Spanish) and numeric. Intended to be
        part of a text normalisation process. If the number contains 'and' or commas, these are
        left behind on the assumption the text will be cleaned further to remove punctuation
        and stop-words.
        """

        query = sentence.replace('-', ' ').split(' ')
        resultwords = [word.strip() for word in query if word not in nums]
        noText = ' '.join(resultwords) # remove string number

        noNums = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", r" ",noText) # remove numeric number
        noNums = re.sub(r"\s\s+",r"\s",noNums)

        return noNums

    def __normalise(self, sentence):
        return self.norm_spa.normalize(text=sentence, normalizations=self.norm_ops)

if __name__ == '__main__':
    import pandas as pd

    file_path_1 = './../data/cikm_english_train_20180516.txt'
    headers_1 = ['eng_1','spa_1','eng_2','spa_2','label']
    df_1 = pd.read_csv(file_path_1,sep='\t',header=None,names=headers_1,encoding='utf-8')


    file_path_2 = './../data/cikm_spanish_train_20180516.txt'
    headers_2 = ['spa_1', 'eng_1', 'spa_2', 'eng_2', 'label']
    df_2 = pd.read_csv(file_path_2,sep='\t',header=None,names=headers_2,encoding='utf-8')


    file_path_3 = "./../data/cikm_test_a_20180516.txt"
    headers_3 = ['spa_1', 'spa_2']
    df_3 = pd.read_csv(file_path_3, sep='\t', header=None, names=headers_3, encoding='utf-8')


    file_path_4 = "./../data/cikm_unlabel_spanish_train_20180516.txt"
    headers_4 = ['spa_1', 'eng_1']
    df_4 = pd.read_csv(file_path_4, sep='\t', header=None, names=headers_4, encoding='utf-8')


    spa_df = pd.DataFrame({'spa':[]})
    eng_df = pd.DataFrame({'eng':[]})
    for df in [df_1, df_2]:
        spa_df = spa_df.append(df['spa_1'].tolist() + df['spa_2'].tolist())
        eng_df = eng_df.append(df['eng_1'].tolist() + df['eng_2'].tolist())

    spa_df = spa_df.append(df_3['spa_1'].tolist() + df_3['spa_2'].tolist() + df_4['spa_1'].tolist())
    eng_df = eng_df.append(df_4['eng_1'].tolist())


    print ("Spanish " + "-" * 20)
    print ("Length: " , spa_df.shape)
    print ("Count Na: ", spa_df.isnull().sum())

    print ("English " + "-" * 20)
    print ("Length: ", eng_df.shape)
    print ("Count Na: ", eng_df.isnull().sum())

    spa_df.to_csv('./../data/new/cikm_all_spanish.txt', sep='\t', header=None, index=None, encoding='utf-8')
    eng_df.to_csv('./../data/new/cikm_all_english.txt', sep='\t', header=None, index=None, encoding='utf-8')


