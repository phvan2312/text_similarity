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
        self.norm_ops = ['remove_stop_words', 'replace_punctuation', 'remove_extra_whitespaces']


    def process(self, sentences):
        result = []

        for sentence in sentences:
            expand_contraction = self.__expand_contraction(sentence.lower())
            steamming = self.__steaming(expand_contraction)
            remove_number = self.__remove_number(steamming)
            normalising = self.__normalise(remove_number)

            result.append(normalising)

        return result

    def __expand_contraction(self, sentence):
        return sentence

    def __steaming(self, sentence):
        return ' '.join(lemEsp.Sentence(lemEsp.parse(sentence, lemmata=True)).lemmata)

    def __remove_number(self, sentence):
        """
        Removes all numbers from strings, both alphabetic (in Spanish) and numeric. Intended to be
        part of a text normalisation process. If the number contains 'and' or commas, these are
        left behind on the assumption the text will be cleaned further to remove punctuation
        and stop-words.
        """

        query = sentence.replace('-', ' ').split(' ')
        resultwords = [word for word in query if word not in nums]
        noText = ' '.join(resultwords).encode('utf-8')
        noNums = noText.translate(None, digits).replace('  ', ' ')

        return noNums

    def __normalise(self, sentence):
        return self.norm_spa.normalize(text=sentence, normalizations=self.norm_ops)

