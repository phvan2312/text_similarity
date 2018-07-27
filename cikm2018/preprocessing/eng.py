"""
five steps:
1. Expanding contractions (exp: he's -> he is)
2. Steaming
3. Remove number
4. Normalising
"""

import re
from string import digits
from cucco import Cucco
import pattern.en as lemEng

# List of number terms
eng_nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
        'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
        'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion',
        'septillion', 'octillion', 'nonillion', 'decillion']

# from https://gist.github.com/nealrs/96342d8231b75cf4bb82
eng_cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

eng_c_re = re.compile('(%s)' % '|'.join(eng_cList.keys()))

class EngPreprocessing:
    def __init__(self):
        self.norm_eng = Cucco(language='en')
        self.norm_ops = ['replace_punctuation', 'remove_extra_whitespaces']

    def process(self, sentences):
        result = []

        for sentence in sentences:
            print('preprocessing sentence: ', sentence)

            expand_contraction = self.__expand_contraction(sentence.lower())
            steamming = self.__steaming(expand_contraction)
            remove_number = self.__remove_number(steamming)
            normalising  = self.__normalise(remove_number)

            result.append(normalising)

        return result

    def __expand_contraction(self, sentence):
        def replace(match):
            return eng_cList[match.group(0)]

        return eng_c_re.sub(replace, sentence)

    def __steaming(self, sentence):
        return ' '.join(lemEng.Sentence(lemEng.parse(sentence, lemmata=True)).lemmata)

    def __remove_number(self, sentence):
        """
        Removes all numbers from strings, both alphabetic (in English) and numeric. Intended to be
        part of a text normalisation process. If the number contains 'and' or commas, these are
        left behind on the assumption the text will be cleaned further to remove punctuation
        and stop-words.
        """

        query = sentence.replace('-', ' ').lower().split(' ')
        resultwords = [word.strip() for word in query if word not in eng_nums]
        noText = ' '.join(resultwords)

        noNums = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", r" ", noText)  # remove numeric number
        noNums = re.sub(r"\s\s+", r"\s", noNums)

        return noNums

    def __normalise(self, sentence):
        return self.norm_eng.normalize(text=sentence,normalizations=self.norm_ops)

