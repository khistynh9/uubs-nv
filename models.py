import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
# dataset
# prepare data set
tingkat_data = pd.read_csv('model/dataset-uubs-nv.csv')
columns = ['loma', 'hormat ka sorangan', 'hormat ka batur', 'kecap']
# rows = []
train_data = pd.DataFrame(tingkat_data, columns=columns)
# train_data = pickle.load(open('model/model-train-data.pkl', 'rb'))
data_set = pickle.load(open('model/model-data-set.pkl', 'rb'))
tkt_loma = pickle.load(open('model/model-tkt-loma.pkl', 'rb'))
tkt_ls = pickle.load(open('model/model-tkt-ls.pkl', 'rb'))
tkt_lb = pickle.load(open('model/model-tkt-lb.pkl', 'rb'))
# sklearn
nb = pickle.load(open('model/uubs-model-nv.pkl', 'rb'))
transformer = pickle.load(open('model/model-transformer.pkl', 'rb'))
vect = pickle.load(open('model/model-vect.pkl', 'rb'))
# file.close()


class Modelnv:

    # init method or constructor
    def __init__(self, text, label):
        self.text = str(text)
        self.label = label

    # load pickle file
    def Pkload(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    # Remove Char Method
    def RemoveChar(self, textr):
        if textr:
            tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
            lst = tokenizer.tokenize(' '.join(textr))
            lstxt = list((map(lambda x: x.lower(), lst)))
            return lstxt

    # filter kata yang termasuk dalam kamus
    def CleanUp(self, ngrams):
        seen = set()
        for ngram in ngrams:
            if ' ' in ngram:
                seen = seen.union(set(ngram.split()))
        return [ngram for ngram in ngrams if ngram not in seen]

    # extract unigram and bigram
    def ExtractGram(self, texte):
        dt_gram = list(nltk.bigrams(texte))
        dt_all = []
        if not texte:
            return dt_all
        else:
            for lst in dt_gram:
                txs = lst[0]
                txs2 = lst[1]
                txf = ' '.join(lst)
                dt_all.append(txs)
                dt_all.append(txf)
                dt_all.append(txs2)
            return dt_all

    # filter kata yang termasuk dalam kamus
    def CheckDict(self, textd):
        result = []
        for token in textd:
            if token in list(data_set['text']):
                result.append(token)
        return result

    # filter dataset and new data predict
    def CheckData(self, text_gram, clean_data):
        result = []
        for token in text_gram:
            if token in list(data_set['text']):
                result.append(token)
            elif token in list(clean_data):
                result.append(token)
        return self.CleanUp(result)

    def CheckTingkat(tingkat):
        if tingkat:
            if tingkat == 0:
                label = 'loma'
            elif tingkat == 1:
                label = 'lemes ka sorangan'
            elif tingkat == 2:
                label = 'lemes ka batur'
        return label

    # Method Preprocessing
    def Preprocessing(self):
        if len(self.text):
            new_sentence = self.text
            new_word_list = word_tokenize(new_sentence.lower())
            clean_data = self.RemoveChar(new_word_list)
            word_gram = self.ExtractGram(clean_data)
            filter_gram = self.CheckData(word_gram, clean_data)
            clean_word_gram = list(dict.fromkeys(filter_gram))

            return clean_word_gram

    # Method Processing
    def Predict(self, cleandata):
        if len(cleandata):
            cln_dict = dict.fromkeys(cleandata, "")
            dict_key = cln_dict.keys()

            dt_p = self.CheckDict(dict_key)

            X_new_counts = vect.transform(dt_p)
            X_new_tfidf = transformer.transform(X_new_counts)

            predicted = nb.predict(X_new_tfidf)
            dt_prd = dict(zip(dt_p, predicted))

            return dt_prd

    # Method Correct
    def Correct(self, label_num, label, data):
        prob_new_l = data
        for key, value in prob_new_l.items():
            if value == label_num:
                prob_new_l[key] = key

            else:
                if key in tkt_loma:
                    hasil = train_data.index[train_data['loma'] == key].tolist(
                    )
                    tmp = hasil[0]
                    prob_new_l[key] = train_data.loc[tmp, label]
                elif key in tkt_ls:
                    hasil = train_data.index[train_data['hormat ka sorangan'] == key].tolist(
                    )
                    tmp = hasil[0]
                    prob_new_l[key] = train_data.loc[tmp, label]
                elif key in tkt_lb:
                    hasil = train_data.index[train_data['hormat ka batur'] == key].tolist(
                    )
                    tmp = hasil[0]
                    prob_new_l[key] = train_data.loc[tmp, label]

        return prob_new_l
