import logging, gensim, string, nltk
from pprint import pprint
from nltk.tokenize import word_tokenize

file_location = input("Enter the full path to your corpus: ")

with open(file_location, 'r', encoding='utf-8') as myfile:
    documents = [i for i in myfile.readlines()]
    documents = list(filter(('\n').__ne__, documents))
    documents = [s.strip('\n') for s in documents]

print('[!]: Document imported')

with open('corenlp_stopwords.txt', 'r', encoding='utf-8') as myfile:
    stopwords = [i for i in myfile.readlines()]
    stopwords = list(filter(('\n').__ne__, stopwords))
    stopwords = [s.strip('\n') for s in stopwords]

extra = ['re', 've', 'nt', 'wo', 'kh', 't', 'st']
for i in extra:
    stopwords.append(i)

print('[!]: Stopwords imported')

texts = []
punct = set(string.punctuation)
punct|= set('-’“”•„…£—‘«»') # set is lacking several characters

# Word tokenizing
for p in documents:
    doc = []
    wds = []
    parag = word_tokenize(p)
    for w in parag:
        doc.append(w)
    
    # Stripping punctuation and stopwords
    for w in doc:
        # This removes punctuation and stopwords from list entries
        if w not in punct and w.lower() not in stopwords:
            # This also removes _some_ punctuation from string
            wds.append(''.join(ch for ch in w.lower() if ch not in set('’“”•„…£‘«»'))) 
    
    # Do not include empty lists
    if wds != []:
        texts.append(wds)
print('[x]: Text created') 

# Preparing trigrams
print('[ ]: 1-grams: Initializing')
phrases = gensim.models.Phrases(texts)
print('[x]: 1-grams: Check')    
print('[ ]: 2-grams: Initializing')
bigram = gensim.models.Phrases(phrases[texts])
print('[x]: 2-grams: Check')    
print('[ ]: 3-grams: Initializing')
trigram = gensim.models.Phrases(bigram[texts])
print('[x]: 3-grams: Check') 

# Creating the model
print('[ ]: Model: Initializing...')
model = gensim.models.Word2Vec([t for t in trigram[bigram[texts]]]
                               , sg=1, window=15, negative=10, size=300, workers=8)
print('[x]: Model is complete!')
print('[!]: DO NOT forget to save the model')

print('[ ]: Saving Model (deprecated format)')
model.save('simplewiki.w2v')
print('[x]: Saved Model (deprecated format)')
print('[ ]: Saving Model')
model.wv.save_word2vec_format('simplewiki.wv')
print('[x]: Saved Model')

print('[!]: Testing the model...')
print('[!]: The lexical similarity between man and woman is: ', str(model.wv.similarity('man', 'woman')))
