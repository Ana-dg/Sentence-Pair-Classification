#Naive Bayes Approach

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from google.colab import files

train_path = '/content/train.json'
validation_path = '/content/validation.json'
test_path = '/content/test.json'


# Load the data into a DataFrame
with open(train_path, 'r') as for_train, open(validation_path, 'r') as for_validation,open(test_path, 'r') as for_test:
    train_data = json.load(for_train)
    validation_data = json.load(for_validation)
    test_data = json.load(for_test)

# Create DataFrames
train_df = pd.DataFrame(train_data)
validation_df = pd.DataFrame(validation_data)
test_df = pd.DataFrame(test_data)

# Remove NaN values
train_df = train_df.dropna()
validation_df = validation_df.dropna()
train_df = pd.DataFrame(train_data)

# Pair the sentences
train_df['pair'] = train_df['sentence1'] + ' ' + train_df['sentence2']
validation_df['pair'] = validation_df['sentence1'] + ' ' + validation_df['sentence2']
test_df['pair'] = test_df['sentence1'] + ' ' + test_df['sentence2']

#Prepare the training and validation dfs
X_train = train_df['pair']
X_validation = validation_df['pair']
X_test = test_df['pair']
y_train = train_df['label']
y_validation = validation_df['label']

my_stopwords = [
    "de", "în", "a", "și", "din", "la", "cu", "o", "care", "este", "pe", "un", "fost", "mai", "pentru",
    "sau", "dar", "pentru", "la", "pe", "al", "un", "o", "mai", "cel", "acest", "aceea", "aceste",
    "aceștia", "acel", "acela", "acele", "acelea", "adica", "am", "ar", "are", "asupra", "asta",
    "astea", "acesta", "acestea", "asupra", "as", "atunci", "au", "avea", "avem", "aveți", "aveți",
    "aș", "așa", "așadar", "așadar", "așadar", "așa", "ați", "bine", "bucur", "buna", "bune", "ca",
    "care", "caut", "cel", "ceva", "cine", "cineva", "ci", "cind", "când", "cum", "cumva", "că", "căci",
    "căreia", "cărei", "cărui", "către", "cătui", "căut", "cît", "cînd", "cît", "cîte", "cîți", "cîtva",
    "cînd", "dacă", "dar", "datorită", "deasupra", "deci", "deja", "deși", "din", "dintr", "după", "ea",
    "ei", "eie", "ele", "eram", "este", "eu", "fi", "fie", "fiecare", "fără", "foarte", "îi", "îl", "îmi",
    "împotriva", "înainte", "înaintea", "înainte", "încît", "încât", "încoace", "încolo",
    ".", ",", "`", "=", "_", "/", "#","*"
]



def my_stemmer(word):
    delete_plurals = ['ul', 'ului', 'aua', 'ea', 'ele', 'elor', 'ii', 'iua', 'iei', 'iile', 'iilor', 'ilor', 'ile', 'atei', 'aţie', 'aţia']
    for suffix in delete_plurals:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_basic = ['at', 'ata', 'ată', 'ati', 'ate', 'ut', 'uta', 'ută', 'uti', 'ute', 'it', 'ita', 'ită', 'iti', 'ite', 'ic', 'ica', 'ice', 'ici', 'ică', 'abil', 'abila',
                         'abile', 'abili', 'abilă', 'ibil', 'ibila', 'ibile', 'ibili', 'ibilă', 'oasa', 'oasă', 'oase', 'os', 'osi', 'oşi', 'ant', 'anta', 'ante', 'anti', 'antă',
                         'ator', 'atori', 'itate', 'itati', 'ităi', 'ităţi', 'iv', 'iva', 'ive', 'ivi', 'ivă']
    for suffix in delete_basic:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_verbs = ['are', 'ere', 'ire', 'âre', 'ind', 'ând', 'indu', 'ându', 'eze', 'ească', 'ez', 'ezi', 'ează', 'esc', 'eşti', 'eşte', 'ăsc', 'ăşti', 'ăşte', 'am', 'ai', 'au',
                     'eam', 'eai', 'ea', 'eaţi', 'eau', 'iam', 'iai', 'ia', 'iaţi', 'iau', 'ui', 'aşi', 'arăm', 'arăţi', 'ară', 'uşi', 'urăm', 'urăţi', 'ură', 'iş', 'irăm', 'irăţi',
                     'iră', 'âi', 'âşi', 'ârăm', 'ârăţi', 'âră', 'asem', 'aseşi', 'ase', 'aserăm', 'aserăţi', 'aseră', 'isem', 'iseşi', 'ise', 'iserăm', 'iserăţi', 'iseră', 'âsem',
                     'âseşi', 'âse', 'âserăm', 'âserăţi', 'âseră', 'usem', 'useşi', 'use', 'userăm', 'userăţi', 'useră']
    for suffix in delete_verbs:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_last_letter = ['a', 'e', 'i', 'ie', 'ă']
    for suffix in delete_last_letter:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    return word

def my_tokenizer(sentence):
    tokens = [my_stemmer(token) for token in str.split(sentence) if token.lower() not in my_stopwords]
    return tokens

# Vectorize train, validation and test with Tfidf
vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, lowercase=True, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_validation_vectorized = vectorizer.transform(X_validation)
X_test_vectorized = vectorizer.transform(X_test)

# Oversampling for the imbalanced classes
oversample_data = RandomOverSampler(random_state=42)
X_train_oversampled, y_train_oversampled = oversample_data.fit_resample(X_train_vectorized, y_train)

# Naive Bayes with alpha=0.1 and Bagging Classifier
naive_bayes = MultinomialNB(alpha=0.1)
model_nb = BaggingClassifier(naive_bayes, n_estimators=10, random_state=42)
model_nb.fit(X_train_oversampled, y_train_oversampled)

pred_train_nb = model_nb.predict(X_train_oversampled)
train_f1_ro_nb = f1_score(y_train_oversampled, pred_train_nb, average='macro')
pred_validation_nb = model_nb.predict(X_validation_vectorized)
val_f1_ro_nb = f1_score(y_validation, pred_validation_nb, average='macro')

# Part 9: Predict and evaluate on the validation set
pred_test_nb = model_nb.predict(X_test_vectorized)
test_df['label'] = pred_test_nb
test_df.to_csv('test_predictions_nb.csv', columns=['guid', 'label'], index=False)
files.download('test_predictions_nb.csv')

#Logistic Regression Approach

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from google.colab import files

train_path = '/content/train.json'
validation_path = '/content/validation.json'
test_path = '/content/test.json'


# Load the data into a DataFrame
with open(train_path, 'r') as for_train, open(validation_path, 'r') as for_validation,open(test_path, 'r') as for_test:
    train_data = json.load(for_train)
    validation_data = json.load(for_validation)
    test_data = json.load(for_test)

# Create DataFrames
train_df = pd.DataFrame(train_data)
validation_df = pd.DataFrame(validation_data)
test_df = pd.DataFrame(test_data)

# Remove NaN values
train_df = train_df.dropna()
validation_df = validation_df.dropna()
train_df = pd.DataFrame(train_data)

# Pair the sentences
train_df['pair'] = train_df['sentence1'] + ' ' + train_df['sentence2']
validation_df['pair'] = validation_df['sentence1'] + ' ' + validation_df['sentence2']
test_df['pair'] = test_df['sentence1'] + ' ' + test_df['sentence2']

#Prepare the training and validation dfs
X_train = train_df['pair']
X_validation = validation_df['pair']
X_test = test_df['pair']
y_train = train_df['label']
y_validation = validation_df['label']

my_stopwords = [
    "de", "în", "a", "și", "din", "la", "cu", "o", "care", "este", "pe", "un", "fost", "mai", "pentru",
    "sau", "dar", "pentru", "la", "pe", "al", "un", "o", "mai", "cel", "acest", "aceea", "aceste",
    "aceștia", "acel", "acela", "acele", "acelea", "adica", "am", "ar", "are", "asupra", "asta",
    "astea", "acesta", "acestea", "asupra", "as", "atunci", "au", "avea", "avem", "aveți", "aveți",
    "aș", "așa", "așadar", "așadar", "așadar", "așa", "ați", "bine", "bucur", "buna", "bune", "ca",
    "care", "caut", "cel", "ceva", "cine", "cineva", "ci", "cind", "când", "cum", "cumva", "că", "căci",
    "căreia", "cărei", "cărui", "către", "cătui", "căut", "cît", "cînd", "cît", "cîte", "cîți", "cîtva",
    "cînd", "dacă", "dar", "datorită", "deasupra", "deci", "deja", "deși", "din", "dintr", "după", "ea",
    "ei", "eie", "ele", "eram", "este", "eu", "fi", "fie", "fiecare", "fără", "foarte", "îi", "îl", "îmi",
    "împotriva", "înainte", "înaintea", "înainte", "încît", "încât", "încoace", "încolo",
    ".", ",", "`", "=", "_", "/", "#","*"
]



def my_stemmer(word):
    delete_plurals = ['ul', 'ului', 'aua', 'ea', 'ele', 'elor', 'ii', 'iua', 'iei', 'iile', 'iilor', 'ilor', 'ile', 'atei', 'aţie', 'aţia']
    for suffix in delete_plurals:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_basic = ['at', 'ata', 'ată', 'ati', 'ate', 'ut', 'uta', 'ută', 'uti', 'ute', 'it', 'ita', 'ită', 'iti', 'ite', 'ic', 'ica', 'ice', 'ici', 'ică', 'abil', 'abila',
                         'abile', 'abili', 'abilă', 'ibil', 'ibila', 'ibile', 'ibili', 'ibilă', 'oasa', 'oasă', 'oase', 'os', 'osi', 'oşi', 'ant', 'anta', 'ante', 'anti', 'antă',
                         'ator', 'atori', 'itate', 'itati', 'ităi', 'ităţi', 'iv', 'iva', 'ive', 'ivi', 'ivă']
    for suffix in delete_basic:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_verbs = ['are', 'ere', 'ire', 'âre', 'ind', 'ând', 'indu', 'ându', 'eze', 'ească', 'ez', 'ezi', 'ează', 'esc', 'eşti', 'eşte', 'ăsc', 'ăşti', 'ăşte', 'am', 'ai', 'au',
                     'eam', 'eai', 'ea', 'eaţi', 'eau', 'iam', 'iai', 'ia', 'iaţi', 'iau', 'ui', 'aşi', 'arăm', 'arăţi', 'ară', 'uşi', 'urăm', 'urăţi', 'ură', 'iş', 'irăm', 'irăţi',
                     'iră', 'âi', 'âşi', 'ârăm', 'ârăţi', 'âră', 'asem', 'aseşi', 'ase', 'aserăm', 'aserăţi', 'aseră', 'isem', 'iseşi', 'ise', 'iserăm', 'iserăţi', 'iseră', 'âsem',
                     'âseşi', 'âse', 'âserăm', 'âserăţi', 'âseră', 'usem', 'useşi', 'use', 'userăm', 'userăţi', 'useră']
    for suffix in delete_verbs:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    delete_last_letter = ['a', 'e', 'i', 'ie', 'ă']
    for suffix in delete_last_letter:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    return word

def my_tokenizer(sentence):
    tokens = [my_stemmer(token) for token in str.split(sentence) if token.lower() not in my_stopwords]
    return tokens

# Vectorize train, validation and test with Tfidf
vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, lowercase=True, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_validation_vectorized = vectorizer.transform(X_validation)
X_test_vectorized = vectorizer.transform(X_test)

# Oversampling for the imbalanced classes
oversample_data = RandomOverSampler(random_state=42)
X_train_oversampled, y_train_oversampled = oversample_data.fit_resample(X_train_vectorized, y_train)

# Logistic Regression and Bagging Classifier
bagging = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=10, penalty='l2', max_iter=1000, random_state=42)
model_lr= BaggingClassifier(bagging, n_estimators=10, random_state=42)
model_lr.fit(X_train_oversampled, y_train_oversampled)

pred_train = model_lr.predict(X_train_oversampled)
train_f1_ro = f1_score(y_train_oversampled, pred_train, average='macro')
pred_validation = model_lr.predict(X_validation_vectorized)
val_f1_ro = f1_score(y_validation, pred_validation, average='macro')

# Part 9: Predict and evaluate on the validation set
pred_test = model_lr.predict(X_test_vectorized)
test_df['label'] = pred_test
test_df.to_csv('test_predictions.csv', columns=['guid', 'label'], index=False)
files.download('test_predictions.csv')
