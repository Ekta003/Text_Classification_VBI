#!/usr/bin/env python
# coding: utf-8

# nlp steps:
# 1.lower case
# 2.tokenization
# 3. punchn removal
# 4. stop words remove
# 5.stemming
# 6.lemmatizing

# In[2]:


import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lematize=WordNetLemmatizer()
import pickle
import re


# In[4]:


nltk.download('punkt')
nltk.download('wordnet')


# # 1.Data Load

# In[5]:


df=pd.read_csv('train.txt',header=None, names=['Text','Label'],sep=";")
df                                          
                                    


# In[6]:


from nltk.corpus import stopwords
nltk.download('stopwords')


# # 2. Lower Case

# In[7]:


df['Text']=df['Text'].str.lower()
df


# # 3.Tokenization

# In[8]:


from nltk.tokenize import word_tokenize


# In[9]:


df['token_words']=df['Text'].apply(word_tokenize)
df


# # 4.Remove Ignore words

# In[10]:


ignore_words=['?','!',',','#','@','&']

for p in ignore_words:
        df['token_words'] = df['token_words'].replace(p, '')
df


# # 5. Remove stop_words

# In[ ]:





# In[11]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


# In[12]:


nltk.download('punkt')
nltk.download('stopwords')


# In[13]:


stop_words = set(stopwords.words('english'))


# In[14]:


def remove_stopwords(textt):
    words=word_tokenize(textt)
    for w in words:
        if w not in stop_words:
            filtered_words=w
       
    return ' '.join(filtered_words)
textt='i feel like this as such a rude comment and i'
remove_stopwords(textt)


# In[9]:


# df['filtered_senc']=df['Text'].apply(remove_stopwords)
# df


# In[15]:


def remove_stopwords(textt):
    words=word_tokenize(textt)
#     for w in words:
#         if w not in stop_words:
#             filtered_words=w
#         filtered_text = ' '.join(filtered_words)
#     return filtered_text
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
textt='i feel like this was such a rude comment and i'
remove_stopwords(textt)


# In[16]:


df['filtered_senc']=df['Text'].apply(remove_stopwords)
df


# # 6.Lemmatization

# In[17]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[36]:


# from nltk.stem.porter import PorterStemmer
# ps=PorterStemmer()
l=[]


# In[ ]:





# In[18]:


lemmatizer=nltk.stem.WordNetLemmatizer()


# In[19]:


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
 

wnl = WordNetLemmatizer()
 

list1 = ['kites', 'babies', 'dogs', 'flying', 'smiling', 
         'driving', 'died', 'tried', 'feet']
for words in list1:
    print(words + " ---> " + wnl.lemmatize(words))


# In[78]:


p=[]


# In[20]:


def word_lemmatize(text):
    word=word_tokenize(text)
    lemmatized_words = [wnl.lemmatize(w) for w in word]
        
    return ' '.join(lemmatized_words)

text='go feeling hopeless damned hopeful around some'
word_lemmatize(text)


# In[21]:


df['lemmatized_senc']=df['filtered_senc'].apply(word_lemmatize)
df


# In[ ]:





# In[ ]:





# # Model

# In[ ]:





# In[ ]:


1.Data loading


# In[22]:


df_train=pd.read_csv('train.txt',names=["Text",'Emotion'], sep=";")
df_test=pd.read_csv('test_nltk.txt',names=["Text",'Emotion'], sep=";")
df_val=pd.read_csv('val.txt',names=["Text",'Emotion'], sep=";")
df_train


# In[23]:


df


# # Model building
# 

# In[24]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


X,Y=Text, Label


# In[25]:


train_data, test_data, train_labels, test_labels = train_test_split(df['lemmatized_senc'], df['Label'], test_size=0.2, random_state=3)


# In[26]:


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
mn= MultinomialNB()
mn.fit(X_train, train_labels)


# In[27]:


predictions = mn.predict(X_test)


# In[28]:


accuracy_score(test_labels, predictions)


# In[ ]:





# In[29]:


confusion_matrix(test_labels, predictions)


# In[30]:


text = ["im feeling rather rotten so im not very ambitious right now"]
l= vectorizer.transform(text)
pred= mn.predict(l)
pred[0]


# In[143]:


# Target_class=["anger", "fear", "joy", "love", "sadness", "surprise"]
# X_class=[df['Text']]


# In[154]:





# In[31]:


text = ["i feel like i m finally losing that stubborn little bit of extra stuff in my lower belly"]
l= vectorizer.transform(text)
pred= mn.predict(l)
pred[0]


# In[32]:


def pred(text):
    l=vectorizer.transform(text)
    pred=mn.predict(l)
    print("the sentiment of the text is:",pred[0])
pred(['feeling grouchy'])


# In[ ]:





# In[30]:


o=[]


# In[48]:


def pred(text):
    
    l=vectorizer.transform([text])
    pred=mn.predict(l)
    return pred[0]
pred('feeling grouchy')


# In[73]:


pred(df['lemmatized_senc'][3])


# In[ ]:





# In[49]:


df['predicted']=df['lemmatized_senc'].apply(pred)
df


# In[35]:


df=df[['lemmatized_senc','Label','predicted']]
df


# In[55]:


def pred(text):
    l=vectorizer.transform([text])
    pred=mn.predict(l)
    print("the sentiment of the text is:",pred[0])
    
    result=df['lemmatized_senc'].isin([text])
    result
    matching_index=df[result]
    
    print(matching_index)
pred('feeling grouchy')


# In[ ]:





# In[40]:


# result=df['lemmatized_senc'].isin([text])
# result
# matching_index=df[result]
# matching_index


# In[ ]:





# In[42]:


df_test=pd.read_csv('test_nltk.txt',header=None, names=['Text','Label'],sep=";")
df_test                                         
      


# In[43]:


df_test['token_words']=df_test['Text'].apply(word_tokenize)
df_test


# In[44]:


df_test['filtered_senc']=df_test['Text'].apply(remove_stopwords)
df_test


# In[45]:


df_test['lemmatized_senc']=df_test['filtered_senc'].apply(word_lemmatize)
df_test


# In[50]:


df_test['predicted']=df_test['lemmatized_senc'].apply(pred)
df_test


# In[54]:


df_test=df_test[['lemmatized_senc','Label','predicted']]
df_test


# In[60]:


def pred_test(text):
    l=vectorizer.transform([text])
    pred=mn.predict(l)
    print("the sentiment of the text is:",pred[0])
    
    result=df_test['lemmatized_senc'].isin([text])
    result
    matching_index=df_test[result]
    
    print(matching_index)
pred_test('im feeling rather rotten im ambitious right')


# In[62]:


# if df_test['Label']!=df_test['predicted']:
#     print('hi')


# In[67]:


df_test['prediction_match'] = df_test['Label'] == df_test['predicted']
df_test


# In[78]:


df_test['correct_prediction']=df_test['prediction_match']==True
df_test


# In[80]:


result=df_test['correct_prediction']==False
matching_index=df_test[result]
matching_index


# In[75]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df_test['prediction_match']=l.fit_transform(df_test['prediction_match'])
df_test


# In[91]:


value=df_test['correct_prediction'].value_counts()
value


# In[ ]:





# In[ ]:





# In[ ]:




