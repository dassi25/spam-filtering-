
# coding: utf-8

# In[ ]:


import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


##nltk.download_shell()


# In[ ]:


messages =[line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[ ]:


print(len(messages))


# In[ ]:


messages[100]


# In[6]:


for mess_no,messages in enumerate(messages[:10]):
    print(mess_no,messages)
    print("\n")


# In[7]:


message1 = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])


# In[8]:


message1.head()


# In[9]:


message1.describe()


# In[10]:


message1.groupby('label').describe()


# In[11]:


message1['length']=message1['message'].apply(len)


# In[12]:


message1['length'].plot.hist(bins=150)


# In[13]:


message1['length'].describe()


# In[14]:


message1[message1['length'] == 910]['message'].iloc[0]


# In[15]:


message1.hist(column='length',by='label',bins=60,figsize=(12,6))


# In[16]:


import string


# In[ ]:


mess = 'this is a really mess! gotcha :"'


# In[18]:


nopunc=[c for c in mess if c not in string.punctuation] 


# In[19]:


string.punctuation


# In[20]:


nopunc


# In[24]:


from nltk.corpus import stopwords


# In[25]:


nopunc=''.join(nopunc)


# In[26]:


nopunc.split()


# In[27]:


clean_mess= [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[28]:


clean_mess


# In[29]:


def func(mess):
    nopunc=[c for c in mess if c not in string.punctuation] 
    nopunc=''.join(nopunc)
    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    


# In[30]:


message1['message'].head().apply(func)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer


# In[32]:


bow_transformer = CountVectorizer(analyzer=func).fit(message1['message'])


# In[33]:


print(len(bow_transformer.vocabulary_))


# In[34]:


mess4 = message1['message'][3]


# In[35]:


print(mess4)


# In[36]:


bow5 = bow_transformer.transform([mess4])


# In[37]:


print(bow5)


# In[50]:


bow_transformer.get_feature_names()[7186]


# In[42]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[43]:


messages_bow = bow_transformer.transform(message1['message'])


# In[44]:


tfidf = TfidfTransformer().fit(messages_bow)


# In[56]:


tfidf4= tfidf.transform(bow5)


# In[46]:


print(tfidf4)


# In[47]:


tfidf.idf_[bow_transformer.vocabulary_['hey']]


# In[48]:


tfidf_messages= tfidf.transform(messages_bow)


# In[38]:


from sklearn.naive_bayes import MultinomialNB


# In[49]:


spam_detect_model = MultinomialNB().fit(tfidf_messages,message1['label'])


# In[54]:


spam_detect_model.predict(tfidf4)[0]


# In[52]:


##message1.head()


# In[57]:


all_pred = spam_detect_model.predict(tfidf_messages)


# In[58]:


all_pred


# In[59]:


from sklearn.cross_validation import train_test_split


# In[63]:


msg_train, msg_test, label_train, label_test = train_test_split(message1['message'], message1['label'], test_size=0.3)


# In[64]:


from sklearn.pipeline import Pipeline


# In[65]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=func)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
    
    
])


# In[66]:


pipeline.fit(msg_train,label_train)


# In[68]:


pred_i = pipeline.predict(msg_test)


# In[69]:


from sklearn.metrics import classification_report


# In[71]:


print(classification_report(label_test,pred_i))

