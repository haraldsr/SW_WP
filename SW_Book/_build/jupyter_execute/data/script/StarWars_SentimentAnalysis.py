#!/usr/bin/env python
# coding: utf-8

# 1 Introduction

# In this kernel we are going to perform a statistical text analysis on the Star Wars scripts from The Original Trilogy Episodes (IV, V and VI), using wordclouds to show the most frequent words. The input files used for the analysis are avaliable here. This post is my particular tribute to the Star Wars Day, on May 4.

# 2 Loading data

# In[1]:


# Load libraries
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from collections import OrderedDict, defaultdict, Counter
import pandas as pd
import csv
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import tokenize
from nltk.tokenize import word_tokenize
from tokenize import tokenize
import itertools
from PIL import Image
from wordcloud import WordCloud
#from ggplot import *

# Read the data
ep4_file = r"C:\\Users\\kahma\\Downloads\\star-wars-movie-scripts\\star-wars-movie-scripts\\SW_EpisodeIV.txt"
ep5_file = r"C:\\Users\\kahma\\Downloads\\star-wars-movie-scripts\\star-wars-movie-scripts\\SW_EpisodeV.txt"
ep6_file = r"C:\\Users\\kahma\\Downloads\\star-wars-movie-scripts\\star-wars-movie-scripts\\SW_EpisodeVI.txt"


# 3 Functions

# The first function performs cleaning and preprocessing steps to a corpus:
# str.replace('[^\w\s]',''). Remove all punctuation marks
# str.replace('  ', ''). Remove excess whitespace
# lambda x: x.lower(). Make all characters lowercase
# lambda x: ' '.join(item for item in x if item not in new_stopwords_list)). Remove some common English stop words (“I”, “she’ll”, “the”, etc.)
# str.replace('\d+', ''). Remove numbers

# In[2]:


# Text transformations
def cleancorpus(txtfile):

    stop_words = set(stopwords.words('english'))
    # add words that aren't in the NLTK stopwords list
    new_stopwords = ['thats','weve','hes','theres','ive','im','will','can','cant','dont','youve','us'
        ,'youre','youll','theyre','whats','didnt']
    new_stopwords_list = stop_words.union(new_stopwords)
    data1 = pd.read_csv(txtfile, delimiter='|')
    pd.set_option('max_colwidth', 200)
    data1["dialogue"] = data1["dialogue"].str.replace('[^\w\s]','')
    data1.dialogue = data1.dialogue.apply(lambda x: x.lower())
    data1.dialogue = data1.dialogue.str.replace('\d+', '')
    data1.dialogue = data1.dialogue.str.split().apply        (lambda x: ' '.join(item for item in x if item not in new_stopwords_list))
    data1.dialogue = data1.dialogue.str.replace('  ', '')
    #print(data1.dialogue.head(120))
    #print(data.dialogue)
    return data1


# The second function constructs the term-document matrix, that describes the frequency of terms that occur in a collection of documents. This matrix has terms in the first column and documents across the top as individual column names.

# In[3]:


def get_top_n_words(corpus):
    word_list = []
    dialogue_list = pd.Series(corpus['dialogue'])
    dialogue_list_temp = dialogue_list.tolist()
    for stat in dialogue_list_temp:
        word_list.extend(stat.split())
    word_series = pd.Series(word_list)
    return word_series.value_counts()


# The next two functions extract tokens containing two words.

# In[4]:


# Define bigram tokenizer 
def bigrams_calculate(bigramfile):
    i = cleancorpus(bigramfile).dialogue         .str.split(expand=True)         .stack()
    j = i + ' ' + i.shift(-1)
    bigrams = j.dropna().reset_index(drop=True)
    return bigrams


# In[5]:


# Most frequent bigrams
def most_frequent_bigrams(freq_bigrams):
    bigrams_list = pd.Series(freq_bigrams)
    count_bigrams = bigrams_list.value_counts().head(20)
    return count_bigrams


# 4 Episode IV: A New Hope

# In[6]:


# How many dialogues?
print('Total Dialogues in Episode 4- A New Hope:', len(cleancorpus(ep4_file).dialogue), '\n')


# In[14]:


# How many characters?
print('Total Characters in Episode 4- A New Hope:', len(cleancorpus(ep4_file).character.unique()), '\n')


# In[15]:


def seriestodf(series):
    df_temp = pd.DataFrame(series)
    df_temp.reset_index(inplace=True)
    df_temp.columns = ('Character', 'Dialogue')
    return df_temp


def seriestodfbigram(series):
    df_temp = pd.DataFrame(series)
    df_temp.reset_index(inplace=True)
    df_temp.columns = ('Bigram', 'Frequency')
    return df_temp

Top20Chars_ep4 = cleancorpus(ep4_file).character.value_counts().head(20)
Top20Chars_ep5 = cleancorpus(ep5_file).character.value_counts().head(20)
Top20Chars_ep6 = cleancorpus(ep6_file).character.value_counts().head(20)

df_ep4 = seriestodf(Top20Chars_ep4)
df_ep5 = seriestodf(Top20Chars_ep5)
df_ep6 = seriestodf(Top20Chars_ep6)
df_ep4_bigram = seriestodfbigram(most_frequent_bigrams(bigrams_calculate(ep4_file)))
df_ep5_bigram = seriestodfbigram(most_frequent_bigrams(bigrams_calculate(ep5_file)))
df_ep6_bigram = seriestodfbigram(most_frequent_bigrams(bigrams_calculate(ep6_file)))


# In[16]:


# Top 20 characters with more dialogues 
Top20Chars_ep4 = cleancorpus(ep4_file).character.value_counts().head(20)
print(Top20Chars_ep4)

# Visualization

df_ep4 = seriestodf(Top20Chars_ep4)
def ggplt(df_ep):
    plt.style.use('ggplot')
    ax = df_ep[['Character', 'Dialogue']].plot(kind='bar', title="Dialogues by a character(Top 20)", figsize=(15, 10),
                                               legend=True, fontsize=12)
    ax.set_xlabel("Character", fontsize=12)
    ax.set_ylabel("Number of Dialogues", fontsize=12)
    return plt.show()

ggplt(df_ep4)


# In[47]:


def wordcloud(data_file):
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    eiv = (get_top_n_words(cleancorpus(data_file)))
    mask = np.array(Image.open(path.join(d, 'C:\\Users\\kahma\\Downloads\\star-wars-movie-scripts'
                                        '\\star-wars-movie-scripts\\wordcloud_masks\\yoda.png')))
    eiv_wc = WordCloud(width=1000, height=1000, background_color='white', mask=mask, random_state=21,
                   max_font_size=110).generate(str(eiv))
    fig = plt.figure(figsize=(32, 16))
    plt.imshow(eiv_wc)

wordcloud(ep4_file)


# In[18]:


# Most frequent bigrams
def ggplt_biagram(df_ep_bigram):
    plt.style.use('ggplot')
    ax = df_ep_bigram[['Bigram', 'Frequency']].plot(kind='bar', title="Most Frequent Bigrams(Top 20)", figsize=(15, 10),
                                                    legend=True, fontsize=12)
    ax.set_xlabel("BIGRAM", fontsize=12)
    ax.set_ylabel("FREQUENCY", fontsize=12)
    return plt.show()


ggplt_biagram(df_ep4_bigram)


# 5 Episode V: The Empire Strikes Back

# In[19]:


# How many dialogues?
print('Total Dialogues in Episode 5 - The Empire Strikes Back:', len(cleancorpus(ep5_file).dialogue), '\n')


# In[20]:


# How many characters?
print('Total Characters in Episode 5 - The Empire Strikes Back:', len(cleancorpus(ep5_file).character.unique()), '\n')


# In[21]:


# Top 20 characters with more dialogues
Top20Chars_ep5 = cleancorpus(ep5_file).character.value_counts().head(20)
print(Top20Chars_ep5)

# Visualization 
ggplt(df_ep5)


# In[22]:


# Wordcloud for Episode V
wordcloud(ep5_file)


# In[23]:


# Most frequent bigrams
ggplt_biagram(df_ep5_bigram)


# 6 Episode VI: Return of the Jedi

# In[24]:


# How many dialogues?
print('Total Dialogues in Episode 6 - Return of the Jedi:', len(cleancorpus(ep6_file).dialogue), '\n')


# In[25]:


# How many characters?
print('Total Characters in Episode 6 - Return of the Jedi:', len(cleancorpus(ep6_file).character.unique()), '\n')


# In[26]:


# Top 20 characters with more dialogues
Top20Chars_ep6 = cleancorpus(ep6_file).character.value_counts().head(20)

# Visualization
ggplt(df_ep6)


# In[27]:


# Wordcloud for Episode VI
wordcloud(ep6_file)


# In[28]:


# Most frequent bigrams
ggplt_biagram(df_ep6_bigram)


# 7 The Original Trilogy

# In this section we are going to compute the previous statistics, but now considering the three movies of The Original Trilogy (Episodes IV, V and VI).

# In[29]:


# The Original Trilogy dialogues 
frames = [cleancorpus(ep4_file), cleancorpus(ep5_file), cleancorpus(ep6_file)]
SW_Trilogy = pd.concat(frames)
Trio = SW_Trilogy.reset_index(drop=True)


# How many dialogues?
print('Total Dialogues in Star Wars Trilogy:', len(Trio), '\n')


# In[30]:


# How many characters?
print('Total Characters in Star Wars Trilogy:', len(Trio.character.unique()), '\n')


# In[31]:


# Top 20 characters with more dialogues
print('Total Dialogues of top 20 characters in Star Wars Trilogy:', '\n', Trio.character.value_counts().head(20),'\n')

# Visualization
Top20Chars_trilogy = SW_Trilogy.character.value_counts().head(20)
df_trilogy = seriestodf(Top20Chars_trilogy)
ggplt(df_trilogy)


# In[32]:


# Wordcloud for The Original Trilogy
dir = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
eiv4 = (get_top_n_words(Trio))
mask = np.array(Image.open(path.join(dir, 'C:\\Users\\kahma\\Downloads\\star-wars-movie-scripts\\'
                                        'star-wars-movie-scripts\\wordcloud_masks\\yoda.png')))
stop_words = set(STOPWORDS)
eiv_wc = WordCloud(width=1000, height=1000, background_color='white', mask=mask, random_state=21,
                   max_font_size=110, stopwords=stop_words).generate(str(eiv4))
fig = plt.figure(figsize=(16, 8))
plt.imshow(eiv_wc)


# In[33]:


# Most frequent bigrams
i = SW_Trilogy.dialogue         .str.split(expand=True)         .stack()
j = i + ' ' + i.shift(-1)
trio_bigrams = j.dropna().reset_index(drop=True)

print('Most Frequent bigrams in Star Wars Trilogy:', '\n',
      most_frequent_bigrams(trio_bigrams), '\n')

df_triology = seriestodfbigram(most_frequent_bigrams(trio_bigrams))

ggplt_biagram(df_triology)


# 7.1 Sentiment analysis

# Let’s address the topic of opinion mining or sentiment analysis. We can use the tools of text mining to approach the emotional content of text programmatically.

# In[34]:


# Transform the text to a tidy data structure with one token per row
SW_Trilogy['dialogue1'] = SW_Trilogy['dialogue'].apply(word_tokenize)
dialogue_list = list(SW_Trilogy.dialogue1)
dialogue_merged = list(itertools.chain.from_iterable(dialogue_list))


# In[210]:


# Frequency of each sentiment
wordList = defaultdict(list)
emotionList = defaultdict(list)
with open('C:\\Users\\kahma\\OneDrive\\Desktop\\NRC-Sentiment-Emotion-Lexicons\\NRC-Sentiment-Emotion-Lexicons\\'
          'NRC-Emotion-Lexicon-v0.92\\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if int(present) == 1:
            #print(word)
            wordList[word].append(emotion)
            emotionList[emotion].append(word)
            
def generate_emotion_count(string, wt):
    emoCount = Counter()
    for token in dialogue_merged:
        emoCount += Counter(wordList[token])
    return emoCount


wt = list(itertools.chain.from_iterable(dialogue_list))
emotionCounts = [generate_emotion_count(SW_Trilogy.dialogue, wt)]
dialogues = SW_Trilogy['dialogue']
emotion_df = pd.DataFrame(emotionCounts)
emotion_df = emotion_df.fillna(0)

plt.style.use('ggplot')
axs = emotion_df[['positive', 'negative', 'trust', 'anticipation',
                 'fear', 'anger', 'joy', 'sadness', 'surprise', 'disgust']] \
    .plot(kind='bar', title="Dialogues by a character(Top 20)", figsize=(15, 10), legend=True, fontsize=12)
axs.set_xlabel("CHARACTER", fontsize=12)
axs.set_ylabel("DIALOGUE", fontsize=12)
plt.show()


# In[49]:


wordList = defaultdict(list)
emotionList = defaultdict(list)
with open('C:\\Users\\kahma\\OneDrive\\Desktop\\NRC-Sentiment-Emotion-Lexicons\\NRC-Sentiment-Emotion-Lexicons\\'
          'NRC-Emotion-Lexicon-v0.92\\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if emotion == 'fear' and int(present) == 1:
            print(word)
#             wordList[word].append(emotion)
#             emotionList[emotion].append(word)
            
# def generate_emotion_count(string, wt):
#     emoCount = Counter()
#     for token in dialogue_merged:
#         emoCount += Counter(wordList[token])
#     return emoCount


# In[62]:


file_nrc = r'C:\\Users\\kahma\\OneDrive\\Desktop\\NRC-Sentiment-Emotion-Lexicons\\NRC-Sentiment-Emotion-Lexicons\\NRC-Emotion-Lexicon-v0.92\\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
sentiment = pd.read_csv(file_nrc, delimiter='\t',header=None)
sentiment.columns = ['Word', 'Emotion', 'isPresent']
sentiment_final = sentiment[sentiment['isPresent'] == 1]
sentiment_final.head()


# In[64]:


sentiment_sample = sentiment_final.groupby(['Emotion']).count()
print(sentiment_sample)


# In[130]:


sentiment_words = pd.DataFrame(columns = ['Word','Emotion'])
#temp = 'abacus'
for temp in dialogue_merged:
    #if sentiment_final.Word == temp:
#print(temp,sentiment_final[['Word']],sentiment_final['Word']==temp)
    df = sentiment_final[sentiment_final['Word']==temp][['Word','Emotion']]
    sentiment_words = sentiment_words.append(df)

#print(sentiment_words.head())
        


# In[131]:


sentiment_words = sentiment_words.reset_index()
#sentiment_words = sentiment_words.drop(['index'])
#print(sentiment_words.head())
#print(sentiment_words.groupby(['Emotion'])['Word'].count())
#sentiment_words = sentiment_words.groupby(['Emotion','Word']).count()#.sort_values(by=''))
#sentiment_words.columns = ['Emotion','Word','Count']
#sentiment_words = sentiment_words.reset_index()
sentiment_words.head()


# In[415]:


sentiment_final2 = sentiment_words
sentiment_final2 = sentiment_final2.groupby(['Emotion','Word']).count()#.sort_values(by=''))
sentiment_final2 = sentiment_final2.reset_index().sort_values(by=['Emotion','index'],ascending=False)#.nlargest(10, 'index')
sentiment_final2 = sentiment_final2[sentiment_final2.Word != 'ill']
sentiment_final3 = sentiment_final2.groupby('Emotion').apply(lambda x: x.nlargest(10, 'index')).reset_index(drop=True) 


# In[416]:


sentiment_final3


# In[287]:


sentiment_final3.groupby('Emotion').size().reset_index(name = 'Word')


# compute the most frequent words for each sentiment.

# In[417]:


p = sentiment_final3.groupby('Emotion').plot(x='Word', y='index', kind = 'barh')
p + geom_bar(stat = "identity") + facet_wrap('Emotion') + coord_flip()

