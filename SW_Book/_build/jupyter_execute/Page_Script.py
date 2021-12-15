#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import re
import os
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from IPython.display import Markdown as md
import seaborn as sns
from PIL import Image

from wordcloud import WordCloud
from nltk.corpus import PlaintextCorpusReader as pcr
import community
from importlib import reload 
reload(community)

import gensim
import gensim.corpora as corpora
from gensim.test.utils import common_corpus
import pyLDAvis
import pyLDAvis.gensim_models
import warnings


import re


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
get_ipython().run_line_magic('matplotlib', 'inline')


# # Movie Scripts Analysis

# In[2]:


characters_df = pd.read_csv('data/characters.csv')
characters_dropna = characters_df.dropna()
corpus_root = os.getcwd() + '/data/characters/'
file_list = characters_dropna['File_Name'] + '.txt'
corpus = pcr(corpus_root, file_list)


# In[3]:


def create_wordclouds(data):
    #The if statement filter the object between a corpus and a dictionary
    if type(data) == dict:
        tc_dict = {document:{term:term_count for term,term_count in FreqDist(data[document]).most_common()} for document in data.keys()}
    else:
        tc_dict = {document:{term:term_count for term,term_count in FreqDist(data.words(document)).most_common()} for document in data.fileids()}
    
    #Dictionary with IDF values for each unique word
    idf_dict = {}
    for term in set([term for i in tc_dict.keys() for term in tc_dict[i].keys()]):
        N = len(tc_dict.keys()) #Total number of words inside document
        nt = 0 
        for d in tc_dict.keys():
            if term in tc_dict[d].keys():
                nt += 1 # nt describes in how many documents the term appears
        idf_dict[term] = round(np.log(N/nt),4) #Calculate Inverse Term Frequency
    
    #Dictionary with TC-IDF values for each word for each document
    tc_idf_dict = {document:{term:tc_dict[document].get(term)*idf_dict.get(term) for term in tc_dict[document].keys()}         for document in tc_dict.keys()}
    
    #Dictionary with TC-IDF values of the top 200 words for each docoment
    most_common_dict = {document:{word:value for word,value in sorted(tc_idf_dict[document].items(),         key=lambda item: item[1],reverse=True)[:200]} for document in tc_dict.keys()}
    

    #Create wordclouds
    col_word = ['Reds', 'Purples', 'BuGn', 'Blues', 'Greens'] #Colormap
    mask = np.array(Image.open(os.getcwd() + '/data/Stormtrooper.jpg')) #Create shape image for wordclouds
    a = 0

    #Plotting the word cloud
    plt.figure(figsize=[16, 24])
    for attribute in most_common_dict.keys():
        plt.subplot(3, 2, a+1)
        wordcloud = WordCloud(mask=mask, collocations = False, background_color="white", colormap=col_word[a],             max_font_size=1024, relative_scaling = 0.6, max_words = 200, 
        width = 3000, height = 3000).generate_from_frequencies(most_common_dict[attribute])
        
        a+=1
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        attribute_clean = str(attribute).replace('.txt','')
        plt.title(f'{attribute_clean}', fontsize=30)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")


# # Word clouds

# ## Word clouds for Anakin/Darth Vader based on movies scripts

# In[4]:


wnl = nltk.WordNetLemmatizer()
def clean_text(text, *args):
    for name in args:
        name = re.sub(r'\(|\)','',name)
        for n in re.split(r'\s',name):
            text = re.sub(n,'',text)

    text = re.sub(r'===.*?===','',text) # removes paterns
    text = re.sub(r'==.*?==','',text) 
    text = re.sub(r'\{\{.*?\}\}','',text)

    tokens = word_tokenize(text) # tokennizing the text
    tokens = [wnl.lemmatize(w.lower()) for w in tokens if (w.lower() not in stopwords.words('english') and w.isalpha())] #changes to lower case and removes unwanted types

    final_words = [word for word in tokens if len(word) > 3] #removing words under length 4
    return final_words


# In[5]:


def first_trilogy_script(path):
    with open(path, 'r', encoding = 'utf8') as file:
        text = file.read() #Read txt file that contains the dialogue

    #Filtering
    text = re.sub(r'\\\n\\\n','\n\n',text) #some scripts have \ for every new line
    text = re.sub(r'\\\n',' ',text) # same as above
    sw = re.findall(r'(?=.*:).*(?=\n\n)',text) # a dialogue pattern is name: dialogue
    sw = [re.sub(r'\s+',' ',elem) for elem in sw] #substituting multiple whitespace with one

    names = [re.findall('(.*)(?=:)',elem)[0] for elem in sw] # find all the names pattern is name: 
    names = np.unique(names) # only getting unique names
    
    # Create df with names and their dialogues
    sw_df = pd.DataFrame(names,columns=['text'], index = names)
    for _, row in sw_df.iterrows():
        pattern = r'(?<=' + re.escape(row['text']) + r':).*(?=\n\n)' #finding character specific dialogue
        row['text'] = ' '.join(re.findall(pattern,text)) #adding it as a single string

    sw_df['clean_text'] = sw_df['text'].apply(clean_text) #adding a cleaned version 
    
    return sw_df


# In[6]:


sw1_df = first_trilogy_script('data/script/Star Wars Episode I - The Phantom Menace.txt')
sw2_df = first_trilogy_script('data/script/Star Wars- Episode II - Attack of the Clones.txt')
sw3_df = first_trilogy_script('data/script/Star Wars- Episode III - Revenge of the Sith.txt')


# In[7]:


# Create function that filters the dialogues
def create_df_SW_OR(path):
    dia = pd.read_csv(path, sep="\t") # the path to find the scripts
    list_char_IV = []
    list_diag_IV = []
    for _, row in dia.iterrows(): # we itterate through one dialogue/line at a time.
        text = row.iloc[0] 
        text_split = text.split('"') # splitting by " to get character and dialogue the format is "character" "dialogue"

        # Add to list the character
        list_char_IV.append(text_split[1]) # 1st element is name

        # Add to daiglog of the character
        list_diag_IV.append(''.join(text_split[3:])) # 3rd element and onwrads is dialogue

    df = pd.DataFrame({'Character': list_char_IV, 'Dialogue': list_diag_IV})
    return df


# In[8]:


star_wars_IV_df = create_df_SW_OR('data/script/SW_EpisodeIV.txt')
star_wars_V_df = create_df_SW_OR('data/script/SW_EpisodeV.txt')
star_wars_VI_df = create_df_SW_OR('data/script/SW_EpisodeVI.txt')


# In[9]:


# Create dataframe for each character 
def aug_df_SW(df):
    name_unqiue = set(df['Character']) #finding the unique names in our dialogue dataframe
    list_name = []
    list_dia = []
    for i in name_unqiue: 
        list_name.append(i)
        list_dia.append(clean_text(' '.join(df[df['Character'] == i]['Dialogue'].values))) # taking every row equal to the given name
        
    return pd.DataFrame({'Character': list_name, 'Dialogue': list_dia})


# In[10]:


star_wars_IV_df1 = aug_df_SW(star_wars_IV_df)
star_wars_V_df1 = aug_df_SW(star_wars_V_df)
star_wars_VI_df1 = aug_df_SW(star_wars_VI_df)


# In[11]:


anakin_dict = dict(zip(('Star Wars Episode 1', 'Star Wars Episode 2', 'Star Wars Episode 3'),
                   (sw1_df["clean_text"]['ANAKIN'], 
                    sw2_df["clean_text"]['ANAKIN'], 
                    sw3_df["clean_text"]['ANAKIN'])))


# In[12]:


# Have troble plotting the word cloud for the wepage, but it works locally, thus we upload an image of it 
#create_wordclouds(anakin_dict)


# ![alliance](data/anakin_p.PNG)

# We see on Anakin's word cloud from the second movie, words that describe his love for Padme such as 'alive' and 'nervous'. The word 'mom' shows us how much he misses his mother in the second movie.
# 
# Also in the third movie, we can see more political words such as the 'senate', 'Jedi council', 'war', and 'Chancellor' (referring to Chancellor Palpatine).

# In[13]:


vader_dict = dict(zip(('Episode 4','Episode 5','Episode 6'),
                   ([word for sublist in star_wars_IV_df1['Dialogue'][star_wars_IV_df1['Character'] == 'VADER'] for word in sublist], 
                    [word for sublist in star_wars_V_df1['Dialogue'][star_wars_V_df1['Character'] == 'VADER'] for word in sublist], 
                    [word for sublist in star_wars_VI_df1['Dialogue'][star_wars_VI_df1['Character'] == 'VADER'] for word in sublist])))


# In[14]:


# Have troble plotting the word cloud for the wepage, but it works locally, thus we upload an image of it 
#create_wordclouds(vader_dict)


# ![alliance](data/vader_o.PNG)

# For the sixth movie, we can see important words for the movie plot such as 'emperor', 'lightsaber', 'dark side', and 'sister'. 
# 
# Also in the fifth movie, we observe that the word 'join' appears (from the famous sentence "join the dark side") and the most valuable word for the trilogy is 'father'.

# In[15]:


#We create a list of words from Anakin's dialogue from the prequel trilogy
anakin_dialogue = sw1_df['clean_text']['ANAKIN'] + sw2_df['clean_text']['ANAKIN']+ sw3_df['clean_text']['ANAKIN']

#We create a list of words from Anakin's dialogue from the original trilogy
vader_dialogue = star_wars_IV_df1['Dialogue'][star_wars_IV_df1['Character'] == 'VADER'].values[0]     + star_wars_V_df1['Dialogue'][star_wars_V_df1['Character'] == 'VADER'].values[0]         + star_wars_VI_df1['Dialogue'][star_wars_VI_df1['Character'] == 'VADER'].values[0]

#Above we create a dictionary of both lists from above as values
anakin_vader_dict = {'Anakin Skywalker':anakin_dialogue, 'Darth Vader':vader_dialogue}


# In[16]:


# Have troble plotting the word cloud for the wepage, but it works locally, thus we upload an image of it 
#create_wordclouds(anakin_vader_dict)


# ![alliance](data/anakin_vader.PNG)

# By comparing Anakin to Darth Vader we can see a lot of descriptive words for example 'Padme' and 'right' which stem from Anakin trying to question if what he does is right. Then words such as 'sorry' and 'mother' create a more emotional thus human aspect to the character, but for Darth Vader's word cloud there are words such as 'emperor', 'destiny', 'rebels fighter', and 'rebellion'. His descriptive words have changed a lot after his transformation to the dark side. Also the word 'Luke' appears on Darth Vader's wordcloud which is his son (spoiler alert).

# ## Word clouds for the two trilogies based on movies scripts

# In[17]:


#We create a list with all the dialogues from Prequel Trilogy
original_dialogue = re.split(' ',' '.join([' '.join(i) for i in sw1_df['clean_text'].values]))    + re.split(' ',' '.join([' '.join(i) for i in sw2_df['clean_text'].values]))        + re.split(' ',' '.join([' '.join(i) for i in sw3_df['clean_text'].values]))

#We create a list with all the dialogues from Original Trilogy
prequel_dialogue = re.split(' ',' '.join([' '.join(i) for i in star_wars_IV_df1['Dialogue'].values]))     + re.split(' ',' '.join([' '.join(i) for i in star_wars_V_df1['Dialogue'].values]))         + re.split(' ',' '.join([' '.join(i) for i in star_wars_VI_df1['Dialogue'].values]))

#Dictionary with the lists above as values
original_prequel_dict = {'Original Trilogy':original_dialogue, 'Prequel Trilogy':prequel_dialogue}


# In[18]:


# Have troble plotting the word cloud for the wepage, but it works locally, thus we upload an image of it 
#create_wordclouds(original_prequel_dict)


# ![alliance](data/o_p.PNG)

# We observe for the first wordcloud some descriptive words (of the trilogy plot) such as 'Chancellor', 'Padme', and the battle of 'Geonosis'.
# 
# For the Prequel Trilogy we observe words related to 'Han solo' and Princess 'Leia' such as 'princess', 'solo' ,'millennium' ,'Calrissian' (Known friend/allie of Han solo), 'Chewie' (his companion) and 'Jabba the Hut'.

# # Sentiment analysis

# ### Part 3.5.1: Analysis of Wikipage
# <a id='S_wiki.'></a>

# In[19]:


# Load the LabMT data and convert to dictionary
LabMT_df = pd.read_csv('data/labMT.txt', sep="\t", header=0, skiprows=[0,1])
word_list_LabMT = LabMT_df['word'].tolist()
LabMT_dict = LabMT_df.set_index('word')['happiness_average'].to_dict()

# VADER form the vaderSentiment library
analyzer = SentimentIntensityAnalyzer()

## Load the wiki page for each character and save in a dictionary 
wiki = {}


# In[20]:


# Define function for compute LabMT sentiment for a text
def compute_avg_sentiment_LabMT(text):
    # Join all sentences to one string

    # Clean text 
    words_final = clean_text(text)
    
    # Initialization to store values
    s = 0
    n = 0

    # Using freqDist to only loop over unique words 
    fdist = FreqDist(w for w in words_final)
    word_unique = [word for word in fdist.keys()]
    word_fre = [f for f in fdist.values()]

    # Loop over words 
    for i in range(len(fdist)):
        w = word_unique[i]
        f = word_fre[i]

        # Get the LabMT score
        S_v = LabMT_dict.get(w, None)

        # Weight the socre by the frequency
        if S_v is not None:
            s += f * LabMT_dict.get(w, None)
            n += f
            
    if n == 0:
        return np.nan
    return s/n


# In[21]:


# Define function for compute sentiment for a text
def compute_avg_sentiment_VADER(text):
    if len(text) == 0:
        return 0
    
    # Convert the text file to the right format, where each line is a  new element
    text = text.split('\n')
    if len(text) == 1:
        text = [text, '']

    # Initialzation 
    s = 0
    len_VADER = 0

    # Compute polarity for each sentence in text 
    for sentence in text:
        if sentence:
            vs = analyzer.polarity_scores(sentence)
            s += vs['compound']
            len_VADER += 1
            
    # Retur the average polarity
    return s/len_VADER


# ## Analysis of Movie Scipts
# <a id='S_script.'></a>

# ### Loading the scripts
# Original trilogy movie scripts are from [github](https://github.com/kamran786/Star-Wars-Movie-Scripts-Text-Analysis).

# In[22]:


# Define function to load script of the original trilogy 
def create_df_SW_OR(path):

    # Load txt file
    dia = pd.read_csv(path, sep="\t")

    # Initilaize list for the character and dialouge 
    list_char_IV = []
    list_diag_IV = []

    # Loop over all rows in data 
    for _, row in dia.iterrows():
        text = row.iloc[0]
        text_split = text.split('"')

        # Add character speaking to list
        list_char_IV.append(text_split[1])

        # Add daiglog to list
        list_diag_IV.append(''.join(text_split[3:]))

    # Create dataframe 
    df = pd.DataFrame({'Character': list_char_IV, 'Dialogue': list_diag_IV})
    df['Sentiment_LabMT'] = df['Dialogue'].apply(lambda x: compute_avg_sentiment_LabMT(x))
    df['Sentiment_VADER'] = df['Dialogue'].apply(lambda x: compute_avg_sentiment_VADER(x))
    return df


# Prequal movie scripts are from [script_I](https://www.actorpoint.com/movie-scripts/scripts/star-wars-the-phantom-menace.html), [script_II](http://sellascript.com/Source/resources/screenplays/attackoftheclones.htm), and [script_III](https://www.actorpoint.com/movie-scripts/scripts/star-wars-revenge-of-the-sith.html).

# In[23]:


def create_df_SW_PREQUAL(path):

    # Open file 
    with open(path, 'r', encoding = 'utf8') as file:
        text = file.read()
    
    # remove newlines 
    text = re.sub(r'\\\n\\\n','\n\n',text)
    text = re.sub(r'\\\n',' ',text)
    text = re.sub(r'\(.*\)', '', text)

    # Finding the correct format for the dialogues
    sw1 = re.findall(r'(?=.*:).*(?=\n\n)',text)
    sw1 = [re.sub(r'\s+',' ',elem) for elem in sw1]

    # Loop through all dialogues and add the speaking character and dialogue
    list_char = []
    list_dia = []
    for row in sw1:
        split_text = row.split(':')
        list_char.append(split_text[0])
        list_dia.append(' '.join(split_text[1:]))
    
    # Create dataframe and compute sentiment 
    df = pd.DataFrame({'Character' : list_char, 'Dialogue' : list_dia})
    df['Sentiment_LabMT'] = df['Dialogue'].apply(lambda x: compute_avg_sentiment_LabMT(x))
    df['Sentiment_VADER'] = df['Dialogue'].apply(lambda x: compute_avg_sentiment_VADER(x))

    return df


# In[24]:


# Load the script for the original trilogy
star_wars_IV_df = create_df_SW_OR('data/script/SW_EpisodeIV.txt')
star_wars_V_df = create_df_SW_OR('data/script/SW_EpisodeV.txt')
star_wars_VI_df = create_df_SW_OR('data/script/SW_EpisodeVI.txt')

# Load the scripts for the prequals 
star_wars_I_df = create_df_SW_PREQUAL('data/script/Star Wars Episode I - The Phantom Menace.txt')
star_wars_II_df = create_df_SW_PREQUAL('data/script/Star Wars- Episode II - Attack of the Clones.txt')
star_wars_III_df = create_df_SW_PREQUAL('data/script/Star Wars- Episode III - Revenge of the Sith.txt')


# In[25]:


star_wars_V_df.head()


# In[26]:


# Create dataframe for all movies 
SW_script_df = pd.concat([star_wars_I_df, star_wars_II_df, star_wars_III_df, star_wars_IV_df, star_wars_V_df, star_wars_VI_df]).reset_index()

## Saving index for where the movies shift for later plot
index_II = len(star_wars_I_df)
index_III = index_II + len(star_wars_II_df) 
index_IV = index_III + len(star_wars_III_df) 
index_V = index_IV +  len(star_wars_VI_df) 
index_VI = index_V + len(star_wars_V_df) 
index_All = [0, index_II, index_III, index_IV, index_V, index_VI]
movies_index =['SW_I', 'SW_II', 'SW_III', 'SW_IV', 'SW_V', 'SW_VI']
color_movie = [ 'hotpink', 'darkorchid', 'indigo', 'green', 'darkgreen', 'olive']


# In[27]:


# Plot sentiment as timeseries
figure(figsize = (25,9))
ax = plt.subplot(1,2,1)
plt.plot(SW_script_df.index, SW_script_df['Sentiment_LabMT'], label = 'Sentiment LabMT', color='steelblue')
plt.ylim([1,9])
plt.ylabel('Sentiment score')
plt.xlabel('Number of dialogue')
plt.title('Timeseries of sentiments using LabMT')
# Add the movie indications 
for i,j,c in zip(index_All, movies_index, color_movie):
    plt.axvline(x=i, label='Movie: {}'.format(j), c=c, lw=5)
plt.legend(loc=(0.95,0.77))


ax2 = plt.subplot(1,2,2)
plt.plot(SW_script_df.index, SW_script_df['Sentiment_VADER'], color='darkred', label = 'Sentiment VADER')
plt.ylim([-1,1])
plt.ylabel('Sentiment score')
plt.xlabel('Number of dialogue')
plt.title('Timeseries of sentiments using VADER')
# Add the movie indications 
for i,j,c in zip(index_All, movies_index, color_movie):
    plt.axvline(x=i, label='Movie: {}'.format(j), c=c, lw=5)
plt.legend(loc=(0.95,0.77))

plt.show()


# We have plotted the sentiment scores as a time series based on when the dialogue appears in all the movies. This plot is created in chronological order meaning the prequel movies come before the original movies.  
# From the plot, we see that the sentiment varies a lot. We cannot see any clear indications of a large period of time where characters are happy or sad.

# In[28]:


# Plot distirubtion of sentiment 
figure(figsize = (18,7))
plt.subplot(1,2,1)
plt.hist(SW_script_df['Sentiment_LabMT'], bins=15, align='left', rwidth = 0.5, color='steelblue')
plt.title('Histogram for sentiments using LabMT')
plt.xlabel('Sentiment score')
plt.ylabel('Frequency')
plt.xlim([1,9])

plt.subplot(1,2,2)
plt.hist(SW_script_df['Sentiment_VADER'], bins=15, align='left', rwidth = 0.5, color='darkred')
plt.title('Histogram for sentiments using VADER')
plt.xlabel('Sentiment score')
plt.ylabel('Frequency')
plt.xlim([-1,1])

plt.show()


# Above we have the distribution plot for the sentiment using the two methods. We see that for both methods we have scored across the whole range, but for both methods majority of the dialogues are categorized as neutral.

# #### Character based analysis

# In[29]:


SW_script_df.groupby('Character').agg({'Dialogue': 'count', 'Sentiment_LabMT':'mean', 'Sentiment_VADER':'mean'})    .sort_values('Dialogue', ascending=False).head(10)


# We can see the top 10 characters with most dialogues for both methods have an average sentiment of around 5 and 0 respectively to LabMT and VADER, which are categorized as neutral.

# In[30]:


def time_sentiment(df, name_list):
    figure(figsize = (25,9))
    plt.subplot(1,2,1)

    # Plot for LabMT
    sns.set_palette("Set1")
    for name in name_list:
        df_temp = df[df['Character'] == name]
        plt.plot(df_temp.index, df_temp['Sentiment_LabMT'], label = name)
    # Add the movie indications 
    for i,j,c in zip(index_All, movies_index, color_movie):
        plt.axvline(x=i, label='Movie: {}'.format(j), c=c, lw=5)
    plt.legend(loc=(0.95,0.77))
    plt.ylabel('Sentiment score')
    plt.xlabel('Number of dialogue')
    plt.ylim([1,9])
    plt.title('Timeseries of sentiments for' + str(name_list) + 'using LabMT')

    plt.subplot(1,2,2)
    # Plot for VADER
    sns.set_palette("Set2")
    for name in name_list:
        df_temp = df[df['Character'] == name]
        plt.plot(df_temp.index, df_temp['Sentiment_VADER'], label = name)
    # Add the movie indications 
    for i,j,c in zip(index_All, movies_index, color_movie):
        plt.axvline(x=i, label='Movie: {}'.format(j), c=c, lw=5)
    plt.legend(loc=(0.95,0.77))
    plt.ylim([-1,1])
    plt.ylabel('Sentiment score')
    plt.xlabel('Number of dialogue')
    plt.title('Timeseries of sentiments for ' + str(name_list) + ' using VADER')

    plt.show()


# From the word cloud we found that Anakin Skywalker (Darth Vader) had an interesting shift in the type of word occurring based on the movies. We will see if we can see a similar tendency in the sentiment based on the dialogues spoken by him. Additionaly, as previously mentioned he shifts form the light side to the dark side, thus we whish to see if this can be seen in the sentiment. 

# In[31]:


time_sentiment(SW_script_df, ['ANAKIN','VADER'])


# From the plot above we see on the left-hand side, that there is no clear tendency using LabMT. For the VADER plot on the right, we see that in the second movie Anakin in the beginning primarily have a more positive sentiment. From the world clouds in section 4.2 positive words occurs such as 'alive' and 'laughing'. We know form the movies that he is in love at that time. 
# 
# But from the plot we also see a tendency towards positive in the second half of the third movie. From watching the movie, we know that this is where he is questioning himself and turning to the dark side. This contradicts with the sentiment, since you would imagine someone in question of his existence will not use a lot of words which are positive. This could also mean that the reason for the positive tendency in the second movie could be a coincidence. 
# 
# We will look at Yoda's sentiment timeseries plot to see if the sentiment also varies.

# In[32]:


time_sentiment(SW_script_df, ['YODA'])


# From the plot for Yoda, we again see that there are a lot of variation in the tendency which is unexpected, since Yoda is supposed to be a wise character. Additionally, as a jedi you are supposed to be neutral. These traits can not be seen in plot.  
# 
# From the sentiment analysis we found that the sentiment for the dialogues varies a lot. The sentiment analysis on the Star Wars scripts are not informative. We cannot see any overall tendency for the movies or characters based on the scripts. One of the reasons for this conclusion could be because the scripts are only part of a movie. How an actor acts out the dialogue will largely affect how the viewer perceive the emotions. If a character cries while saying 'I am happy' in the sentiment analysis will categorized it as being positive, but in reality, the actor and the screen is trying to express a sad emotion. The movie scripts are only a small part of expressing the emotions of characters, and thus if we wish to analysis the emotions characters are expressing, we need to take more into account than solely the dialogues.

# ## Part 3.6: Hidden topic modeling
# <a id='htm.'></a> 

# ### Part 3.6.1: Movie Scipts
# <a id='htm_script.'></a> 

# In[33]:


list_of_scripts = [re.split(' ',' '.join([' '.join(i) for i in sw1_df['clean_text'].values])),
re.split(' ',' '.join([' '.join(i) for i in sw2_df['clean_text'].values])),
re.split(' ',' '.join([' '.join(i) for i in sw3_df['clean_text'].values])),
re.split(' ',' '.join([' '.join(i) for i in star_wars_IV_df1['Dialogue'].values])),
re.split(' ',' '.join([' '.join(i) for i in star_wars_V_df1['Dialogue'].values])),
re.split(' ',' '.join([' '.join(i) for i in star_wars_VI_df1['Dialogue'].values]))]


# In[34]:


id2word = corpora.Dictionary(list_of_scripts) # creating dictionary

common_corpus = [id2word.doc2bow(index) for index in list_of_scripts] # maping to index

model = gensim.models.LdaModel(corpus = common_corpus, id2word = id2word, num_topics = 6, chunksize = 1, random_state = 3) #building model
model.show_topics(num_topics=6, num_words = 8) # printing out the words for the topics


# With this topic modelling we see some clear topic-movie correlation. For topic 0 we have 'princess' and 'uncle' which are specific to the 4th movie. For topic 4 we see 'father' and 'sister' which is also specific to movie 6. But we there are also problems, looking at topic 3 Luke and Anaking are only mentioned together in the 6th movie, but we already assigned topic 4 to movie 6. Additionaly the words have small weight and are thus not descriptive of the topic. 
# 
# When mapping the topics with PCA we also see this problem:

# In[35]:


temp = pyLDAvis.gensim_models.prepare(topic_model=model, corpus=common_corpus, dictionary=id2word)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
pyLDAvis.enable_notebook()
pyLDAvis.display(temp)


# Here topic 5 (in the plot) corrosponds to topic 0 (in our model print), and we see that this topic is neglegeble in size. Topic 1 (plot) corrosponds to topic 5 (print output) and given it's large size and the fact that it contains words like (Luke, Vader, Emporer, Leia, Force etc.) it most likely describes the entire original trilogy (movies 4-6). 
# 
# The fact that LDA does not capture all 6 movies and just the generel themes is likely due to two reasons. 
# 1) It's an unsupervised model and thus the result will differ everytime, which is why we set the random_state. This also means it's probably possible to create 6 clearly different topics for the 6 movies, but statistically it will be unlikely because
# 
# 2) the movies are very similar. They are all from the same cinematic universe and movies 4 and 6 had the same plot of destroying the death star. Additionaly, the main characters are the same in the sequal, and in the prequals.

# In[36]:


model = gensim.models.LdaModel(corpus = common_corpus, id2word = id2word, num_topics = 2, chunksize = 1, random_state = 5) #building model
model.show_topics(num_topics=2, num_words = 8) # printing out the words for the topics


# Luke, father, and Chewie are from the sequals, thus the hidden topic modeling is still not capturing prequals/sequals. 
# 
# Finally, we'll do a hidden topic analyses for each movie sepperately:

# In[37]:


i = 1

for movie in list_of_scripts:
    id2word = corpora.Dictionary([movie]) # creating dictionary

    common_corpus = [id2word.doc2bow(index) for index in [movie]] # maping to index

    model = gensim.models.LdaModel(corpus = common_corpus, id2word = id2word, num_topics = 1) #building model
    print(f'Movie: {i}')
    print(f'{model.show_topics(num_topics=1, num_words = 6)}\n') # printing out the words for the topics
    i += 1


# Unsuprisngly the topic modelling better captures the plot of the movies. Geonosis a planet important to the second movie, and 'Mesa' Jar-Jar's catchphrase from the first movie. However, we still se the topics for movies 1 and 5 are undescriptive given their low word score.
