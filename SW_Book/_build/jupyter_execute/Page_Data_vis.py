#!/usr/bin/env python
# coding: utf-8

# # The Star Wars data 

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# The data used for the analysis is a data frame containing all the characters in the chosen Star Wars movies, meaning all movies in the Skywalker saga (Star Wars 1-9). The data frame is created such that we will get an easy overview of all characters and their associated attributes. The information of characters will come from the [Star Wars fandom wiki](https://starwars.fandom.com/wiki/Main_Page). There exists a unique wiki page for each movie, where all the characters appearing in the movie will be listed. An example is:  
# ![wiki_appearances](data/wiki_appearances.PNG)
# 
# The characters in our data frame will be from _Characters_ and _Creatures_ in the web pages. Furthermore, we will only extract the characters categorized as cannon, which means we will ignore legend characters.
# We have chosen to add _homeworld, species, gender, affiliation_ and _alliance_. 
# We have in total 1397 unique characters. 
# Here we an example of the data frame. 

# In[2]:


characters_df = pd.read_csv('data/characters.csv')
characters_df[characters_df['Name'].isin(['Anakin Skywalker','R2-D2', 'Yané','Finn', 'Unidentified female First Order officer'])]

