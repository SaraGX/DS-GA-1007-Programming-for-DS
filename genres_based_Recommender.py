
# coding: utf-8

# # Movies Recommender System

# This is the second part of my Springboard Capstone Project on Movie Data Analysis and Recommendation Systems. In my first notebook ( [The Story of Film](https://www.kaggle.com/rounakbanik/the-story-of-film/) ), I attempted at narrating the story of film by performing an extensive exploratory data analysis on Movies Metadata collected from TMDB. I also built two extremely minimalist predictive models to predict movie revenue and movie success and visualise which features influence the output (revenue and success respectively).
# 
# In this notebook, I will attempt at implementing a few recommendation algorithms (content based, popularity based and collaborative filtering) and try to build an ensemble of these models to come up with our final recommendation system. With us, we have two MovieLens datasets.
# 
# * **The Full Dataset:** Consists of 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
# * **The Small Dataset:** Comprises of 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users.
# 
# I will build a Simple Recommender using movies from the *Full Dataset* whereas all personalised recommender systems will make use of the small dataset (due to the computing power I possess being very limited). As a first step, I will build my simple recommender system.

# In[2]:

#get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
                                                                                                                                                                                                                            
import warnings; warnings.simplefilter('ignore')


# ## Simple Recommender
# 
# The Simple Recommender offers generalized recommnendations to every user based on movie popularity and (sometimes) genre. The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user. 
# 
# The implementation of this model is extremely trivial. All we have to do is sort our movies based on ratings and popularity and display the top movies of our list. As an added step, we can pass in a genre argument to get the top movies of a particular genre. 

# In[3]:

moviedata = pd. read_csv('movies_metadata.csv')
#moviedata.head(30)
#moviedata.columns.values


# In[4]:

moviedata['genres']= moviedata['genres'].fillna('[]')


# In[5]:

moviedata['genres']=moviedata['genres'].apply(literal_eval)


# In[6]:

moviedata['genres']= moviedata['genres'].apply(lambda x : [i['name'] for i in x] if isinstance(x,list) else [])


# In[7]:

# for j in range(len(moviedata['genres'])):
#     if isinstance (moviedata['genres'][j],list):
#         genres=[]
#         for i in moviedata['genres'][j]:
#             moviedata['genres'][j]=genres.append(i['name'])
#             #moviedata['genres']=i['name']
#     else:
#         moviedata['genres'][j] =[]


# In[8]:

#moviedata['genres']


# In[9]:


# > From the IMDb site: 
# 
# >The formula for calculating the Top Rated 250 Titles gives a true Bayesian estimate:
# 
# >Weighted Rating (WR) = $(\frac{v}{v+m} * R) + (\frac{m}{(v+m)}* C) $
# 
# >Where:
# >R = average for the movie (mean) = (Rating)
# 
# >v = number of votes for the movie = (votes)
# 
# >m = minimum votes required to be listed in the Top 250 (currently 25000)
# 
# >C = the mean vote across the whole report (currently 7.0)
# 
# >For the Top 250, only votes from regular voters are considered.@
# 
# ** We are going to use this formula to rate our moviedata. We need to redefine m and C **

# In[10]:

# moviedata.head(30)


# In[11]:

# moviedata.describe()


# In[12]:

# len(moviedata[moviedata['vote_count'] > 433])


# **According to the descriptive chart, we noticed that there are some movies in our list with relatively low and even 0 voters to rating. So we are going to fliter movies which has votes more than  at least 95% of the movies.**

# In[13]:

moviedata['vote_count']=moviedata['vote_count'].fillna(0)


# In[14]:

m=int(np.percentile(moviedata['vote_count'], 95))


# In[15]:

moviedata[moviedata['vote_count']>=m]


# In[16]:

C=moviedata[moviedata['vote_average'].notnull()]['vote_average'].mean()


# In[17]:

# C


# In[18]:

# moviedata[['vote_count','vote_average']].groupby('vote_count').mean()[:50].plot(kind='bar')


# In[19]:

# moviedata['vote_count'].plot(kind='line')


# In[20]:

df= moviedata.groupby('vote_count').size()[:100]
# df.plot.bar(figsize=(20,10))


# In[21]:

# moviedata.shape


# In[22]:

# print(df.index)


# In[23]:

moviedata['year'] = pd.to_datetime(moviedata['release_date'], errors='coerce').map(lambda x: x.year)


# In[24]:




# In[25]:

'''fliter movies with vote_counts more than 95% movies '''
newmoviedata= moviedata[((moviedata['vote_count']>=m) & (moviedata['vote_average'].notnull()))]


# In[26]:




# In[27]:




# In[28]:



# In[29]:

'''extract uesful columns '''
usefulcol=['title','year','genres','popularity','vote_average','vote_count']
usefulm_data=newmoviedata[usefulcol]


# In[30]:

# usefulm_data.shape
## our data to be considered for our recommendation system has 2282 movies with 6 importane features


# In[31]:

'''create the column of weighted rating scores'''
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[32]:

usefulm_data['weighted_rating']=weighted_rating(usefulm_data)


# In[33]:

# usefulm_data.head()


# In[34]:

for x in usefulm_data['year']:
    if x!=np.nan:
        x=int(x)
    else:
        x=np.nan


# In[35]:

usefulm_data['year']=usefulm_data['year'].astype(int)


# In[36]:

top250=usefulm_data.sort_values('weighted_rating',ascending= False)[:250]


# In[37]:



# In[38]:

# top250[['title','weighted_rating']][:10].plot(kind='bar')


# In[39]:

# top250[:15].plot(x='title',y='weighted_rating',kind='bar',color='lightblue',title='Top 30 movies weighted ratings')


# ### Top Movies

# In[40]:

sortmovie=usefulm_data.sort_values('weighted_rating',ascending= False)


# #### Top Crime Movie ####

# In[41]:

sortmovie[sortmovie['genres'].apply(lambda x:  'Crime' in x)]


# In[42]:

top20Crimemovie=sortmovie[sortmovie['genres'].apply(lambda x:  'Crime' in x)][:20]
# top20Crimemovie.plot(x='title',y='weighted_rating',kind='bar',color='black',title='Top 20 crime movies rating')
# plt.savefig('top crime movie.png')


# #### Top Action Movie ####

# In[43]:

sort_action_mv=sortmovie[sortmovie['genres'].apply(lambda x:  'Action' in x)]


# In[44]:



# In[45]:

# sort_action_mv[:20].plot(x='title',y='weighted_rating', kind='bar',fontsize=15, title='Top 20 action movies rating')
# plt.savefig('top action movie.png')


# #### Top Romance Movie####

# In[46]:

sort_romance_mv=sortmovie[sortmovie['genres'].apply(lambda x:  'Romance' in x)]
# len(sort_romance_mv)


# In[47]:

# sort_romance_mv[:20].plot(x='title',y='weighted_rating', kind='bar',color='lightpink',fontsize=15, title='Top 20 romance movies rating')
# plt.savefig('top romance movie.png')


# #### sorted movies with specified genre function ####

# In[48]:

def TopMovieWithSpecifiedGenre(genre):
    #print(sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)])
    return sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)][:20][['title','year']]
def PlotTop20MovieWithGenre(genre):
    sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)][:20].plot(x='title',y='weighted_rating', kind='bar',fontsize=15, title='Top 20 '+ genre+' movies rating')
    plt.savefig('top '+genre+' movie.png')
    return plt


# In[49]:

# TopMovieWithSpecifiedGenre('Thriller')


# # In[50]:

# PlotTop20MovieWithGenre('History')


# # In[51]:

# sortmovie.head(15)


# In[52]:

# s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
# s.name = 'genre'
# gen_md = md.drop('genres', axis=1).join(s)


# In[53]:

# def build_chart(genre, percentile=0.85):
#     df = gen_md[gen_md['genre'] == genre]
#     vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
#     vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
#     C = vote_averages.mean()
#     m = vote_counts.quantile(percentile)
    
#     qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
#     qualified['vote_count'] = qualified['vote_count'].astype('int')
#     qualified['vote_average'] = qualified['vote_average'].astype('int')
    
#     qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
#     qualified = qualified.sort_values('wr', ascending=False).head(250)
    
#     return qualified


# Let us see our method in action by displaying the Top 15 Romance Movies (Romance almost didn't feature at all in our Generic Top Chart despite  being one of the most popular movie genres).
# 

# In[55]:

user_choice = input("Input the genre of movie you'd like to watch (!note!Plz input with first letter captialized)\n Input 'Q' to quit \n>>> ")
while user_choice != 'Q':
    try:
        print(TopMovieWithSpecifiedGenre(user_choice))
    except IndexError:
        print('Oops! we have limited database So we cannot provied you with related movie T_T')
    user_choice=input("Input the genre of movie you'd like to watch (!note!Plz input with first letter captialized)\n Input 'Q' to quit \n>>> ")
print('Have a nice day! Bye ^o^')
# In[ ]:



