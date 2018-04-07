
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

# In[1]:

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

# In[2]:

moviedata = pd. read_csv('movies_metadata.csv')
#moviedata.head(30)
moviedata.columns.values


# In[3]:

moviedata['genres']= moviedata['genres'].fillna('[]')


# In[4]:

moviedata['genres']=moviedata['genres'].apply(literal_eval)


# In[5]:

moviedata['genres']= moviedata['genres'].apply(lambda x : [i['name'] for i in x] if isinstance(x,list) else [])


# In[ ]:

# for j in range(len(moviedata['genres'])):
#     if isinstance (moviedata['genres'][j],list):
#         genres=[]
#         for i in moviedata['genres'][j]:
#             moviedata['genres'][j]=genres.append(i['name'])
#             #moviedata['genres']=i['name']
#     else:
#         moviedata['genres'][j] =[]


# In[6]:

#moviedata['genres']


# In[7]:

df=pd.DataFrame([1,2,3])
df.apply(np.sqrt).apply(lambda x: x+1)


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

# In[8]:

#moviedata.head(30)


# In[9]:

#moviedata.describe()


# In[10]:

len(moviedata[moviedata['vote_count'] > 433])


# **According to the descriptive chart, we noticed that there are some movies in our list with relatively low and even 0 voters to rating. So we are going to fliter movies which has votes more than  at least 95% of the movies.**

# In[11]:

moviedata['vote_count']=moviedata['vote_count'].fillna(0)


# In[12]:

m=int(np.percentile(moviedata['vote_count'], 95))


# In[13]:

moviedata[moviedata['vote_count']>=m]


# In[14]:

C=moviedata[moviedata['vote_average'].notnull()]['vote_average'].mean()


# In[15]:

#C


# In[16]:

#moviedata[['vote_count','vote_average']].groupby('vote_count').mean()[:50].plot(kind='bar')


# In[17]:

#moviedata['vote_count'].plot(kind='line')


# In[18]:

# df= moviedata.groupby('vote_count').size()[:100]
# df.plot.bar(figsize=(20,10))


# In[20]:

#moviedata.shape


# In[20]:

#print(df.index)


# In[21]:

moviedata['year'] = pd.to_datetime(moviedata['release_date'], errors='coerce').map(lambda x: x.year)


# In[22]:

#moviedata['year']


# In[23]:

'''fliter movies with vote_counts more than 95% movies '''
newmoviedata= moviedata[((moviedata['vote_count']>=m) & (moviedata['vote_average'].notnull()))]


# In[24]:

#newmoviedata.head()


# In[25]:

#newmoviedata.columns


# In[26]:

# import seaborn as sns
# corr=moviedata.corr()
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr,  cmap=cmap, vmax=.3, center=0,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[26]:

'''extract uesful columns '''
usefulcol=['title','year','genres','popularity','vote_average','vote_count']
usefulm_data=newmoviedata[usefulcol]


# In[27]:

#usefulm_data.shape
## our data to be considered for our recommendation system has 2282 movies with 6 importane features


# In[28]:

'''create the column of weighted rating scores'''
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[29]:

usefulm_data['weighted_rating']=weighted_rating(usefulm_data)


# In[30]:

#usefulm_data.head()


# In[31]:

for x in usefulm_data['year']:
    if x!=np.nan:
        x=int(x)
    else:
        x=np.nan


# In[32]:

usefulm_data['year']=usefulm_data['year'].astype(int)


# In[33]:

top250=usefulm_data.sort_values('weighted_rating',ascending= False)[:250]


# In[34]:

#top250


# In[35]:

#top250[['title','weighted_rating']][:10].plot(kind='bar')


# In[37]:

#top250[:15].plot(x='title',y='weighted_rating',kind='bar',color='lightblue',title='Top 30 movies weighted ratings')


# ### Top Movies

# In[38]:

sortmovie=usefulm_data.sort_values('weighted_rating',ascending= False)


# #### Top Crime Movie ####

# In[39]:

#sortmovie[sortmovie['genres'].apply(lambda x:  'Crime' in x)]


# In[40]:

# top20Crimemovie=sortmovie[sortmovie['genres'].apply(lambda x:  'Crime' in x)][:20]
# top20Crimemovie.plot(x='title',y='weighted_rating',kind='bar',color='black',title='Top 20 crime movies rating')
# plt.savefig('top crime movie.png')


# #### Top Action Movie ####

# In[41]:

# sort_action_mv=sortmovie[sortmovie['genres'].apply(lambda x:  'Action' in x)]


# In[42]:

# len(sort_action_mv)


# In[43]:

# sort_action_mv[:20].plot(x='title',y='weighted_rating', kind='bar',fontsize=15, title='Top 20 action movies rating')
# plt.savefig('top action movie.png')


# #### Top Romance Movie####

# In[44]:

# sort_romance_mv=sortmovie[sortmovie['genres'].apply(lambda x:  'Romance' in x)]
# len(sort_romance_mv)


# In[45]:

# sort_romance_mv[:20].plot(x='title',y='weighted_rating', kind='bar',color='lightpink',fontsize=15, title='Top 20 romance movies rating')
# plt.savefig('top romance movie.png')


# #### sorted movies with specified genre function ####

# In[42]:

def TopMovieWithSpecifiedGenre(genre):
    #print(sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)])
    return sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)][:20][['title','year']]
def PlotTop20MovieWithGenre(genre):
    sortmovie[sortmovie['genres'].apply(lambda x:  genre in x)][:20].plot(x='title',y='weighted_rating', kind='bar',fontsize=15, title='Top 20 '+ genre+' movies rating')
    plt.savefig('top '+genre+' movie.png')
    return plt


# In[43]:

#TopMovieWithSpecifiedGenre('Thriller')


# In[47]:

#PlotTop20MovieWithGenre('History')


# In[48]:

#sortmovie.head(15)


# In[12]:

# s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
# s.name = 'genre'
# gen_md = md.drop('genres', axis=1).join(s)


# In[13]:

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

# ## Content Based Recommender
# 
# The recommender we built in the previous section suffers some severe limitations. For one, it gives the same recommendation to everyone, regardless of the user's personal taste. If a person who loves romantic movies (and hates action) were to look at our Top 15 Chart, s/he wouldn't probably like most of the movies. If s/he were to go one step further and look at our charts by genre, s/he wouldn't still be getting the best recommendations.
# 
# For instance, consider a person who loves *Dilwale Dulhania Le Jayenge*, *My Name is Khan* and *Kabhi Khushi Kabhi Gham*. One inference we can obtain is that the person loves the actor Shahrukh Khan and the director Karan Johar. Even if s/he were to access the romance chart, s/he wouldn't find these as the top recommendations.
# 
# To personalise our recommendations more, I am going to build an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked. Since we will be using movie metadata (or content) to build this engine, this also known as **Content Based Filtering.**
# 
# I will build two Content Based Recommenders based on:
# * Movie Overviews and Taglines
# * Movie Cast, Crew, Keywords and Genre
# 
# Also, as mentioned in the introduction, I will be using a subset of all the movies available to us due to limiting computing power available to me. 

# In[44]:

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[45]:

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[46]:

moviedata['id'] = moviedata['id'].apply(convert_int)


# In[47]:

moviedata[moviedata['id'].isnull()]


# In[48]:

moviedata = moviedata.drop([19730, 29503, 35587])


# In[49]:


moviedata['id'] = moviedata['id'].astype('int')


# In[50]:

selectedmd = moviedata[moviedata['id'].isin(links_small)]
#selectedmd.shape


# We have **9099** movies avaiable in our small movies metadata dataset which is 5 times smaller than our original dataset of 45000 movies.

# ### Movie Description Based Recommender
# 
# Let us first try to build a recommender using movie descriptions and taglines. We do not have a quantitative metric to judge our machine's performance so this will have to be done qualitatively.

# In[51]:

selectedmd['tagline'] = selectedmd['tagline'].fillna('')
selectedmd['description'] = selectedmd['overview'] + selectedmd['tagline']
selectedmd['description'] = selectedmd['description'].fillna('')


# In[52]:

#selectedmd['description']


# In[53]:

#selectedmd.info()


# In[54]:

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(selectedmd['description'])


# In[55]:

#tfidf_matrix.shape


# #### Cosine Similarity
# 
#  
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $
# 
# 

# In[56]:

'''calculate Cosine Similarity'''
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[57]:

#cosine_sim[0]#第一个movie和包括自己的所有movie的similarity 


# In[58]:

#cosine_sim[1]


# Write a function that returns the 30 most similar movies based on the cosine similarity score.

# In[76]:

selectedmd = selectedmd.reset_index()
# titles = selectedmd['title']
# indices = pd.Series(selectedmd.index, index=selectedmd['title'])


# In[77]:

# indices['The Godfather']


# In[75]:

# hahaha = pd.Series(titles, index=range(len(titles)))
# hahaha


# In[77]:

newdf=pd.DataFrame(selectedmd['title'], selectedmd.index)


# In[78]:

#newdf.iloc[0]['title']


# In[79]:

def Get_Movie_Location(moviename):
    for i in newdf.index:
         m = re.search(moviename,newdf.iloc[i]['title'],re.I)
#     if newdf.iloc[i]['title']==patternmoviename:
         if bool(m):
            return i 
        # if newdf.iloc[i]['title']==moviename:
            
        #     return i 


# In[80]:

#Get_Movie_Location('The Godfather')


# In[81]:

#sim_scores= list(enumerate(cosine_sim[Get_Movie_Location('The Godfather')]))


# In[82]:

#top30smv=sorted(sim_scores, key=lambda x: x[1],reverse = True)[1:31]


# In[83]:

#newdf['title'].iloc[]
# for i in range(30):
#         print(newdf.iloc[top30smv[i][0]]['title'])


# In[84]:

def GetSimilarMovie(moviename):
    similarity_scores=list(enumerate(cosine_sim[Get_Movie_Location(moviename)]))
    top30smv=sorted(similarity_scores, key=lambda x: x[1],reverse = True)[1:31]
#     similar_list=[]
    movieidx=[]
    for i in range(30):
#         similar_list.append(newdf.iloc[top30smv[i][0]]['title'])
        movieidx.append(top30smv[i][0])
    movies = selectedmd.iloc[movieidx][['title','vote_count','vote_average','year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(10)
    
    return qualified


# In[85]:

#GetSimilarMovie('The Godfather')


# In[86]:

#GetSimilarMovie('Force Majeure')


# In[87]:

#GetSimilarMovie('The Dark Knight')


# We see that for **The Dark Knight**, our system is able to identify it as a Batman film and subsequently recommend other Batman films as its top recommendations. But unfortunately, that is all this system can do at the moment. This is not of much use to most people as it doesn't take into considerations very important features such as cast, crew, director and genre, which determine the rating and the popularity of a movie. Someone who liked **The Dark Knight** probably likes it more because of Nolan and would hate **Batman Forever** and every other substandard movie in the Batman Franchise.
# 
# Therefore, we are going to use much more suggestive metadata than **Overview** and **Tagline**. In the next subsection, we will build a more sophisticated recommender that takes **genre**, **keywords**, **cast** and **crew** into consideration.

# In[ ]:

user_choice = input("Input some keywords of a movie!\nInput 'Q' to quit. \n>>> ")
while user_choice != 'Q':
    try:
        print(GetSimilarMovie(user_choice))
    
    except IndexError:
        print('Oops! we have limited database So we cannot related movie T_T')
    user_choice = input("Input some keywords of a movie!\n Input 'Q' to quit. \n>>> ")
print('Have a nice day! Bye ^o^')
