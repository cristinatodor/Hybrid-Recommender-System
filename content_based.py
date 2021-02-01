# code based on 
# https://github.com/youonf/recommendation_system/blob/master/content_based_filtering/content_based_recommender_approach2_v2.ipynb

import os
import collections
import csv
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from rake_nltk import Rake 
import nltk 


def prep_data(delivery):
    df = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df = df.drop(['cool', 'funny', 'useful', 'date', 'count', 'review_id'], axis = 1)

    # businesses that aren't temporary closed
    df_selected_business_covid = pd.read_csv('./data/selected_business_Restaurants_Toronto_covid.csv')
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['business_id'].isin(df['business_id'])]

    df_selected_business_covid = df_selected_business_covid.drop(['Unnamed: 0', 'Temporary Closed Until', 'Covid Banner', 'Virtual Services Offered'], axis=1)
    
    if delivery:
        df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]

    # only keep businesses that have at least 20 ratings
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['review_count'] >= 20]

    df = df[df['business_id'].isin(df_selected_business_covid['business_id'])]

    # only for testing purposes
    '''
    print(df.loc[4322])
    df = df.drop(index=4322)
    print(df.loc[11400])
    df = df.drop(index=11400)
    #print(df.loc[11740])
    #df = df.drop(index=11740) 
    '''

    unique_business_id = df['business_id'].unique()
    unique_user_id = df['user_id'].unique()

    ratings = df[['user_id', 'stars', 'business_id', 'name', 'categories']]

    df_merged = df_selected_business_covid

    unique_business_id = df_merged['business_id'].unique()
    business_shape = unique_business_id.shape
    business_df = pd.DataFrame({
            'business_id' :unique_business_id,
            'business_index': range(business_shape[0])
        })

    df_merged = pd.merge(df_merged, business_df, how = 'inner', left_on='business_id', right_on = 'business_id')

    
    df_merged = df_merged.drop(['avg_stars', 'review_count'], axis=1)

    contents_df = df_merged[['business_id', 'name', 'city', 'categories', 'attributes']] 

    contents_df['categories'] = contents_df['categories'].map(lambda x: x.lower().split(','))

    copy_contents_df = contents_df.copy()

    def remove_spaces(cat_list):
            x_list = []

            for x in cat_list:
                x = ''.join(x.split())
                x_list.append(x)
                
            return x_list

    for index, row in contents_df.iterrows():
        '''
        if index == 0:
            print(row['categories'])
            print(copy_unique_df.iloc[index]['categories'])
        '''        
        x_list = remove_spaces(row['categories'])
       
        copy_contents_df.at[index, 'categories'] = x_list

        x = ''.join(row['city'].split())
        copy_contents_df.at[index, 'city'] = x.lower()


    contents_df['categories'] = copy_contents_df['categories']
    contents_df['city'] = copy_contents_df['city']

    contents_df = contents_df.drop(['attributes'], axis=1)
   
    # item profile 
    df_item = contents_df[['business_id', 'categories']]

    categories = df_item['categories']
    df_categories = df_item.explode('categories')

    # one-hot encoding for category
    df_oh_categories = pd.get_dummies(df_categories['categories'])
    #normalized
    df_categories_normalized = df_oh_categories.apply(lambda x: x/np.sqrt(df_oh_categories.sum(axis=1)))

    #create item profile
    df_item = pd.concat([df_item, df_categories_normalized], axis=1)
    df_item = df_item.drop(columns='categories')

    df_item_unique = df_item.groupby('business_id').sum()

    return ratings, df_item_unique, contents_df, unique_business_id


def recommend_cbf(user_id, delivery):

    ratings, df_item_unique, contents_df, unique_business_id = prep_data(delivery)

    rating_df = pd.pivot_table(ratings, values='stars', index=['business_id'], columns = ['user_id'])

    # get number of users
    users_no = rating_df.columns

    df_users = pd.DataFrame(columns=df_item_unique.columns)

    # normally, a matrix containing all users is built
    # because this takes a while, only the row for the active user is computed here
    # this doesn't affect the recommender, as it builds the user profile based only on the active user's preferences 

    '''
    for i in tqdm(range(len(users_no))):
        working_df = df_item_unique.mul(rating_df.iloc[:,i], axis=0)
        working_df.replace(0, np.NaN, inplace=True)    
        df_users.loc[users_no[i]] = working_df.mean(axis=0)
    '''
    user_list = users_no.tolist()
    index = user_list.index(user_id)

    #i = 0 
    working_df = df_item_unique.mul(rating_df.iloc[:,index], axis=0)
    working_df.replace(0, np.NaN, inplace=True)    
    df_users.loc[users_no[index]] = working_df.mean(axis=0)

    document_frequency = df_item_unique.sum()
    idf = (len(unique_business_id)/document_frequency).apply(np.log) #log inverse of DF
    #The dot product of article vectors and IDF vectors gives us the weighted scores of each article.
    idf_df_item = df_item_unique.mul(idf.values)

    # make prediction by tfidf
    df_predict = pd.DataFrame()

    '''
    for i in tqdm(range(len(users_no))):
        working_df = idf_df_item.mul(df_users.iloc[i], axis=1)
        df_predict[users_no[i]] = working_df.sum(axis=1)
    '''

    #i = 0
    working_df = idf_df_item.mul(df_users.iloc[0], axis=1)
    df_predict[users_no[index]] = working_df.sum(axis=1)

    # get all business id
    business_id = df_predict.index

    #user predicted rating to all businesses
    user_predicted_rating = df_predict[user_id]

    # combine ratings and business details
    user_rating_business = pd.concat([user_predicted_rating, contents_df.set_index('business_id')], axis=1)

    already_rated = ratings[ratings['user_id'].isin([user_id])]

    #recommendation without places being rated by user
    all_rec = user_rating_business[~user_rating_business.index.isin(already_rated['business_id'])]

    return all_rec.sort_values(by=[user_id], ascending=False)
 




