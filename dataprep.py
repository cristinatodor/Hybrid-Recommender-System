# code based on 
# https://github.com/gyhou/yelp_dataset/blob/master/notebooks/yelp_dataset_csv_conversion.ipynb

import json
import os
import collections
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
 

def prep_covid():
    # local path 
    covid_json_path = './covid_19_dataset/yelp_academic_dataset_covid_features.json'

    df_covid = pd.read_json(covid_json_path, lines = True)
    df_covid = df_covid.drop(['highlights', 'Grubhub enabled'], axis=1)
    print(df_covid.sample(20))
    print(df_covid.shape)
    print(df_covid.columns)

    closed = df_covid[df_covid['Temporary Closed Until'] != 'FALSE']
    print('Closed: ', closed.shape)
    print(closed.head())
    closed = closed.drop(['Call To Action enabled', 'Request a Quote Enabled'], axis=1)

    opened = df_covid[df_covid['Temporary Closed Until'] == 'FALSE']
    opened = opened.drop(['Call To Action enabled', 'Request a Quote Enabled'], axis=1)

    banner = df_covid[df_covid['Covid Banner'] != 'FALSE']
    print('Banner: ', banner.shape)

    df = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv.csv')
    df = df.drop(['cool', 'funny', 'useful', 'date', 'count', 'review_id'], axis = 1)

    unique_business_id = df['business_id'].unique()
    unique_user_id = df['user_id'].unique()
    print('Unique businesses: ', len(unique_business_id))
    print('Unique users: ', len(unique_user_id))

    df_selected_business = pd.read_csv('./data/selected_business_Restaurants.csv')
    df_selected_business = df_selected_business[df_selected_business['business_id'].isin(df['business_id'])]

    print(df_selected_business.shape)

    df_merge = pd.merge(df_selected_business, opened, how='inner', on='business_id')
    print(df_merge.shape)
    print(df_merge.head(10))

    df_merge.to_csv('./data/selected_business_Restaurants_covid.csv', encoding = 'utf-8', index = True)


def prep_business_review():
    # local path 
    business_json_path = './yelp_dataset/yelp_academic_dataset_business.json'
    
    df_business = pd.read_json(business_json_path, lines = True)
    print(df_business.head())
    print(df_business.info())
    print(df_business.shape)

    # create Pandas DataFrame filters
    names = df_business.columns
    shape = df_business.shape
    print(names)

    # filter by open business
    df_business = df_business[df_business['is_open']==1]
    print(df_business.head())
    print(df_business.shape)

    # create one row for each series that contain comma-separated items
    df_explode = df_business.assign(categories = df_business.categories.str.split(', ')).explode('categories')
    #print(df_explode.shape)
    #print(df_explode.sample(3))

    print('Total number of categories:', len(df_explode.categories.value_counts()))
    print('Top 20 categories:')
    print(df_explode.categories.value_counts()[:20])

    # finding categories that contain restaurants + food
    print(df_explode[df_explode['categories'].str.contains('Restaurant', case=True, na=False)].categories.value_counts())
    print(df_explode[df_explode['categories'].str.contains('Food', case=True, na=False)].categories.value_counts())
    
    # keep only business with categories that are restaurant related -- check if others are needed!
    df_filtered = df_business[df_business['categories'].str.contains(
                         'Restaurants|Pop-Up Restaurants|Food|Fast Food|Specialty Food|Food Trucks|Ethnic Food|Comfort Food|Soul Food|Do-It-Yourself Food|Live/Raw Food|Pub Food|Swiss Food', 
                         case=False, na=False)]
    
    selected_features = ['business_id', 'name', 'city', 'stars', 'review_count', 'attributes', 'categories']

    # make a DataFrame that contains only the abovementioned columns
    df_selected_business = df_filtered[selected_features]
    print(df_selected_business.head())

    # rename the column name "stars" to "avg_stars" to avoid naming conflicts with review dataset
    df_selected_business =  df_selected_business.rename(columns={'stars': 'avg_stars'})
    #print(df_selected_business.head())
    print(df_selected_business.size)

    print('Number of cities: ', len(df_selected_business.city.value_counts()))
    print('Top 20:\n', df_selected_business.city.value_counts()[:20])

    '''
    print('Number of states: ', len(df_selected_business.state.value_counts()))
    print('Top 20:\n', df_selected_business.state.value_counts()[:20])
    '''

    df_selected_business_toronto = df_selected_business[df_selected_business['city'] == 'Toronto']

    print('Selected busineess Toronto: ', df_selected_business_toronto.shape) 
    print('Selected busineess: ', df_selected_business.shape) 

    # save to CSV 
    df_selected_business.to_csv('./data/selected_business_Restaurants.csv', encoding = 'utf-8', index = False)
    df_selected_business_toronto.to_csv('./data/selected_business_Restaurants_Toronto.csv', encoding = 'utf-8', index = False)

    # local path 
    review_json_path = './yelp_dataset/yelp_academic_dataset_review.json'

    df_review = pd.read_json(review_json_path, lines = True)
    print(df_review.head())
    print(df_review.info())
    print(df_review.shape)

    print('Prep business df')

    # prepare the business dataframe and set index to column "business_id", and name it as df_left
    #df_left = df_selected_business_toronto.set_index('business_id')
    df_left = df_selected_business.set_index('business_id')

    # prepare the review dataframe and set index to column "business_id", and name it as df_right
    df_right = df_review.set_index('business_id')

    # inner join df_left and df_right   
    df_merge = pd.merge(df_left, df_right, left_index = True, right_index = True, how = 'inner')
    df_merge.reset_index()

    # make a filter that selects date after 2019-09-10
    df_merge['date'] = pd.to_datetime(df_merge['date'])

    df_merge_filtered = df_merge.copy()
    boundary = pd.to_datetime('2019-10-01')
    

    # filter the joined DataFrame 
    df_final = df_merge_filtered[df_merge_filtered['date'] >= boundary].copy()

    # e.g. calculate counts of reviews per business entity, and plot it
    df_final['count'] = 1

    df_review_num = df_final[['count']].groupby('business_id').sum()
    # ouput the reviews per business entity
    print(df_review_num.head())

    df_reviews_text = df_merge_filtered[df_merge_filtered['date'] >= boundary].copy()
    df_reviews_text.groupby('business_id')
    print(df_reviews_text.head())

    df_reviews_merge = pd.merge(df_reviews_text, df_review_num, how = 'inner', left_index = True, right_index = True)
    print(df_reviews_merge.head())
    print('Review: ', df_reviews_merge.shape)

    df_reviews_merge.to_csv('./data/2019_3months_Restaurants_reviews.csv', encoding = 'utf-8', index = True)
    

prep_business_review()
prep_covid()


