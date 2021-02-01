import os
import collections
import csv
import operator
import itertools

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import math

from content_based import recommend_cbf
from collaborative import recommend_cf

from texttable import Texttable

def main():
    df = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df = df.drop(['cool', 'funny', 'useful', 'date', 'count', 'review_id'], axis = 1)

    # businesses that aren't temporary closed 
    df_selected_business_covid = pd.read_csv('./data/selected_business_Restaurants_Toronto_covid.csv')
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['business_id'].isin(df['business_id'])]

    print('\n')
    df_selected_business_covid = df_selected_business_covid.drop(['Unnamed: 0', 'Temporary Closed Until', 'Covid Banner', 'Virtual Services Offered'], axis=1)

    print('Welcome to Toronto restaurant recommender!\n')
    delivery_ans = input('Do you want delivery/takeaway?[Y/N]')

    if delivery_ans == 'y' or delivery_ans == 'Y':
        delivery = True
    else:
        delivery = False

    if delivery:
        df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]
    
    # only keep businesses that have at least 20 ratings
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['review_count'] >= 20]

    df = df[df['business_id'].isin(df_selected_business_covid['business_id'])]

    df_delivery = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]
 
    # only for testing purposes
    '''
    print(df.loc[4322])
    df = df.drop(index=4322)
    print(df.loc[11400])
    df = df.drop(index=11400)
    #print(df.loc[11740])
    #df = df.drop(index=11740) 
    '''
    
    # reconstruct business id
    unique_business_id = df['business_id'].unique()
    business_shape = unique_business_id.shape
    print('\nThere are: %d businesses' % business_shape[0])

    business_df = pd.DataFrame({
            'business_id' :unique_business_id,
            'business_index': range(business_shape[0])
        })

    business_dict = pd.Series(business_df.business_id, index = business_df.business_index)


    # reconstruct user id
    unique_users_id = df['user_id'].unique()
    user_shape = unique_users_id.shape
    print('\nThere are: %d active users' % user_shape[0])

    user_df = pd.DataFrame({
            'user_id': unique_users_id,
            'user_index': range(user_shape[0])
        })

    user_dict = pd.Series(user_df.user_id, index = user_df.user_index)

    user_index = int(input('\nPlease enter a user index (number): '))
    user_id = user_dict[user_index]
    print('The corresponding user id is: %s' % user_id, '\n')

    # used for measuring accuracy of usage predictions
    test_user_id = 'jI43jHCXx1R-gg_vYcQWww'
    #user_id = test_user_id

    # content-based recommendations
    cbf = recommend_cbf(user_id, delivery)
    
    # collaborative filtering recommendations
    cf = recommend_cf(user_id, delivery, num_recs=len(business_dict))

    print('\n') 

    cbf_index_list = cbf.index.tolist()
    cf_index_list = cf['business_id'].tolist()

    cbf_ranking_dict = {}
    cf_ranking_dict = {}

     
    for i in range(len(cbf_index_list)):
        cbf_ranking_dict[cbf_index_list[i]] = i + 1
    
    
    for i in range(len(cf_index_list)):
        cf_ranking_dict[cf_index_list[i]] = i + 1


    num_items = len(cf_ranking_dict)

    score_dict = {}
    
    # compute hybrid score 
    for business_id in cf_ranking_dict:
        score = 1 + num_items - cf_ranking_dict[business_id]

        if business_id in cbf_ranking_dict:
            score = score * (1 + num_items - cbf_ranking_dict[business_id])
        
        score_dict[business_id] = score
    
    sorted_dict = dict(sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True))
 
    n_items = list(itertools.islice(sorted_dict.items(), 20))

    business_ids = []
    for pair in n_items:
        business_ids.append(pair[0])


    recommendations = pd.DataFrame()
    
    
    sum_log = 0

    for business_id in business_ids:
        recommendation = df_selected_business_covid[df_selected_business_covid['business_id'] == business_id]
        recommendations = recommendations.append(recommendation)

        #sum_log += math.log(recommendation['review_count'], 2) # used for testing novelty 
       
    # testing novelty
    '''
    novelty = -1 * sum_log / len(business_ids)
    print('Novelty: ', novelty, '\n')
    '''

    user_rated = df[df['user_id'] == user_id]
    user_rated = user_rated[['business_id', 'name', 'categories', 'text', 'stars']]
    user_rated = user_rated.drop_duplicates()

    places_rated = list(user_rated['name'])
    categories_rated = list(user_rated['categories'])
    text_rated = list(user_rated['text'])
    stars_rated = list(user_rated['stars'])

    print('User {0} has already rated: {1} place(s)'.format(user_index, user_rated.shape[0]))

    rated_table = Texttable()
    rated_table_text = Texttable()
    rated_table.add_rows([['Restaurant', 'Categories', 'Stars']])
    rated_table_text.add_rows([['Restaurant', 'Categories', 'Text']])

    rated_table_text.set_cols_align(["c", "c", "c"])
    rated_table_text.set_cols_width([15, 40, 120])

    for i in range(len(places_rated)):
        rated_table.add_row([places_rated[i], categories_rated[i], stars_rated[i]])
        rated_table_text.add_row([places_rated[i], categories_rated[i], text_rated[i]])

    print(rated_table.draw())

    recs1 = recommendations.iloc[:10, :]
    recs2 = recommendations.iloc[10:, :]
   

    print('\nRecommendations')
    rec_table = Texttable()
    rec_table.add_rows([['', 'Restaurant', 'Avg stars', 'Categories', 'Reviews', 'Delivery/Takeout']])

    rec_table.set_cols_align([ "c", "c", "c", "c", "c", "c"])
    rec_table.set_cols_width([5, 15, 9, 40, 7, 9])

    name_rec1 = list(recs1['name'])
    stars_rec1 = list(recs1['avg_stars'])
    categories_rec1 = list(recs1['categories'])
    reviews_rec1 = list(recs1['review_count'])
    attrib_rec1 = list(recs1['attributes'])
    delivery_rec1 = list(recs1['delivery or takeout'])

    name_rec2 = list(recs2['name'])
    stars_rec2 = list(recs2['avg_stars'])
    categories_rec2 = list(recs2['categories'])
    reviews_rec2 = list(recs2['review_count'])
    attrib_rec2 = list(recs2['attributes'])
    delivery_rec2 = list(recs2['delivery or takeout'])

    for i in range(len(name_rec1)):
        rec_table.add_row([i + 1, name_rec1[i], stars_rec1[i], categories_rec1[i], reviews_rec1[i], delivery_rec1[i]])

    print(rec_table.draw())

    more_rec = input('\nWould you like to see more recommendations?[Y/N]')
    
    if more_rec == 'y' or more_rec == 'Y':
        rec_table = Texttable()
        rec_table.add_rows([['', 'Restaurant', 'Avg stars', 'Categories', 'Reviews', 'Delivery/Takeout']])

        rec_table.set_cols_align([ "c", "c", "c", "c", "c", "c"])
        rec_table.set_cols_width([5, 15, 9, 40, 7, 9])

        for i in range(len(name_rec1)):
            rec_table.add_row([i + 11, name_rec2[i], stars_rec2[i], categories_rec2[i], reviews_rec2[i], delivery_rec2[i]])

        print(rec_table.draw())


    # testing diversity
    '''
    cosine_sim, inddict = compute_cosine(df_selected_business_covid)

    id1 = inddict[test_business_id1]
    id2 = inddict[test_business_id2]

    print(cosine_sim)
    print(business_dict)

    business_dict_reverse = {y:x for x, y in business_dict.items()}

    print('\n')
    ext_sum = 0
    int_sum = 0

    for i in business_ids:
        id1 = business_dict_reverse[i]

        for j in business_ids:
            id2 = business_dict_reverse[j]
            #if i != j:
            int_sum += cosine_sim[id1][id2]
        
        ext_sum += (1 - int_sum)
    
  
    n = len(business_ids)
    print(n)
    diversity = ext_sum / ((n / 2) * (n-1))

    print('Diversity', diversity)
    '''

    # testing precision, recall
    '''
    indexes = [4322, 11400] #, 11740]   
    business_ids = ['LBQD0H2109oltJNF1raLWA', 't7psd5uRy5NpadoONmTY7w'] #, 'uZPU3Vgo7q72EKYuT-bRTQ']
    names = ['The Haam', 'Tractor Foods'] #, 'LOV King West']

    rec_list = list(recommendations['business_id'])
    #print('\n\n', rec_list)
    print('Recommendations len:', len(rec_list))

    tp = 0
    for x in business_ids:
        if x in rec_list:
            print('yes')
            tp += 1
        else:
            print('no')
    print('TP:', tp)
    '''

 
def compute_cosine(df):
    places = df
    places = places.sort_values(by=['business_id'])
    #print('\n',places.shape)

    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')

    # build business name tfidf matrix 
    tfidf_matrix = vectorizer.fit_transform(places['name'])
    tfidf_feature_name = vectorizer.get_feature_names()
    #print(tfidf_matrix.shape)

    places['all_content'] = places['name'] + places['categories'] #+ places_10['city'] 
    tfidf_all_content = vectorizer.fit_transform(places['all_content'])
    #print(tfidf_all_content.shape)

    # computing cosine similarity matrix using linear_kernal of sklearn
    cosine_similarity_all_content = linear_kernel(tfidf_all_content, tfidf_all_content)
    #print('Cosine sim', cosine_similarity_all_content.shape)

    indices_n = pd.Series(places['business_id'])
    print(indices_n)
    inddict = indices_n.to_dict()
    inddict = dict((v,k) for k,v in inddict.items())
    #print(places['business_id'])#


    return cosine_similarity_all_content, inddict


if __name__ == "__main__":
    main()