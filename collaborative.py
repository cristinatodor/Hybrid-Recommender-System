# code based on 
# https://beckernick.github.io/matrix-factorization-recommender/ 
# https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65

import os
import collections
import csv

import math 
import numpy as np
import pandas as pd

from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def prep_data(df, delivery):
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

    review_df = df[['business_id', 'user_id', 'stars']]

    # reconstruct business id
    unique_business_id = review_df['business_id'].unique()
    business_shape = unique_business_id.shape

    # reconstruct user id
    unique_users_id = review_df['user_id'].unique()
    user_shape = unique_users_id.shape

    return df, df_selected_business_covid, review_df


def recommend_places(predictions_df, user_id, original_df, num_recommendations):
    # get the places already rated by the user
    user_rated = original_df[original_df['user_id'] == user_id]
    user_rated = user_rated.drop(['review_count', 'review_id', 'text', 'count'], axis = 1)
    user_rated = user_rated.drop_duplicates()

    # sort predictions
    sorted_user_predictions = predictions_df.loc[user_id].sort_values(ascending=False)

    recommendations = pd.DataFrame(sorted_user_predictions).rename(columns = {user_id: 'predictions'})

    # get info from the original dataframe
    recommendations_merge = pd.merge(recommendations, original_df, how = 'right', on = 'business_id').sort_values('predictions', ascending = False)
    recommendations_merge = recommendations_merge.drop(['city', 'review_count'], axis=1)
    recommendations_merge = recommendations_merge.drop_duplicates(subset='business_id')
    recommendations_merge = recommendations_merge[['business_id', 'predictions', 'name', 'avg_stars', 'attributes', 'categories']]
    
    # only keep places not already rated by the user 
    recommendations_filtered = recommendations_merge[~recommendations_merge['business_id'].isin(user_rated['business_id'])]
    test_filtered = recommendations_merge[recommendations_merge['business_id'].isin(user_rated['business_id'])]

    return recommendations_filtered.head(num_recommendations)


def compute_svd(R_df, k):
    # compute svd    
    util_matrix = np.array(R_df)

    # the nan or unavailable entries are masked
    mask = np.isnan(util_matrix)
    masked_arr = np.ma.masked_array(util_matrix, mask)
    item_means = np.mean(masked_arr, axis=0)

    # replace nan ratings by the average rating for each item
    utilMat = masked_arr.filled(item_means)

    x = np.tile(item_means, (util_matrix.shape[0],1))

    # remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x

    U, s, V = np.linalg.svd(util_matrix, full_matrices=False)
    s = np.diag(s)

    # only take the k most significant features
    s = s[0:k, 0:k]
    U = U[:, 0:k]
    V = V[0:k, :]

    s_root = sqrtm(s)

    Usk = np.dot(U, s_root)
    skV = np.dot(s_root, V)
    UsV = np.dot(Usk, skV)
    UsV = UsV + x
    
    preds_df = pd.DataFrame(UsV, index = R_df.index, columns = R_df.columns)

    return preds_df


def recommend_cf(user_id, delivery, num_recs):
    df_original_3 = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df_original_3 = df_original_3.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    
    df, df_selected_business_covid, review_df = prep_data(df_original_3, delivery)

    # create utility matrix
    R_df = df.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 

    # compute svd
    preds_df = compute_svd(R_df, k=12)

    # make predictions
    cf = recommend_places(preds_df, user_id, df, num_recs)

    return cf


test_user_id1 = 'jI43jHCXx1R-gg_vYcQWww'


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

    indices_n = pd.Series(places['business_id'])
    print(indices_n)
    inddict = indices_n.to_dict()
    inddict = dict((v,k) for k,v in inddict.items())
    #print(places['business_id'])#

    return cosine_similarity_all_content, inddict


def baseline_test():
    df_original_3 = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df_original_3 = df_original_3.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original_3.shape)
    
    df, df_selected_business_covid, review_df = prep_data(df_original_3, delivery=False)
    
    test_user_id = 'jI43jHCXx1R-gg_vYcQWww'  

    '''
    print(df.loc[4322])
    df = df.drop(index=4322)
    print(df.loc[11400])
    df = df.drop(index=11400)    
    print(df.loc[11740])
    df = df.drop(index=11740) 
    '''

    recs = recommend_cf(test_user_id, delivery=False, num_recs=40)
    print('\nRecommendations\n', recs)

    rec_list = list(recs['business_id'])
    print('Recommendations', len(rec_list))
    print(rec_list)

    # testing precision, recall
    '''
    business_ids = ['LBQD0H2109oltJNF1raLWA', 't7psd5uRy5NpadoONmTY7w']#, 'uZPU3Vgo7q72EKYuT-bRTQ']

    tp = 0
    for x in business_ids:
        if x in rec_list:
            print('yes')
            tp += 1
        else:
            print('no')
    print('TP:', tp)
    '''
    
    # testing novelty 
    '''
    sum_log = 0
    for business_id in rec_list:
        recommendation = df_selected_business_covid[df_selected_business_covid['business_id'] == business_id]

        sum_log += math.log(recommendation['review_count'], 2)
    
    novelty = -1 * sum_log / len(rec_list)
    print('Novelty: ', novelty, '\n')
    '''

    # reconstruct business id
    unique_business_id = df['business_id'].unique()
    business_shape = unique_business_id.shape
    print('There are: %d businesses' % business_shape[0])

    business_df = pd.DataFrame({
            'business_id' :unique_business_id,
            'business_index': range(business_shape[0])
        })

    business_dict = pd.Series(business_df.business_id, index = business_df.business_index)

    cosine_sim, inddict = compute_cosine(df_selected_business_covid)

    print(cosine_sim)
    print(business_dict)

    business_dict_reverse = {y:x for x, y in business_dict.items()}

    print('\n')
    ext_sum = 0
    int_sum = 0

    for i in rec_list:
        id1 = business_dict_reverse[i]

        for j in rec_list:
            id2 = business_dict_reverse[j]
            #if i != j:
            int_sum += cosine_sim[id1][id2]
        
        ext_sum += (1 - int_sum)
    
  
    n = len(rec_list)
    print(n)
    diversity = ext_sum / ((n / 2) * (n-1))

    print('Diversity', diversity)

#baseline_test()

def test_MAE():
    '''
    df_original_2 = pd.read_csv('./data/2019_6months_Restaurants_reviews.csv')
    df_original_2 = df_original_2.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original_2.shape)
    df, df_selected_business_covid, review_df = prep_data(df_original_2)
    '''

    df_original_3 = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df_original_3 = df_original_3.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original_3.shape)
    
    df, df_selected_business_covid, review_df = prep_data(df_original_3, delivery=False)
    
    #test_user_id = '--UOvCH5qEgdNQ8lzR8QYQ'
    test_user_id = 'jI43jHCXx1R-gg_vYcQWww'                
    
    df_test = df_original_3[df_original_3['user_id'] == test_user_id]
    print('\nTest df \n', df_test)

    '''
    test_user_id = 'if0ciRlu2sO0PIyYM0JRkg'
    df_loc = df_test.drop(index=3255) # business id = 'G24p1oGGfY3t-m8Z2lPCaQ'
    df_loc = df_loc.drop(index=12589 ) # business id = 'zFR99jgMi-qzaJXIx8MXHA'
    '''

    true_ratings = []
    pred_ratings = []
 
    df_loc = df_test.drop(index=11740) # uZPU3Vgo7q72EKYuT-bRTQ      
    #df_loc = df_loc.drop(index=11400)  # t7psd5uRy5NpadoONmTY7w    
    #df_loc = df_loc.drop(index=4322)   # LBQD0H2109oltJNF1raLWA               
    print('Loc df \n', df_loc)
    
    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[11740])
    df_original_3 = df_original_3.drop(index=11740)
    print('Shape:', df_original_3.shape)

    '''
    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[11400])
    df_original_3 = df_original_3.drop(index=11400)
    print('Shape:', df_original_3.shape)

    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[4322])
    df_original_3 = df_original_3.drop(index=4322)
    print('Shape:', df_original_3.shape)
    '''

    #df_original = df_original.drop(index=1160)
    #df_original = df_original.drop(index=38571)
    #df, df_selected_business_covid, review_df = prep_data(df_original_3)
    #print(df.shape)
    print('\nTesting\n')
    print('Shape:', df.shape)
    print(df.loc[11740])
    #print(df.loc[11400])
    #print(df.loc[4322])
    true_ratings.append(df.loc[11740]['stars'])
    #true_ratings.append(df.loc[11400]['stars'])
    #true_ratings.append(df.loc[4322]['stars'])
    df = df.drop(index=11740)
    #df = df.drop(index=11400)
    #df = df.drop(index=4322)
    print('Shape:', df.shape)

    # create utility matrix
    R_df = df.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 
    print(R_df.shape)

    preds_df = compute_svd(R_df, k=10)
    print('Pred shape:', preds_df.shape)
    print(preds_df)

    cf = recommend_places(preds_df, test_user_id, df, 3655)

    print(cf[cf['business_id'] == 'uZPU3Vgo7q72EKYuT-bRTQ'])
    #print(cf[cf['business_id'] == 't7psd5uRy5NpadoONmTY7w'])
    #print(cf[cf['business_id'] == 'LBQD0H2109oltJNF1raLWA'])
    pred_ratings.append(cf[cf['business_id'] == 'uZPU3Vgo7q72EKYuT-bRTQ']['predictions'])
    #pred_ratings.append(cf[cf['business_id'] == 't7psd5uRy5NpadoONmTY7w']['predictions'])
    #pred_ratings.append(cf[cf['business_id'] == 'LBQD0H2109oltJNF1raLWA']['predictions'])
    
    mae = mean_absolute_error(true_ratings, pred_ratings)
    print('\nMAE:', mae)


