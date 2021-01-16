# source code
# https://beckernick.github.io/matrix-factorization-recommender/

import os
import collections
import csv

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

#from content_based_v3 import recommend_cosine

def prep_data(df, delivery):
    '''
    df = df.rename(columns={"review_count": "business_review_count"})

    # users with at least 10 reviews
    df_users = pd.read_csv('./data/users_at_least_10.csv')
    df_users = df_users.drop(['Unnamed: 0', 'name'], axis=1)

    df = pd.merge(df, df_users, how='left', on='user_id')
    #df = df[['user_id', 'review_count', 'business_id']]
    df = df.dropna()
    print(df.head(10))
    print(df.shape)
    '''

    # businesses that aren't temporary closed
    df_selected_business_covid = pd.read_csv('./data/selected_business_Restaurants_Toronto_covid.csv')
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['business_id'].isin(df['business_id'])]

    print('\n')
    df_selected_business_covid = df_selected_business_covid.drop(['Unnamed: 0', 'Temporary Closed Until', 'Covid Banner', 'Virtual Services Offered'], axis=1)

    if delivery:
        df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]
    print('Selected business covid shape: ', df_selected_business_covid.shape)
    print(df_selected_business_covid.columns)

    df = df[df['business_id'].isin(df_selected_business_covid['business_id'])]
    print('Df shape:', df.shape)

    # only for testing purposes
    '''
    print(df.loc[4322])
    df = df.drop(index=4322)
    print(df.loc[11400])
    df = df.drop(index=11400)
    '''
    
    #print(df.loc[11740])
    #df = df.drop(index=11740) 

    print('Reviews: \n')
    review_df = df[['business_id', 'user_id', 'stars']]
    print(review_df.sample(10))
    print(review_df.shape)

    # reconstruct business id
    unique_business_id = review_df['business_id'].unique()
    business_shape = unique_business_id.shape
    print('Number of Unique Business ID: %d' % business_shape[0])

    # reconstruct user id
    unique_users_id = review_df['user_id'].unique()
    user_shape = unique_users_id.shape
    print('Number of Unique User ID: %d' % user_shape[0])

    return df, df_selected_business_covid, review_df


def make_predictions(review_df):
    R_df = review_df.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 
    print('Shape:', R_df.shape)

    # convert to numpy array and de-mean (normalise by each user's mean)
    R = R_df.to_numpy()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    # appply svd
    U, sigma, Vt = svds(R_demeaned, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, index = R_df.index, columns = R_df.columns)
    print(preds_df.head(10))
    print(preds_df.shape)

    return preds_df


def recommend_places(predictions_df, user_id, original_df, num_recommendations):
    # get the places already rated by the user
    user_rated = original_df[original_df['user_id'] == user_id]
    user_rated = user_rated.drop(['review_count', 'review_id', 'text', 'count'], axis = 1)
    user_rated = user_rated.drop_duplicates()

    print('User has already rated: {0} places'.format(user_rated.shape[0]))
    print(user_rated)

    # sort predictions
    sorted_user_predictions = predictions_df.loc[user_id].sort_values(ascending=False)

    recommendations = pd.DataFrame(sorted_user_predictions).rename(columns = {user_id: 'predictions'})

    # get info from the original dataframe
    recommendations_merge = pd.merge(recommendations, original_df, how = 'right', on = 'business_id').sort_values('predictions', ascending = False)
    #recommendations_merge = recommendations_merge.drop(['city', 'review_id', 'user_id', 'stars', 'text', 'count', 'review_count'], axis=1)
    recommendations_merge = recommendations_merge.drop(['city', 'review_count'], axis=1)
    recommendations_merge = recommendations_merge.drop_duplicates(subset='business_id')
    recommendations_merge = recommendations_merge[['business_id', 'predictions', 'name', 'avg_stars', 'attributes', 'categories']]
    print('Recommendations merge')
    print(recommendations_merge)
    
    #print(recommendations_merge.shape)

    # only keep places not already rated by the user 
    recommendations_filtered = recommendations_merge[~recommendations_merge['business_id'].isin(user_rated['business_id'])]
    test_filtered = recommendations_merge[recommendations_merge['business_id'].isin(user_rated['business_id'])]
    print('Test shape: ', test_filtered.shape)
    print(test_filtered)
    print('Rec shape: ', recommendations_filtered.shape)

    return recommendations_filtered.head(num_recommendations)

#recommend_places(preds_df, test_user_id1, df, df, 20)
'''
test_user_id1 = 'TqSZTCaIE7Rmnia_4X4iIA'
test_user_id2 = 'TUMlJiM6Aw-xogVcF876qw'
test_user_id3 = 'ltvHnOfno4e5wUdmMDU72A'

test_business_id1 = 'K5Q2vkF5UpytV9Q1rB-5Yg'

cb = recommend_cosine(test_business_id1)
cb.drop(['all_content'], axis = 1)
print(cb)
print(cb.columns)

preds_df = make_predictions()
#recommend_places(preds_df, test_user_id1, df, cb, 25) this keeps all the places recommedend bf CBF, ordered by prediction 
cbf = recommend_places(preds_df, test_user_id1, df, cb, 10)
print(cbf)
'''

# source
#https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65
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
    print("svd done")
    
    preds_df = pd.DataFrame(UsV, index = R_df.index, columns = R_df.columns)

    return preds_df


def testing(R_df, test):
    k = 10

    preds_df = compute_svd(R_df, k)
    print('Pred shape:', preds_df.shape)
    print(preds_df)

    business_ids = preds_df.columns
    users_ids = preds_df.index
  
    pred = []
    count_zero = 0

    for _, row in test.iterrows():
        user = row['user_id']
        item = row['business_id']

        #prediction = preds_df.loc[(preds_df['user_id'] == user) & (preds_df['business_id'] == item)]

        if user in users_ids:
            predictions = preds_df.loc[user]

            if item in business_ids:
                prediction = predictions.loc[item]
                pred_rating = prediction
            else:
                pred_rating = predictions.mean()
            
        else:
            pred_rating = 0
            count_zero += 1
        
        pred.append(pred_rating)
    
    print('Count_zero:', count_zero)
    
    return pred
    

def rmse(true, pred):
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)


def recommend_cf(user_id, delivery):
    '''
    df_original = pd.read_csv('./data/last_1_year_Restaurants_reviews.csv')
    df_original = df_original.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original.shape)
    '''

    df_original_3 = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df_original_3 = df_original_3.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original_3.shape)
    
    df, df_selected_business_covid, review_df = prep_data(df_original_3, delivery)

    '''
    preds_df = make_predictions(review_df)
    print('Preds', preds_df)

    cf = recommend_places(preds_df, test_user_id1, df, df, 10)

    return cf
    '''
    # create utility matrix
    R_df = df.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 
    print(R_df.shape)

    preds_df = compute_svd(R_df, k=12)
    print('Pred shape:', preds_df.shape)
    print(preds_df)

    cf = recommend_places(preds_df, user_id, df, 3655)

    return cf

test_user_id1 = '--UOvCH5qEgdNQ8lzR8QYQ'
test_user_id1 = 'jI43jHCXx1R-gg_vYcQWww'
recommend_cf(test_user_id1, delivery=True)

def test():
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
    
    test_user_id = '--UOvCH5qEgdNQ8lzR8QYQ'
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
    df_loc = df_loc.drop(index=11400)  # t7psd5uRy5NpadoONmTY7w    
    df_loc = df_loc.drop(index=4322)   # LBQD0H2109oltJNF1raLWA               
    print('Loc df \n', df_loc)
    
    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[11740])
    df_original_3 = df_original_3.drop(index=11740)
    print('Shape:', df_original_3.shape)

    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[11400])
    df_original_3 = df_original_3.drop(index=11400)
    print('Shape:', df_original_3.shape)

    print('Shape:', df_original_3.shape)
    print(df_original_3.loc[4322])
    df_original_3 = df_original_3.drop(index=4322)
    print('Shape:', df_original_3.shape)
    #df_original = df_original.drop(index=1160)
    #df_original = df_original.drop(index=38571)
    #df, df_selected_business_covid, review_df = prep_data(df_original_3)
    #print(df.shape)
    print('\nTesting\n')
    print('Shape:', df.shape)
    print(df.loc[11740])
    print(df.loc[11400])
    print(df.loc[4322])
    true_ratings.append(df.loc[11740]['stars'])
    true_ratings.append(df.loc[11400]['stars'])
    true_ratings.append(df.loc[4322]['stars'])
    df = df.drop(index=11740)
    df = df.drop(index=11400)
    df = df.drop(index=4322)
    print('Shape:', df.shape)

    # create utility matrix
    R_df = df.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 
    print(R_df.shape)

    preds_df = compute_svd(R_df, k=10)
    print('Pred shape:', preds_df.shape)
    print(preds_df)

    cf = recommend_places(preds_df, test_user_id, df, 3655)

    print(cf[cf['business_id'] == 'uZPU3Vgo7q72EKYuT-bRTQ'])
    print(cf[cf['business_id'] == 't7psd5uRy5NpadoONmTY7w'])
    print(cf[cf['business_id'] == 'LBQD0H2109oltJNF1raLWA'])
    pred_ratings.append(cf[cf['business_id'] == 'uZPU3Vgo7q72EKYuT-bRTQ']['predictions'])
    pred_ratings.append(cf[cf['business_id'] == 't7psd5uRy5NpadoONmTY7w']['predictions'])
    pred_ratings.append(cf[cf['business_id'] == 'LBQD0H2109oltJNF1raLWA']['predictions'])
    
    mae = mean_absolute_error(true_ratings, pred_ratings)
    print('\nMAE:', mae)
    #print(cf)

#test()


def main2():
    df_original = pd.read_csv('./data/last_1_year_Restaurants_reviews.csv')
    df_original = df_original.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    print('Data shape: ', df_original.shape)
    
    df, df_selected_business_covid, review_df = prep_data(df_original)

    print(df.shape)

    train, test = train_test_split(review_df, test_size=0.2)
    print('Train:', train.shape)
    print('Test:', test.shape)

    #review_df_train = train[['business_id', 'user_id', 'stars']]
    #print(review_df_train.shape)

    # create utility matrix
    R_df = train.pivot_table(index = 'user_id', columns ='business_id', values = 'stars').fillna(0) 
    print(R_df.shape)

    #test = test.drop(['city', 'avg_stars', 'review_count', 'attributes', 'review_id', 'text', 'count'], axis=1)
    print(test)
    print('Unique users: ', len(test['user_id'].unique()))

    print('Stars:', len(test['stars']))
    
    pred = testing(R_df, test)
    print('Pred: ', len(pred))

    print('RMSE, k=10:', rmse(test['stars'], pred))
  