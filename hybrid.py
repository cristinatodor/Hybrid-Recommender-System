
import os
import collections
import csv
import operator
import itertools

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds

#from content_based_v3 import recommend_cosine
#from content_based_v2_copy import recommend
#from collaborative_v2 import make_predictions, recommend_places

from content_based import recommend_cbf
from collaborative import recommend_cf

def main():
    '''
    df = pd.read_csv('./data/last_1_year_Restaurants_reviews.csv')
    df = df.drop(['useful', 'funny', 'cool', 'date'], axis = 1)
    df_selected_business = pd.read_csv('./data/selected_business_Restaurants_Toronto.csv')
    df_selected_business = df_selected_business[df_selected_business['business_id'].isin(df['business_id'])]
    '''

    df = pd.read_csv('./data/2019_3months_Restaurants_reviews.csv')
    df = df.drop(['cool', 'funny', 'useful', 'date', 'count', 'review_id'], axis = 1)

    #df_selected_business = pd.read_csv('./data/selected_business_Restaurants_Toronto.csv')
    #df_selected_business = df_selected_business[df_selected_business['business_id'].isin(df['business_id'])]

    # businesses that aren't temporary closed
    df_selected_business_covid = pd.read_csv('./data/selected_business_Restaurants_Toronto_covid.csv')
    df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['business_id'].isin(df['business_id'])]

    print('\n')
    df_selected_business_covid = df_selected_business_covid.drop(['Unnamed: 0', 'Temporary Closed Until', 'Covid Banner', 'Virtual Services Offered'], axis=1)

    delivery_ans = input('Do you want delivery/takeaway?[Y/N]')

    if delivery_ans == 'y' or delivery_ans == 'Y':
        delivery = True
    else:
        delivery = False

    if delivery:
        df_selected_business_covid = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]

    print('Selected business covid shape: ', df_selected_business_covid.shape)
    print(df_selected_business_covid.columns)

    df = df[df['business_id'].isin(df_selected_business_covid['business_id'])]

    df_delivery = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == True]
    df_no_delivery = df_selected_business_covid[df_selected_business_covid['delivery or takeout'] == False]
 
    # only for testing purposes
    '''
    print(df.loc[4322])
    df = df.drop(index=4322)
    print(df.loc[11400])
    df = df.drop(index=11400)
    '''

    #print(df.loc[11740])
    #df = df.drop(index=11740) 
    
    #print(df.loc[2783])
    #df = df.drop(index=2783)

    # reconstruct business id
    unique_business_id = df['business_id'].unique()
    business_shape = unique_business_id.shape
    print('There are: %d businesses' % business_shape[0])

    business_df = pd.DataFrame({
            'business_id' :unique_business_id,
            'business_index': range(business_shape[0])
        })

    business_dict = pd.Series(business_df.business_id, index = business_df.business_index)

    #business_index = int(input('Please enter a business index: '))
    #business_id = business_dict[business_index]
    #print('The corresponding business id is: %s' % business_id)

    print('\n')
    # reconstruct user id
    unique_users_id = df['user_id'].unique()
    user_shape = unique_users_id.shape
    print('There are: %d users' % user_shape[0])

    user_df = pd.DataFrame({
            'user_id': unique_users_id,
            'user_index': range(user_shape[0])
        })

    user_dict = pd.Series(user_df.user_id, index = user_df.user_index)

    user_index = int(input('Please enter a user index: '))
    user_id = user_dict[user_index]
    print('The corresponding user id is: %s' % user_id)

    '''
    test_user_id =  '--7gjElmOrthETJ8XqzMBw'
    test_user_id = '--BumyUHiO_7YsHurb9Hkw'
    test_user_id = 'ltvHnOfno4e5wUdmMDU72A'
    '''
    # used for measuring accuracy of usage predictions
    #test_user_id = 'jI43jHCXx1R-gg_vYcQWww'
    #user_id = test_user_id

    # content-based recommendations
    cbf = recommend_cbf(user_id, delivery)
    
    # collaborative filtering recommendations
    #recommend_places(preds_df, test_user_id1, df, cb, 25) this keeps all the places recommedend bf CBF, ordered by prediction 4
    cf = recommend_cf(user_id, delivery)

    print('\nHybrid')
    print('\nContent_based: \n', cbf)
    print('\n')
    print('Collaborative: \n', cf)

    #print(cb.index)
    
    num_items = 3655

    cbf_index_list = cbf.index.tolist()
    cf_index_list = cf['business_id'].tolist()
    #print('Cb index list: \n', cb_index_list)
    #print('Cf index list: \n', cf_index_list)

    cbf_ranking_dict = {}
    cf_ranking_dict = {}

     
    for i in range(len(cbf_index_list)):
        cbf_ranking_dict[cbf_index_list[i]] = i + 1
    
    print('Length of content-based dict: ', len(cbf_ranking_dict))
    
    for i in range(len(cf_index_list)):
        cf_ranking_dict[cf_index_list[i]] = i + 1

    print('Length of collaborative dict: ', len(cf_ranking_dict))

    score_dict = {}
    
    for business_id in cf_ranking_dict:
        score = 1 + num_items - cf_ranking_dict[business_id]
        #print(business_id, score, '\n')

        if business_id in cbf_ranking_dict:
            #print('yes', business_id)
            score = score * (1 + num_items - cbf_ranking_dict[business_id])
        
        score_dict[business_id] = score
    
    #print('Score dict:', score_dict)
    #sorted_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))
    sorted_dict = dict(sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True))
    print('Sorted dict len:', len(sorted_dict))
    #print(sorted_dict)

    #n_items = take(20, sorted_dict.iteritems())
    n_items = list(itertools.islice(sorted_dict.items(), 20))
    print(n_items)

    business_ids = []
    for pair in n_items:
        business_ids.append(pair[0])
    
    print('\nBusiness ids:', business_ids)

    print(df_selected_business_covid.shape)
    print('\nBusiness_details\n')

    recommendations = pd.DataFrame()

    i = 1
    for business_id in business_ids:
        recommendation = df_selected_business_covid[df_selected_business_covid['business_id'] == business_id]
        recommendations = recommendations.append(recommendation)
        '''
        business_name = recommendation['name']
        city = recommendation['city']
        avg_stars = recommendation['avg_stars']
        categories = recommendation['categories']
        review_count = recommendation['review_count']
        delivery = recommendation['delivery or takeout']

        print(i)
        print(business_name)
        print('City', city)
        print('Avg_stars', avg_stars)
        print('Categories', categories)
        print('Reviews:', review_count)
        print('Delivery or takeout', delivery)
        print('\n')
        i += 1
        '''

    print(recommendations)

    recommendations_show = recommendations.drop(['business_id'], axis=1)
    recommendations_show = recommendations_show[['name', 'city', 'avg_stars', 'categories', 'review_count', 'attributes', 'delivery or takeout']]
    print(recommendations_show)

    
    '''
    top_all = dict(itertools.islice(sorted_dict.items(), 60))
    all_business_ids = top_all.keys()
    all_recommendations = df_selected_business_covid[df_selected_business_covid['business_id'].isin(all_business_ids)]
    print(all_recommendations.shape)
    print('All rec top 20:\n', all_recommendations)
    
    #top20 = OrderedDict(itertools.islice(sorted_dict.iteritems(), 20))
    top20 = dict(itertools.islice(sorted_dict.items(), 20))
    print('\n', top20)

    print(df_selected_business_covid.shape)
    print('\nBusiness_details\n')

    business_ids = top20.keys()
    #details_df = df.drop_duplicates()
    recommendations = df_selected_business_covid[df_selected_business_covid['business_id'].isin(business_ids)]
    print(recommendations.columns)

    recommendations_show = recommendations.drop(['business_id'], axis=1)
    print(recommendations_show)
    '''
    
    '''
    for business_id in top500:
        print(df_selected_business_covid[df_selected_business_covid['business_id'] == business_id][['name', 'categories', 'avg_stars']])
        print('\n')
    '''
    user_rated = df[df['user_id'] == user_id]
    #user_rated = user_rated.drop(['review_count', 'review_id', 'text', 'count'], axis = 1)
    user_rated = user_rated[['business_id', 'name', 'categories', 'text']]
    user_rated = user_rated.drop_duplicates()

    print('User has already rated: {0} places'.format(user_rated.shape[0]))
    print(user_rated)

    '''
    indexes = [4322, 11400] #,11740]   
    business_ids = ['LBQD0H2109oltJNF1raLWA', 't7psd5uRy5NpadoONmTY7w']#, 'uZPU3Vgo7q72EKYuT-bRTQ']
    names = ['The Haam', 'Tractor Foods']#, 'LOV King West']

    rec_list = list(recommendations['business_id'])
    print('\n\n', rec_list)
    print(len(rec_list))

    tp = 0
    for x in business_ids:
        if x in rec_list:
            print('yes')
            tp += 1
        else:
            print('no')
    print('TP:', tp)
    '''
       


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

if __name__ == "__main__":
    main()