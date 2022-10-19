
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#required libraries
import os
from logging import exception
#import numpy as np

from flask import Flask, jsonify
app = Flask(__name__)

#import plotly.express as px
#import numpy as np
import pandas as pd
#import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#import seaborn as sns
#import operator as op
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
#from flask import render_template
for dirname, _, filenames in os.walk('test_Master_data.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#!pip install apriori

app = Flask(__name__)

data = pd.read_csv('test_Master_data.csv')
# Converting Date column into correct datatype which is datetime
data.columns = ['memberID', 'Date', 'itemName','orderID','coupon','total_cost','user_name']
data.Date = pd.to_datetime(data.Date)
data.memberID = data['memberID'].astype('str')

Recency = data.groupby(['memberID'])['Date'].max().reset_index()
Recency.columns = ['memberID', 'LastDate']
# Finding last date for our dataset
last_date_dataset = Recency['LastDate'].max()
# Calculating Recency by subtracting (last transaction date of dataset) and (last purchase date of each customer)
Recency['Recency'] = Recency['LastDate'].apply(lambda x: (last_date_dataset - x).days)
Frequency = data.drop_duplicates(['Date', 'memberID']).groupby(['memberID'])['Date'].count().reset_index()
# Frequency of the customer visits

Frequency.columns = ['memberID', 'Visit_Frequency']
Monetary = data.groupby(["memberID"])['total_cost'].sum().reset_index()
Monetary.columns = ['memberID','Monetary']
# Combining all scores into one DataFrame
RFM = pd.concat([Recency['memberID'], Recency['Recency'], Frequency['Visit_Frequency'], Monetary['Monetary']],
                axis=1)

# 5-5 score = the best customers
bin_labels_5 = ['1', '2', '3', '4', '5']
bin_labels = ['5,','4', '3', '2']
RFM['Recency_quartile'] = pd.qcut(RFM['Recency'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=bin_labels_5)

#RFM['Recency_quartile'] = pd.qcut(RFM['Recency'],
                                  #q= [0, .2, .4, .6, .8, 1],labels = bin_labels_5)
#RFM['Frequency_quartile'] = pd.qcut(RFM['Visit_Frequency'],
                           #   q=[1, .75, .5, .25 , 0 ],
                            #  labels=bin_labels_5, duplicates = 'drop' )
RFM['Frequency_quartile'] = pd.qcut(RFM['Visit_Frequency'], 5,[4,3,2,1], duplicates = 'drop')

RFM['RF_Score'] = RFM['Recency_quartile'].astype(str) + RFM['Frequency_quartile'].astype(str)
segt_map = {  # Segmentation Map [Ref]
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

RFM['RF_Segment'] = RFM['RF_Score'].replace(segt_map, regex=True)

#needed for combining with transaction data
new_data= RFM.drop(['Recency','Visit_Frequency','Monetary','Recency_quartile','Frequency_quartile','RF_Score'],axis =1)
segmented_data = pd.merge(data,
                      new_data,
                      on ='memberID',
                      how ='inner')
Discount_list = segmented_data.groupby(['memberID']).coupon.agg([('count', 'count'), ('coupon', ', '.join)])
Discount_list.reset_index()
#read the transaction data and do merging with segments
#new_data= RFM.drop(['Recency','Visit_Frequency','Monetary','Recency_quartile','Frequency_quartile','RF_Score'],axis =1)
merged_data = pd.read_csv('merged_data.csv')
merged_data.Member_number = merged_data['Member_number'].astype('str')
#merged_data.Date = data['Date'].astype('Date')
merged_data.Date = pd.to_datetime(merged_data.Date)
merged_data.item_brand = merged_data['item_brand'].astype('str')
merged_data.category = merged_data['category'].astype('str')
merged_data.rename(columns = {'Member_number':'memberID'}, inplace = True)

#dictionary of coupons
dictionary_segments = {
    'hibernating':'WELCOMEBACK',
    'at risk':'20xSuperpoints',
    'can\'t loose': '20%discount',
    'about to sleep' :'3for2',
    'need attention': 'BUY1GET1',
    'loyal customers':'Referal50',
    'promising' : 'mysteryDeal',
    'new customers':'FREE_DELIVERY',
    'potential loyalists':'gift_coupon',
    'champions':'10OFF'
}
#segmets with custome data
cust_seg_history = pd.merge(merged_data,new_data, on = 'memberID', how = 'outer')
customer_data = pd.read_csv('customer_data.csv')

@app.route('/')
def welcome():
    return jsonify("welcome to the page")

@app.route("/api/top_users", methods = ['GET'])
def api_top_users():
    try:
        user_item = data.groupby(pd.Grouper(key='memberID')).size().reset_index(name='count') \
            .sort_values(by='count', ascending=False)
        top_25 = user_item.memberID.count() / 4
        top_25 = round(top_25)
        user_items = user_item.reset_index()
        user_items = user_items.iloc[:top_25]
        result = user_items.to_json(orient='index')
        return result
    except exception as e:
        return jsonify(e)

@app.route('/api/segments', methods = ['GET'])
def api_segments():
    try:
        x = RFM.RF_Segment.value_counts()
        result = x.to_json(orient='index')
        return result
    except exception as e:
        return "error occured"

@app.route("/api/at-risk", methods=['GET'])
def api_at_risk():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'at risk']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('at risk')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['atrisk_data'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/champions", methods = ['GET'])
def api_champions():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'champions']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    #merge other data : discount list customer info with grouped prediction
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('champions')
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['champoins'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/hibernating", methods = ['GET'])
def api_hibernating():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'hibernating']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('hibernating')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['hibernating'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/cant-loose", methods = ['GET'])
def api_cant_lose():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'can\'t loose']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('can\'t loose')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['can\'t loose'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/about-to-sleep", methods = ['GET'])
def api_about_to_sleep():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'about to sleep']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('about to sleep')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['about_to_sleep'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/need-attention", methods = ['GET'])
def api_need_attention():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'need attention']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('need attention')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['need_attention'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/loyal-customers", methods = ['GET'])
def api_loyal_customers():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'loyal customers']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    suggested_coupon = dictionary_segments.get('loyal customers')

    values = dict()
    values['loyal_customers'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/promising", methods = ['GET'])
def api_promising():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'promising']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    suggested_coupon = dictionary_segments.get('promising')

    values = dict()
    values['promising'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/new-customers", methods = ['GET'])
def api_new_customers():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'new customers']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('new customers')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['new_customers'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values

@app.route("/api/potential-loyalists", methods = ['GET'])
def api_potential_loyalists():
    # seperate dataset with segments
    atrisk_data = cust_seg_history[cust_seg_history['RF_Segment'] == 'potential loyalists']
    most_revenue = atrisk_data.groupby(['memberID', 'category'], sort=True)['price'].sum().reset_index()
    most_revenue = most_revenue.sort_values(['memberID', 'price'], ascending=False)
    most_revenue = most_revenue.drop_duplicates(['memberID'], keep='first')
    most_revenue = most_revenue.sort_values(['memberID'], ascending=False)
    test = most_revenue.reset_index()
    test.drop(['index'], axis=1)
    # test.to_csv('C:\\Users\\nageshs\\Documents\\Projects\\GroceryData\\test.csv')
    suggested_coupon = dictionary_segments.get('potential loyalists')
    customer_data.rename(columns={'Member_number': 'memberID'}, inplace=True)
    customer_data.memberID = customer_data['memberID'].astype('str')
    customer_segment_data = pd.merge(test, customer_data, on='memberID', how='left')
    Suggested_coupon_category = pd.merge(customer_segment_data, Discount_list, on='memberID', how='left')
    Suggested_coupon_category = Suggested_coupon_category.drop(['price', 'count'], axis=1)
    Suggested_coupon_category = Suggested_coupon_category.to_json(orient='index')
    values = dict()
    values['potential_loyalists'] = Suggested_coupon_category
    values['suggested_coupon'] = suggested_coupon
    return values


port = os.getenv('PORT',8080)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=port)
