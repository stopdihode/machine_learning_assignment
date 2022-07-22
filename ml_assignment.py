# import all necessary packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

## import dataframe
# df_test = pd.read_json('C:/Users/Malik Rigot/Desktop/Tilburg/Master/Machine learning/Challenge/test.json')
# df_train = pd.read_json('C:/Users/Malik Rigot/Desktop/Tilburg/Master/Machine learning/Challenge/train-1.json')

# function for calculating H-Index
def H_index(citations):
      
    # sorting in ascending order
    citations.sort()
      
    # iterating over the list
    for i, cited in enumerate(citations):
          
        # finding current result
        result = len(citations) - i
          
        # if result is less than or equal
        # to cited then return result
        if result <= cited:
            return result
           
    return 0

# https://www.geeksforgeeks.org/what-is-h-index/

# Split dataset based on authors
df_train = df_train.explode('authors').reset_index()

### Features which are going to be used
# unique citations each author (necessary for the H_index)
author_set = df_train.sort_values('citations').groupby('authors').agg( {'citations': ['unique']})
author_set.columns = ['unique_citation']

#H_index
h_index = pd.DataFrame(author_set['unique_citation'].map(lambda x: H_index(x)) )
h_index.columns = ['H_index']

# mean, sum and max citations of venue
df_venue = df_train.groupby('venue').agg({'citations': ['mean', 'sum', 'max']})
df_venue.columns = ['venue_mean', "venue_sum", 'venue_max']

# mean, sum and max citations authors
df_authors = df_train.groupby('authors').agg({'citations': ['mean', 'sum', 'max']})
df_authors.columns = ['author_mean', "author_sum", 'author_max']

# length of title
df_train['length_title'] = df_train['title'].str.split().str.len()

# Merge features with train data
df_train = pd.merge(df_train, author_set, on="authors", how='inner')
df_train = pd.merge(df_train, df_venue, on='venue', how='outer')
df_train = pd.merge(df_train, df_authors, on='authors', how='outer')
df_train = pd.merge(df_train, h_index, on='authors', how='left')

# remove unnecesary features and split into X and Y dataframes
X = df_train.drop(['doi', 'title', 'abstract', 'topics', 'is_open_access', 'fields_of_study', 'unique_citation', 'authors', 'venue', "index", 'year'], axis=1)
Y = np.log1p(X.pop('citations').values)

#split dataframe into train test set
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = .20, random_state = 42)

#scale test and validation sets
scaler_x = StandardScaler()
X_train = pd.DataFrame(scaler_x.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(scaler_x.transform(X_val.values), columns=X_val.columns, index=X_val.index)

Scaler_Y = StandardScaler()
Y_train = Scaler_Y.fit_transform(Y_train.reshape(-1, 1))
Y_val = Scaler_Y.transform(Y_val.reshape(-1, 1))

# function for gridsearch
from sklearn.model_selection import GridSearchCV

def gridsearch(trainset, citations):
    model = SVR()
    param = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}
    grid = GridSearchCV(model, param_grid=param, cv=5, scoring='r2', verbose=1, return_train_score=True)
    grid.fit(trainset, citations)
    return grid.best_estimator_

# https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Support%20Vector%20Regression.ipynb

#fit model with parameters found using gridsearch
svr_best=SVR(kernel='rbf', C=5, epsilon=0.2)
svr_best.fit(X_train, Y_train)

# preprocess testing data
df_test = df_test.explode("authors").reset_index()
df_test['length_title'] = df_test['title'].str.split().str.len()
df_test = df_test.drop(['title', 'abstract', 'topics', 'is_open_access', 'fields_of_study', 'index'], axis=1)

#merge test set with relevant features of training set
df_merged = pd.merge(df_test, author_set, on="authors", how='left')
df_merged = pd.merge(df_merged, df_venue, on='venue', how='left')
df_merged = pd.merge(df_merged, df_authors, on='authors', how='left')
df_merged = pd.merge(df_merged, h_index, on='authors', how='left')

doi = df_merged.pop('doi')
df_merged = df_merged.drop(['venue', 'authors', 'unique_citation', 'year'], axis=1)

# fillna for missing features venues
for stat in ['venue_mean', 'venue_sum', 'venue_max']:
    df_merged[stat].fillna(value= df_merged[stat].median(), inplace=True)

# reorder columns and impute missing data
cols = list(X_train.columns.values)
df_merged = df_merged.reindex(columns = cols)

imputer = KNNImputer(n_neighbors=7)
df_merged = pd.DataFrame(scaler_x.transform(df_merged), columns=df_merged.columns)
df_merged = pd.DataFrame(imputer.fit_transform(df_merged),columns = df_merged.columns)
# https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e

#Pred test set
pred = svr_best.predict(df_merged)
pred = pred.reshape(-1, 1)

#rescale test prediction
pred_rescaled = np.expm1(Scaler_Y.inverse_transform(pred))
df_test['citations'] = pred_rescaled

#group prediction to each doi
output = df_test.groupby('doi', as_index=False)['citations'].mean()
# https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas


#open json file and write predicted citations
import json
json_list = json.loads(json.dumps(list(output.T.to_dict().values())))

# https://stackoverflow.com/questions/39257147/convert-pandas-dataframe-to-json-format?answertab=active#tab-top

with open('predicted.json', 'w') as outfile:
    json.dump(json_list, outfile)