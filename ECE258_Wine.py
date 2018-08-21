from io import StringIO
import requests
import json
import pandas as pd

winedata = pd.read_csv('winemag-data_first150k.csv')
country_list = ['US','Italy','France','Spain','Chile','Argentina','Portugal','Australia','New Zealand','Germany','South Africa']
sub_data1 = winedata[winedata['country'].isin(country_list)]

#Lets build a model to predict the top 5 wine varieties in US reviews.
varietylist = ['Pinot Noir','Cabernet Sauvignon','Chardonnay','Syrah','Red Blend','Bordeaux-style Red Blend','Sauvignon Blanc','Merlot']
subdata = sub_data1[sub_data1['variety'].isin(varietylist)]

#encoding the labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(subdata['variety'])
label_encoded_y = label_encoder.transform(subdata['variety'])
subdata['encoded_winevariety'] = label_encoded_y
subdata.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    min_df=5, max_features=200, strip_accents='unicode',lowercase =True,
    analyzer='word', token_pattern=r'\w+', use_idf=True,
    smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(subdata["description"])

features = tfidf.get_feature_names()
print(features)

X_tfidf_text = tfidf.transform(subdata["description"])
subdata_2 = pd.DataFrame(X_tfidf_text.toarray())
subdata = subdata.reset_index()
subdata_2['encoded_winevariety'] = subdata['encoded_winevariety']
#Also adding variety for better readibility
subdata_2['variety'] = subdata['variety']

from sklearn.cross_validation import train_test_split
seed = 7

#Split into train and test
test_size = 0.2
y = subdata_2['encoded_winevariety']
X = subdata_2.drop(['encoded_winevariety','variety'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data
import xgboost as xgb
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

clf.fit(X_train, y_train)

#Measuring accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))