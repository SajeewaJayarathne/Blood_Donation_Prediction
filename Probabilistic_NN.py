import numpy as np
import pandas as pd
from neupy.algorithms import PNN

# read data from the csv files
train_data = pd.DataFrame(pd.read_csv('processed_input/train.csv'))
test_data = pd.DataFrame(pd.read_csv('processed_input/test.csv'))

# print train_data.describe

# split data in to target and data
X_train = train_data.drop(columns='Made Donation in March 2007', axis=1)
y_train = train_data['Made Donation in March 2007']
X_test = test_data.drop(columns='Made Donation in March 2007', axis=1)
IDs = test_data['ID']

# instantiate the PNN model
pnn = PNN(verbose=True)

# fit to training data and then predict
prediction = pnn.fit(X_train, y_train).predict_proba(X_test)

# concatenate IDs and the prediciton
pred = pd.concat([IDs, pd.DataFrame(prediction.astype(float), columns=['Made Donation in March 2007'])], axis=1)

#write prediction to csv
pred.to_csv('output/pnn_prediction.csv', index=False)