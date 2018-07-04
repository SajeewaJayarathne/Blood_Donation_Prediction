# coding: utf-8

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import preprocessing


# reading csv files to pandas data frames
train_data = pd.DataFrame(pd.read_csv('./input/train.csv'))
test_data = pd.DataFrame(pd.read_csv('./input/test.csv'))
data = [train_data, test_data]

for set in data:
    # checking for missing values
    m_values = set.isnull().sum()
    if m_values.sum()==0:
        print ('No missing values found in dataset.')
    else:
        for i in range(len(m_values)):
            if m_values.data[i] != 0:
                print ('Column {} has {} missing values'.format(m_values.index[i], m_values.data[i]))

# plotting the correlation matrix
corr = train_data.corr()
frame, heatmap = plt.subplots()
# adding correlation data
heatmap.imshow(corr, cmap='inferno')
# setting x,y axes ticks
heatmap.set_xticks(np.arange(train_data.columns.size))
heatmap.set_yticks(np.arange(train_data.columns.size))
# setting x.y axes labels
heatmap.set_xticklabels(train_data.columns)
heatmap.set_yticklabels(train_data.columns)
# rotating x axes labels by 45 degrees for convenience
plt.setp(heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# tight layout
frame.tight_layout
# plotting the heat map
# plt.show()

# dropping Total Volume Donated (c.c.) column as it shows the strongest linear relationship with Number of Donations
train_data.drop(columns=['Total Volume Donated (c.c.)'], inplace=True)
test_data.drop(columns=['Total Volume Donated (c.c.)'], inplace=True)

# adding 'Days since First Donation' and 'Days since Last Donation' features
train_data = train_data.rename({'Months since Last Donation':'Days since Last Donation'}, axis='columns')
train_data = train_data.rename({'Months since First Donation':'Days since First Donation'}, axis='columns')

train_data['Days since First Donation'] = train_data['Days since First Donation'].apply(lambda x: x*30)
train_data['Days since Last Donation'] = train_data['Days since Last Donation'].apply(lambda x: x*30)

test_data = test_data.rename({'Months since Last Donation':'Days since Last Donation'}, axis='columns')
test_data = test_data.rename({'Months since First Donation':'Days since First Donation'}, axis='columns')

test_data['Days since First Donation'] = test_data['Days since First Donation'].apply(lambda x: x*30)
test_data['Days since Last Donation'] = test_data['Days since Last Donation'].apply(lambda x: x*30)

# calculate the mean time period between two donations in days
mean_wait = np.mean((train_data['Days since First Donation'] - train_data['Days since Last Donation'])/train_data['Number of Donations'])

# add donation eligibility. True if Days since last donation > 60, False otherwise.
eligibility_vector_train = pd.DataFrame(np.array(train_data['Days since First Donation'] > 60).T.astype(int),columns=['Donation Eligiblity'])
eligibility_vector_test = pd.DataFrame(np.array(test_data['Days since First Donation'] > 60).T.astype(int),columns=['Donation Eligiblity'])

# add donation likelihood
likelihood_vector_train = pd.DataFrame(np.array(((train_data['Days since First Donation'] - train_data['Days since Last Donation'])/train_data['Number of Donations']) > mean_wait).T.astype(int), columns=['Donation Likelihood'])
likelihood_vector_test = pd.DataFrame(np.array(((test_data['Days since First Donation'] - test_data['Days since Last Donation'])/test_data['Number of Donations']) > mean_wait).T.astype(int), columns=['Donation Likelihood'])

# add new vectors to the original dataset
train_data = pd.concat([pd.concat([train_data, eligibility_vector_train], axis=1), likelihood_vector_train], axis=1)
test_data = pd.concat([pd.concat([test_data, eligibility_vector_test], axis=1), likelihood_vector_test], axis=1)

# normalizing the columns
standard_scaler = preprocessing.StandardScaler()
train_data.iloc[:,1:4] = standard_scaler.fit_transform(train_data.iloc[:,1:4])
test_data.iloc[:,1:4] = standard_scaler.fit_transform(test_data.iloc[:,1:4])

# write processed dataframes to files
train_data.to_csv('processed_input/train.csv', index=False)
test_data.to_csv('processed_input/test.csv', index=False)