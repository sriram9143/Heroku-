# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

banks = pd.read_csv('bank-additional-full.csv',delimiter=";")
banks_clients = banks.iloc[:, :7]

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
banks_clients['job']=lb.fit_transform(banks_clients['job'])
banks_clients['marital']=lb.fit_transform(banks_clients['marital'])
banks_clients['education'] = lb.fit_transform(banks_clients['education'])
banks_clients['default']=lb.fit_transform(banks_clients['default'])
banks_clients['housing']=lb.fit_transform(banks_clients['housing'])
banks_clients['loan']=lb.fit_transform(banks_clients['loan'])

bank_campaign = banks.iloc[:,7:11]
look_up = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
bank_campaign['month'] = bank_campaign['month'].map(look_up)

#change day of a month into number 
day_num = {'mon':'01','tue':'02','wed':'03','thu':'04','fri':'05'}
bank_campaign['day_of_week'] = bank_campaign['day_of_week'].map(day_num)

bank_campaign['contact'] = bank_campaign['contact'].map({'telephone':'01','cellular':'02'})

bank_se = banks.iloc[:,11:-1]
bank_se['poutcome'] = lb.fit_transform(bank_se['poutcome'])


banks = pd.concat([banks_clients, bank_campaign,bank_se,banks['y'].map({'yes':1,'no':0})], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(banks.drop('y',axis=1))
scaled_feature = scaler.transform(banks.drop('y',axis=1))



X = pd.DataFrame(scaled_feature, columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed'])

y = banks['y']

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=4)


#Fitting model with trainig data
Knn.fit(X_train, y_train)

# Saving model to disk
pickle.dump(Knn, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[	-0.6740214166855213	,-1.0361837914214256, -0.2837414980846948,
1.054463930632832,	-0.5135996909374948	, 0.9421274250179283,	-0.45249061649750644,
	0.7585699681458711,	2.1519661045847998,	-0.6940017972496559, 0.9708376072195214	, 
    -0.5659219741930245, 0.1954139001271294	,1.6711360607672916	,-2.563097932616333,
	-0.7523425444073057	, 2.0581680488803893,	-2.2249534387270913	,-1.4911505545597972,
	-2.8156965971113914]]))