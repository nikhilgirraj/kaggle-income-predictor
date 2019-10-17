import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn import preprocessing
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel

path_to_trainingcsv = r'res\tcd ml 2019-20 income prediction training (with labels).csv'
path_to_UnLabelledcsv = r'res\tcd ml 2019-20 income prediction test (without labels).csv'

print("Reading the UnLabelled file.")
dfUnLabelled = pd.read_csv(path_to_UnLabelledcsv, sep=',', keep_default_na=False, na_values=['#N/A'])
dataraw = pd.read_csv(path_to_trainingcsv, sep=',', keep_default_na=False, na_values=['#N/A'])

YY = None
DD = None
YScaler = None
featSel=None
t1, t2, t3, t4, t5 = None, None, None, None, None

dataraw.isnull().any()


def S2(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:2]
    else:
        return s1

# Pipeline to process the datasets - Training - Test and Prediction
def ProcessRawData(df, schemaCols=None):

    medianSimpleImputer = SimpleImputer(strategy='median')
    standardScaler = preprocessing.StandardScaler()

    # Adding extra features AgeLog and HeightLog
    df['AgeLog'] = np.log(df['Age'].values)
    df['HeightLog'] = np.log(df['Body Height [cm]'].values)

    # Fill missing values
    df[['Year of Record', 'Age', 'AgeLog', 'HeightLog']] = medianSimpleImputer.fit_transform(df[['Year of Record', 'Age', 'AgeLog', 'HeightLog']].values)

    # Scale numeric columns 1
    df[['Year of Record', 'Size of City', 'Body Height [cm]', 'Age', 'AgeLog']] = standardScaler.fit_transform(df[['Year of Record', 'Size of City', 'Body Height [cm]', 'Age', 'AgeLog']].values)

    # Scale numeric columns 2
    if 'Income in EUR' in df.columns:
        global YScaler
        YScaler = preprocessing.StandardScaler()
        df[['Income in EUR']] = YScaler.fit_transform(df[['Income in EUR']].values)

    # Reducing complexity of features
    df.Profession = list(df.Profession.map(S2))

    # To be used while writing results to CSV
    instances = df['Instance'].values
    df = df.drop(['Instance'], axis=1)

    print('Columns available 1 - ', df.columns)

    # Target encoding the data - could've been done with a single encoder object, will try later,
    if (schemaCols is None): # condition to skip fitting on Prediction dataset and only transform then
        global t1, t2, t3, t4, t5
        t1 = TargetEncoder()
        t2 = TargetEncoder()
        t3 = TargetEncoder()
        t4 = TargetEncoder()
        t5 = TargetEncoder()
        t1.fit(df.Country.values, df['Income in EUR'].values)
        t2.fit(df.Profession.values, df['Income in EUR'].values)
        t3.fit(df.Gender.values, df['Income in EUR'].values)
        t4.fit(df['University Degree'].values, df['Income in EUR'].values)
        t5.fit(df['Hair Color'].values, df['Income in EUR'].values)

    df.Country = t1.transform(df.Country.values)
    df.Profession = t2.transform(df.Profession.values)
    df.Gender = t3.transform(df.Gender.values)
    df['University Degree'] = t4.transform(df['University Degree'].values)
    df['Hair Color'] = t5.transform(df['Hair Color'].values)

    if (schemaCols is not None):
        newdf = pd.DataFrame()
        for columnName in schemaCols:
            if columnName not in df.columns:
                newdf[columnName] = 0
            else:
                newdf[columnName] = df[columnName].values
        df = newdf

    df = df.sort_index(axis=1)

    # standardize datasets prediction and training to use the same code from there on
    if 'Income in EUR' not in df.columns:
        df['Income in EUR'] = np.zeros(df.values.shape[0])

    if 'Income' in df.columns:
        df.drop('Income')

    X = df.drop('Income in EUR', axis=1).values
    Y = df['Income in EUR'].values

    print('Shape - ', df.shape)

    global featSel
    if featSel is None:
        print('k = ? ')
        featSel = SelectKBest(f_regression, k=10)
        featSel.fit(X, Y)

    X = featSel.transform(X)
    print('Shape after feature selection - ', X.shape)
    return instances, X, Y, df.columns

# Preprocess data
instanceId, X, Y, finalColumns = ProcessRawData(dataraw.copy(deep=True))

# Split labelled dataset into training and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

# Train
regressor = GradientBoostingRegressor(n_estimators=500)
fitResult = regressor.fit(Xtrain, Ytrain)

# Calculate cost
YPredTest = regressor.predict(Xtest)
rmsError = np.sqrt(metrics.mean_squared_error(YScaler.inverse_transform(Ytest), YScaler.inverse_transform(YPredTest)))
print("RMS Error on Training-Test data: ", rmsError)

# Predict unlabelled dataset values
print('Processing unlabelled data..')
instanceIdUnLabelled, XUnLabelled, y_whatever, col_whatever = ProcessRawData(dfUnLabelled.copy(deep=True), schemaCols=finalColumns)
print('Generating predictions...')
YUnLabelled = regressor.predict(XUnLabelled)
YScaler.inverse_transform(YUnLabelled, copy=False)
outfilename = datetime.now().strftime('%Y%m%d%H%M%S') + " - [" + str(rmsError) + "].csv"
print('Writing CSV output - ', outfilename)
pd.DataFrame({'Instance': instanceIdUnLabelled, 'Income': YUnLabelled}).to_csv(outfilename, index=False)

print('Done.')