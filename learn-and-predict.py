import os
from datetime import datetime
from os import system
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, IsolationForest, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel

os.chdir(r"C:\Work\Projects\TCD Assignments\ML-JB-Kaggle-1")
path_to_trainingcsv = 'tcd ml 2019-20 income prediction training (with labels).csv'
path_to_challengecsv = r"tcdml1920-income-ind\tcd ml 2019-20 income prediction test (without labels).csv"

print("Reading the challenge file.")
dfChallenge = pd.read_csv(path_to_challengecsv, sep=',', keep_default_na=False, na_values=['#N/A'])
dataraw = pd.read_csv(path_to_trainingcsv, sep=',', keep_default_na=False, na_values=['#N/A'])

listOfCountries = np.union1d(np.union1d(dfChallenge['Country'].values, dataraw['Country'].values), ['missing'])
# ["Afghanistan", "Aland Islands", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "CÃ´te dâ€™Ivoire", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Czechia", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "DR Congo", "East Timor", "East Timor (Timor-Leste)", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestinian Territory", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome & Principe", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "State of Palestine", "Sudan", "Suriname", "Swaziland", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "The Bahamas", "The Gambia", "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "USA", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"]
listOfProfessions = np.union1d(np.union1d(dfChallenge['Profession'].values.astype(str), dataraw['Profession'].values.astype(str)), ['missing'])
listOfHairColor = np.union1d(np.union1d(dataraw['Hair Color'].astype('str').values, dfChallenge['Hair Color'].astype('str').values), ['missing'])

labelEncoderCountry = preprocessing.LabelEncoder()
labelEncoderCountry.fit(listOfCountries)

labelEncoderProf = preprocessing.LabelEncoder()
labelEncoderProf.fit(listOfProfessions)

labelEncoderGender = preprocessing.LabelEncoder()
labelEncoderGender.fit(np.union1d(dataraw['Gender'].astype('str').values, ['missing']))

labelEncoderDegree = preprocessing.LabelEncoder()
labelEncoderDegree.fit(np.union1d(dataraw['University Degree'].astype('str').values, ['missing']))

labelEncoderHairColor = preprocessing.LabelEncoder()
labelEncoderHairColor.fit(listOfHairColor)

YY = None
DD = None
YScaler = None
featSel=None
t1, t2, t3, t4, t5 = None, None, None, None, None

dataraw.isnull().any()

def I5 (y1):
    if (y1 is not np.nan):
        return (y1/5)*5
    else:
        return y1

def I10 (y1):
    if (y1 is not np.nan):
        return (y1/10)*10
    else:
        return y1


def S2(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:2]
    else:
        return s1


def ProcessRawData(df, schemaCols=None):

    medianSimpleImputer = SimpleImputer(strategy='median')
    minMaxScaler = preprocessing.StandardScaler()

    # if 'Income in EUR' in df.columns:
    #     df = df.drop(index=df.loc[df['Income in EUR'] < 0].index)

    #medianSimpleImputer.fit_transform(dataraw['Year of Record'].values.reshape(-1,1))
    df['AgeLog'] = np.log(df['Age'].values)
    df['HeightLog'] = np.log(df['Body Height [cm]'].values)
    df['Age'] = np.power(df['Age'].values, 1) #np.log(df['Age'].values)
    df['Body Height [cm]'] = np.power(df['Body Height [cm]'].values, 1) #np.log(df['Body Height [cm]'].values)

    # FUZZY 1
    # df['Size of City'] = list(df['Size of City'].map(I10))
    # df['Body Height [cm]'] = list(df['Body Height [cm]'].map(I5))
    df[['Year of Record', 'Age', 'AgeLog', 'HeightLog']] = medianSimpleImputer.fit_transform(df[['Year of Record', 'Age', 'AgeLog', 'HeightLog']].values)
    df[['Year of Record', 'Size of City', 'Body Height [cm]', 'Age', 'AgeLog']] = minMaxScaler.fit_transform(df[['Year of Record', 'Size of City', 'Body Height [cm]', 'Age', 'AgeLog']].values)
    if 'Income in EUR' in df.columns:
        global YScaler
        YScaler = preprocessing.StandardScaler()
        df[['Income in EUR']] = YScaler.fit_transform(df[['Income in EUR']].values)

    # FUZZY 2
    df.Profession = list(df.Profession.map(S2))
    # assuming normally distributed Age data, 6 sigma outlier removing
    # if 'Income in EUR' in df.columns:
    #     print('Doing the Six Sigma thing -')
    #     print('Before - ', len(df))
    #     mean = np.average(df['Age'].values)
    #     sd = np.std(df['Age'].values)
    #     df = df.drop(index = df.loc[df['Age'] > (mean + 3 * sd)].index)
    #     df = df.drop(index = df.loc[df['Age'] < (mean - 3 * sd)].index)
    #     print('After - ', len(df))

    #df['Age'] = np.divide(1, df['Age'].values)  # np.log(df['Age'].values)

    # df['Gender'] = df['Gender'].fillna('missing')
    # df['Hair Color'] = df['Hair Color'].fillna('missing')
    # df['University Degree'] = df['University Degree'].fillna('missing')
    # df['Profession'] = df['Profession'].fillna('missing')
    # df['Country'] = df['Size of City'].astype(str) + df.Country
    #dataraw['Age'] = np.divide(1, dataraw['Age'].values)
    instances = df['Instance'].values
    df = df.drop(['Instance'], axis=1)
    print('Columns - ', df.columns)
    dataNonCateg = df #pd.get_dummies(df, prefix_sep='_')

    if (schemaCols is None):
        global t1, t2, t3, t4, t5
        t1 = TargetEncoder()
        t2 = TargetEncoder()
        t3 = TargetEncoder()
        t4 = TargetEncoder()
        t5 = TargetEncoder()
        t1.fit(dataNonCateg.Country.values, dataNonCateg['Income in EUR'].values)
        t2.fit(dataNonCateg.Profession.values, dataNonCateg['Income in EUR'].values)
        t3.fit(dataNonCateg.Gender.values, dataNonCateg['Income in EUR'].values)
        t4.fit(dataNonCateg['University Degree'].values, dataNonCateg['Income in EUR'].values)
        t5.fit(dataNonCateg['Hair Color'].values, dataNonCateg['Income in EUR'].values)

    dataNonCateg.Country = t1.transform(dataNonCateg.Country.values)
    dataNonCateg.Profession = t2.transform(dataNonCateg.Profession.values)
    dataNonCateg.Gender = t3.transform(dataNonCateg.Gender.values)
    dataNonCateg['University Degree'] = t4.transform(dataNonCateg['University Degree'].values)
    dataNonCateg['Hair Color'] = t5.transform(dataNonCateg['Hair Color'].values)

    # for country in listOfCountries:
    #     tmpstr = "Country_" + country
    #     if (tmpstr not in dataNonCateg.columns):
    #         dataNonCateg[tmpstr] = 0
    #
    # for prof in listOfProfessions:
    #     tmpstr = "Profession_" + prof
    #     if (tmpstr not in dataNonCateg.columns):
    #         dataNonCateg[tmpstr] = 0

    if (schemaCols is not None):
        newdf = pd.DataFrame()
        for columnName in schemaCols:
            if columnName not in dataNonCateg.columns:
                newdf[columnName] = 0
            else:
                newdf[columnName] = dataNonCateg[columnName].values
        dataNonCateg = newdf

    dataNonCateg = dataNonCateg.sort_index(axis=1)

    del df
    colsToDrop = [col for col in dataNonCateg.columns if 'missing' in col]
    dataNonCategNonMissing = dataNonCateg.drop(colsToDrop, axis=1)

    if 'Income in EUR' not in dataNonCategNonMissing.columns:
        dataNonCategNonMissing['Income in EUR'] = np.zeros(dataNonCategNonMissing.values.shape[0])

    if 'Income' in dataNonCategNonMissing.columns:
        dataNonCategNonMissing.drop('Income')

    # global XX, YY
    # YY = dataNonCategNonMissing
    # print('FFF')
    # print(dataNonCategNonMissing.shape)
    # # preprocessing.Normalizer(copy=False).fit_transform(dataNonCategNonMissing)
    # print(dataNonCategNonMissing.shape)
    # if (schemaCols is None):
    #     dataNonCategNonMissing = dataNonCategNonMissing[(np.abs(stats.zscore(dataNonCategNonMissing)) < 3).all(axis=1)]
    # print(dataNonCategNonMissing.shape)
    # print(type(dataNonCategNonMissing))
    #
    # isf = IsolationForest(behaviour='new', max_samples=50, random_state=1, contamination='auto')
    # outliersPred = isf.fit_predict(dataNonCategNonMissing.drop('Income in EUR', axis=1).values)
    # outliers = [i for i, o in enumerate(outliersPred) if o == -1]
    #
    # print('Found outliers - ', outliers)
    # dataNonCategNonMissing = dataNonCategNonMissing.drop(index=outliers)
    # print(dataNonCategNonMissing.shape)

    X = dataNonCategNonMissing.drop('Income in EUR', axis=1).values
    Y = dataNonCategNonMissing['Income in EUR'].values

    print('Shape - ', dataNonCategNonMissing.shape)
    global featSel
    if featSel is None:
        print('k = ? ')
        featSel = SelectKBest(f_regression, k=10)
        featSel.fit(X, Y)
        # featSel = ExtraTreesRegressor(n_estimators=10)
        # featSel = featSel.fit(X, Y)

        # from sklearn.svm import SVC
        # from sklearn.feature_selection import RFECV
        #
        # # Create the RFE object and rank each pixel
        # svc = SVC(kernel="linear", C=1)
        # featSel = RFECV(estimator=svc, step=1)
        # featSel.fit(X, Y)
    # if schemaCols is None:
    X = featSel.transform(X)
    # X = SelectFromModel(featSel, preFit=True).transform(X)
    print('Shape new - ', X.shape)
    #
    return instances, X, Y, dataNonCategNonMissing.columns

instanceId, X, Y, finalColumns = ProcessRawData(dataraw.copy(deep=True))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

regressor = LinearRegression(normalize=True, n_jobs=4)
# regressor = BayesianRidge(n_iter=100, alpha_1=1e-06, alpha_2=1e-06)
# regressor = RandomForestRegressor(n_estimators=500)
# regressor = GradientBoostingRegressor(n_estimators=500)
# regressor = XGBRegressor()
fitResult = regressor.fit(Xtrain, Ytrain)
#coeff_df = pd.DataFrame(regressor.coef_, dataNonCategNonMissing.columns.drop('Income in EUR'), columns=['Thetas'])

YPredTest = regressor.predict(Xtest)
#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
rmsError = np.sqrt(metrics.mean_squared_error(YScaler.inverse_transform(Ytest), YScaler.inverse_transform(YPredTest)))
print("RMS Error on Training-Test data: ", rmsError)


"""
print('Processing unlabelled data..')
instanceIdChallenge, XChallenge, y_whatever, col_whatever = ProcessRawData(dfChallenge.copy(deep=True), schemaCols=finalColumns)
print('Generating predictions...')
YChallenge = regressor.predict(XChallenge)
# YScaler.inverse_transform(YChallenge, copy=False)
outfilename = datetime.now().strftime('%Y%m%d%H%M%S') + " - [" + str(rmsError) + "].csv"
print('Writing CSV output - ', outfilename)
pd.DataFrame({'Instance': instanceIdChallenge, 'Income': YChallenge}).to_csv(outfilename, index=False)
#pd.DataFrame({'Instance': instanceIdChallenge, 'Income': YChallenge}).to_csv(outfilename, index=False)
"""