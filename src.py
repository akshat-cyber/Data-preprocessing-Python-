# main
import numpy
import pandas

# gets the missing data impute
from sklearn.impute import SimpleImputer

# encoding the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# splitting the data into -> train and test
from sklearn.model_selection import train_test_split

# fetch the data
Input = 0
getData = pandas.read_csv('Data.csv')
x = getData.iloc[:, :-1].values
y = getData.iloc[:, -1].values

# gets the missing data
imputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")

# binding
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])  # returns the updated value
# encoding the data {Column}

# columnTransformer transforms a particular selected column
columnTransformer = ColumnTransformer(transformers=[  # expects a list of operations
    ('encoder', OneHotEncoder(), [0])  # encoding operation applied
],
    remainder='passthrough')

# bind it with the main matrix , returns a numpy array
x = numpy.array(columnTransformer.fit_transform(x))

lc = LabelEncoder()  # for Yes No
y = lc.fit_transform(y)

# splitting the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# feature scaling using standardisation = {x - mean(x)}/{StandardDeviation(x)}

from sklearn.preprocessing import StandardScaler

standardScale = StandardScaler()
t