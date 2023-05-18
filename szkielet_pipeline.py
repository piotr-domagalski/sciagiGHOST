import pandas as pd
from sklearn.model_selection import train_test_split

#wczytywanie
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

y = data.Price
X = data.drop(['Price'], axis=1)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                        X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

#przetwarzanie
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #pozostałe w sklearn.impute: IterativeImputer, MissingImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder

numerical_transformer = SimpleImputer(strategy='constant', #lub: mean, median, most_frequent
                                      fill_value=None,     #jeśli strategy='constant', użyj tej wartości (defaultowo None, czyli 0)
                                      add_indicator=False  #jeśli True, dodaje kolumnę wskazującą czy był NaN
                                      )
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols), 
        ('cat', categorical_transformer, categorical_cols)
    ])

#model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

#uczenie
from sklearn.metrics import mean_absolute_error
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
