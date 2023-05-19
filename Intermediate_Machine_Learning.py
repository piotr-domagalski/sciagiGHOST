cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

#drop - usuń kolumny z NaN'ami
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

#impute - zmień NaN'y na szacunkową wartość, np. minimum, średnią
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

#impute + flag - dodatkowo kolumna informująca, czy wartość była NaN
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

### 3. Categorical Variables ###
#drop
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

#ordinal encoding
#zastępuje kolumny kategoryczne int'ami, przypisując każdemu unikalnemu stringowi inta.
#np. dla zmiennej przyjmującej 3 wartości ['Never', 'Sometimes', 'Always']: 
#'Never'     -> 0, 
#'Sometimes' -> 1, 
#'Always'    -> 2
from sklearn.preprocessing import OrdinalEncoder
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

#onehot encoding
#kolumny kategoryczne z n unikalnymi wartościami zastępuje n kolumnami z wartościami 1 lub 0 (flagi, czy konkretna wartość wystąpiła)
#np. kolumna 'Colour' z wartościami ['Red', 'Yellow', 'Green'] zastąpiona 3 kolumnami 'Red', 'Yellow' i 'Green'
# 'Red'    -> 1, 0, 0
# 'Yellow' -> 0, 1, 0
# 'Green'  -> 0, 0, 1
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

#zwykle onehot > ordinal > drop
#onehot niepraktyczne przy dużej liczbie unikalnych wartości

### 4. Pipelines ###
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

### 5. Cross-Validation ###
#lepsza alternatywa train_test_split'a, szczególnie przy małym zbiorze danych. Dane podzielone na n podzbiorów (tzw. foldów) i każdy użyty osobno jako walidacyjny.
from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5, #liczba foldów
                              scoring='neg_mean_absolute_error') #konwencja większy wynik = lepszy, ale w kursach używane samo mae (czyli mniejszy = lepszy)

#hyperparameter optimization
from sklearn.model_selection import GridSearchCV
#TODO

### 6. XGBoost ###
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                        early_stopping_rounds=5)
my_model.fit(X_train, y_train, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
n_estimators = my_model.best_iteration+1 #zaczyna się od 0;
                                         #po znalezieniu fitować jeszcze raz, bez early_stopping_rounds

### 7. Data Leakage ###
#tylko teoria: data leakage i train test contamination
