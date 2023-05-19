import pandas as pd

melbourne_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv') 
melbourne_data.describe() #podstawowe statystyki
melbourne_data.head() #pierwsze rekordy
melbourne_data.columns #lista nazw kolumn

y = melbourne_data.Price
X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(max_leaf_nodes=None, 
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        random_state=1)
melbourne_model.fit(train_X, train_y)
melbourne_model.predict(val_X)

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=100,
                                     bootstrap=True,   #czy wybiera podpróbki z X
                                     max_samples=None, #rozmiar podpróbek(None->wszystkie)
                                     n_jobs=None,      #-1->wszystkie rdzenie, None=1->1 rdzeń
                                     warm_start=False, #czy fit() dodaje drzewa a nie zamienia
                                                       #+ te same co w DecisionTreeRegressor
                                     random_state=1)
forest_model.fit(train_X, train_y)
forest_model.predict(val_X)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y, melbourne_model.predict(val_X))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
