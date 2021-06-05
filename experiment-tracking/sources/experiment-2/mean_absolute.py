
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mean_absolute(max_leaf_nodes, train_X, val_X, train_y, val_y):
    temp_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    temp_model.fit(train_X, train_y)
    preds_val = temp_model.predict(val_X)
    mean_absoulte = mean_absolute_error(val_y, preds_val)
    return mean_absoulte

