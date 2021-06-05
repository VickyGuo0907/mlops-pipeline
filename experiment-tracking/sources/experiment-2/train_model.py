import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from mean_absolute import get_mean_absolute

# set local mlflow path
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("home-data")


# Path of the file to read
iowa_file_path = 'train.csv'

# read csv file
df_home_data = pd.read_csv(iowa_file_path)

# define features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Create X
X = df_home_data[features]

# Create target object and call it y
y = df_home_data.SalePrice

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from
scores = {}
# Store the best value of max_leaf_nodes ( it will be either 5, 25, 50, 100, 250 or 500)

for leaf_size in candidate_max_leaf_nodes:
    scores[leaf_size] = get_mean_absolute(leaf_size, train_X, val_X, train_y, val_y)
#     scores.append()
# scores = {leaf_size: get_mean_absolute(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print("Best Tree Size: ", best_tree_size)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes= best_tree_size, random_state=1)

final_model.fit(X, y)


# define and update in mlflow
metrics = {"train_score": final_model.score(train_X, train_y),
"validation_score": final_model.score(val_X, val_y)}
print(metrics)

# log params to mlflow and artifacts to minio
mlflow.log_params({"random_state": 1})
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(final_model, "decision_tree_regress")
