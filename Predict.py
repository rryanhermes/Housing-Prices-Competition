import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from Functions import *

target_variable = 'SalePrice'
feature_columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'Median Real Estate Price', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

data1 = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# Data2 is a dataset I put together aggregating real estate prices from a credible real estate website.
data2 = pd.read_csv('Median Real Estate Prices.csv')

data = pd.merge(data1, data2[['Neighborhood', 'Median Real Estate Price']], on='Neighborhood', how='left')
test_data = pd.merge(test_data, data2[['Neighborhood', 'Median Real Estate Price']], on='Neighborhood', how='left')
data = pd.get_dummies(data)
test_data = pd.get_dummies(test_data)

data = data.interpolate(method='polynomial', order=2)
test_data = test_data.interpolate(method='polynomial', order=2)

feature_columns = [col for col in data.columns for partial_match in feature_columns if partial_match in col]

y, x = data[target_variable], data[feature_columns]
train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.2)
model = RandomForestRegressor(random_state=1, max_leaf_nodes=10000)
model.fit(train_x, train_y)
predictions = model.predict(validation_x)
rmse, smape = evaluate(validation_y, predictions)
print(f"Average error of ${round(rmse, 2)} ({round(smape, 2)}%)")

# create_chart(validation_y, predictions, rmse, smape, index=validation_y.index)

missing_columns = [col for col in train_x.columns if col not in test_data.columns]
for col in missing_columns: test_data[col] = 0

test_predictions = model.predict(test_data[feature_columns])

submission_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})
submission_df.to_csv('submission.csv', index=False)