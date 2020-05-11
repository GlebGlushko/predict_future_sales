import pandas as pd
import numpy as np
import gc
import lightgbm
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor, plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
pd.set_option("display.width", 250)
pd.set_option('display.max_columns', 75)

sales = pd.read_csv('./data/sales_train.csv')
shops = pd.read_csv('./data/shop_town.csv')
items = pd.read_csv('./data/items.csv')
item_cats = pd.read_csv('./data/item_categories.csv')
test = pd.read_csv('./data/test.csv').set_index('ID')
submissions = pd.read_csv('./data/sample_submission.csv')

# sales = sales[sales['shop_id'].isin([25, 26])]
# test = test[test['shop_id'].isin([25, 26])]

print(test.shape)
test['date_block_num'] = 34

# remove outliers
sales = sales[sales['item_price'] < 50000]
sales = sales[sales['item_cnt_day'] < 500]

# calculate revenue for each item-day
sales['revenue'] = sales['item_price'] * sales['item_cnt_day']


# print(test.shape[0])
index_cols = ['shop_id', 'item_id', 'date_block_num']

# gb = sales.groupby(by=index_cols, as_index=False)['item_cnt_day'].sum()
# gb = gb.rename(columns={'item_cnt_day': 'target'})
# sales = pd.merge(sales, gb, how='left', on=index_cols).fillna(0)
# # sales['target'].fillna(0).clip(0, 20)


sales = pd.concat([sales, test], ignore_index=True, sort=False, keys=index_cols)
sales.fillna(0, inplace=True)

print('concatenated test + train')

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items], [block_num])), dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns=index_cols)


print('created grid')

# group by data to get shop-item-month aggregates
gb = sales.groupby(by=index_cols, as_index=False)['item_cnt_day'].sum()
gb = gb.rename(columns={'item_cnt_day': 'target'})
gb['target'] = gb['target'].clip(0, 20)
# gb = gb.clip(0, 20)
data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

del grid
# group by data to get shop-month aggregates
gb = sales.groupby(by=['shop_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()
gb = gb.rename(columns={'item_cnt_day': 'target_shop'})
data = pd.merge(data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# group by data to get item-month aggregates
gb = sales.groupby(by=['item_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()
gb = gb.rename(columns={'item_cnt_day': 'target_item'})
data = pd.merge(data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# group by data to get shop-item-month revenue
gb = sales.groupby(by=index_cols, as_index=False)['revenue'].sum()
gb = gb.rename(columns={'revenue': 'revenue_shop_item'})
data = pd.merge(data, gb, how='left', on=index_cols).fillna(0)

# group by data to get shop-month-revenue
gb = sales.groupby(by=['shop_id', 'date_block_num'], as_index=False)['revenue'].sum()
gb = gb.rename(columns={'revenue': 'revenue_shop'})
data = pd.merge(data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# group by data to get item-month-revenue
gb = sales.groupby(by=['item_id', 'date_block_num'], as_index=False)['revenue'].sum()
gb = gb.rename(columns={'revenue': 'revenue_id'})
data = pd.merge(data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

del gb

print('created features')
cols_to_rename = list(data.columns.difference(index_cols))
shift_range = [1, 2, 3, 4, 6, 9, 12]

for month_shift in shift_range:
    train_shift = data[index_cols+cols_to_rename].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    train_shift = train_shift.rename(columns=(lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x))
    data = pd.merge(data, train_shift, on=index_cols, how='left').fillna(0)

data['month'] = data['date_block_num'] % 12 + 1
sales['month'] = sales['date_block_num'] % 12 + 1
days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
data['days'] = data['month'].map(days)
# data['year'] = data['date_block_num'] // 12 + 2013

# sales = pd.merge(sales, shops[['shop_id', 'town_id']], on=['shop_id'], how='left')
# data = pd.merge(data, shops[['shop_id', 'town_id']], on=['shop_id'], how='left')

sales = pd.merge(sales, items[['item_category_id', 'item_id']], on=['item_id'], how='left')
data = pd.merge(data, items[['item_category_id', 'item_id']], on=['item_id'], how='left')
# 2.09 2.019
# print(sales.describe())

# gb = sales.groupby(by=['item_category_id', 'shop_id'], as_index=False)['revenue'].mean()
# gb = gb.rename(columns={'revenue': 'shop_category_revenue_mean'})
# data = pd.merge(data, gb, how='left', on=['item_category_id', 'shop_id']).fillna(0)
#
#
# gb = sales.groupby(by=['item_category_id', 'shop_id'], as_index=False)['revenue'].sum()
# gb = gb.rename(columns={'revenue': 'shop_category_revenue_sum'})
# data = pd.merge(data, gb, how='left', on=['item_category_id', 'shop_id']).fillna(0)

gb = sales.groupby(by=['shop_id', 'month'], as_index=False)['revenue'].sum()
gb = gb.rename(columns={'revenue': 'shop_month_revenue'})
data = pd.merge(data, gb, how='left', on=['shop_id', 'month']).fillna(0)
#
#
# gb = sales.groupby(by=['item_id', 'month'], as_index=False)['revenue'].sum()
# gb = gb.rename(columns={'revenue': 'item_month_revenue'})
# data = pd.merge(data, gb, how='left', on=['item_id', 'month']).fillna(0)
#
# gb = sales.groupby(by=['shop_id', 'item_id', 'month'], as_index=False)['revenue'].sum()
# gb = gb.rename(columns={'revenue': 'shop_item_month_revenue'})
# data = pd.merge(data, gb, how='left', on=['shop_id', 'item_id', 'month']).fillna(0)
#
# # gb = sales.groupby(by=['shop_id', 'month'], as_index=False)['revenue'].mean()
# # gb = gb.rename(columns={'revenue': 'shop_month_mean'})
# # data = pd.merge(data, gb, how='left', on=['shop_id', 'month']).fillna(0)
#
# gb = sales.groupby(by=['shop_id', 'item_id'], as_index=False)['date_block_num'].min()
# gb = gb.rename(columns={'date_block_num': 'first_sale_date_block'})
# data = pd.merge(data, gb, how='left', on=['shop_id', 'item_id']).fillna(34)
#
gb = sales.groupby(by=['item_id'], as_index=False)['date_block_num'].min()
gb = gb.rename(columns={'date_block_num': 'first_item_sale_date_block'})
data = pd.merge(data, gb, how='left', on=['item_id']).fillna(34)
#
# gb = sales.groupby(by=['item_id'], as_index=False)['date_block_num'].max()
# gb = gb.rename(columns={'date_block_num': 'last_item_sale_date_block'})
# data = pd.merge(data, gb, how='left', on=['item_id']).fillna(0)
# exit()



# gb = sales.groupby(by=['town_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()
# gb = gb.rename(columns={'item_cnt_day': 'items_cnt_date_block_by_town'})
# data = pd.merge(data, gb, how='left', on=['town_id', 'date_block_num']).fillna(0)
#
# gb = sales.groupby(by=['town_id', 'date_block_num'], as_index=False)['revenue'].sum()
# gb = gb.rename(columns={'revenue': 'revenue_by_town_date_block'})
# data = pd.merge(data, gb, how='left', on=['town_id', 'date_block_num']).fillna(0)


cols_month = ['month', 'item_category_id', 'shop_month_revenue', 'first_item_sale_date_block',
              'shop_month_revenue', 'days']
    # ['shop_category_revenue_sum', 'shop_category_revenue_mean', 'item_category_id',
    #           'last_item_sale_date_block', 'first_sale_date_block', 'first_item_sale_date_block', 'year', 'month',
    #           'shop_month_mean', 'item_month_revenue', 'shop_month_revenue', 'shop_item_month_revenue']


# exit()
print('created_lags')

# list of lagged features(columns)
fit_cols = [col for col in data.columns if col[-1] in [str(item) for item in shift_range]]

# will drop on fitting stage
to_drop_cols = list(set(list(data.columns))-(set(fit_cols) | set(index_cols) | set(cols_month)))+['date_block_num']


# item_category_mapping = items[['item_id', 'item_category_id']].drop_duplicates()
# data = pd.merge(data, item_category_mapping, how='left', on='item_id')

dates = data['date_block_num']
last_block = dates.max()

dates_train = dates[dates < last_block]
dates_test = dates[dates == last_block]

x_train = data.loc[dates < last_block-1].drop(to_drop_cols, axis=1)
x_valid = data.loc[dates == last_block-1].drop(to_drop_cols, axis=1)
x_test = data.loc[dates == last_block].drop(to_drop_cols, axis=1)


y_train = data.loc[dates < last_block-1, 'target']
y_valid = data.loc[dates == last_block-1, 'target']
'''

x_train.to_csv('./data/created/x_train.csv', index=False)
y_train.to_csv('./data/created/y_train.csv', index=False)

x_valid.to_csv('./data/created/x_valid.csv', index=False)
y_valid.to_csv('./data/created/y_valid.csv', index=False)

x_test.to_csv('./data/created/x_test.csv', index=False)
print('exit')
exit()
#

x_train = pd.read_csv('./data/created/x_train.csv')
y_train = pd.read_csv('./data/created/y_train.csv')

x_valid = pd.read_csv('./data/created/x_valid.csv')
y_valid = pd.read_csv('./data/created/y_valid.csv')

x_test = pd.read_csv('./data/created/x_test.csv')

'''

x_train = pd.concat([x_train, x_valid], ignore_index=True, sort=False)
y_train = pd.concat([y_train, y_valid], ignore_index=True, sort=False)




lr = LinearRegression()
xgb = XGBRegressor(
    max_depth=10,
    n_estimators=25,
    min_child_weight=2**7,
    colsample_bytree=0.8,
    subsample=0.75,
    eta=0.1,
    tree_method='hist',
    numthreads=8,
    grow_police='lossguide',
    predictor='cpu_predictor',
    rate_drop=0.2
)
lgb_params = {
    'feature_fraction': 0.75,
    'metric': 'rmse',
    'min_data_in_leaf': 2**6,
    'bagging_fraction': 0.75,
    'learning_rate': 0.1,
    'objective': 'mse',
    'bagging_seed': 2**7,
    'num_leaves': 2**8,
    'tree_learner': 'feature',
    'bagging_freq': 1,
    'verbose': 0,
    'n_jobs': 8,
}


lr.fit(x_train.values, y_train.values)
lr_preds = lr.predict(x_test.values).clip(0, 20)
print('lr done')
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_test).clip(0, 20)
print('xgb done')
knn = lightgbm.train(lgb_params, lightgbm.Dataset(x_train, label=y_train), num_boost_round=19)
pred_knn = knn.predict(x_test).clip(0, 20)
print('knn done')
x_test_level2 = np.c_[lr_preds, pred_xgb, pred_knn]
# print(len(pred_xgb) )
# print(len(x_test_level2))

test_preds = 0.4 * x_test_level2[:, 1] + 0.3 * x_test_level2[:, 0] + \
             0.3 * x_test_level2[:, 2]

used_blocks = [ 29, 30, 31, 32]
dates_train_level2 = dates_train[dates_train.isin(used_blocks)]
y_train_level2 = y_train[dates_train.isin(used_blocks)]

x_train_level2 = np.zeros([y_train_level2.shape[0], 3])

print('start ensembling')

for cur_block_num in used_blocks:
    x_train = data.loc[dates < cur_block_num].drop(to_drop_cols, axis=1)
    x_test = data.loc[dates == cur_block_num].drop(to_drop_cols, axis=1)
    y_train = data.loc[dates < cur_block_num].target.values

    print('cur_block_num: ', cur_block_num)

    lr.fit(x_train.values, y_train)
    pred_lr = lr.predict(x_test.values).clip(0, 20)

    xgb.fit(x_train, y_train)
    pred_xgb = xgb.predict(x_test).clip(0, 20)

    knn = lightgbm.train(lgb_params, lightgbm.Dataset(x_train, label=y_train), num_boost_round=19)
    pred_knn=knn.predict(x_test).clip(0, 20)

    x_train_level2[dates_train_level2 == cur_block_num] = np.c_[pred_lr, pred_xgb, pred_knn]

    del x_train
    del x_test
    del y_train
    gc.collect()

best_alpha = 0
best_beta = 0
best_mse = 10
for alpha in np.arange(0, 0.7, 0.01):
    for beta in np.arange(0, 1-alpha, 0.01):
        mix = alpha * x_train_level2[:, 1] + (1-alpha-beta) * x_train_level2[:, 2] + beta*x_train_level2[:, 0]
        mse = mean_squared_error(y_train_level2, mix)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_beta = beta

test_preds = best_alpha * x_test_level2[:, 1] + best_beta * x_test_level2[:, 0] + \
             (1-best_beta-best_alpha) * x_test_level2[:, 2]

print('x_test+lvl2', len(x_test_level2))
print(len(test_preds))

submissions_enseble = pd.DataFrame({
    'ID': range(len(test_preds)),
    'item_cnt_month': test_preds
})

submissions_enseble.to_csv('submission_ensemble.csv', index=False)

exit()
lgb_params = {
    'feature_fraction': 0.75,
    'metric': 'rmse',
    'min_data_in_leaf': 2**6,
    'bagging_fraction': 0.75,
    'learning_rate': 0.1,
    'objective': 'mse',
    'bagging_seed': 2**7,
    'num_leaves': 2**8,
    'tree_learner': 'feature',
    'bagging_freq': 1,
    'verbose': 0,
    'n_jobs': 8,
}
print('start training')
lgbm = lightgbm.train(lgb_params, lightgbm.Dataset(x_train, label=y_train), num_boost_round=19)
                      # valid_sets=lightgbm.Dataset(x_valid, label=y_valid), early_stopping_rounds=10)

pred_lgb = lgbm.predict(x_test).clip(0, 20)

submissions = pd.DataFrame({
    'ID': x_test.index,
    'item_cnt_month': pred_lgb
})

submissions.to_csv('submission_lightgbm.csv', index=False)
print('lightgbm done')

xgb = XGBRegressor(
    max_depth=10,
    n_estimators=25,
    min_child_weight=2**7,
    colsample_bytree=0.8,
    subsample=0.75,
    eta=0.1,
    tree_method='hist',
    numthreads=8,
    grow_police='lossguide',
    predictor='cpu_predictor',
    rate_drop=0.2
)
xgb.fit(x_train, y_train)
        # eval_metric='rmse', eval_set=[(x_valid, y_valid)],
        # verbose=True, early_stopping_rounds=10)

pred_xgb = xgb.predict(x_test).clip(0, 20)
submissions = pd.DataFrame({
    'ID': x_test.index,
    'item_cnt_month': pred_xgb
})
submissions.to_csv('submission_xgb.csv', index=False)

submissions = pd.DataFrame({
    'ID': x_test.index,
    'item_cnt_month': (pred_xgb + pred_lgb)/2
})
submissions.to_csv('submission_xgb_lgb.csv', index=False)


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


plot_features(xgb, (7, 8))
plt.show()

# # 4.038 4.1929

# добавить кол-во дней в месяце
# обрезать данные сначала (0,20)

