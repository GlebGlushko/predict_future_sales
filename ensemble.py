import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
best_l = 0.25
best_lg = 0.2
best_x = 0.45
best_k = 0.1
x_test_level2 = pd.read_csv('./data/created/level2_knn.csv')

test_preds = best_l * x_test_level2['lr'] + best_x * x_test_level2['xgb'] + best_k * x_test_level2['knn'] + best_lg * x_test_level2['lgbm']

submissions_enseble = pd.DataFrame({
    'ID': range(len(test_preds)),
    'item_cnt_month': test_preds
})

submissions_enseble.to_csv('./submission_ensemble.csv',index=False)
exit()
best_mse = 100
y_valid = pd.read_csv('./data/created/y_valid.csv')
x_valid_2 = pd.read_csv('./data/created/valids2_knn.csv')
# print(y_valid.describe())
# print(x_valid_2.describe())
# exit()
for l in np.arange(0, 1, 0.05):
    for x in np.arange(0, 1-l, 0.05):
        for k in np.arange(0, 1-l-x, 0.05):
            lg=1-x-l-k
            mix = l * x_valid_2['lr'] + x * x_valid_2['xgb'] + k * x_valid_2['knn'] + lg * x_valid_2['lgbm']
            mse = mean_squared_error(y_valid, mix)
            # print(mse)
            if mse < best_mse:
                best_mse = mse
                best_l = l
                best_k = k
                best_lg = lg
                best_x = x
                print('linear={0}\nxgboost={1}\nknn={2}\nlightgbm={3}\n'.format(best_l, best_x, best_k, best_lg))

# best_lg = 1 - best_lg - best_k - best_x

print('linear={0}\nxgboost={1}\nknn={2}\nlightgbm={3}\n'.format(best_l, best_x, -1, best_lg))
