import numpy as np
import pandas as pd

pred_idx = np.load("../data/pred_colission_idx.npy")
full_data = pd.read_excel("../data/8-30-17 U18 Games - corroborated matched with dbase export.xlsx")

#get predicted colission data
pred_data = full_data.iloc[pred_idx]
#get false positives i.e. data predicted as colission but isn't
# for idx, data in enumerate(pred_data[pred_data.keys()[0]]):
#     if data == 'X':
#         false_pos_idx.append(idx)
#get all impact_idx
impact_idx = []
for idx, data in enumerate(full_data[full_data.keys()[0]]):
    if data == 'I':
        impact_idx.append(idx)
false_pos_idx = np.setdiff1d(pred_idx, impact_idx)
true_pos_idx = np.intersect1d(pred_idx, impact_idx)
false_neg_idx = np.setdiff1d(true_pos_idx, impact_idx)

#get data at these idxs
true_pos_data = full_data.iloc[true_pos_idx].to_excel('../data/true_pos.xlsx')
false_neg_data = full_data.iloc[false_neg_idx].to_excel('../data/false_neg.xlsx')
false_pos_data = full_data.iloc[false_pos_idx].to_excel('../data/false_pos.xlsx')
