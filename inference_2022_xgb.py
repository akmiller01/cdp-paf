import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm



def main():
    crs = pd.read_csv('large_data/crs_2022_predictions.csv')
    crs_dummies = crs.rename(
        columns={
            'donor_name':'DonorName',
            'sector_name':'SectorName',
            'purpose_name': 'PurposeName',
            'flow_name': 'FlowName',
            'channel_name': 'ChannelName'
        }
    )
    crs_dummies = pd.get_dummies(crs_dummies, columns=["DonorName", "SectorName", "PurposeName", "FlowName", "ChannelName"], dtype=float)

    with open('models/xgb_multi.pkl', 'rb') as f:
        cf_xgb, paf_xgb, aa_xgb, dummy_cols = pickle.load(f)
    dummy_dim = len(dummy_cols)

    for dummy_col in dummy_cols:
        crs_dummies[dummy_col] = crs_dummies.get(dummy_col, 0)

    crs_dummies = crs_dummies[['Crisis finance confidence', 'PAF confidence', 'AA confidence'] + dummy_cols]

    cf_preds = list()
    cf_probs = list()
    paf_preds = list()
    paf_probs = list()
    aa_preds = list()
    aa_probs = list()
    for _, row in tqdm(crs_dummies.iterrows(), total=len(crs_dummies)):
        X = np.reshape(row.values.astype('float'), (1, dummy_dim + 3))
        cf_pred =  cf_xgb.predict(X)[0] == 1
        cf_pred_prob = cf_xgb.predict_proba(X)[0]
        paf_pred =  paf_xgb.predict(X)[0] == 1
        paf_pred_prob =  paf_xgb.predict_proba(X)[0]
        aa_pred =  aa_xgb.predict(X)[0] == 1
        aa_pred_prob =  aa_xgb.predict_proba(X)[0]
        cf_preds.append(cf_pred)
        cf_probs.append(cf_pred_prob[1])
        paf_preds.append(paf_pred)
        paf_probs.append(paf_pred_prob[1])
        aa_preds.append(aa_pred)
        aa_probs.append(aa_pred_prob[1])
    crs["Crisis finance predicted XGB"] = cf_preds
    crs["Crisis finance confidence XGB"] = cf_probs
    crs["PAF predicted XGB"] = paf_preds
    crs["PAF confidence XGB"] = paf_probs
    crs["AA predicted XGB"] = aa_preds
    crs["AA confidence XGB"] = aa_probs
    crs.to_csv("large_data/crs_2022_predictions_xgb.csv", index=False)


if __name__ == '__main__':
    main()

