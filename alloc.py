import numpy as np
import pandas as pd
import scipy

def percentage_change_computed_df(df):
    for x in range(0,9):   
        df['shift'+ str(x)] = df[df.columns[x]].shift(1)
        df['PercentChange'+ str(x)] = (df[df.columns[x]] - df['shift'+ str(x)]).div(df['shift'+ str(x)]) * 100
        df=df.drop(columns=('shift'+str(x)))
    return df

def covmatrix(returns):
    return pd.DataFrame(np.cov(np.transpose(returns)))


asset_prices_df = pd.DataFrame()
asset_price_predictions_1_df = pd.DataFrame()
asset_price_predictions_2_df = pd.DataFrame()
asset_price_predictions_3_df = pd.DataFrame()
shares_outstanding = pd.DataFrame()

asset_price_predictions_1_df = asset_price_predictions_1_df[::21]
asset_price_predictions_2_df = asset_price_predictions_2_df[::21]
asset_price_predictions_3_df = asset_price_predictions_3_df[::21]
asset_price_predictions_df = (asset_price_predictions_1_df + asset_price_predictions_2_df + asset_price_predictions_3_df)/3
asset_price_predictions_df = percentage_change_computed_df(asset_price_predictions_df)

def allocate_portfolio(asset_prices, asset_price_predictions_1,asset_price_predictions_2,asset_price_predictions_3):
    global asset_prices_df
    print(asset_prices_df)
    asset_prices_df = asset_prices_df.append(asset_prices, ignore_index=True)
    print(asset_prices_df)
    asset_price_prediction = (asset_price_predictions_1 + asset_price_predictions_2 + asset_price_predictions_3)/3
    
    asset_prices_df_returns = percentage_change_computed_df(asset_prices_df)
    returns = asset_prices_df_returns.loc[:,"PercentChange0":"PercentChange8"].iloc[1:,:]
    cov_matrix = covmatrix(returns)
    
    market_caps = shares_outstanding.iloc[0] * asset_prices
    total_market_cap = sum(market_caps)
    wmkt = market_caps / total_market_cap
    lamb = 1
    implied_returns = lamb*(np.dot(cov_matrix, wmkt))
    
    prediction_cov = covmatrix(asset_price_predictions_df.loc[:,"PercentChange0":"PercentChange8"].iloc[1:,:])/21
    bl_cov = np.linalg.inv((np.linalg.inv(cov_matrix)+ np.linalg.inv(prediction_cov)))
    Q = ((asset_price_prediction/100 + 1)**(1/21)-1)*100
    bl_returns = bl_cov@(np.linalg.inv(cov_matrix)@implied_returns + np.linalg.inv(prediction_cov)@Q)
    bl_weights = np.linalg.inv(cov_matrix)@bl_returns/lamb
    bl_weights = bl_weights/sum(bl_weights)
    
    return bl_weights
