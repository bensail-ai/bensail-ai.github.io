# Evaluation of Model on Final unseen races of 2022 & Project Summary

Author : Ben Palmer\
Date : 02/11/2022

## Summary:
This notebook has analysed the predictive power of the trained models on the last 4 races of 2022, which is a completely unseen dataset as the qualifying and races occured during the project after the original data was downloaded.

The results show that:
- The models predicted the overall trends well, were able to predict the top drivers from the bottom drivers in both position and lap time delta
- The position model predicts a float and not ranked results, therefore to compare the predictions need to be ranked 1 to 20. For example it can predict 3.3 and 3.7 for two drivers but these positions don't exist in reality. 
- Predicted Max Verstappen to get pole correctly twice.
- Predicted Hamilton's position quite closely

- The models had lower R2 and Mean Aboslute Error Scores than the original Test set, which is expected of a reality test set
- The model struggled to predict lap time delta especially in Singapore, often over predicting the top drivers and under predicting the bottom drivers
- Over the course of the season the models did well especially for the races they had been trained on, and less well on races they had not. Indicating there may be slight over training in the models.

Overall this project has learned a number of interesting features, including:
- The variance & maximum RPM (revolution per minute) of the engine on the straight
- The maximum speed on the corners
- The amount of distance a driver spent on the brakes in a given lap
- Initial Sector 1 times 
- Speed on the straights

The models have a number of limitations as well:
- Both the laptime delta and position prediction models show signs of over training with the test sets doing worse than the training datasets
- The dataset was small for machine learning with 1890 rows and high number of features the models will suffer with dimensionality and the lack of data
- The lap time delta model performed poorly on the "test" test set with mean absolute errors of 0.8 seconds
- As a regression technique was used the models especially for position did not predict the range of outcomes (1-20) and the data needed to be ranked afterwards to match
- Some of the features have outliers which can highly impact the predicted results. 
- The model struggled with unseen circuits, especially on lap time delta. This is likely because the features show more variability between circuits than driver performance

Recommendations for further work, include: 

- Normalising the input features to the circuit characteristics.
- Generating features that look at the past Grand Prix's performance as an indicator of future results.
- Using automatic feature engineering techniques such as auto encoders.
- Investigating classification models with bin size of 1 for position predictions.
- Build models to predict off practice 3 performance features. 
- Predict results at a lap level not at a race level. 

## Introduction:

The aim of this notebook is to evaluate the performance of the optimised models on the last 4 races of the 2022 season. This races occured whilst the project was being conducted and therefore were not included in the original dataset for analysis and modelling. These races are close to a production test set, a "test" test set. The performance of the models on this data will give us an indication of how the models are likely to perform in reality. 

In addition this notebook would summarise the performance of the models over the 2022 season up to present day to provide a summary of performance and predictive power of the models as a whole.

## Table of Contents:
1. [Predicting New Data Results](#newdata)\
    1.1[Prepare data for Modelling](#prep)\
    1.2[Predict & Evaluate Qualifying Results](#pos)\
    1.3[Predict & Evaluate Qualifying Lap time](#delta)
   

2. [Predictions vs True for 2022](#2022)\
    1.1 [Predict early 2022](#predict)\
    1.2 [combine predictions](#combine)\
    1.3 [2022 Results](#summary)

3. [Project Summary](#projsummary)

4. [Recommendations for Further Work](#recommendations)
    

### Imports


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats

from scripts.ds_ultils import *
from scripts.fastf1_data_download import *
from scripts.model_ultils import *

plt.style.use('./scripts/plotstyle.mplstyle')

import joblib
from mlxtend.feature_selection import ColumnSelector
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, plot_roc_curve, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from xgboost import XGBRegressor

```

The last 4 races Singapore, Suzuka, Austin and Mexico City, in addittion these races where not included in the original Ergast Database download. Therefore no results data has been collected. The data will be collected from the fastf1 api link to F1 live. However the data will be needed to be processed to match the same format as the previous data especially regarding format of the categorical variables. This is done for the results and categorical data before pulling all the telemetry data and creating the features as done for the previous dataset.

This code block has been run and the data saved to the model_data folder as:

`formula1_2022_last_races_model_data.pkl`


```python
#download the data from the api, this takes a few hours and has been completed and results saved for you.
#file='./data/clean/combined_ergast_clean.csv'
#event_df=get_year_quali()
#event_df=new_sessions(file,event_df)
#new_results_df=get_new_results_dataframe(file,event_df)
#new_data = pull_new_races_aggregate_telemetry(new_results_df)
#new_data.to_pickle('./data/model_data/formula1_2022_last_races_model_data.pkl',compression='gzip')
```

# Predicting New Data Results for 2022
<a id="newdata"></a>

First read in the new data, which the above code block saved as a temporary file.


```python
new_data_df=pd.read_pickle('./data/model_data/formula1_2022_last_races_model_data.pkl',compression='gzip')
```

## Preparing data for modelling
<a id="prep"></a>

Similar to the larger dataset, the data needs to be prepared before any predictions can be made. This is to ensure it is in the same format as the dataset that the models were trained on. The prepare modelling function:
- converts columns which are numeric to float
- Creates the target features and bins
- Creates the home country feature
- Cleans the dataframe 
- Fixes the GrandPrix where the DRS telemetry data was not working. 

However because we are only predicting and not training, test split is False to return just a X and y dataframe

The y features are split into position features and lap time delta.

The next step is to select the manual features which were considered important from the analysis in notebook [6_Formula1_Initial_Modelling](./6_Formula1_Initial_Modelling.ipynb). Similar to previous analysis the Sector Times of Fastest Lap are removed since these are just the break down of the times of the target and therefore not considered an input feature



```python

X_new_races, y_new_races = prepare_modelling_df(new_data_df,test_split=False)
ypos_new_races =y_new_races['quali_position'].copy()
ydelta_new_races=abs(y_new_races['lap_timedelta_milliseconds'].copy())

feature_importance=pd.read_pickle('./data/model_data/feature_importance_random_forest.pkl')
manual_features = list(feature_importance[feature_importance['Random_forest_result']>0.0105].index)
features_remove =[ 'numerical_transform__fastestlap_Sector1',
 'numerical_transform__fastestlap_Sector2',
 'numerical_transform__fastestlap_Sector3', 
 ] 
manual_features = [x for x in manual_features if x not in features_remove]
X_new_races_manual = apply_manual_features_X(X_new_races,features=manual_features)

```

## Predict Position
<a id="pos"></a>

This section will predict the Qualifying position for the last 4 races of the 2022 season


```python
#load model
Position_RF_model=joblib.load('./models/pickled_best_RF_regression_model_position.pkl')
```

Summary of the model, it consisents of a column transformer to scale the numerical columns and encode the categorical columns. Then a Random Forest Regressor to predict Qualifying position


```python
Position_RF_model
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;column_transform&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical_transform&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;circuit_total_corner_length&#x27;,
                                                   &#x27;circuit_total_corner_curvature&#x27;,
                                                   &#x27;circuit_mean_corner_curvature&#x27;,
                                                   &#x27;circuit_max_corner_curvature&#x27;,
                                                   &#x27;circuit_std_corner_curvature&#x27;,
                                                   &#x27;max_max_speed&#x27;,
                                                   &#x27;mean_straight_speed&#x27;,
                                                   &#x27;max_fastest_accleration&#x27;,
                                                   &#x27;mean_max_lap_accler...
                                                   &#x27;fl_lap_distance_on_brake&#x27;,
                                                   &#x27;fl_lap_distance_on_DRS&#x27;,
                                                   &#x27;fl_lap_max_speed_corner&#x27;,
                                                   &#x27;avglap_Sector1&#x27;,
                                                   &#x27;avglap_Sector3&#x27;,
                                                   &#x27;fastestlap_track_temperature&#x27;,
                                                   &#x27;avg_lap_track_temperature&#x27;,
                                                   &#x27;avg_lap_humidty&#x27;, &#x27;age&#x27;]),
                                                 (&#x27;hot_encode&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;driverRef&#x27;,
                                                   &#x27;constructorRef&#x27;])])),
                (&#x27;random_forest&#x27;,
                 RandomForestRegressor(max_depth=35, max_features=20,
                                       n_estimators=160))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;column_transform&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical_transform&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;circuit_total_corner_length&#x27;,
                                                   &#x27;circuit_total_corner_curvature&#x27;,
                                                   &#x27;circuit_mean_corner_curvature&#x27;,
                                                   &#x27;circuit_max_corner_curvature&#x27;,
                                                   &#x27;circuit_std_corner_curvature&#x27;,
                                                   &#x27;max_max_speed&#x27;,
                                                   &#x27;mean_straight_speed&#x27;,
                                                   &#x27;max_fastest_accleration&#x27;,
                                                   &#x27;mean_max_lap_accler...
                                                   &#x27;fl_lap_distance_on_brake&#x27;,
                                                   &#x27;fl_lap_distance_on_DRS&#x27;,
                                                   &#x27;fl_lap_max_speed_corner&#x27;,
                                                   &#x27;avglap_Sector1&#x27;,
                                                   &#x27;avglap_Sector3&#x27;,
                                                   &#x27;fastestlap_track_temperature&#x27;,
                                                   &#x27;avg_lap_track_temperature&#x27;,
                                                   &#x27;avg_lap_humidty&#x27;, &#x27;age&#x27;]),
                                                 (&#x27;hot_encode&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;driverRef&#x27;,
                                                   &#x27;constructorRef&#x27;])])),
                (&#x27;random_forest&#x27;,
                 RandomForestRegressor(max_depth=35, max_features=20,
                                       n_estimators=160))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">column_transform: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numerical_transform&#x27;, StandardScaler(),
                                 [&#x27;circuit_total_corner_length&#x27;,
                                  &#x27;circuit_total_corner_curvature&#x27;,
                                  &#x27;circuit_mean_corner_curvature&#x27;,
                                  &#x27;circuit_max_corner_curvature&#x27;,
                                  &#x27;circuit_std_corner_curvature&#x27;,
                                  &#x27;max_max_speed&#x27;, &#x27;mean_straight_speed&#x27;,
                                  &#x27;max_fastest_accleration&#x27;,
                                  &#x27;mean_max_lap_accleration&#x27;,
                                  &#x27;max_fastest_lap_rpm&#x27;,
                                  &#x27;var_fa...
                                  &#x27;avg_lap_distance_on_brake&#x27;,
                                  &#x27;avg_lap_distance_on_DRS&#x27;,
                                  &#x27;avg_lap_max_speed_corner&#x27;,
                                  &#x27;fl_lap_distance_on_brake&#x27;,
                                  &#x27;fl_lap_distance_on_DRS&#x27;,
                                  &#x27;fl_lap_max_speed_corner&#x27;, &#x27;avglap_Sector1&#x27;,
                                  &#x27;avglap_Sector3&#x27;,
                                  &#x27;fastestlap_track_temperature&#x27;,
                                  &#x27;avg_lap_track_temperature&#x27;,
                                  &#x27;avg_lap_humidty&#x27;, &#x27;age&#x27;]),
                                (&#x27;hot_encode&#x27;,
                                 OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                 [&#x27;driverRef&#x27;, &#x27;constructorRef&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">numerical_transform</label><div class="sk-toggleable__content"><pre>[&#x27;circuit_total_corner_length&#x27;, &#x27;circuit_total_corner_curvature&#x27;, &#x27;circuit_mean_corner_curvature&#x27;, &#x27;circuit_max_corner_curvature&#x27;, &#x27;circuit_std_corner_curvature&#x27;, &#x27;max_max_speed&#x27;, &#x27;mean_straight_speed&#x27;, &#x27;max_fastest_accleration&#x27;, &#x27;mean_max_lap_accleration&#x27;, &#x27;max_fastest_lap_rpm&#x27;, &#x27;var_fastest_lap_straight_rpm&#x27;, &#x27;mean_fastest_lap_straight_rpm&#x27;, &#x27;max_max_rpm&#x27;, &#x27;mean_var_straight_rpm&#x27;, &#x27;mean_straight_rpm&#x27;, &#x27;avg_lap_distance_on_brake&#x27;, &#x27;avg_lap_distance_on_DRS&#x27;, &#x27;avg_lap_max_speed_corner&#x27;, &#x27;fl_lap_distance_on_brake&#x27;, &#x27;fl_lap_distance_on_DRS&#x27;, &#x27;fl_lap_max_speed_corner&#x27;, &#x27;avglap_Sector1&#x27;, &#x27;avglap_Sector3&#x27;, &#x27;fastestlap_track_temperature&#x27;, &#x27;avg_lap_track_temperature&#x27;, &#x27;avg_lap_humidty&#x27;, &#x27;age&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">hot_encode</label><div class="sk-toggleable__content"><pre>[&#x27;driverRef&#x27;, &#x27;constructorRef&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(max_depth=35, max_features=20, n_estimators=160)</pre></div></div></div></div></div></div></div>



The qualifying results can predicted with simiply calling predict on the X features


```python
position_new_predictions= Position_RF_model.predict(X_new_races_manual)
```


```python
position_new_predictions
```




    array([ 3.7       ,  4.69375   ,  5.825     ,  7.05      , 11.22786458,
            5.2875    ,  9.66875   ,  5.7125    , 13.4625    ,  8.39375   ,
           10.6640625 , 12.425     , 11.86875   , 13.8875    , 15.2875    ,
           14.21875   , 10.41875   , 13.3       , 12.96875   , 15.63125   ,
            3.39375   ,  6.84375   ,  5.15      ,  5.35      , 11.2875    ,
            4.5625    , 10.14375   , 11.66875   , 12.21469595, 11.8890625 ,
           13.375     , 11.60625   , 10.75625   , 13.85625   , 14.96875   ,
           13.76875   , 14.23125   , 15.8125    , 17.56875   ,  6.55625   ,
            6.9375    ,  5.2875    ,  5.93125   , 11.3875    ,  9.45625   ,
           12.2       ,  4.06875   , 13.1875    , 13.16875   ,  7.0375    ,
           14.9       , 14.54375   , 14.51875   , 14.30625   , 12.45      ,
           12.21875   , 13.775     , 14.91875   , 15.7       ,  3.66875   ,
            2.94375   ,  2.9125    ,  4.6875    ,  5.05625   ,  5.99375   ,
           14.41666667,  8.4       , 11.21130952, 12.1375    , 16.7       ,
           12.6575    , 11.423125  , 13.673125  , 10.6125    , 12.82916667,
           10.24375   , 12.1578125 , 15.05625   , 17.98125   ])



The predictions are for the last 4 races per driver, they are also floats where as the true results are integers. This is because the models is a regression model and can predict any number. One alternative method to avoid this would be a classification model with a bin size of 1. To correct for this the results can be also be ranked in order of prediction to give a result more indicative of position.

First what are the model evaluation scores for the last 4 races of 2022


```python
print(f'Lap Delta Random Forest regression Initial R2 score with manual selected featuers {r2_score(ypos_new_races,position_new_predictions)}')
print(f'Lap Delta Random Forest regression Initial MSE score with manual selected feaure {mean_squared_error(ypos_new_races,position_new_predictions)}')
print(f'Lap Delta Random Forest regression Initial MAE score with manual selected feaure {mean_absolute_error(ypos_new_races,position_new_predictions)}')
```

    Lap Delta Random Forest regression Initial R2 score with manual selected featuers 0.650600540444255
    Lap Delta Random Forest regression Initial MSE score with manual selected feaure 11.575361682061502
    Lap Delta Random Forest regression Initial MAE score with manual selected feaure 2.8325292095327694
    

The R2 is similar to the previous test set with a Mean Aboslute error of 2.8 slightly higher to what was predicted previously. This can be compared by  looking at the predictions versus true values on a scatter plot


```python
plt.scatter(ypos_new_races,position_new_predictions)
plt.plot(np.arange(1,20,1),np.arange(1,20,1),c='r')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Predictions vs True for last 4 races of 2022 season')
plt.show()
```


    
![png](output_21_0.png)
    


The red line shows the perfect predictions where True=Predictions. The model seems to over predict results for the top drivers and under predict (predict better results) for the bottom drivers. This may be due to features in these GrandPrix's which the model thinks all the drivers are more equal. Maybe as the season has progressed the difference beteween drivers and cars has reduced such that the model thinks the drivers are more similar.

How about if we rank the results for 1 Grandprix and compared the model ranked results with the actual results.


```python
print('Singapore GrandPrix 2022 Results')
pos_comparison = new_data_df[['quali_position','driverRef','circuitRef']]
pos_comparison['position_predicted'] = position_new_predictions
singapore_position = pos_comparison[pos_comparison['circuitRef'] =='marina_bay'].copy()
singapore_position
```

    Singapore GrandPrix 2022 Results
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quali_position</th>
      <th>driverRef</th>
      <th>circuitRef</th>
      <th>position_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>leclerc</td>
      <td>marina_bay</td>
      <td>6.55625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>perez</td>
      <td>marina_bay</td>
      <td>6.93750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>hamilton</td>
      <td>marina_bay</td>
      <td>5.28750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>sainz</td>
      <td>marina_bay</td>
      <td>5.93125</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>alonso</td>
      <td>marina_bay</td>
      <td>11.38750</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>norris</td>
      <td>marina_bay</td>
      <td>9.45625</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>gasly</td>
      <td>marina_bay</td>
      <td>12.20000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>max_verstappen</td>
      <td>marina_bay</td>
      <td>4.06875</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>kevin_magnussen</td>
      <td>marina_bay</td>
      <td>13.18750</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>tsunoda</td>
      <td>marina_bay</td>
      <td>13.16875</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>russell</td>
      <td>marina_bay</td>
      <td>7.03750</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>stroll</td>
      <td>marina_bay</td>
      <td>14.90000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>mick_schumacher</td>
      <td>marina_bay</td>
      <td>14.54375</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>vettel</td>
      <td>marina_bay</td>
      <td>14.51875</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15.0</td>
      <td>zhou</td>
      <td>marina_bay</td>
      <td>14.30625</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16.0</td>
      <td>bottas</td>
      <td>marina_bay</td>
      <td>12.45000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.0</td>
      <td>ricciardo</td>
      <td>marina_bay</td>
      <td>12.21875</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18.0</td>
      <td>ocon</td>
      <td>marina_bay</td>
      <td>13.77500</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.0</td>
      <td>albon</td>
      <td>marina_bay</td>
      <td>14.91875</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20.0</td>
      <td>latifi</td>
      <td>marina_bay</td>
      <td>15.70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Use scipy function to rank the predictions for singapore
singapore_position['position_predicted_ranked'] = rankdata(singapore_position['position_predicted'])
```

Plot the predictions ranked such that the predictions are 1-20 in order of how the model assigned the position scores. 


```python
plt.scatter(singapore_position['quali_position'],singapore_position['position_predicted_ranked'])
plt.plot(np.arange(1,20,1),np.arange(1,20,1),c='r')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Predictions vs True for Singapore GP 2022')
plt.show()
```


    
![png](output_26_0.png)
    


As can be observed the model captures the trend but gets a few drivers badly wrong. It generally predicts the top 10 drivers to do worse than they did, but mostly in roughly the right order. Max Verstappen who has dominated 2022 is predicted to get pole, therefore it maybe affected by the categorical dummy variable for max more than his driving data. It under predicts for Russell and Bottas as well probably because of either the team they driver for or there previous results. It over predicts for Mick Schumacher and Stroll probably related to their previous results.  

On a positive note apart from clear errors it gets the order for the top 15 more or less right, with the exception of Max Verstappen it gets the top 3 almost right just in the wrong order. Plus the drivers from 5th to 10th are also in the more or less right order just offsetted.


```python
print(f"Lap Delta Random Forest regression Initial R2 score with manual selected featuers {r2_score(singapore_position['quali_position'],singapore_position['position_predicted_ranked'])}")
print(f"Lap Delta Random Forest regression Initial MSE score with manual selected feaure {mean_squared_error(singapore_position['quali_position'],singapore_position['position_predicted_ranked'])}")
print(f"Lap Delta Random Forest regression Initial MAE score with manual selected feaure {mean_absolute_error(singapore_position['quali_position'],singapore_position['position_predicted_ranked'])}")
```

    Lap Delta Random Forest regression Initial R2 score with manual selected featuers 0.58796992481203
    Lap Delta Random Forest regression Initial MSE score with manual selected feaure 13.7
    Lap Delta Random Forest regression Initial MAE score with manual selected feaure 3.0
    

As we can see ranking the data an impact on the metrics this is because the errors are maginfied. In the unranked predictions the top drivers were predicted worse and the worse drivers predicted better, averaging out the drivers performance which can reduce the error as the model becomes less extreme or sure it its predictions. It predicts some drivers to do well but not too sure and some to do poor but also not too sure. Therefore this reduces the Mean absoute error. By ranking the Mean Absolute Error increases as the the unsure points are forced to fill the full range. 


# Lap time delta predictions
<a id="delta"></a>

This section will predict the lap time delta for the last 4 races of 2022.

First load the XGboost model for predicting lap time deltas


```python
Lapdelta_XG_model=joblib.load('./models/pickled_best_XGboost_regression_model_lap_delta.pkl')
```

Predict the lap time deltas for the last 4 races:


```python
lapdelta_new_predictions= Lapdelta_XG_model.predict(X_new_races_manual)
```


```python
lapdelta_new_predictions
```




    array([ 880.1565 , 1438.6971 , 1180.8447 , 1296.8682 , 1812.8461 ,
           1169.3201 , 1822.2637 , 1158.1523 , 2426.7173 , 1508.983  ,
           1993.9811 , 2669.0835 , 2078.9802 , 2662.2546 , 2359.1501 ,
           2584.7327 , 1815.3125 , 2248.5618 , 2016.9613 , 2652.7021 ,
            639.421  ,  566.3785 ,  829.21826, 1065.0433 , 1807.4713 ,
            652.2429 , 1421.7653 , 1822.8589 , 1865.5409 , 1492.1318 ,
           2021.5482 , 1279.7029 , 1409.1636 , 1970.5421 , 2274.2869 ,
           2047.0016 , 1727.6038 , 3276.9106 , 3028.2742 , 1890.5687 ,
           2393.826  , 2320.7075 , 2755.3943 , 3008.1597 , 3130.3552 ,
           2855.4634 , 2050.2947 , 3926.7761 , 3014.654  , 2223.9104 ,
           3557.371  , 6459.6665 , 3566.9734 , 3932.539  , 6575.056  ,
           6152.094  , 7681.5635 , 6657.079  , 6511.405  ,   48.03798,
            429.90155,  311.8454 ,  813.8326 ,  894.0303 ,  809.2479 ,
           2507.0876 ,  773.581  , 1525.0428 , 2789.0137 , 3380.0232 ,
           2032.303  , 1540.2434 , 2076.641  , 1449.4377 , 2444.9236 ,
           1716.5864 , 2195.368  , 2876.5132 , 2899.111  ], dtype=float32)



The output is an array of lap time deltas for the last 4 races of 2022. How did the model perform against the true values?


```python
print(f'Lap Delta Random Forest regression Initial R2 score with manual selected featuers {r2_score(ydelta_new_races,lapdelta_new_predictions)}')
print(f'Lap Delta Random Forest regression Initial MSE score with manual selected feaure {mean_squared_error(ydelta_new_races,lapdelta_new_predictions)}')
print(f'Lap Delta Random Forest regression Initial MAE score with manual selected feaure {mean_absolute_error(ydelta_new_races,lapdelta_new_predictions)}')
```

    Lap Delta Random Forest regression Initial R2 score with manual selected featuers 0.6986873848786612
    Lap Delta Random Forest regression Initial MSE score with manual selected feaure 1028441.6571452693
    Lap Delta Random Forest regression Initial MAE score with manual selected feaure 775.7193984019606
    

In this "test" test set, the lap time delta did worse than the with the previous test set, with a worse MAE error. The Model had a mean error of 800 milliseconds or 0.8s (8/10ths of the second). Lets see how the results were distributed. Was the model worse at predicting certain results?


```python
plt.scatter(ydelta_new_races,lapdelta_new_predictions)
plt.plot(np.arange(1,8000,100),np.arange(1,8000,100),c='r')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Predictions vs True for Lap time delta last 4 races of 2022 season')
plt.show()
```


    
![png](output_38_0.png)
    


WHen looking at the predictions vs true values the majority of points are captured between 0 to 4 seconds (4000 milliseconds) away from the pole sitter. The majority of these points are close to the true prediction line (red), similar to the position the model averages out the score over predicting the top drivers delta and under prediction the worse drivers results, especially in the small tail of points from 4 to 8s. These are anomalous data points and it can be expected that the model may struggle with these data points.

There is another interesting data cloud, sitting above the true prediction line predicted at 3 s behind the pole sitter, there are a cluster of predicted points where the model predicted 3s delta for both the front drivers to the drivers who were actually 3 seconds back. This is probably one single GrandPrix where the model struggled. As shown in the evaluation of the model in [8_Formula1_Regression_models]('./8_Formula1_Regression_models.ipynb') the lap time delta model is affected most by features like avglap sector 1 times and RPM on the straight, if the GrandPrix has a partically long sector 1 time or the engines can't run at the high RPM because the straights are short the model will predict high lap time deltas which are in fact wrong.

This confirms the challenges of predicting lap time delta due to the fact there are so many variables that complicate the actual time result and the difference between results. As discussed addittional features which are normalised for different circuit lengths may help the predictive power of lap time delta. If avg sector time was also a lap time delta between the drivers that may help the model. If the metrics of speed and rpm on the straights was nominalised for straight length as well may help the model.

Lets investigate further by looking at the results of a couple of GrandPrix's


```python
print('Singapore GrandPrix 2022 Results')
delta_comparison = new_data_df[['lap_timedelta_milliseconds','driverRef','circuitRef']]
delta_comparison['lap_timedelta_milliseconds'] = abs(delta_comparison['lap_timedelta_milliseconds'])
delta_comparison['laptime_delta_predicted'] = lapdelta_new_predictions
delta_comparison['Residuals'] = delta_comparison['laptime_delta_predicted'] - delta_comparison['lap_timedelta_milliseconds']
singapore_position = delta_comparison[delta_comparison['circuitRef'] =='marina_bay'].copy()
singapore_position
```

    Singapore GrandPrix 2022 Results
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lap_timedelta_milliseconds</th>
      <th>driverRef</th>
      <th>circuitRef</th>
      <th>laptime_delta_predicted</th>
      <th>Residuals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>leclerc</td>
      <td>marina_bay</td>
      <td>1890.568726</td>
      <td>1890.568726</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>perez</td>
      <td>marina_bay</td>
      <td>2393.825928</td>
      <td>2371.825928</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.0</td>
      <td>hamilton</td>
      <td>marina_bay</td>
      <td>2320.707520</td>
      <td>2266.707520</td>
    </tr>
    <tr>
      <th>3</th>
      <td>171.0</td>
      <td>sainz</td>
      <td>marina_bay</td>
      <td>2755.394287</td>
      <td>2584.394287</td>
    </tr>
    <tr>
      <th>4</th>
      <td>554.0</td>
      <td>alonso</td>
      <td>marina_bay</td>
      <td>3008.159668</td>
      <td>2454.159668</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1172.0</td>
      <td>norris</td>
      <td>marina_bay</td>
      <td>3130.355225</td>
      <td>1958.355225</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1799.0</td>
      <td>gasly</td>
      <td>marina_bay</td>
      <td>2855.463379</td>
      <td>1056.463379</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1983.0</td>
      <td>max_verstappen</td>
      <td>marina_bay</td>
      <td>2050.294678</td>
      <td>67.294678</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2161.0</td>
      <td>kevin_magnussen</td>
      <td>marina_bay</td>
      <td>3926.776123</td>
      <td>1765.776123</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2571.0</td>
      <td>tsunoda</td>
      <td>marina_bay</td>
      <td>3014.654053</td>
      <td>443.654053</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4600.0</td>
      <td>russell</td>
      <td>marina_bay</td>
      <td>2223.910400</td>
      <td>-2376.089600</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4799.0</td>
      <td>stroll</td>
      <td>marina_bay</td>
      <td>3557.371094</td>
      <td>-1241.628906</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4958.0</td>
      <td>mick_schumacher</td>
      <td>marina_bay</td>
      <td>6459.666504</td>
      <td>1501.666504</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4968.0</td>
      <td>vettel</td>
      <td>marina_bay</td>
      <td>3566.973389</td>
      <td>-1401.026611</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5963.0</td>
      <td>zhou</td>
      <td>marina_bay</td>
      <td>3932.539062</td>
      <td>-2030.460938</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6671.0</td>
      <td>bottas</td>
      <td>marina_bay</td>
      <td>6575.056152</td>
      <td>-95.943848</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6814.0</td>
      <td>ricciardo</td>
      <td>marina_bay</td>
      <td>6152.094238</td>
      <td>-661.905762</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6925.0</td>
      <td>ocon</td>
      <td>marina_bay</td>
      <td>7681.563477</td>
      <td>756.563477</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7573.0</td>
      <td>albon</td>
      <td>marina_bay</td>
      <td>6657.079102</td>
      <td>-915.920898</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8120.0</td>
      <td>latifi</td>
      <td>marina_bay</td>
      <td>6511.404785</td>
      <td>-1608.595215</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(singapore_position['lap_timedelta_milliseconds'],singapore_position['Residuals'])
plt.hlines(0,0,8000,color='r')
plt.xlabel('True values')
plt.ylabel('Residuals')
plt.title('Residuals for Lap time delta for Singapore 2022 season')
plt.show()
```


    
![png](output_41_0.png)
    


As can be seen from the Singapore residuals this is the GrandPrix that corresponded to the cloud of points high above the true prediction red line in the overall plot, and also the grandprix that had the results with a large tail of points going out to 8seconds back. This GrandPrix the model clearly struggled with to get the lap time delta right. This GrandPrix is a street circuit, with lots of short straights and tight corners. Hence the cars can not get up to full speed and are trying to thread the needle between the tight walled off streets of Singapore. Therefore larger variation of lap times is expected. In adittion the cars do not reach their top speeds they would expect on a larger more open circuit. Therefore the features of RPM, Speed and sector times would be very different to other GrandPrix's hence this is likely why the model performed so poorly for this Grand Prix


```python
suzaka_position = delta_comparison[delta_comparison['circuitRef'] =='suzuka'].copy()

plt.scatter(suzaka_position['lap_timedelta_milliseconds'],suzaka_position['Residuals'])
plt.hlines(0,0,3500,color='r')
plt.xlabel('True values')
plt.ylabel('Residuals')
plt.title('Residuals for Lap time delta for Suzuka 2022 season')
plt.show()
```


    
![png](output_43_0.png)
    


For Suzaka the model performed better with all the drivers within 1.4 seconds of there true values. However similar to Singapore the model over predicted the times of most drivers. Suzuka is a long but fast circuit and it may have been more affected by features like avg lap sector time

# Predictions versus Reality for 2022
<a id="2022"></a>

How good are the model predictions for the 2022 season as a whole?

To answer this question would require combining the predictions from the previous dataset with the predictions of the last 4 races in this notebook.

To do this we will extract the 2022 races from the previous dataset and re run the predictions for just the 2022 races. This would include races the model was trained on and ones it was not trained on.

First read the previous dataset:


```python
qualify_df=pd.read_pickle('./data/model_data/formula1_complete_2018_2022_complete_301022.pkl',compression='gzip')
qualifying_df=qualify_df.copy()

```

Select only the 2022 races:


```python
initial_2022_races_df = qualifying_df[qualifying_df['year'] == 2022].copy()
```

## Predict position and Lap time delta 
<a id="predict"></a>

For the early 2022 races predict the position and laptime delta

This requires preparing the data for modelling and applying the manaul feature selection


```python
X_22_races, y_22_races = prepare_modelling_df(initial_2022_races_df,test_split=False)
ypos_22_races =y_22_races['quali_position'].copy()
ydelta_22_races=abs(y_22_races['lap_timedelta_milliseconds'].copy())

X_22_races_manual = apply_manual_features_X(X_22_races,features=manual_features)

```

Run the predictions for position and Lap time delta:


```python
position_22_predictions= Position_RF_model.predict(X_22_races_manual)
lapdelta_22_predictions= Lapdelta_XG_model.predict(X_22_races_manual)
```

## Combine into a Summary Table
<a id="combine"></a>

How did the model perform over the whole 2022 season? To evaluate its performance we need to combine the results from 2022 races in the original dataset and last 4 races.

Create a summary table from the early 2022 predictions:


```python
summary_22_df = initial_2022_races_df[['driverRef','circuitRef','quali_position','lap_timedelta_milliseconds']].copy()
summary_22_df['predicted_position_model'] = position_22_predictions
summary_22_df['predicted_laptime_delta'] = lapdelta_22_predictions
summary_22_df['lap_timedelta_milliseconds']= abs(summary_22_df['lap_timedelta_milliseconds'])
```

For the position predictions, they need to be ranked to force the results to go from positions 1 to 20


```python
circuits = summary_22_df['circuitRef'].unique()
for circuit in circuits:
    query = summary_22_df['circuitRef'] == circuit
    summary_22_df.loc[query,'predicted_position_ranked'] = rankdata(summary_22_df.loc[query,'predicted_position_model'])
    

```

The summary dataframe will contain information of:
- driver
- circuit
- True qualifying position
- Lap time detla prediction
- Predicted position & ranked position
- Predicted lap time delta


```python
summary_22_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>driverRef</th>
      <th>circuitRef</th>
      <th>quali_position</th>
      <th>lap_timedelta_milliseconds</th>
      <th>predicted_position_model</th>
      <th>predicted_laptime_delta</th>
      <th>predicted_position_ranked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>leclerc</td>
      <td>albert_park</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.44375</td>
      <td>17.939283</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>perez</td>
      <td>albert_park</td>
      <td>3.0</td>
      <td>372.0</td>
      <td>4.40000</td>
      <td>379.915253</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>norris</td>
      <td>albert_park</td>
      <td>4.0</td>
      <td>835.0</td>
      <td>5.92500</td>
      <td>843.338257</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>hamilton</td>
      <td>albert_park</td>
      <td>5.0</td>
      <td>957.0</td>
      <td>5.15000</td>
      <td>959.565918</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>russell</td>
      <td>albert_park</td>
      <td>6.0</td>
      <td>1065.0</td>
      <td>6.23125</td>
      <td>1043.465088</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



This dataframe is created for the results of the last 4 races


```python
newraces_22_df = new_data_df[['driverRef','circuitRef','quali_position','lap_timedelta_milliseconds']].copy()
newraces_22_df['predicted_position_model'] = position_new_predictions
newraces_22_df['predicted_laptime_delta'] = lapdelta_new_predictions
newraces_22_df['lap_timedelta_milliseconds']= abs(newraces_22_df['lap_timedelta_milliseconds'])
```

The predicted positions are ranked 0 to 20:


```python
circuits = newraces_22_df['circuitRef'].unique()
for circuit in circuits:
    query = newraces_22_df['circuitRef'] == circuit
    newraces_22_df.loc[query,'predicted_position_ranked'] = rankdata(newraces_22_df.loc[query,'predicted_position_model'])
    
```

Now to combine the two dataframes of 2022 season together


```python
summary_22_combined = pd.concat([summary_22_df,newraces_22_df])
summary_22_combined.reset_index(drop=True)
summary_22_combined['laptime_delta_residual'] = summary_22_combined['lap_timedelta_milliseconds'] - summary_22_combined['predicted_laptime_delta']
summary_22_combined.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>driverRef</th>
      <th>circuitRef</th>
      <th>quali_position</th>
      <th>lap_timedelta_milliseconds</th>
      <th>predicted_position_model</th>
      <th>predicted_laptime_delta</th>
      <th>predicted_position_ranked</th>
      <th>laptime_delta_residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>leclerc</td>
      <td>albert_park</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.44375</td>
      <td>17.939283</td>
      <td>1.0</td>
      <td>-17.939283</td>
    </tr>
    <tr>
      <th>42</th>
      <td>perez</td>
      <td>albert_park</td>
      <td>3.0</td>
      <td>372.0</td>
      <td>4.40000</td>
      <td>379.915253</td>
      <td>2.0</td>
      <td>-7.915253</td>
    </tr>
    <tr>
      <th>43</th>
      <td>norris</td>
      <td>albert_park</td>
      <td>4.0</td>
      <td>835.0</td>
      <td>5.92500</td>
      <td>843.338257</td>
      <td>4.0</td>
      <td>-8.338257</td>
    </tr>
    <tr>
      <th>44</th>
      <td>hamilton</td>
      <td>albert_park</td>
      <td>5.0</td>
      <td>957.0</td>
      <td>5.15000</td>
      <td>959.565918</td>
      <td>3.0</td>
      <td>-2.565918</td>
    </tr>
    <tr>
      <th>45</th>
      <td>russell</td>
      <td>albert_park</td>
      <td>6.0</td>
      <td>1065.0</td>
      <td>6.23125</td>
      <td>1043.465088</td>
      <td>5.0</td>
      <td>21.534912</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary_22_combined.to_pickle("./data/model_data/Summary_2022_model_predicitions.pkl",compression='gzip')
```

# Summary of Results 
<a id="summary"></a>

With the summary dataframe of results and predictions, we can ask questions like;
 - How good was the model at predicting Pole Position?
 - How well did it do for certain drivers throughout the year?

Lets create a dataframe of pole winners by selecting the driver who was on True Pole and selecting the driver the model predicted to be pole for each circuit.


```python
pole_winner=pd.DataFrame()
pole_winner['circuitRef'] = summary_22_combined['circuitRef'].unique()
pole_winner['True_pole_sitter']=''
pole_winner['predicted_pole_sitter']=''
circuits = summary_22_combined['circuitRef'].unique()
for circuit in circuits:

    circuit_query = pole_winner['circuitRef'] == circuit
    circuit_query2 = summary_22_combined['circuitRef'] == circuit    
    query1 = summary_22_combined['quali_position'] == 1.0
    if len(summary_22_combined.loc[(query1) & (circuit_query2),'driverRef'].values) > 0: # some races do not have rows for the pole position driver because no telemetry data was recored for that driver in qualifying. Therefore we need to check for than and assign a NaN value for that entry
        true_position =summary_22_combined.loc[(query1) & (circuit_query2),'driverRef'].values[0]
    else:
        true_position = np.NaN 

    pole_winner.loc[circuit_query,'True_pole_sitter'] = true_position
    
    query2 = summary_22_combined['predicted_position_ranked'] == 1.0
    pole_winner.loc[circuit_query,'predicted_pole_sitter'] = summary_22_combined.loc[(query2) & (circuit_query2),'driverRef'].values[0]
   
```


```python
pole_winner
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>circuitRef</th>
      <th>True_pole_sitter</th>
      <th>predicted_pole_sitter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>albert_park</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red_bull_ring</td>
      <td>NaN</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>baku</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bahrain</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spa</td>
      <td>NaN</td>
      <td>sainz</td>
    </tr>
    <tr>
      <th>5</th>
      <td>silverstone</td>
      <td>sainz</td>
      <td>sainz</td>
    </tr>
    <tr>
      <th>6</th>
      <td>villeneuve</td>
      <td>NaN</td>
      <td>sainz</td>
    </tr>
    <tr>
      <th>7</th>
      <td>zandvoort</td>
      <td>NaN</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>8</th>
      <td>imola</td>
      <td>NaN</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ricard</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hungaroring</td>
      <td>russell</td>
      <td>russell</td>
    </tr>
    <tr>
      <th>11</th>
      <td>monza</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>12</th>
      <td>miami</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>13</th>
      <td>monaco</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>14</th>
      <td>jeddah</td>
      <td>perez</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>15</th>
      <td>catalunya</td>
      <td>leclerc</td>
      <td>leclerc</td>
    </tr>
    <tr>
      <th>16</th>
      <td>suzuka</td>
      <td>max_verstappen</td>
      <td>max_verstappen</td>
    </tr>
    <tr>
      <th>17</th>
      <td>rodriguez</td>
      <td>max_verstappen</td>
      <td>max_verstappen</td>
    </tr>
    <tr>
      <th>18</th>
      <td>marina_bay</td>
      <td>leclerc</td>
      <td>max_verstappen</td>
    </tr>
    <tr>
      <th>19</th>
      <td>americas</td>
      <td>sainz</td>
      <td>max_verstappen</td>
    </tr>
  </tbody>
</table>
</div>



Here we have a table where we can compare how the model did predicting pole position against the true pole sitter. A couple of observations jump out immediately:
- Where do the NaN values come from in True pole sitter?
- Overall the model does well only in correctly predicting the pole sitter where there is data 3 times

Firstly the NaN values come from Max Verstappen, as mentioned when the data was downloaded from the API ([6_Formula1_Initial_Modelling]('./6_Formula1_Initial_Modelling.ipynb')) there are occurences where no telemetry data is collected for a driver for a particular race. Max has the most drop outs of telemetry data for every driver and in 2022 all those drop outs occur when we gets pole position...
Therefore all the NaN values are for MaxVerstappen. 

At face value the model seems to do well overall, however the model was trained on most of the races, only 6 races are Test races. These are:
- Red Bull Ring
- Zandvoort 
from the original dataset and the last 4 races:
- Suzuka
- Rodriguez
- Marina Bay
- Americas 

The two test races in the original dataset Max got pole and there was no data for the model to predict Max. The last 4 races the model predicted Max all the time, it got it right 2 out of 4 times. 

Therefore the model maybe showing signs of over training. 

How does it vary for certain drivers over the year?

Select the results for certain drivers:


```python
hamilton_results = summary_22_combined[summary_22_combined['driverRef'] == 'hamilton']
```


```python
Verstappen_results = summary_22_combined[summary_22_combined['driverRef'] == 'max_verstappen']
```


```python
kevin_magnussen = summary_22_combined[summary_22_combined['driverRef'] == 'kevin_magnussen']
```

How are Hamilton's results:


```python
temp = hamilton_results[['driverRef','circuitRef','quali_position','predicted_position_ranked']]
temp2 = pd.melt(temp,id_vars=['driverRef','circuitRef'], value_vars=['quali_position','predicted_position_ranked'])
```

Plot the results of predicted and true positions for Hamilton throughout the year


```python
temp = hamilton_results[['driverRef','circuitRef','quali_position','predicted_position_ranked']]
temp2 = pd.melt(temp,id_vars=['driverRef','circuitRef'], value_vars=['quali_position','predicted_position_ranked'])
plt.figure(figsize=(4,7))
sns.barplot(temp2,x='value',y='circuitRef',hue='variable')
plt.xlabel('Qualifying position')
plt.ylabel('Grand Prix')
plt.title("Hamilton's positions and predictions for 2022")
plt.show()
```


    
![png](output_78_0.png)
    


Overal for Hamilton the model seemed to do ok. In the last 4 races, it predicted correctly for Singapore, Americas and was very close in Rodriquez. Overall it seemed to predict he would perform better than he did. This is understanable as Hamilton has had his first year of results in a long time, from dominating Formula 1 for the last 7 years, he has now had a car which can not competete at the front. Since the model was trained mostly on his good results, it is understandable that it predicted him to do better than expected.

How was the lap time delta predictions?


```python
colors = ['#405678' if c >= 0 else '#E10600' for c in hamilton_results['laptime_delta_residual']]
sns.barplot(data=hamilton_results,x='laptime_delta_residual',y='circuitRef',palette=colors)
plt.title("Hamilton's Lap time delta differences for 2022")
plt.xlabel('Milliseconds difference')
plt.ylabel('Grand Prix')
plt.tight_layout()
plt.savefig('./images/Hamiltons_lap_time_delta_predictions.jpg',dpi=300)
plt.show()
```


    
![png](output_80_0.png)
    



```python
hamilton_results_test= hamilton_results[hamilton_results['circuitRef'].isin(['red_bull_ring','zandvoort','suzuka','rodriquez','marina_bay','americas'])]
colors = ['#405678' if c >= 0 else '#E10600' for c in hamilton_results_test['laptime_delta_residual']]
sns.barplot(data=hamilton_results_test,x='laptime_delta_residual',y='circuitRef',palette=colors)
plt.title("Hamilton's Lap time delta differences for 2022")
plt.xlabel('Milliseconds difference')
plt.ylabel('Grand Prix')
plt.tight_layout()
plt.savefig('./images/Hamiltons_lap_time_delta_predictions_testset.jpg',dpi=300)
plt.show()
```


    
![png](output_81_0.png)
    


For the majority of the races it predicted Hamilton's time delta to within half a second, which on face value is rather good considering the complexities of Formula1. For the last 4 races it over predicted his time delta, and for Singapore (Marina Bay circuit) it was almost 3 seconds off his true time delta. In addittion on the other test races Red Bull Ring and Zandvoort it also over predicted the results. Indicating the model maybe over training slightly.

The model maybe over predicting the lap time delta in the test cases because it does not learn the link between sector times and lap time delta or that it has not learnt the relationships between this years cars and the circuit.

# Project Summary 
<a id="projsummary"></a>

This project aimed to help race teams and fans of Formula1 by analysing and creating models which can predict the results of Qualifying. In this study a number of key insights have been found. Some of which are:

- Overall the models of position and lap time delta (time difference to first place) where able to pick up on the overal trends and predict the results with a R2 square score of 0.5 to 0.6. 
- The best models were regression based as the mean absolute error was on average less than the bin size of the classification models
- The model learned a number of interesting features that helped the model predict the correct results, these included:
    - The variance & maximum RPM (revolution per minute) of the engine on the straight
    - The Maximum speed on the corners
    - The amount of distance a driver spent on the brakes in a given lap
    - Initial Sector 1 times 
    - Speed on the straights

Some of the features the model picked up on were expected, for example speed on the straight, speed in the corners and distance on the brakes because the fast drivers who brake less are likely to do better. However the other featuers such as RPM which was ranked highly for position and lap time delta prediction was un expected. This maybe picking up on the differences between the engines in the cars, cars which are more consistently hitting high RPMS are likely to do better than ones who performance is more erradict and don't reach the high numbers. Therefore, this analysis would recommend teams to investigate how to maximise the performance of the engine and reduce variance in RPM. Further investigation is required in these features and there impact.

The models have a number of limitations as well:

- Both the laptime delta and position prediction models show signs of over training with the test sets doing worse than the training datasets
- The dataset was small for machine learning with 1890 rows and high number of features the models will suffer with dimensionality and the lack of data
- The lap time delta model performed poorly on the "test" test set with mean absolute errors of 0.8 seconds
- As a regression technique was used the models especially for position did not predict the range of outcomes (1-20) and the data needed to be ranked afterwards to match
- Some of the features have outliers which can highly impact the predicted results. 
- The model struggled with unseen circuits, especially on lap time delta. This is likely because the features show more variability between circuits than driver performance

This limiations in the models are likely caused by the lack of data and that the variance in the features may be more related to the circuit that the driver performance. The lack of data is challenging, for example in 2022 the cars and rules were heavily changed so the cars are very different to 2021, this not only impacts which teams and drivers were performing well but likely has a large impact on the features for the model. This would make it extremely hard to accurately predict the results for circuits the model had not been trained on for 2022, as the features maybe very different to what was expected for that circuit.

In adittion, the fact that model struggles with certain GrandPrix's such as Sinapore in 2022 may indicate that a lot of the variance in the features may come from the circuit (Singapore is challenging street circuit) and not the drivers and teams performance.

Overall, this project has learned a number of interesting insights about Formula1 Qualifying, built models that have resonable level of accuracies (the position prediction performing better than lap time delta). In addittion it found out that Max Verstappen conviently has the largest number of missing telemetry days of the field, which may be considered suspect as one of the biggest protagonists. The models faces a number of limitations which made the predictions challenging, however they picked up on interesting features, for initial models they show promise. Further work may be able to reduce the impact of the limitations by generating and engineering more predictive features. In addittion as F1 live publishes more of this data, in the coming years the dataset will increase which will also help the models learn.

There are number of recommendations for future work on this project.

# Recomendations for Further Work
<a id="recommendations"></a>

As discussed the models have a number of limitations related to the predictive power of the features, lack of data and the fact that each year the rules are slightly different hence the cars have different characteristics. It is very difficult to tackle the challenges of limited data without waiting for more data to be collected over the next few years. However more work could be done on feature engineering.

Due to the limited time of the project a number of ideas couldnt be investigated.These include:

- Normalising the input features to the circuit characteristics, this could reduce issues seen especially in lap time delta, where anomalous features related to a particular long circuit or tight circuit impact the result.

- Generating features that look at the past Grand Prix's performance as an indicator of future results, since drivers results show a trend in any given season and using the average result of the last 5 GrandPrix's in that season may help the model predict the results in the next GrandPrix.

- Using automatic feature engineering techniques such as auto encoders, training machines to encode all the data in the lap telemetry into a single vector. The neural network may pick up and learn addittional features that were not thought of in the manual programming feature engineering approach taken in this project.

- Investigating classification models with bin size of 1 for position predictions to avoid the issues of averaging out position predictions. This project showed that classification with a large bin size miss classified because results were likely close to edges of the bins and the model predicted the wrong neighbour bin. However pure regression also struggled since the position result is ranked variable with always integers between 1-20. Therefore a classification with bin size of 1 may help.

- Build models to predict off practice 3 performance features. The challenge of these models is there use the data after the event to predict the result. Therefore limiting there use. The models could be more useful if they were forward predictive. That could be engineered by inputing the feature for certain driver, car circuit combinations based on previous races. Or by using practice 3 data, in practice 3 the last practice session before qualifying the teams often tune the car for a fast hot lap and therefore the data of those laps is usually indicative of who will perform well in qualifying

- Predict results at a lap level not at a race level. Due to the limited number of years of telemetry data, more data can be created if the model predicts the lap times at a lap level and not a race level and therefore the final aggregations from lap to race level are not needed and the dataset would be dramatically bigger.

It is recommended to focus on next on improving the feature engineering through manual and automatic techniques, if those show improved results, look to translate the models to predicting based on practice 3 data. If the approaches on feature engineering fail to improve the models, it is recommended to look at predicting lap performance based on individual lap data. This would teach teams what are the most predictive features at a lap level and can help provide insights on where they should focus to improve performance at different circuits. 
