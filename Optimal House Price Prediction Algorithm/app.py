import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
from dash import html, dcc, callback, Input, Output  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
import traceback  # type: ignore
import warnings
import xgboost as xgb # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.svm import SVR # type: ignore
from dash import dash_table # type: ignore
from joblib import Parallel, delayed # type: ignore


warnings.simplefilter('ignore')

app = dash.Dash(__name__, title = 'House Price Predictor', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP], assets_folder ="assets", assets_url_path="/assets")


train_df = pd.read_csv("train.csv")  
test_df = pd.read_csv("test.csv")  
target_df = pd.read_csv("target.csv")  

# Divide Indian Housing Dataset in Training nd Testing Data
india_df = pd.read_csv('HousePriceIndia.csv')
split_idx = int(0.8 * len(india_df))
train_df_india = india_df.iloc[:split_idx]
test_df_india = india_df.iloc[split_idx:]
target_df_india = test_df_india[['id', 'Price']]
test_df_india.drop(columns = ['Price'], inplace = True, errors= 'ignore')


# Drop columns from all dataframes as they are mostly Null values
for df in [train_df, test_df]:
    df.drop(columns=['Alley', 'Pool QC', 'Fence', 'Misc Feature', 'Fireplace Qu'], inplace=True, errors='ignore')

india_df.drop(columns= ['Date'], inplace = True, errors= 'ignore')

# Identify numerical and categorical columns
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols_india = train_df_india.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_india = test_df_india.select_dtypes(exclude=[np.number]).columns.tolist()

# Ensure 'SalePrice' is not processed in test_df
if "SalePrice" in num_cols:
    num_cols.remove("SalePrice")
if "Price" in num_cols_india:
    num_cols_india.remove("Price")

# Fill missing values in numerical columns
for col in num_cols:
    median_value = train_df[col].median()
    train_df[col].fillna(median_value, inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(median_value, inplace=True)

for col in cat_cols:
    mode_value = train_df[col].mode()[0]
    train_df[col].fillna(mode_value, inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(mode_value, inplace=True)

for col in num_cols_india:
    median_value = train_df_india[col].median()
    train_df_india[col].fillna(median_value, inplace=True)
    if col in test_df_india.columns:
        test_df_india[col].fillna(median_value, inplace=True)

for col in cat_cols_india:
    mode_value = train_df_india[col].mode()[0]
    train_df_india[col].fillna(mode_value, inplace=True)
    if col in test_df_india.columns:
        test_df_india[col].fillna(mode_value, inplace=True)



# Encode categorical columns using LabelEncoder
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].astype("category")

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        test_df[col] = test_df[col].astype("category")

for col in train_df_india.columns:
    if train_df_india[col].dtype == 'object':
        train_df_india[col] = train_df_india[col].astype("category")

for col in test_df_india.columns:
    if test_df_india[col].dtype == 'object':
        test_df_india[col] = test_df_india[col].astype("category")


for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])
    
    # Fill missing values with mode, in categorical columns
    mode_value = train_df[col].mode()[0]
    train_df[col].fillna(mode_value, inplace=True)
    test_df[col].fillna(mode_value, inplace=True)

for col in cat_cols_india:
    le = LabelEncoder()
    train_df_india[col] = le.fit_transform(train_df_india[col])
    
    if col in test_df_india.columns:
        test_df_india[col] = le.transform(test_df_india[col])
    
    # Fill missing values with mode, in categorical columns
    mode_value = train_df_india[col].mode()[0]
    train_df_india[col].fillna(mode_value, inplace=True)
    test_df_india[col].fillna(mode_value, inplace=True)


# Drop duplicate rows if any
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
train_df_india = train_df_india.drop_duplicates()
test_df_india = test_df_india.drop_duplicates()



def no_data_plot():
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": "No Data Available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14}
                }
            ]
        }
    }

def no_table_plot():
    return html.Div(
        "No Data Available",
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontSize': '18px',
            'color': 'gray',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'margin': '10px'
        }
    )



def scatter_plot(df, x_feature, y_feature, dot_color= 'blue', line_color= 'red'):
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x= df[x_feature], 
                y= df[y_feature],
                mode= 'markers',
                marker= dict(color= dot_color, opacity= 0.6)
            )
        )

        X = df[x_feature]
        Y = df[y_feature]
        m, b = np.polyfit(X, Y, 1) 

        trendline_y = m * X + b

        fig.add_trace(
            go.Scatter(
                x= X, 
                y= trendline_y,
                mode='lines',
                line= dict(color= line_color, width= 2, dash= 'dash')
            )
        )

        fig.update_layout(
            title= f'{x_feature} vs {y_feature}',
            xaxis_title= x_feature,
            yaxis_title= y_feature,
            xaxis= dict(showgrid= True),
            yaxis= dict(showgrid= True),
            legend= dict(title= 'Legend'),
            showlegend= False,
            template='plotly_white'
        )

        return fig
    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot()
    

def bar_plot(df, x_feature, y_feature, bar_color= '#77DD77', line_color= 'red'):
    try:
        grouped_df = df.groupby(x_feature)[y_feature].mean().reset_index()
        grouped_df = grouped_df.sort_values(by=x_feature)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=grouped_df[x_feature], 
                y=grouped_df[y_feature], 
                marker=dict(color= bar_color, opacity=0.7),
                name='Average SalePrice'
            )
        )

        window_size = 3
        y_smooth = np.convolve(grouped_df[y_feature], np.ones(window_size)/window_size, mode='same')

        fig.add_trace(
            go.Scatter(
                x=grouped_df[x_feature],
                y=y_smooth,
                mode='lines',
                line=dict(color= line_color, width=2, dash='dash'),
                name='Smoothed Line'
            )
        )

        fig.update_layout(
            title=f'Average {y_feature} by {x_feature}',
            xaxis_title=x_feature,
            yaxis_title=f'Average {y_feature}',
            xaxis=dict(type='category', showgrid=False),
            yaxis=dict(showgrid=True),
            template='plotly_white',
            showlegend= False
        )

        return fig
    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot()
    

def correlation_heatmap(df=None, dataset = 'ames'):
    try:
        if dataset == 'ames':
            df.drop(columns= ['Order'], inplace = True, errors= 'ignore')
        elif dataset == 'india':
            df.drop(columns= ['id'], inplace = True, errors= 'ignore')
        else:
            return no_data_plot()
        numeric_df = df.select_dtypes(include=[np.number])
        cor_matrix = numeric_df.corr()

        cor_matrix = cor_matrix.dropna(how= 'all', axis= 0).dropna(how= 'all', axis= 1)

        fig = go.Figure(
            data= go.Heatmap(
                z= cor_matrix.values,
                x= cor_matrix.columns,
                y= cor_matrix.columns,
                colorscale= "RdBu",
                zmin= -1,
                zmax= 1,
                colorbar= dict(title= "Correlation")
            )
        )

        fig.update_layout(
            title= "Correlation Heatmap",
            xaxis= dict(tickangle= -60),
            height= 600,
            width= 800
        )

        return fig

    except Exception as e:
        return no_data_plot()
    

def linear_regression_model(train_df, test_df, target_df, features, target_col='SalePrice', dataset='ames'):
    try:
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]

        # ---------- Manual Hyperparameter Selection ----------
        # Rule: Use fit_intercept only if the mean of y_train is significantly different from 0
        fit_intercept = abs(y_train.mean()) > 1e-3

        # Rule: Enable positive weights if all features are positively correlated with the target
        positive_corr = all(train_df[feature].corr(y_train) > 0.2 for feature in features)
        positive = True if positive_corr else False

        model = LinearRegression(fit_intercept=fit_intercept, positive=positive)
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        test_df['Predicted'] = test_preds

        if dataset == 'ames':
            merged_df = test_df.merge(target_df, on='Order', how='left')
        elif dataset == 'india':
            merged_df = test_df.merge(target_df, on='id', how='left')
        else:
            return no_data_plot(), 0.00, 0.00, 0.00, 0.00

        actual = merged_df[target_col].values
        predicted = merged_df['Predicted'].values
        valid_mask = ~np.isnan(actual)

        actual = actual[valid_mask]
        predicted = predicted[valid_mask]

        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if len(actual) > 0 else 0.00
        rmse = np.sqrt(np.mean((actual - predicted) ** 2)) if len(actual) > 0 else 0.00

        n = len(actual)
        p = len(features)
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else 0.00

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, 
            y=predicted, 
            mode='markers', 
            marker=dict(color='#FF8C00', opacity=0.7), 
            name='Predictions'
        ))

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines', 
            line=dict(color='#4682B4', dash='dash'), 
            name='Perfect Fit'
        ))

        fig.update_layout(
            title=f"Predicted vs Actual SalePrice (Linear Regression)",
            xaxis_title="Actual SalePrice",
            yaxis_title="Predicted SalePrice",
            template='plotly_white',
            showlegend=True
        )

        return fig, mape, adjusted_r2, rmse, r2

    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 0.00, 0.00, 0.00, 0.00
    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 0.00, 0.00, 0.00, 0.00
    

def random_forest_model(train_df, test_df, target_df, features, target_col='SalePrice', dataset = 'ames'):
    try:
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]

        # ==== Manual Hyperparameter Tuning Based on Data Stats ====

        # 1. Number of trees
        n_samples = len(X_train)
        n_estimators = 100 if n_samples > 1000 else 50

        # 2. Max depth based on feature count (limit overfitting)
        max_depth = 10 if len(features) > 10 else None

        # 3. Minimum samples per leaf: fewer for large datasets, more to avoid overfitting on small
        min_samples_leaf = 1 if n_samples > 1000 else 2

        # 4. Max features: square root heuristic
        max_features = 'sqrt' if len(features) > 5 else None

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)

        test_df['Predicted'] = test_preds
        if dataset == 'ames':
            merged_df = test_df.merge(target_df, on='Order', how='left')
        elif dataset == 'india':
            merged_df = test_df.merge(target_df, on='id', how='left')
        else:
            return no_data_plot(), 0.00, 0.00, 0.00, 0.00

        actual = merged_df[target_col].values
        predicted = merged_df['Predicted'].values

        mape = np.mean(np.abs((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)]) / actual[~np.isnan(actual)])) * 100
        if mape is None or np.isnan(mape):
            mape = 0.00

        rmse = np.sqrt(np.mean((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)]) ** 2))
        if rmse is None or np.isnan(rmse):
            rmse = 0.00

        n = len(actual[~np.isnan(actual)])
        p = len(features)  
        ss_total = np.sum((actual[~np.isnan(actual)] - np.mean(actual[~np.isnan(actual)]))**2)
        ss_residual = np.sum((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)])**2)
        r2 = 1 - (ss_residual / ss_total)

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else None 
        if adjusted_r2 is None or np.isnan(adjusted_r2):
            adjusted_r2 = 0.00

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual, 
            y=predicted, 
            mode='markers', 
            marker=dict(color='#FF8C00', opacity=0.7), 
            name='Predictions'
        ))

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines', 
            line=dict(color='#4682B4', dash='dash'), 
            name='Perfect Fit'
        ))

        fig.update_layout(
            title="Predicted vs Actual SalePrice (Random Forest)",
            xaxis_title="Actual SalePrice",
            yaxis_title="Predicted SalePrice",
            template='plotly_white',
            showlegend=True
        )

        return fig, mape, adjusted_r2, rmse, r2

    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 0.00, 0.00, 0.00, 0.00    


def xgboost_model(train_df, test_df, target_df, features, target_col='SalePrice', dataset='ames'):
    try:
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        max_depth = 6 if n_features <= 10 else 10
        min_child_weight = 20 if n_samples < 1000 else 50 if n_samples < 5000 else 100
        gamma = 0.1 if n_samples < 1000 else 0.01 if n_samples < 10000 else 0.001

        model = xgb.XGBRegressor(
            objective= 'reg:squarederror',
            gamma= gamma,
            max_depth= max_depth,
            min_child_weight= min_child_weight,
            enable_categorical= True,
            random_state= 42
        )

        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_df['Predicted'] = test_preds

        if dataset == 'ames':
            merged_df = test_df.merge(target_df, on='Order', how='left')
        elif dataset == 'india':
            merged_df = test_df.merge(target_df, on='id', how='left')
        else:
            return no_data_plot(), 0.00, 0.00, 0.00, 0.00

        actual = merged_df[target_col].values
        predicted = merged_df['Predicted'].values

        mask = ~np.isnan(actual)
        actual = actual[mask]
        predicted = predicted[mask]

        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if len(actual) else 0.00
        rmse = np.sqrt(np.mean((actual - predicted) ** 2)) if len(actual) else 0.00

        n = len(actual)
        p = len(features)
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total else 0.00
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else 0.00

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            marker=dict(color='#FF8C00', opacity=0.7),
            name='Predictions'
        ))

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#4682B4', dash='dash'),
            name='Perfect Fit'
        ))

        fig.update_layout(
            title="Predicted vs Actual SalePrice (XGBoost)",
            xaxis_title="Actual SalePrice",
            yaxis_title="Predicted SalePrice",
            template='plotly_white',
            showlegend=True
        )

        return fig, mape, adjusted_r2, rmse, r2

    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 0.00, 0.00, 0.00, 0.00



def svm_model(train_df, test_df, target_df, features, target_col='SalePrice', dataset='ames'):
    try:
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        # Dynamic SVM hyperparameters based on data scale
        if n_samples < 1000:
            C = 10
            gamma = 0.5
        elif n_samples < 5000:
            C = 5
            gamma = 0.1
        else:
            C = 1
            gamma = 'scale'

        model = SVR(C= C, gamma= gamma)
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        test_df['Predicted'] = test_preds

        if dataset == 'ames':
            merged_df = test_df.merge(target_df, on='Order', how='left')
        elif dataset == 'india':
            merged_df = test_df.merge(target_df, on='id', how='left')
        else:
            return no_data_plot(), 0.00, 0.00, 0.00, 0.00

        actual = merged_df[target_col].values
        predicted = merged_df['Predicted'].values

        mape = np.mean(np.abs((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)]) / actual[~np.isnan(actual)])) * 100
        if mape is None or np.isnan(mape):
            mape = 0.00

        rmse = np.sqrt(np.mean((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)]) ** 2))
        if rmse is None or np.isnan(rmse):
            rmse = 0.00

        n = len(actual[~np.isnan(actual)])  
        p = len(features)  
        ss_total = np.sum((actual[~np.isnan(actual)] - np.mean(actual[~np.isnan(actual)]))**2)
        ss_residual = np.sum((actual[~np.isnan(actual)] - predicted[~np.isnan(actual)])**2)
        r2 = 1 - (ss_residual / ss_total)

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else 0.00
        if adjusted_r2 is None or np.isnan(adjusted_r2):
            adjusted_r2 = 0.00

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=predicted,
            mode='markers',
            marker=dict(color='#FF8C00', opacity=0.7),
            name='Predictions'
        ))

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#4682B4', dash='dash'),
            name='Perfect Fit'
        ))

        fig.update_layout(
            title="Predicted vs Actual SalePrice (SVM)",
            xaxis_title="Actual SalePrice",
            yaxis_title="Predicted SalePrice",
            template='plotly_white',
            showlegend=True
        )

        return fig, mape, adjusted_r2, rmse, r2

    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 0.00, 0.00, 0.00, 0.00
    

app.layout = html.Div([
    html.Div(
        children = [
            html.Div(
                className='container-fluid',
                children = [
                    dcc.Location(id='url', refresh=False),
                    dcc.Store(id='trained-models-store', data={}),
                    html.Div([
                        html.Hr(
                            style={
                                "borderWidth": "0.3vh",
                                "width": "100%",
                                "borderColor": "#00589c",
                                "opacity": "unset",
                            }
                        ),
                        html.H1('House Price Prediction Using XGBoost', style={'textAlign': 'center', 'font-size': 28}),
                        html.Hr(
                            style={
                                "borderWidth": "0.1vh",
                                "width": "100%",
                                "borderColor": "#A9A9A9",
                                "opacity": "unset",
                            }
                        ),
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label('Dataset', style={'font-size': 14}),
                                    dcc.Dropdown(
                                        id='dataset',
                                        options=[{'label': 'AMES Iowa', 'value': 'ames'}, {'label' : 'Indian Housing', 'value' : 'india'}],
                                        placeholder= 'Select a Dataset',
                                        value= 'ames',
                                        style={'font-size': 15, 'margin': 10}
                                    )
                                ],
                                className = 'col-sm-2'
                            ),
                        ]
                    ),

                    dbc.Tabs(
                        [
                            dbc.Tab(
                                label="Analysis", 
                                tab_id="analysis-tab", 
                                label_style={"color": "#00AEF9"}, 
                                activeTabClassName="fw-bold fst-italic",
                                active_label_style={"color": "#00589c"},

                                children = [
                                    html.Hr(
                                        style={
                                            "borderWidth": "0.3vh",
                                            "width": "100%",
                                            "borderColor": "#00589c",
                                            "opacity": "unset",
                                        }
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader([ 
                                                        html.I(className= "fa fa-info-circle", style = {'color':'green'}),
                                                        html.H5(' Exploratory Data Analysis', style = {'display':'inline'})
                                                    ], style = {"font-size": 14}),

                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                html.Hr(
                                                                    style = {
                                                                        "borderWidth": "0.3vh",
                                                                        "width": "100%",
                                                                        "borderColor": "#00589c",
                                                                        "opacity": "unset",
                                                                    }
                                                                ),
                                                            ),

                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children = [
                                                                                    dbc.Card([
                                                                                        dbc.CardHeader([
                                                                                            html.I(className="fa fa-th", style={'color': 'red', 'margin-right': '8px'}),
                                                                                            html.H5(' Feature Correlation Matrix', style={'display': 'inline'})
                                                                                        ], style={"font-size": 14}),

                                                                                        dbc.CardBody(
                                                                                            [
                                                                                               dcc.Graph(
                                                                                                    id = 'corrplot',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                ) 
                                                                                            ]
                                                                                        )
                                                                                    ])
                                                                                ]
                                                                            )
                                                                        ], className='col-sm-8'
                                                                    )
                                                                ], justify = 'center'
                                                            ),

                                                            html.Hr(
                                                                style={
                                                                    "borderWidth": "0.3vh",
                                                                    "width": "100%",
                                                                    "borderColor": "#00589c",
                                                                    "opacity": "unset",
                                                                }
                                                            ),

                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig1',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),

                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig2',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),                                            
                                                                ]
                                                            ),

                                                            html.Hr(
                                                                style={
                                                                    "borderWidth": "0.3vh",
                                                                    "width": "100%",
                                                                    "borderColor": "#00589c",
                                                                    "opacity": "unset",
                                                                }
                                                            ),

                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig3',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),

                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig4',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),
                                                                    
                                                                ]
                                                            ),

                                                            html.Hr(
                                                                style={
                                                                    "borderWidth": "0.3vh",
                                                                    "width": "100%",
                                                                    "borderColor": "#00589c",
                                                                    "opacity": "unset",
                                                                }
                                                            ),

                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig5',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),

                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                children= [
                                                                                    dbc.Card([
                                                                                        dbc.CardBody(
                                                                                            [
                                                                                                dcc.Graph(
                                                                                                    id = 'fig6',
                                                                                                    className = 'h-100',
                                                                                                    config = {'displaylogo': False}
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                    ]),
                                                                                ]
                                                                            ),
                                                                        ], className='col-sm-6'
                                                                    ),
                                                                    
                                                                ]
                                                            ),

                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            

                            dbc.Tab(
                                label="Prediction", 
                                tab_id="prediction-tab", 
                                label_style={"color": "#00AEF9"}, 
                                activeTabClassName="fw-bold fst-italic",
                                active_label_style={"color": "#00589c"},

                                children = [
                                    html.Hr(
                                        style={
                                            "borderWidth": "0.3vh",
                                            "width": "100%",
                                            "borderColor": "#00589c",
                                            "opacity": "unset",
                                        }
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label('Features', style={'font-size': 14}),
                                                    dcc.Dropdown(
                                                        id='features',
                                                        options=[{'label': col, 'value': col} for col in test_df.columns if col != 'Order'],
                                                        placeholder='All',
                                                        value=['Gr Liv Area', 'Total Bsmt SF', 'Garage Cars', 'Overall Qual', 'Garage Area', '1st Flr SF'],
                                                        multi=True,
                                                        style={'font-size': 15, 'margin': 10}
                                                    )
                                                ],
                                                className = 'col-sm-4'
                                            ),

                                            dbc.Col(
                                                [
                                                    html.Label('Model', style={'font-size': 14}),
                                                    dcc.Dropdown(
                                                        id='model',
                                                        options=[
                                                            {'label': 'XGBoost', 'value': 'xgb'},
                                                            {'label': 'Linear Regression', 'value': 'lr'},
                                                            {'label': 'Random Forest Regression', 'value': 'rfr'},
                                                            {'label': 'Support Vector Regressor', 'value': 'svm'}
                                                        ],
                                                        placeholder='Select a Model',
                                                        value='xgb',
                                                        style={'font-size': 13, 'margin-top': '20px'}
                                                    )
                                                ],
                                                className = 'col-sm-2'
                                            ),

                                            dbc.Col(
                                                dbc.Button(
                                                    "Reset", 
                                                    id="reset-btn", 
                                                    color="secondary", 
                                                    size="sm", 
                                                    style={'width': '100px', 'margin-top': '50px', 'margin-left': '25px'}
                                                ),
                                                className = 'col-sm-2'
                                            )
                                        ],
                                    ),


                                    html.Hr(
                                        style={
                                            "borderWidth": "0.3vh",
                                            "width": "100%",
                                            "borderColor": "#00589c",
                                            "opacity": "unset",
                                        }
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader([ 
                                                        html.I(className= "fa fa-info-circle", style = {'color':'green'}),
                                                        html.H5(' Forecast', style = {'display':'inline'})
                                                    ], style = {"font-size": 14}),

                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dcc.Graph(
                                                                        id = 'xgboostPrediction',
                                                                        className = 'h-100',
                                                                        config = {'displaylogo': False}
                                                                    )
                                                                ]
                                                            ),
                                                            
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                id = 'mape',
                                                                                style={'textAlign': 'center', 'fontSize': 16, 'marginTop': '10px'},
                                                                            )
                                                                        ], className = 'col-sm-2'
                                                                    ),
                                                                    
                                                                    dbc.Col(
                                                                        [    
                                                                            html.Div(
                                                                                id = 'accuracy',
                                                                                style={'textAlign': 'center', 'fontSize': 16, 'marginTop': '10px'},
                                                                            )
                                                                        ], className = 'col-sm-2'
                                                                    ),

                                                                    dbc.Col(
                                                                        [    
                                                                            html.Div(
                                                                                id = 'r2',
                                                                                style={'textAlign': 'center', 'fontSize': 16, 'marginTop': '10px'},
                                                                            )
                                                                        ], className = 'col-sm-2'
                                                                    ),

                                                                    dbc.Col(
                                                                        [    
                                                                            html.Div(
                                                                                id = 'adj-r2',
                                                                                style={'textAlign': 'center', 'fontSize': 16, 'marginTop': '10px'},
                                                                            )
                                                                        ], className = 'col-sm-2'
                                                                    ),

                                                                    dbc.Col(
                                                                        [    
                                                                            html.Div(
                                                                                id = 'rmse',
                                                                                style={'textAlign': 'center', 'fontSize': 16, 'marginTop': '10px'},
                                                                            )
                                                                        ], className = 'col-sm-2'
                                                                    ),
                                                                ], justify = 'center'
                                                            )
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),

                                    html.Hr(
                                        style={
                                            "borderWidth": "0.3vh",
                                            "width": "100%",
                                            "borderColor": "#00589c",
                                            "opacity": "unset",
                                        }
                                    ),

                                    dbc.Row(
                                        [
                                           dbc.Col(
                                               [
                                                   dbc.Card(
                                                        [
                                                            dbc.CardHeader([ 
                                                                    html.I(className= "fa fa-info-circle", style = {'color':'green'}),
                                                                    html.H5(' Statistical Summary', style = {'display':'inline'})
                                                                ], style = {"font-size": 14}),

                                                            dbc.CardBody(
                                                                [
                                                                    html.Div(
                                                                        id='stats-table',
                                                                        style={'textAlign': 'center', 'fontSize': 14},
                                                                    )
                                                                ]
                                                            )       
                                                        ]
                                                    ) 
                                               ], className = 'col-sm-6', 
                                           )
                                        ], justify = 'center'
                                    )
                                ]
                            ),
                        ]
                    )
                ]
            )
        ]
    )
])



@app.callback(
    Output('corrplot', 'figure'),
    Output('fig1', 'figure'),
    Output('fig4', 'figure'),
    Output('fig3', 'figure'),
    Output('fig2', 'figure'),
    Output('fig5', 'figure'),
    Output('fig6', 'figure'),
    Input('dataset', 'value')
)
def update_layout_analysis(dataset):
    try:
        if dataset == 'ames':
            return correlation_heatmap(df= train_df.copy()), scatter_plot(df= train_df.copy(), x_feature= 'Gr Liv Area', y_feature= 'SalePrice', dot_color= '#77B5FE', line_color= 'red'), scatter_plot(df= train_df.copy(), x_feature= 'Total Bsmt SF', y_feature= 'SalePrice', line_color= '#8A2BE2', dot_color= '#FFD700'), bar_plot(df= train_df.copy(), x_feature= 'Garage Cars', y_feature= 'SalePrice', bar_color= '#77DD77', line_color= 'red'), bar_plot(df= train_df.copy(), x_feature= 'Overall Qual', y_feature= 'SalePrice', bar_color= '#6FAE75', line_color= '#FF4500'), scatter_plot(df= train_df.copy(), x_feature= 'Garage Area', y_feature= 'SalePrice', dot_color= '#FF7F50', line_color= '#0B3D91'), scatter_plot(df= train_df.copy(), x_feature= '1st Flr SF', y_feature= 'SalePrice', dot_color= '#7EB6E6', line_color= 'red')

        elif dataset == 'india':
            return correlation_heatmap(df = train_df_india.copy(), dataset= 'india'), scatter_plot(df= train_df_india.copy(), x_feature= 'living area', y_feature= 'Price', dot_color= '#77B5FE', line_color= 'red'), scatter_plot(df= train_df_india.copy(), x_feature= 'Area of the basement', y_feature= 'Price', line_color= '#8A2BE2', dot_color= '#FFD700'),  bar_plot(df= train_df_india.copy(), x_feature= 'number of bathrooms', y_feature= 'Price', bar_color= '#77DD77', line_color= 'red'), bar_plot(df= train_df_india.copy(), x_feature= 'condition of the house', y_feature= 'Price', bar_color= '#6FAE75', line_color= '#FF4500'), scatter_plot(df= train_df_india.copy(), x_feature= 'Area of the house(excluding basement)', y_feature= 'Price', dot_color= '#FF7F50', line_color= '#0B3D91'), scatter_plot(df= train_df_india.copy(), x_feature= 'living_area_renov', y_feature= 'Price', dot_color= '#7EB6E6', line_color= 'red')

        return no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot()
  
    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot(), no_data_plot()
    

@app.callback(
    Output('trained-models-store', 'data'),    
    Output('stats-table', 'children'),
    Input('features', 'value'),
    Input('dataset', 'value'),
)
def train_models(features= None, dataset= None):
    try:
        if dataset is None:
            return {}, no_table_plot()
        
        if dataset == 'ames':
            if features is None or len(features) == 0:
                features = [col for col in test_df.columns if col != 'Order' and col != 'Predicted']

            features = list(features)

            def train_xgb():
                return xgboost_model(train_df.copy(), test_df.copy(), target_df.copy(), features, 'SalePrice')

            def train_lr():
                return linear_regression_model(train_df.copy(), test_df.copy(), target_df.copy(), features, 'SalePrice')

            def train_rfr():
                return random_forest_model(train_df.copy(), test_df.copy(), target_df.copy(), features, 'SalePrice')

            def train_svm():
                return svm_model(train_df.copy(), test_df.copy(), target_df.copy(), features, 'SalePrice')

            results = Parallel(n_jobs=-1)(delayed(func)() for func in [train_xgb, train_lr, train_rfr, train_svm])

            (xgb_fig, xgb_mape, xgb_adj_r2, xgb_rmse, xgb_r2,
            lr_fig, lr_mape, lr_adj_r2, lr_rmse, lr_r2,
            rfr_fig, rfr_mape, rfr_adj_r2, rfr_rmse, rfr_r2,
            svm_fig, svm_mape, svm_adj_r2, svm_rmse, svm_r2) = [item for res in results for item in res]

            xgb_acc = 100 - xgb_mape
            lr_acc = 100 - lr_mape
            rfr_acc = 100 - rfr_mape
            svm_acc = 100 - svm_mape

            models_data = {
                'xgb': {
                    'fig': xgb_fig,
                    'mape': xgb_mape, 'acc': xgb_acc,
                    'r2': xgb_r2, 'adj_r2': xgb_adj_r2, 'rmse': xgb_rmse
                },
                'lr': {
                    'fig': lr_fig,
                    'mape': lr_mape, 'acc': lr_acc,
                    'r2': lr_r2, 'adj_r2': lr_adj_r2, 'rmse': lr_rmse
                },
                'rfr': {
                    'fig': rfr_fig,
                    'mape': rfr_mape, 'acc': rfr_acc,
                    'r2': rfr_r2, 'adj_r2': rfr_adj_r2, 'rmse': rfr_rmse
                },
                'svm': {
                    'fig': svm_fig,
                    'mape': svm_mape, 'acc': svm_acc,
                    'r2': svm_r2, 'adj_r2': svm_adj_r2, 'rmse': svm_rmse
                }
            }

            metrics_data = [
                {
                    "Model": "XGBoost",
                    "MAPE": f"{xgb_mape:.2f}%",
                    "Accuracy": f"{xgb_acc:.2f}%",
                    "Adjusted R\u00b2": f"{xgb_adj_r2:.4f}",
                    "RMSE": f"{xgb_rmse:.4f}",
                    "R\u00b2": f"{xgb_r2:.4f}"
                },
                {
                    "Model": "Linear Regression",
                    "MAPE": f"{lr_mape:.2f}%",
                    "Accuracy": f"{lr_acc:.2f}%",
                    "Adjusted R\u00b2": f"{lr_adj_r2:.4f}",
                    "RMSE": f"{lr_rmse:.4f}",
                    "R\u00b2": f"{lr_r2:.4f}"
                },
                {
                    "Model": "Random Forest",
                    "MAPE": f"{rfr_mape:.2f}%",
                    "Accuracy": f"{rfr_acc:.2f}%",
                    "Adjusted R\u00b2": f"{rfr_adj_r2:.4f}",
                    "RMSE": f"{rfr_rmse:.4f}",
                    "R\u00b2": f"{rfr_r2:.4f}"
                },
                {
                    "Model": "SVM",
                    "MAPE": f"{svm_mape:.2f}%",
                    "Accuracy": f"{svm_acc:.2f}%",
                    "Adjusted R\u00b2": f"{svm_adj_r2:.4f}",
                    "RMSE": f"{svm_rmse:.4f}",
                    "R\u00b2": f"{svm_r2:.4f}"
                }
            ]

            metrics_table = dash_table.DataTable(
                id='metrics-table',
                columns=[
                    {"name": "Model", "id": "Model"},
                    {"name": "MAPE", "id": "MAPE"},
                    {"name": "Accuracy", "id": "Accuracy"},
                    {"name": "R\u00b2", "id": "R\u00b2"},
                    {"name": "Adjusted R\u00b2", "id": "Adjusted R\u00b2"},
                    {"name": "RMSE", "id": "RMSE"}
                ],
                data=metrics_data,
                style_table={'overflowX': 'auto', 'width': '100%', 'margin': '20px 0'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'font-family': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': '#4682B4',
                    'fontWeight': 'bold',
                    'color': 'white',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'row_index': 'even'}, 'backgroundColor': '#ffffff'}
                ],
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_size=10,
            )
        
        elif dataset == 'india':
            if features is None or len(features) == 0:
                features = [col for col in test_df_india.columns if col != 'id' and col != 'Price']

            features = list(features)

            def train_xgb():
                return xgboost_model(train_df_india.copy(), test_df_india.copy(), target_df_india.copy(), features, 'Price', dataset= 'india')

            def train_lr():
                return linear_regression_model(train_df_india.copy(), test_df_india.copy(), target_df_india.copy(), features, 'Price', dataset= 'india')

            def train_rfr():
                return random_forest_model(train_df_india.copy(), test_df_india.copy(), target_df_india.copy(), features, 'Price', dataset= 'india')

            def train_svm():
                return svm_model(train_df_india.copy(), test_df_india.copy(), target_df_india.copy(), features, 'Price', dataset= 'india')

            results = Parallel(n_jobs=-1)(delayed(func)() for func in [train_xgb, train_lr, train_rfr, train_svm])

            (xgb_fig, xgb_mape, xgb_adj_r2, xgb_rmse, xgb_r2,
            lr_fig, lr_mape, lr_adj_r2, lr_rmse, lr_r2,
            rfr_fig, rfr_mape, rfr_adj_r2, rfr_rmse, rfr_r2,
            svm_fig, svm_mape, svm_adj_r2, svm_rmse, svm_r2) = [item for res in results for item in res]

            xgb_acc = 100 - xgb_mape
            lr_acc = 100 - lr_mape
            rfr_acc = 100 - rfr_mape
            svm_acc = 100 - svm_mape

            models_data = {
                'xgb': {
                    'fig': xgb_fig,
                    'mape': xgb_mape, 'acc': xgb_acc,
                    'r2': xgb_r2, 'adj_r2': xgb_adj_r2, 'rmse': xgb_rmse
                },
                'lr': {
                    'fig': lr_fig,
                    'mape': lr_mape, 'acc': lr_acc,
                    'r2': lr_r2, 'adj_r2': lr_adj_r2, 'rmse': lr_rmse
                },
                'rfr': {
                    'fig': rfr_fig,
                    'mape': rfr_mape, 'acc': rfr_acc,
                    'r2': rfr_r2, 'adj_r2': rfr_adj_r2, 'rmse': rfr_rmse
                },
                'svm': {
                    'fig': svm_fig,
                    'mape': svm_mape, 'acc': svm_acc,
                    'r2': svm_r2, 'adj_r2': svm_adj_r2, 'rmse': svm_rmse
                },
            }

            metrics_data = [
                {
                    "Model": "XGBoost",
                    "MAPE": f"{xgb_mape:.2f}%",
                    "Accuracy": f"{xgb_acc:.2f}%",
                    "Adjusted R\u00b2": f"{xgb_adj_r2:.4f}",
                    "RMSE": f"{xgb_rmse:.4f}",
                    "R\u00b2": f"{xgb_r2:.4f}"
                },
                {
                    "Model": "Linear Regression",
                    "MAPE": f"{lr_mape:.2f}%",
                    "Accuracy": f"{lr_acc:.2f}%",
                    "Adjusted R\u00b2": f"{lr_adj_r2:.4f}",
                    "RMSE": f"{lr_rmse:.4f}",
                    "R\u00b2": f"{lr_r2:.4f}"
                },
                {
                    "Model": "Random Forest",
                    "MAPE": f"{rfr_mape:.2f}%",
                    "Accuracy": f"{rfr_acc:.2f}%",
                    "Adjusted R\u00b2": f"{rfr_adj_r2:.4f}",
                    "RMSE": f"{rfr_rmse:.4f}",
                    "R\u00b2": f"{rfr_r2:.4f}"
                },
                {
                    "Model": "SVM",
                    "MAPE": f"{svm_mape:.2f}%",
                    "Accuracy": f"{svm_acc:.2f}%",
                    "Adjusted R\u00b2": f"{svm_adj_r2:.4f}",
                    "RMSE": f"{svm_rmse:.4f}",
                    "R\u00b2": f"{svm_r2:.4f}"
                },
            ]

            metrics_table = dash_table.DataTable(
                id='metrics-table',
                columns=[
                    {"name": "Model", "id": "Model"},
                    {"name": "MAPE", "id": "MAPE"},
                    {"name": "Accuracy", "id": "Accuracy"},
                    {"name": "R\u00b2", "id": "R\u00b2"},
                    {"name": "Adjusted R\u00b2", "id": "Adjusted R\u00b2"},
                    {"name": "RMSE", "id": "RMSE"}
                ],
                data=metrics_data,
                style_table={'overflowX': 'auto', 'width': '100%', 'margin': '20px 0'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'font-family': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': '#4682B4',
                    'fontWeight': 'bold',
                    'color': 'white',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'row_index': 'even'}, 'backgroundColor': '#ffffff'}
                ],
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_size=10,
            )

        return models_data, metrics_table

    except Exception as e:
        print(traceback.format_exc())
        return {}, no_table_plot()



@app.callback(
    Output('xgboostPrediction', 'figure'),
    Output('mape', 'children'),
    Output('accuracy', 'children'),
    Output('r2', 'children'),
    Output('adj-r2', 'children'),
    Output('rmse', 'children'),
    Input('model', 'value'),
    Input('trained-models-store', 'data'),
)
def update_figure(model= None, models_data= None):
    try:        
        if models_data is None or model not in models_data or len(models_data) == 0:
            return no_data_plot(), 'MAPE : --.--%', 'Accuracy : --.--%', 'R\u00b2 : --.--', 'Adjusted R\u00b2 : --.--', 'RMSE : --.--'

        model_data = models_data[model]

        fig = model_data['fig']
        mape = model_data['mape']
        acc = model_data['acc']
        r2 = model_data['r2']
        adj_r2 = model_data['adj_r2']
        rmse = model_data['rmse']

        return (
            fig,
            f'MAPE : {mape:.2f}%',
            f'Accuracy : {acc:.2f}%',
            f'R\u00b2 : {r2:.4f}',
            f'Adjusted R\u00b2 : {adj_r2:.4f}',
            f'RMSE : {rmse:,.2f}'
        )

    except Exception as e:
        print(traceback.format_exc())
        return no_data_plot(), 'MAPE : --.--%', 'Accuracy : --.--%', 'R\u00b2 : --.--', 'Adjusted R\u00b2 : --.--', 'RMSE : --.--'
    

@app.callback(
    Output('features', 'value'),
    Output('model', 'value'),
    Input('reset-btn', 'n_clicks'),
    Input('dataset', 'value'),
    prevent_initial_call=True
)
def reset_dropdown(n_clicks, dataset= None):
    try:
        if dataset is None:
            return [], None
        if dataset == 'ames':
            return ['Gr Liv Area', 'Total Bsmt SF', 'Garage Cars', 'Overall Qual', 'Garage Area', '1st Flr SF'], 'xgb'
        elif dataset == 'india':
            return ['living area', 'Area of the basement', 'number of bathrooms', 'condition of the house', 'Area of the house(excluding basement)', 'living_area_renov'], 'xgb'
        return [], None
    except Exception as e:
        print(traceback.format_exc())
        return [], None
    

@app.callback(
    Output('features', 'options'),
    Input('dataset', 'value')
)
def reset_dropdown_options(dataset):
    try:
        if dataset is None:
            return []
        if dataset == 'ames':
            return [{'label': col, 'value': col} for col in test_df.columns if col != 'Order']
        elif dataset == 'india':
            return [{'label': col, 'value': col} for col in test_df_india.columns if col != 'id']
        return []
    except Exception as e:
        print(traceback.format_exc())
        return []

if __name__ == '__main__':
    app.run(debug= True, port= 9052)