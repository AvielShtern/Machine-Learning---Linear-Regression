######################################################################################
# FILE: EX2.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 2 - Linear Regression 2021
######################################################################################
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    """run end plot the result for EX2"""
    path_for_data = '/Users/avielshtern/Desktop/semb/iml/IML.HUJI-master/data/kc_house_data (1).csv'
    design_matrix, response_vector = load_data(path_for_data)
    putting_it_all_together_1(design_matrix, response_vector)
    putting_it_all_together_2(design_matrix, response_vector)
    feature_evaluation(design_matrix, response_vector)


def fit_linear_regression(design_matrix, response_vector):
    """return the coefficients_vector and singular values of X"""
    s = np.linalg.svd(design_matrix, compute_uv=False)
    pseudo_inverse_design_matrix = np.linalg.pinv(design_matrix)
    w = pseudo_inverse_design_matrix @ response_vector
    return w, s


def predict(design_matrix, coefficients_vector):
    """returns a numpy array with the predicted values by the model"""
    return design_matrix @ coefficients_vector


def mse(response_vector, prediction_vector):
    """returns the MSE over the received samples"""
    return np.power(response_vector - prediction_vector, 2).mean()


def load_data(path_for_data):
    """Given a path to the data file. Cleans it of bad information. And in the case of categorical variables adds them.
    At the end returns design_matrix and the response_vector (price)"""
    lest_age = 2016
    last_date = datetime.strptime('20160128', '%Y%m%d')

    data1 = pd.read_csv(path_for_data)
    data = clean_data(data1)  # clean the data from nan, negative price etc.

    response_vector = data['price'].to_numpy().reshape(-1, 1)  # y.shape = (M,1)

    # 1. Make the sale date the number of days between the sale and a specific date (for all dates) - there may be an
    #    effect on the price.
    # 2. Make "Year of built" and "Year of Last Renovation" one feature of "Apartment Age"
    # 3. Turn the longitude and latitude into coordinates in space. (Houses that are close to each other will be close
    #    in all 3 coordinates)
    data['date'] = [(last_date - datetime.strptime(x[:8], '%Y%m%d')).days for x in data['date'].values]
    data['age'] = [lest_age - x[0] if x[1] == 0 else lest_age - x[1] for x in
                   data[['yr_built', 'yr_renovated']].values]
    data['x_coordinate'] = [math.cos(x[0]) * math.cos(x[1]) for x in data[['lat', 'long']].values]
    data['y_coordinate'] = [math.cos(x[0]) * math.sin(x[1]) for x in data[['lat', 'long']].values]
    data['z_coordinate'] = [math.sin(x) for x in data[['lat']].values]

    dummies_zipcode = pd.get_dummies(data['zipcode'])  # this is categorical feature (70 options)

    # id irrelevant. We took care of everything else
    data.drop(['id', 'long', 'lat', 'yr_built', 'yr_renovated', 'zipcode', 'price'], axis='columns', inplace=True)

    update_data = pd.concat([data, dummies_zipcode], axis=1)

    design_matrix_before_ones = update_data.to_numpy().astype(float)

    # Add the bias column to the matrix
    design_matrix_after_ones = np.concatenate(
        (np.ones((design_matrix_before_ones.shape[0], 1)), design_matrix_before_ones), axis=1)
    return design_matrix_after_ones, response_vector


def clean_data(data):
    """Deleted corrupted data. Apartments that have neither rooms nor bathrooms (neither). Any negative numerical value
     (except latitude and longitude)"""
    data.dropna(inplace=True)
    for feature in data:
        if ((feature != 'lat') and (feature != 'long') and (feature != 'date')):
            data.drop(data[(data[feature] < 0)].index, inplace=True)
    data.drop(data[(data['price'] == 0)].index, inplace=True)
    data.drop(data[(data['bedrooms'] == 0) & (data['bathrooms'] == 0.0)].index, inplace=True)
    return data


def plot_singular_values(singular_values):
    """scree-plot of singular values"""
    fig = go.Figure([go.Scatter(x=np.arange(1, len(singular_values) + 1), y=singular_values, name="Singular values",
                                showlegend=True,
                                mode='markers+lines',
                                marker=dict(color="black", opacity=1),
                                line=dict(color="black", dash="dash", width=1))],
                    layout=go.Layout(title=r"$\text{(14) Scree-plot of singular values}$",
                                     xaxis={"title": "x - num of singular value"},
                                     yaxis={"title": "y - Singular value"},
                                     height=300))
    fig.show()


def putting_it_all_together_1(design_matrix, response_vector):
    """loads the dataset, performs the preprocessing and plots the singular values plot"""
    w, s = fit_linear_regression(design_matrix, response_vector)
    plot_singular_values(s)


def putting_it_all_together_2(design_matrix, response_vector):
    """fit a model and test it over the data. Begin with writing code that splits the data into train- and test-sets
     randomly, such that the size of the test set is 1/4 of the total data"""
    res = []
    X_train, X_test, y_train, y_test = train_test_split(design_matrix, response_vector, test_size=0.25)
    num_rows_in_X_train = X_train.shape[0]
    for p in range(1, 101):
        num_rows = int(num_rows_in_X_train * (p / 100)) + 1
        coefficients_vector, s = fit_linear_regression(X_train[:num_rows, :], y_train[:num_rows, :])
        prediction_vector = predict(X_test, coefficients_vector)
        res.append(mse(prediction_vector, y_test))

    fig, ax = create_fig("MSE over the test set as a function of p%", "p", "MSE over the test set", (-1, 101),
                         (2 * math.pow(10, 10), max(res)))
    ax.plot(np.arange(1, 101), res, color='g')
    fig.show()


def feature_evaluation(design_matrix, response_vector):
    """given the design matrix and response vector, plots for every non-categorical feature, a graph (scatter plot) of
     the feature values and the response values. It then also computes and shows on the graph the Pearson Correlation
      between the feature and the response"""

    response_vector = response_vector.flatten()
    y_lim = (np.min(response_vector), np.max(response_vector))
    name_of_feature = ['date', 'bedrooms', 'bathrooms', 'sqft_living',
                       'sqft_lot', 'floors', 'waterfront', 'view',
                       'condition', 'grade', 'sqft_above', 'sqft_basement',
                       'sqft_living15', 'sqft_lot15', 'age', 'x_coordinate',
                       'y_coordinate', 'z_coordinate']
    for feature in range(len(name_of_feature)):
        cur_feature = design_matrix[:, [feature + 1]].flatten()
        x_lim = (np.min(cur_feature) - 1, np.max(cur_feature) + 1)

        cov = np.cov(cur_feature, response_vector)
        pearson_corr = cov[0][1] / (math.sqrt(cov[0][0]) * math.sqrt(cov[1][1]))
        fig, ax = create_fig(f" Feature is: {name_of_feature[feature]} and Pearson Correlation = {pearson_corr}",
                             name_of_feature[feature], "response", x_lim, y_lim)
        ax.scatter(design_matrix[:, [feature + 1]].flatten(), response_vector.flatten(), s=1, marker='.')
        fig.show()


def create_fig(title, xlabel, ylabel, xlim, ylim):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


if __name__ == '__main__':
    main()
