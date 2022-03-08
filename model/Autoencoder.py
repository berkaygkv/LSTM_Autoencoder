import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objects as go
from tensorflow import keras
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
import ta
from scipy.signal import argrelextrema
import warnings
import json


def extract_layers():
    with open("layers.json") as rd:
        js = json.load(rd)
    return js


layers = extract_layers()


class Model(keras.Model):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            columns: list = ['Close'],
            n_of_units: int = 512,
            dropout_rate: int = 0.4,
            time_steps: int = 5,
            learning_rate: float = 0.0001,
            KFilter_covariance: float = .1):

        super(Model, self).__init__()
        self.df = dataframe[columns]
        self.columns = columns
        self.KFilter_covariance = KFilter_covariance
        self.n_of_units = n_of_units
        self.dropout_rate = dropout_rate
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.scaled_columns = {}
        self.history = None
        self.model = None

    def __repr__(self):
        return f"""Model(
        DataFrame = df, 
        columns = {self.columns}, 
        n_of_units = {self.n_of_units}, 
        dropout_rate = {self.dropout_rate}, 
        batch_size = {self.batch_size}, 
        epochs = {self.epochs}, 
        time_steps = {self.time_steps}
        )"""

    def apply_kfilter(self, dataframe):
        kf = KalmanFilter(
            initial_state_mean=dataframe.iloc[0]["Close"],
            n_dim_obs=1,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=self.KFilter_covariance)
        dataframe["Close_original"] = dataframe["Close"]
        dataframe["Close"] = kf.filter(dataframe["Close"])[0]
        return dataframe

    def plot_kfilter(self):
        fig_kf = px.line(x='Date', y='Close', data_frame=self.df.reset_index(
        ), color_discrete_sequence=['red'])
        fig_original = px.line(x='Date', y='Close_original', data_frame=self.df.reset_index(
        ), color_discrete_sequence=['green'])
        fig_original.update_traces(line={'width': 0.5})
        fig = go.Figure(fig_kf.data + fig_original.data)
        return fig

    def train_split(self):
        train_size = int(len(self.df) * 0.95)
        self.train, self.test = self.df.iloc[0:train_size], self.df.iloc[train_size:len(
            self.df)]
        if self.KFilter_covariance:
            self.train = self.apply_kfilter(self.train)
        print(
            f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def scaling(self):
        for column in self.df.columns:
            if not self.scaled_columns.get(column, False):
                scaler = StandardScaler()
                scaler.fit(self.train[[column]])
                self.train[column] = scaler.transform(self.train[[column]])
                self.test[column] = scaler.transform(self.test[[column]])
                self.scaled_columns.update({column: True})
            else:
                print(f'Column {column} Already Scaled!')

    def reshape_dataset(self, X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def prepare_data(self):
        # reshape to [samples, time_steps, n_features]
        self.X_train, self.y_train = self.reshape_dataset(
            self.train[self.columns],
            self.train["Close"],
            self.time_steps
        )
        self.X_test, self.y_test = self.reshape_dataset(
            self.test[self.columns],
            self.test["Close"],
            self.time_steps
        )
        print(
            f"X_train Shape: {self.X_train.shape}, X_test Shape: {self.X_test.shape}")

    def build_model(self):
        keras.backend.clear_session()
        self.model = keras.Sequential()
        for layer in layers['encoder']:
            if layer == "input_layer":
                self.model.add(keras.layers.LSTM(
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    units=layers['encoder'][layer]['n_of_units'],
                    return_sequences=layers['encoder'][layer]['return_sequences'],
                    name=layer
                ))

            else:
                self.model.add(keras.layers.LSTM(
                    units=layers['encoder'][layer]['n_of_units'],
                    return_sequences=layers['encoder'][layer]['return_sequences'],
                    name=layer
                ))

        self.model.add(keras.layers.RepeatVector(n=self.X_train.shape[1]))

        for layer in layers['decoder']:
            self.model.add(keras.layers.LSTM(
                units=layers['decoder'][layer]['n_of_units'],
                return_sequences=layers['decoder'][layer]['return_sequences'],
                name=layer
            ))

        self.model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(units=1)
            ))

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='mae', optimizer=opt)

    def train_model(self, epochs=250, batch_size=128, verbose=1):
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=verbose
        )

    def plot_history(self):
        return plt.plot(self.history.history['loss'], label='train')
