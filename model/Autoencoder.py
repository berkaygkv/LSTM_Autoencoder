import json
import os

import jmespath
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from pykalman import KalmanFilter
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

matplotlib.rcParams["figure.figsize"] = (15, 8)


class Model(keras.Model):
    """Class Object that handles all the operations to build an end-to-end LSTM Autoencoder Keras Model for BTC data.
    Upon providing a JSON file that contains Neural Net architecture (e.g., layers.json) including an 'encoder' and
    a 'decoder' layer, the model is ready be built with the default arguments. Accepts a dataframe object, a list of 
    features, time steps, learning rate and Kalman Filter Covariance Constant. Once the model is built it can be saved/loaded 
    in order to be cached for later use. In addition, there are utility tools that can be used to output training loss graphs,
    predictions and charts.

    Args:
        dataframe: BTC data with Close Price values, optionally, in addition to other features (e.g., Volume)
        columns: a list object that will be used to train the model
        time_steps: number of look-back time steps
        learning_rate: keyword argument for keras model
        KFilter_covariance: Kalman Filter Covariance constant to be used in preprocessing step. The filter smoothens the data
        to make the model as sensitive to strong deviations as possible.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        columns: list = ["Close"],
        time_steps: int = 5,
        learning_rate: float = 0.0001,
        KFilter_covariance: float = 0.1,
    ):

        super(Model, self).__init__()
        self.df = dataframe[columns]
        self.columns = columns
        self.KFilter_covariance = KFilter_covariance
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.scaled_columns = {}
        self.history = None
        self.model = None
        self.model_id = None
        self.kf_plot_df = pd.DataFrame()
        self.layers_json = self.extract_layers()

    def __repr__(self):
        return f"Model(dataframe=df, columns={self.columns}, time_steps={self.time_steps}, KFilter_covariance={self.KFilter_covariance}, learning_rate={self.learning_rate})"

    def extract_layers(self):
        "Function to read Neural Net architecture."
        with open("layers.json") as rd:
            js = json.load(rd)
        return js

    def generate_model_id(self, epochs, batchsize):
        """Generates unique model id for model storing purposes."""
        cell_numbers = "-".join(
            jmespath.search("encoder.*.to_string(n_of_units)", self.layers_json)
        )
        time_steps = self.time_steps
        kf_covariance_constant = self.KFilter_covariance
        columns = ", ".join(self.columns)
        id_string = f"{cell_numbers}seq_{epochs}eps_{batchsize}bs_{time_steps}ts_{kf_covariance_constant}KFconst-{columns}"
        self.model_id = id_string

    def apply_kfilter(self, dataframe):
        """Applies Kalman Filter to smooth the inputted data.

        Args:
            dataframe: Target Dataframe to be processed.

        Returns:
            pandas.DataFrame: Returns the processed dataframe.
        """
        kf = KalmanFilter(
            initial_state_mean=dataframe.iloc[0]["Close"],
            n_dim_obs=1,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=self.KFilter_covariance,
        )
        dataframe["Close_original"] = dataframe["Close"]
        dataframe["Close"] = kf.filter(dataframe["Close"])[0]
        return dataframe

    def plot_kfilter(self):
        """Plots the input data with and without Kalman Filter applied.

        Returns:
            plotly.graph_objs._figure.Figure: Plotly line chart.
        """
        fig_kf = px.line(
            x="Date",
            y="Close",
            data_frame=self.kf_plot_df.reset_index(),
            color_discrete_sequence=["red"],
        )
        fig_original = px.line(
            x="Date",
            y="Close_original",
            data_frame=self.kf_plot_df.reset_index(),
            color_discrete_sequence=["green"],
        )
        fig_original.update_traces(line={"width": 0.5})
        fig = go.Figure(fig_kf.data + fig_original.data)
        return fig

    def train_split(self):
        """Split the data into train and test sets.
        """
        train_size = int(len(self.df) * 0.95)
        self.train, self.test = (
            self.df.iloc[0:train_size],
            self.df.iloc[train_size: len(self.df)],
        )
        if self.KFilter_covariance:
            self.train = self.apply_kfilter(self.train)
        self.kf_plot_df = self.train.copy()
        print(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def scaling(self):
        """Applies Standard Scale to the train and test data.
        """
        for column in self.df.columns:
            if not self.scaled_columns.get(column, False):
                scaler = StandardScaler()
                scaler.fit(self.train[[column]])
                self.train[column] = scaler.transform(self.train[[column]])
                self.test[column] = scaler.transform(self.test[[column]])
                self.scaled_columns.update({column: scaler})
                if column == "Close":
                    self.close_scaler = scaler

            else:
                print(f"Column {column} Already Scaled!")

    def predict(self, df, **kwargs):
        """Detects anomalies in the given dataframe.

        Args:
            df: Dataframe that will be predicted.

        Returns:
            numpy.array: Returns an array of indices with the peak loss values and an array of losses
        """
        for column in self.df.columns:
            df[column] = self.scaled_columns[column].transform(df[[column]])

        X, _ = self.reshape_dataset(df[self.columns], df["Close"], self.time_steps)
        col_idx = df.columns.get_loc("Close")
        X_pred = self.model.predict(X)
        X_pred = self.close_scaler.inverse_transform(X_pred)
        X = self.close_scaler.inverse_transform(X[:, col_idx][:, col_idx])
        test_mae_loss = np.abs(X_pred[:, col_idx].flatten() - X).flatten()
        peaks, _ = find_peaks(test_mae_loss, **kwargs)
        return peaks, test_mae_loss

    def reshape_dataset(self, X, y, time_steps):
        """Reshapes the data set into sequences of 5 time steps.

        Args:
            X: Feature values that will be used to make predictions.
            y: Array of target values that will be predicted.
            time_steps: Number of time steps that will be looked into the history

        Returns:
            numpy.array: Returns an array of feature and target values in the given time step sequence. 
        """
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i: (i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def prepare_data(self):
        """Main function to initiate reshaping operations for both train and test data.
        """
        self.X_train, self.y_train = self.reshape_dataset(
            self.train[self.columns], self.train["Close"], self.time_steps
        )
        self.X_test, self.y_test = self.reshape_dataset(
            self.test[self.columns], self.test["Close"], self.time_steps
        )
        print(f"X_train Shape: {self.X_train.shape}, X_test Shape: {self.X_test.shape}")

    def build_model(self):
        """Model building function that reads the provided Neural Net Architecture JSON and compiles an LSTM Autoencoder model.
        """
        keras.backend.clear_session()
        self.model = keras.Sequential()
        for layer in self.layers_json["encoder"]:
            if layer == "input_layer":
                self.model.add(
                    keras.layers.LSTM(
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                        units=self.layers_json["encoder"][layer]["n_of_units"],
                        return_sequences=self.layers_json["encoder"][layer]["return_sequences"],
                        name=layer,
                    )
                )

            else:
                self.model.add(
                    keras.layers.LSTM(
                        units=self.layers_json["encoder"][layer]["n_of_units"],
                        return_sequences=self.layers_json["encoder"][layer]["return_sequences"],
                        name=layer,
                    )
                )

        self.model.add(keras.layers.RepeatVector(n=self.X_train.shape[1]))

        for layer in self.layers_json["decoder"]:
            self.model.add(
                keras.layers.LSTM(
                    units=self.layers_json["decoder"][layer]["n_of_units"],
                    return_sequences=self.layers_json["decoder"][layer]["return_sequences"],
                    name=layer,
                )
            )

        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss="mae", optimizer=opt)

    def train_model(self, epochs=250, batch_size=128, verbose=1):
        """Function initating training session with the given epoch and batch size parameters.
        """
        _ = self.generate_model_id(epochs, batch_size)
        target_model_path = "saved_models/" + self.model_id.split('seq_')[0] + "/" + self.model_id + ".h5"
        if not os.path.exists(target_model_path):
            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                verbose=verbose,
            )

        else:
            self.load_model(self.model_id, override=True)
            print(f"Model: {self.model_id} is loaded!")

    def plot_history(self):
        """Plots loss values across epochs.

        Returns:
            matplotlib.pyplot.plot: Loss vs. epoch chart.
        """
        return plt.plot(self.history.history["loss"], label="train")

    def write_history(self):
        """Stores and aggregates the loss values of the trained model into a JSON file.
        """
        if os.path.exists("saved_models/history.json"):
            with open("saved_models/history.json", "r") as read:
                data = json.load(read)

        else:
            data = {}

        data.update({self.model_id: self.history.history})
        with open("saved_models/history.json", "w") as wr:
            json.dump(data, wr, indent=4)

    def predict_test(self):
        """Predicts test data.
        """
        self.X_test_pred = self.model.predict(self.X_test)

    def calculate_loss(self):
        """Calculates losses of each predictions.

        Returns:
            numpy.array: Returns an array of losses.
        """
        col_idx = self.df.columns.get_loc("Close")
        X_test_pred = self.close_scaler.inverse_transform(self.X_test_pred)
        X_test = self.close_scaler.inverse_transform(self.X_test[:, col_idx][:, col_idx])
        test_mae_loss = np.abs(X_test_pred[:, col_idx].flatten() - X_test).reshape(-1, 1)

        return test_mae_loss

    def create_df(self, **kwargs):
        """Creates a dataframe that contains the actual close prices and loss values of each predictions
        and extract significantly deviated predictions as anomalies.
        """
        test_score_df = pd.DataFrame(index=self.test[self.time_steps:].index)
        test_mae_loss = self.calculate_loss()
        test_score_df["loss"] = test_mae_loss[:, 0]
        test_score_df["Close"] = self.test[self.time_steps:].Close
        test_score_df.reset_index(inplace=True)
        anomaly_indices, _ = find_peaks(test_score_df["loss"], **kwargs)
        test_score_df["anomaly"] = False
        test_score_df.iloc[
            anomaly_indices, test_score_df.columns.get_loc("anomaly")
        ] = True
        test_score_df.set_index("Date", inplace=True)
        self.test_score_df = test_score_df

    # This method will be merged with 'close_vs_loss' method
    def plot_anomaly(self, **kwargs):
        """Plots the actual close prices with the anomalous datapoints pinned.

        Returns:
            plotly.graph_objs._figure.Figure: Figure that shows the anomalies in the actual prices data.
        """
        self.create_df(**kwargs)
        fig_original = px.line(
            x="Date", y="Close", data_frame=self.test_score_df.reset_index()
        )
        fig_original.update_yaxes(secondary_y=True)
        fig_anomaly = px.scatter(
            x="Date",
            y="Close",
            data_frame=self.test_score_df.reset_index()
            .query("anomaly == True")
            .reset_index(),
            color_discrete_sequence=["green"],
            hover_data={"index": True},
        )
        fig_anomaly.update_yaxes(secondary_y=True)
        fig = go.Figure(fig_original.data + fig_anomaly.data)
        return fig

    def close_vs_loss(self):
        """Plots the actual close prices along with the corresponding loss values.

        Returns:
            plotly.graph_objs._figure.Figure: Figure that shows the anomalies in the actual prices data and the losses.
        """
        col_idx = self.df.columns.get_loc("Close")
        X_test_pred = self.close_scaler.inverse_transform(self.X_test_pred[:, 0][:, 0])
        X_test = self.close_scaler.inverse_transform(
            self.X_test[:, col_idx][:, col_idx]
        )
        data = {"test": X_test, "pred": X_test_pred}
        df = pd.DataFrame(data, index=self.test_score_df.index)
        df["loss"] = np.abs(df["test"] - df["pred"])
        df["anomaly"] = self.test_score_df["anomaly"]
        fig4 = px.line(
            x="Date",
            y="loss",
            data_frame=df.reset_index(),
            color_discrete_sequence=["blue"],
        )
        fig4.update_traces(opacity=0.25)
        fig = go.Figure(fig4.data)
        fig.add_trace(go.Scatter(x=df.index, y=df["test"], name="Actual", yaxis="y2"))
        fig.add_trace(
            go.Scatter(
                x=df.query("anomaly == True").index,
                y=df.query("anomaly == True")["test"],
                name="Anomaly",
                yaxis="y2",
                mode="markers",
                marker=dict(color="green"),
            )
        )
        fig.update_layout(
            yaxis2=dict(anchor="free", overlaying="y1", side="right", position=1.0)
        )
        return fig

    def save_model(self, override=False):
        """Function to save the model in the saved models folder.

        Args:
            override (bool, optional): In case of an already stored model in the folder this option will 
            be used to override the current. Defaults to False. 
        """
        target_folder_path = "saved_models/" + self.model_id.split("seq_")[0]
        self.target_model_path = target_folder_path + f"/{self.model_id}.h5"
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        if self.model_id + ".h5" not in os.listdir(target_folder_path):
            self.model.save(self.target_model_path)
            if self.history:
                self.write_history()

        else:
            if not override:
                print(
                    "Another model with the same name was already saved... Use override arg to proceed"
                )

            else:
                self.model.save(self.target_model_path)
                if self.history:
                    self.write_history()

    def load_model(self, target_model_name, override=False):
        """Function to load the model from saved models folder. In case of trying to train an already stored model,
        this function will call the stored model from the folder instead of initating the training model from scratch.

        Args:
            target_model_name (string): Name of the model will be loaded.
            override (bool, optional): True if an active instance model will be replaced with the stored one. Defaults to False.
        """
        target_model_path = "saved_models/" + target_model_name.split('seq_')[0] + "/" + target_model_name + ".h5"
        self.model_id = target_model_name
        if os.path.exists(target_model_path):
            if not self.model:
                self.model = keras.models.load_model(target_model_path)

            else:
                if not override:
                    print("Another model in use... Use override arg to proceed")

                else:
                    self.model = keras.models.load_model(target_model_path)

        else:
            print(f"No model named {self.model_id}")
