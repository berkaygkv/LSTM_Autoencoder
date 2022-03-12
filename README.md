# LSTM Autoencoder Engine for Bitcoin Data
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
  </ol>
</details>



## About The Project
### Aim
This personal project aims to create an anomaly detection algorithm that is capable of capturing strong deviations in the Bitcoin dataset for self learning purposes. I have assembled some of the popular techniques used in the field and customized them in a way that, I believed/experimented, worked best for me. I am completely open to any suggestions and feedbacks. The data used in the project was obtained from [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data?select=bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv).

---


### Rationale
Often times you come across people trying to predict cryptocurrency or stock prices using various ML techniques. One of the most popular and powerful method in achiving this, not surprisingly; would be Long-Short Term Memory network, a.k.a LSTM. It, indeed, does a great job in predicting the price value in the next time step. That being said, being able to predict the price in the next 1-minute (or 1-hour) would not mean much unless you are able to predict the value in the 2nd, 3rd, 4th... etc. minutes. This can, surely, be achieved by using multi-step LSTM tecniques but with a **cost of performance and scope**. Using multi-step approach causes drastic declines in prediction performance and the scope of the model would be limited to the number of steps you had given. The latter is an essential problem when it comes to crypto/stock prices as the increases or declines happen to occur at various time frame length.  That is when Autoencoder comes into play.
<br>
<br>
>_P.S: I can only argue in my humble opinion that the Autoencoder approach "seems to" present a solution for the **scope problem**_

<br>


**Autoencoder** is a neural net architecture that allows labeling anomalies in the data without the necessity of explicitly feeding any labels into the model in the training phase. In the most basic sense, unlike the other prediction approaches you create a model that predicts the input sequence itself instead of a supposedly unknown value in the future. The model predicts the input sequence as accurate as possible and if the predicted value is significantly poor one, this is a sign that the given sequence is anomalous. By doing this you make the model learn the frequently occuring patterns so much that the model becomes super-sensitive to the deviations in the data. 

One novel (I bet it is) approach I came up with is that what if I "truncate" the peaks and dips in the training data by using smoothing techniques. This leads the spikes will be regarded as anomalous by the model since the model has not been exposed to any of the peaks or dips in the training phase.
For this approach, I have applied Kalman Filter and the results seemed confirming the efficiency of the experimented model.


---



## Getting Started


The project will be dockerized soon. Until then, you can access the [Nbviewer](https://nbviewer.org/github/berkaygkv/LSTM_Autoencoder/blob/master/anomaly_detection_nb.ipynb) version of the notebook to interact with the Plotly charts.


## Contact
Berkay Gökova - berkaygokova@gmail.com


