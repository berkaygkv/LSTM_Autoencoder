# LSTM Autoencoder Engine for Bitcoin Data
## Aim
This personal project aims to create an anomaly detection algorithm that is capable of capturing strong deviations in the Bitcoin dataset for self learning purposes. I have assembled some of the popular techniques used in the field and customized them in a way that, I believed/experimented, worked best for me. I am completely open to any suggestions or critiques

---


## Rationale
Often times you come across people trying to predict cryptocurrency or stock prices using various ML techniques. One of the most popular and powerful method in achiving this, not surprisingly; would be Long-Short Term Memory network, a.k.a LSTM. It, indeed, does a great job in predicting the price value in the next time step. That being said, being able to predict the price in the next 1-minute (or 1-hour) would not mean much unless you are able to predict the value in 2nd, 3rd, 4th... etc. minutes. This can, surely, be achieved by using multi-step LSTM tecniques but with a **cost of performance and scope**. Using multi-step approach causes drastic declines in prediction performance and the scope of the model would be limited to the number of steps you had given. The latter is an essential problem when it comes to crypto/stock prices as the increases or declines happen to occur at various time frame length.  That is when Autoencoder comes into play.

_P.S: I can only argue in my humble opinion that the Autoencoder approach "seems to" present a solution for the **scope problem**_

**Autoencoder** is a neural net architecture that allows labeling anomalies in the data without the necessity of explicitly feeding any labels into the model in the training phase. In the most basic sense, unlike the other prediction approaches you create a model that predicts the input sequence itself instead of a supposedly unknown value in the future. The model predicts the input sequence as accurate as possible and if the predicted value is significantly poor one, this is a sign that the given sequence is anomalous. By doing this you make the model learn the frequently occuring patterns so much that the model becomes super-sensitive to the deviations in the data. 

One novel approach (hopefully it is!) I came up with is that what if I "truncate" the peaks and dips in the training data by using smoothing techniques. This results in the peaks and dips will be regarded as anomalous by the model since the model has not seen any peaks or dips in the training phase.
For this approach, I have applied Kalman Filter and the results seemed confirming the efficiency of the experimented model.