import sklearn
import requests
import pandas as pd
import yfinance as yf
import datetime
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, optimizers, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def get_stock_data(stock):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(weeks=10 * 52)

    stock_final = pd.DataFrame()
    stock_data = yf.download(stock, start=start, end=end, progress=False)
    stock_df = stock_final.append(stock_data, sort=False)

    html_doc = requests.get('https://www.marketwatch.com/investing/stock/{}'.format(stock)).text
    soup = BeautifulSoup(html_doc, "html.parser")
    market = soup.select('.company__market')[0].text.split(':')[1].strip()
    market_idx = {'Nasdaq': '^IXIC', 'NYSE': '^NYA'}
    df2 = yf.download(market_idx[market], start=start, end=end, progress=False)
    df2.drop('Close', axis=1, inplace=True)
    df2 = df2.add_prefix('m_')
    df_f = pd.merge(stock_df, df2, on='Date')
    return stock_df

def preprocess(df):
    #Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target', 'Target2']


    df['ma5'] = df['Close'].rolling(5).mean()
    #df['ma30'] = df['Close'].rolling(30).mean()
    df['ma5'].fillna(df['Close'].mean(), inplace=True)
    #df['ma30'].fillna(df['Close'].mean(), inplace=True)
    #df = df.drop('ma30', axis=1)
    df['Target'] = df['Adj Close']

    df['Target2'] = df['Adj Close'] - df['Adj Close'].shift(1)

    df['Target2'] = df['Adj Close'] - df['Adj Close'].shift(1).fillna(0)
    df['Target2'] = df['Target2'] > 0
    df.drop("Close", axis=1, inplace=True)

    #print(df)
    return df

def my_model(df, case):
    LENGTH = 60
    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(df.iloc[:, :-1])

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(LENGTH, df.shape[0]):
        X_train.append(training_set_scaled[i - LENGTH:i, 0:-1])
        y_train.append(training_set_scaled[i, -1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))




    x = tf.convert_to_tensor(X_train, dtype=tf.float32)
    if case == 1:
        y = tf.convert_to_tensor(np.array(df.iloc[LENGTH:, -1]), dtype=tf.bool)
    else:
        y = tf.convert_to_tensor(y_train, dtype=tf.float32)

    train_size = int(df.shape[0] * 0.8)
    print(x.shape, y.shape)

    BATCH_SIZE = 16
    # Creating a data structure with 60 time-steps and 1 output

    x_train, x_val = tf.split(x, axis=0, num_or_size_splits=[train_size, x.shape[0] - train_size])
    y_train, y_val = tf.split(y, axis=0, num_or_size_splits=[train_size, x.shape[0] - train_size])


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(BATCH_SIZE)  # drop last batch remainder
    print(train_dataset)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    print(x_train.shape, y_train.shape)

    if case == 1:
        print(y_train.numpy())
        print(x_train.numpy())

    if case == 1:
        model = tf.keras.Sequential([
            layers.LSTM(64, dropout=0.2, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            #layers.LSTM(128, dropout=0.2, return_sequences=True),
            layers.LSTM(64, dropout=0.2),
            #layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
    else:
        model = tf.keras.Sequential([

            layers.LSTM(128, dropout=0.2, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            layers.LSTM(128, dropout=0.2, return_sequences=True),
            layers.LSTM(64, dropout=0.2),
            layers.Dense(1)
        ])


    # Compiling the RNN
    if case == 1:
        model.compile(optimizer=keras.optimizers.Adam(0.01), loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=['binary_accuracy'])
        #model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy,
        #              metrics=[keras.metrics.binary_accuracy])
        # sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9)
        # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=sgd,
        #               metrics=['accuracy'])#'binary_crossentropy' tf.keras.optimizers.Adam(1e-4)

    else:
        model.compile(optimizer='adam', loss='mean_squared_error')
    # model.fit(X_train, y_train, epochs = 10, batch_size = 32)
    #model.fit(x_train, y_train, epochs=10)
    model.fit(train_dataset, epochs=20, validation_data=test_dataset)

    price = model.predict(test_dataset)
    if case == 1:
        dataset_test = df.iloc[-y_val.shape[0]:, -1]
        print(price)
    else:
        dataset_test = df.iloc[-y_val.shape[0]:, -2:-1]
        price = np.tile(price, df.shape[1]-1)
        print(price.shape)
        predicted_stock_price = sc.inverse_transform(price)[:, 0]
        p = df.iloc[-predicted_stock_price.shape[0]:,-1]

        a=[(predicted_stock_price[1:] > predicted_stock_price[:-1])==p[1:]]
        unique, counts = np.unique(a, return_counts=True)
        dict(zip(unique, counts))

    if case!=1:
        # Visualising the results
        plt.plot(dataset_test.values, color='red', label='Real TESLA Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted TESLA Stock Price')
        # plt.plot(price)
        # plt.plot(y_val)
        plt.xticks(np.arange(0, 459, 50))
        plt.title('TESLA Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('TESLA Stock Price')
        plt.legend()
        plt.show()


def my_model2(df):
    LENGTH = 20
    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    print(df.columns)
    training_set_scaled = sc.fit_transform(df.iloc[:, :-2])

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []

    for i in range(LENGTH, df.shape[0]):
        X_train.append(training_set_scaled[i - LENGTH:i, :])
    data, target = np.array(X_train), np.array(df.iloc[LENGTH:,-1])
    #data, target = np.array(training_set_scaled), np.array(df['Target2'])
    print(data.shape)
    data = data.reshape(data.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    #clf = RandomForestClassifier(max_depth=3, min_samples_split=16)
    clf = SVC(kernel='poly', degree=3)
    print(X_train.shape, X_test.shape,y_train.shape, y_test.shape)
    clf.fit(X_train, y_train)  # train model
    print(clf.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(clf.score(X_train, y_train), clf.score(X_test, y_test))
    print(np.mean(cross_val_score(clf, data, target, cv=5)))


def model2():
    end = datetime.datetime.today()
    start = end - datetime.timedelta(weeks=10 * 52)

    stock_final = pd.DataFrame()
    stock_data = yf.download('TSLA', start=start, end=end, progress=False)
    df = stock_final.append(stock_data, sort=False)
    df['Target2'] = df['Close'] - df['Close'].shift(1)

    df['Target2'] = df['Close'] - df['Close'].shift(1).fillna(0)
    df['Target2'] = df['Target2'] > 0
    df['predictForUp'] = 0

    target = df['Target2']
    length = len(df)
    trainNum = int(length * 0.8)
    predictNum = length - trainNum
    feature = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    feature = preprocessing.scale(feature)
    featureTrain = feature[1:trainNum - 1]
    targetTrain = target[1:trainNum - 1]
    svmTool = SVC(kernel='poly')
    svmTool.fit(featureTrain, targetTrain)
    print(svmTool.score(featureTrain, targetTrain))


if __name__ == '__main__':
    #df = get_stock_data('TSLA')
    #df = preprocess(df)
    #my_model(df,1)
    #print(df.columns)
    #my_model2(df)
    model2()

