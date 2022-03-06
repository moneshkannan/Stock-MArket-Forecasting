# from urllib import request
import pandas_datareader as pdr
import pandas as pd
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import tensorflow as tf

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

# demonstrate prediction for next 10 days
from numpy import array

import matplotlib.pyplot as plt

app = Flask(__name__)

key = "757f6d561ad6c59ce1d9b6d4ca73328956fdea48"


@app.route('/', methods=['POST'])
def json():
    request_data = request.get_json()
    company = request_data['comp']
    n_epoch = request_data['epoch']
    n_days= request_data['days']
    df = pdr.get_data_tiingo(company, api_key=key)
    df.to_csv(f'dataset/{company}.csv')
    df=pd.read_csv(f'dataset/{company}.csv')
    head = df.head()
    print(head)
    df1=df.reset_index()['close']
    print ("df1",df1)

    plt.plot(df1)
    print("first plot")
    plt.show()
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler_df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    print(scaler_df1)

    ##splitting dataset into train and test split
    training_size = int(len(scaler_df1)*0.65)
    test_size=len(scaler_df1)-training_size
    train_data,test_data=scaler_df1[0:training_size,:],scaler_df1[training_size:len(scaler_df1),:1]

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)
    
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    # print(X_train.shape), print(y_train.shape)
    # print(X_test.shape), print(ytest.shape)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train_shape =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test_shape = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    ### Create the Stacked LSTM model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()
    model_summary = model.to_json()

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=n_epoch,batch_size=64,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    print("train_predict",train_predict)
    test_predict = model.predict(X_test)
    print("test_predict",test_predict)

    print(tf.__version__)

    ##Transformback to original form
    train_predict_inverse_transform = scaler.inverse_transform(train_predict)
    print("train_predict_inverse_transform",train_predict_inverse_transform)
    test_predict_inverse_transform =scaler.inverse_transform(test_predict)
    print("test_predict_inverse_transform",test_predict_inverse_transform)

    ### Calculate RMSE performance metrics
    math.sqrt(mean_squared_error(y_train,train_predict))
    print("ytrain_predict",math.sqrt(mean_squared_error(y_train,train_predict)))

    ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))
    print("ytest_predict",math.sqrt(mean_squared_error(ytest,test_predict)))

    ### Plotting 
    # shift train predictions for plotting
    look_back = 100
    trainPredictPlot = numpy.empty_like(scaler_df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    shift_train_predictions = trainPredictPlot
    print("shift_train_predictions",shift_train_predictions)

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(scaler_df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaler_df1)-1, :] = test_predict
    shift_test_predictions = trainPredictPlot
    print ("shift_test_predictions",shift_test_predictions)
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(scaler_df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    print("plot baseline and predictions")
    plt.show()

    len(test_data)
    print (len(test_data))

    x_input=test_data[341:].reshape(1,-1)
    print("x_input",x_input.shape)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    print("temp_input",temp_input)

    # demonstrate prediction for next 10 days
    lst_output=[]
    n_steps=100
    i=0
    while(i<n_days):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        

    print(lst_output)

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    length_dataframe = len(scaler_df1)
    print(len(scaler_df1))
    # print("hi",day_new, scaler.inverse_transform(df1[1159:]))
    # print("welcome",day_pred,scaler.inverse_transform(lst_output))
    plt.plot(day_new,scaler.inverse_transform(scaler_df1[(length_dataframe-100):]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    print("day_new,day_pred")
    plt.show()

    df3=scaler_df1.tolist()
    df3.extend(lst_output)
    plt.plot(df3[1200:])
    print("df3 with lst_output")
    plt.show();

    df3=scaler.inverse_transform(df3).tolist()
    plt.plot(df3)
    print("scaler inverse transform")
    plt.show()

    return jsonify(
        head = head.to_json(),
        df1 = df1.to_json(),
        scaler_df1 = scaler_df1.tolist(),
        training_size = training_size,
        test_size = test_size,
        train_data = train_data.tolist(),
        test_data = test_data.tolist(),
        time_step = int(time_step),
        x_train = X_train.tolist(),
        y_train = y_train.tolist(),
        x_test = X_test.tolist(),
        y_test = ytest.tolist(),
        x_train_shape = X_train_shape.tolist(),
        x_test_shape = X_test_shape.tolist(),
        model_summary = model_summary,
        tensorflow_version = tf.__version__,
        train_predict = train_predict.tolist(),
        test_predict = test_predict.tolist(),
        train_predict_inverse_transform = train_predict_inverse_transform.tolist(),
        test_predict_inverse_transform = test_predict_inverse_transform.tolist(),
        ytrain_predict = math.sqrt(mean_squared_error(y_train,train_predict)),
        ytest_predict = math.sqrt(mean_squared_error(ytest,test_predict)),
        shift_train_predictions = shift_train_predictions.tolist(),
        shift_test_predictions = shift_test_predictions.tolist(),
        test_data_length = len(test_data),
        x_input = x_input.tolist(),
        temp_input = temp_input,
        lst_output = lst_output
    )


if __name__ == '__main__':
    app.run(debug=True)
