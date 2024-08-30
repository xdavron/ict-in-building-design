import paho.mqtt.client as PahoMQTT
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense,  Conv2D, MaxPooling2D, TimeDistributed, Flatten, InputLayer, Reshape, Conv1D, MaxPooling1D, Bidirectional, Dropout, ReLU
from tensorflow.keras.models import Sequential, load_model
import math

regressands = [
    'Temp_IN'
]
past_hours = 23
next_hours = 1


def test_data():
    f = open('sample.json')
    sample_json = json.load(f)
    data = []
    for item in sample_json['data']:
        # print(item)
        d = {'timestamp': item['time_stamp'], 'Temp_IN': item['value']}
        data.append(d)
    return data


def minmax_value():
    # Minimum Maximum value from train dataset
    # used for normalizing and reverse
    minmax = pd.DataFrame()
    minmax['day'] = [1, 31]
    minmax['month'] = [1, 12]
    minmax['year'] = [2018, 2021]
    minmax['hour'] = [0, 23]
    minmax['minute'] = [0, 0]
    minmax['Temp_IN'] = [14.412768, 31.387367]
    return minmax


def cnn_lstm_model():
    model = Sequential()
    # normalized_X_test.shape[1] = 5
    model.add(Reshape((past_hours, 5, 1),
                      input_shape=(past_hours, 5)))
    model.add(Conv2D(filters=64, kernel_size=(2, 1), strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 1)))
    model.add(Conv2D(filters=64, kernel_size=(2,1), strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(1, 1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dense(32))
    model.add(Dense(next_hours * len(regressands)))
    model.add(ReLU())
    model.add(Reshape((next_hours, len(regressands))))

    model.compile(loss='mse', optimizer='adam', metrics='mean_absolute_error')

    return model


# IMPLEMENT MQTT SUBSCRIBER TO
class MySubscriber:
    def __init__(self, clientID, prediction_model):
        self.clientID = clientID
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, False)

        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        self.topic = 'ict4bd/sensor_reading'
        # self.messageBroker = 'localhost'
        self.messageBroker = 'test.mosquitto.org'
        # if {'name': clientID} not in self.client.get_list_database():
        #     self.client.create_database(clientID)

        self.model = prediction_model
        # list of previously received temperature readings
        self.previous_readings = []

    def start(self):
        # manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()
        # subscribe for a topic
        self._paho_mqtt.subscribe(self.topic, 2)

    def stop(self):
        self._paho_mqtt.unsubscribe(self.topic)
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.messageBroker, rc))

    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        # A new message is received
        print("Topic:'" + msg.topic+"', QoS: '"+str(msg.qos)+"' Message: '"+str(msg.payload) + "'")
        try:
            data = json.loads(msg.payload)
            if data["measurement"] == "Temperature":
                # add new prediction
                test_data = {
                    'Temp_IN': data['value'],
                    'timestamp': data['time_stamp']
                             }
                self.previous_readings.append(test_data)
                print(self.previous_readings)


                # TODO
                # INSERT TEMPERATURE PREDICTION HERE
                # ---------------------------------
                features = self.previous_readings[-25:]  # get last 25 readings
                prediction = data['value']  # TODO USE MODEL TO COMPUTE HOUR PREDICTION
                result = 0

                # PREDICTION PART START

                # Loading mock data
                # t_data = test_data()
                if len(features) > 24:
                    # Creating dataframe with last 25 readings
                    df = pd.DataFrame(features[-25:])

                    # changing type and format of timestamp
                    ts = []
                    for item in df['timestamp']:
                        datetime_object = datetime.strptime(item, '%Y-%m-%d %H:%M:%S')
                        ts.append(datetime_object)
                    df['timestamp'] = pd.to_datetime(ts, format='%m/%d/%Y %H:%M:%S')

                    # Creating additional columns for data and separating date
                    days = []
                    months = []
                    years = []
                    hours = []
                    minutes = []
                    for timestamp in df['timestamp']:
                        days.append(timestamp.day)
                        months.append(timestamp.month)
                        years.append(timestamp.year)
                        hours.append(timestamp.hour)
                        minutes.append(timestamp.minute)

                    X_test = pd.DataFrame()
                    X_test['day'] = days
                    X_test['month'] = months
                    X_test['year'] = years
                    X_test['hour'] = hours
                    X_test['minute'] = minutes
                    X_test.index = df.index
                    X_test['Temp_IN'] = df['Temp_IN']
                    X_test = X_test.astype('float32')

                    # Normilizing test data
                    normalized_X_test = (X_test-minmax_value().min())/(minmax_value().max()-minmax_value().min())
                    normalized_X_test.dropna(axis=1, inplace=True)

                    # Prepare test data and convert to numpy
                    testX = []
                    for t in range(past_hours, len(normalized_X_test.index)-next_hours, next_hours):
                        testX.append(normalized_X_test.values[t-past_hours:t, :])
                    testX = np.array(testX)

                    # Call model
                    # print(self.model.summary())

                    # Loading wights to model
                    self.model.load_weights(f'Temp_IN/cnn_lstm.h5')

                    # Predicting
                    out = self.model.predict(testX)

                    # Normilized result
                    result = out[0, 0, 0] * (minmax_value()['Temp_IN'].max() - minmax_value()['Temp_IN'].min()) + minmax_value()['Temp_IN'].min()
                    print(result)

                    # PREDICTION PART END
                    # self.client
                    # ---------------------------------
                    date_format = '%Y-%m-%d %H:%M:%S'
                    tmp = datetime.strptime(data['time_stamp'], date_format)
                    new_date = tmp + timedelta(hours=1)

                    json_body = {
                            "measurement": data['measurement'],
                            "node": "Tin_predicted",
                            "location": data['location'],
                            "time_stamp": new_date.strftime(date_format),
                            "value": result
                        }
                    self.myPublish("ict4bd/predicted", json.dumps(json_body))
        except Exception as e:
            print(e)

    def myPublish(self, topic, message):
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, message, 2)


if __name__ == "__main__":
    print("START TEST")
    # PREDICTION PART START
    # Create model
    model = cnn_lstm_model()
    test_publ = MySubscriber("test_publish", model)
    test_publ.start()
    while True:
        pass


