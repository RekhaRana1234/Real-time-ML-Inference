from flask import Flask, render_template, request
import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__, template_folder='template')

model = load_model('new_model.h5')

@app.route('/')

def home():
    return render_template("main-page.html")

def windowed_dataset(client_data, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(client_data)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset



def min_max_scaler(data):
   scaler = MinMaxScaler(feature_range=(0,1))
   dataset = scaler.fit_transform(data)
   return dataset

def get_input():
   value1 = request.form.get('value1')
   value2 = request.form.get('value2')
   value3 = request.form.get('value3')
   value4 = request.form.get('value4')

   data = np.float32([[value1], [value2], [value3], [value4]])

   
   return data

def inverse_data(data):
   scaler = MinMaxScaler(feature_range=(0,1))
   result_transformed = scaler.inverse_transform(data)
   return result_transformed 


@app.route('/send', methods=['POST'])

def show_results():
   df = get_input()
   df = df.astype('float32')
   Values = pd.DataFrame(df)
   scaler = MinMaxScaler(feature_range=(0,1))
   dataset = scaler.fit_transform(df)
   Test_data = windowed_dataset(dataset, 1, 10, len(df))
   testPredict = model.predict(Test_data)
   testPredict = scaler.inverse_transform(testPredict)
   outcome = testPredict[-1]
   return render_template('results.html', tables = [Values.to_html(classes='data', header=False)], result=outcome)
   




if __name__=="__main__":
    app.run(debug=True)