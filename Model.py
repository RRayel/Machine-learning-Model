# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:57:28.458673Z","iopub.execute_input":"2025-09-29T04:57:28.459442Z","iopub.status.idle":"2025-09-29T04:57:28.464432Z","shell.execute_reply.started":"2025-09-29T04:57:28.459416Z","shell.execute_reply":"2025-09-29T04:57:28.463446Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:57:28.465984Z","iopub.execute_input":"2025-09-29T04:57:28.466461Z","iopub.status.idle":"2025-09-29T04:57:28.481472Z","shell.execute_reply.started":"2025-09-29T04:57:28.466435Z","shell.execute_reply":"2025-09-29T04:57:28.480403Z"},"jupyter":{"outputs_hidden":false}}
feet = np.linspace(0, 100, 100)   # 100 values from 0 to 100 feet this is use as input
meters = feet * 0.3048 #creates a 1 dimensional numpy array

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:57:28.482354Z","iopub.execute_input":"2025-09-29T04:57:28.482708Z","iopub.status.idle":"2025-09-29T04:57:28.497974Z","shell.execute_reply.started":"2025-09-29T04:57:28.482670Z","shell.execute_reply":"2025-09-29T04:57:28.496661Z"},"jupyter":{"outputs_hidden":false}}
# Reshape to (n_samples, 1) because Keras expects 2D inputs
X = feet.reshape(-1, 1)
y = meters.reshape(-1, 1)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:57:28.499691Z","iopub.execute_input":"2025-09-29T04:57:28.500054Z","iopub.status.idle":"2025-09-29T04:57:28.526512Z","shell.execute_reply.started":"2025-09-29T04:57:28.500026Z","shell.execute_reply":"2025-09-29T04:57:28.525764Z"},"jupyter":{"outputs_hidden":false}}
#here we create the model and call the keras API
model = keras.Sequential([
    layers.Dense(units=1 , input_shape=[1]) #units is the 1 output neuron, and input shape tells keras to expect vector of length 1
])
#the equation for the following neuron ouput = weight * input+bias

#optimizer depending on the learning process is the learning speed you are giving the machine
model.compile(optimizer="adam", loss="mean_squared_error") 
#loss being mean squared error is just to show how far the predicted value is from the actual value and adjust the weight accordingly

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:57:28.527262Z","iopub.execute_input":"2025-09-29T04:57:28.527623Z","iopub.status.idle":"2025-09-29T04:58:13.065490Z","shell.execute_reply.started":"2025-09-29T04:57:28.527532Z","shell.execute_reply":"2025-09-29T04:58:13.064620Z"},"jupyter":{"outputs_hidden":false}}
history = model.fit(X, y, epochs=1000, verbose=0)
#x and y are the training data
#epochs being 500 means the dataset is being shown to the network 500 times and if you change it to 1000 then it shows 1000 time

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:58:13.066738Z","iopub.execute_input":"2025-09-29T04:58:13.067027Z","iopub.status.idle":"2025-09-29T04:58:13.185240Z","shell.execute_reply.started":"2025-09-29T04:58:13.067008Z","shell.execute_reply":"2025-09-29T04:58:13.184098Z"},"jupyter":{"outputs_hidden":false}}
test_values = np.array([10,15,20, 25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])  # feet
predictions = model.predict(test_values)
#the test values are the number of feet that will be converted
#the predictions now puts those values in the model 


for f, p in zip(test_values, predictions):
    print(f"{f} feet → predicted {p[0]:.4f} meters | actual {f*0.3048:.4f} meters")
#pairs zip function pairs the predicted values with the test values
#p[0] accesses the scalar inside the 1-element prediction array for that sample

# %% [code] {"execution":{"iopub.status.busy":"2025-09-29T04:58:13.186217Z","iopub.execute_input":"2025-09-29T04:58:13.186439Z","iopub.status.idle":"2025-09-29T04:58:13.193215Z","shell.execute_reply.started":"2025-09-29T04:58:13.186422Z","shell.execute_reply":"2025-09-29T04:58:13.192082Z"},"jupyter":{"outputs_hidden":false}}
# Check learned weight (should be close to 0.3048)
weights, bias = model.layers[0].get_weights()
print("\nLearned weight:", weights[0][0], " | Learned bias:", bias[0])