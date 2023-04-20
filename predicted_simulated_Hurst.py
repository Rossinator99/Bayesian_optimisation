# -*- coding: utf-8 -*-
"""
Author: Thomas Ross
Finds predicted Hurst values for Hurst generated fBm realisation
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from stochastic.processes.continuous import FractionalBrownianMotion
from tensorflow.keras.activations import swish
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
# define the attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = K.dot(x, self.w)
        a = K.softmax(e, axis=1)
        output = x * a
        return output
#generate our data for training and testing
nsamples = 10000
ntimes = 10
traindata = np.empty((nsamples,ntimes))
trainlabels = np.empty((nsamples,1))

for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.0,1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp,t=1,rng=None)
    x = fbm.sample(ntimes)
    #apply differencing and normalization on the data
    dx = (x[1:]-x[0:-1])/(np.amax(x)-np.amin(x))
    traindata[i,:] = dx
    trainlabels[i,:] = hurst_exp
testdata = np.empty((nsamples,ntimes))
testlabels = np.empty((nsamples,1))
for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.0,1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp,t=1,rng=None)
    x = fbm.sample(ntimes)
    dx = (x[1:]-x[0:-1])/(np.amax(x)-np.amin(x))
    testdata[i,:] = dx
    testlabels[i,:] = hurst_exp
np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")
model = tf.keras.models.load_model("./model3dense_n"+str(ntimes)+".h5") #"./model3dense_n"+str(ntimes)+".h5"
# with custom_object_scope({'AttentionLayer': AttentionLayer}):
#     model = tf.keras.models.load_model("best_model_2.h5")

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
#evaluate the model generalizes by using the test data set
loss, mae, mse = model.evaluate(testdata, testlabels, verbose=1)
print("Testing set Mean Abs Error: {:5.2f}".format(mae))
#predict values using data in the testing set
test_predictions = model.predict(testdata)
#save predicted values
np.savetxt("H_NNestimated_n"+str(ntimes)+".csv",test_predictions,delimiter=",")