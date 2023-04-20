'''
Author: Thomas Ross
Code for making RNN model
'''
# import tensorflow and other necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN
from stochastic.processes.continuous import FractionalBrownianMotion
from tensorflow.keras.layers import GRU, Conv1D, Dropout, LSTM
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from tensorflow.keras import regularizers
np.random.seed(42)
tf.random.set_seed(42)
start_time = time.time()
# generate our data for training and testing
nsamples = 10000
ntimes = 100
traindata = np.empty((nsamples, ntimes))
trainlabels = np.empty((nsamples, 1))
for i in range(0, nsamples):
    hurst_exp = np.random.uniform(0., 1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp, t=1, rng=None)
    x = fbm.sample(ntimes)
    dx = (x[1:] - x[:-1]) / (np.amax(x) - np.amin(x))
    traindata[i, :] = dx
    trainlabels[i, :] = hurst_exp

testdata = np.empty((nsamples, ntimes))
testlabels = np.empty((nsamples, 1))
for i in range(0, nsamples):
    hurst_exp = np.random.uniform(0., 1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp, t=1, rng=None)
    x = fbm.sample(ntimes)
    dx = (x[1:] - x[:-1]) / (np.amax(x) - np.amin(x))
    testdata[i, :] = dx
    testlabels[i, :] = hurst_exp

# Reshape the training and testing data to have the correct shape
traindata_rnn = np.reshape(traindata, (traindata.shape[0], traindata.shape[1], 1))
testdata_rnn = np.reshape(testdata, (testdata.shape[0], testdata.shape[1], 1))
np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")

# create the model
model = tf.keras.Sequential([
    GRU(15, input_shape=(ntimes, 1), activation='softsign', return_sequences=True),
    Dropout(0.04200159719931207),
    Bidirectional(GRU(87, activation='softsign', return_sequences=True)),
    Dropout(0.1838302390304667),
    Bidirectional(GRU(281, activation='relu', return_sequences=True)),
    Dropout(0.00925324747526878),
    LSTM(251, activation='hard_sigmoid', return_sequences=True),
    Dropout(0.018737470096543685),
    Bidirectional(GRU(199, activation='elu')),
    Dropout(0),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# add optimizer, a loss function and metrics#
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0012335839423630618, clipvalue=1.0)
model.compile(optimizer=optimizer,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error','mean_squared_error']
)
model.summary()

#train the model
EPOCHS = 100
BATCH_SIZE = 19
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0) #0.001
callbacks = [early_stopping, reduce_lr]
history = model.fit(traindata_rnn, trainlabels, epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks)
#Save model
print("Saving model")
model.save("./model3dense_n"+str(ntimes)+".h5")
del model
model = tf.keras.models.load_model("./model3dense_n"+str(ntimes)+".h5")

#evaluate the model generalizes by using the test data set
loss, mae, mse = model.evaluate(testdata_rnn, testlabels, verbose=1)
print("Testing set Mean Abs Error: {:5.2f}".format(mae))
#predict values using data in the testing set
test_predictions = model.predict(testdata_rnn)
#save predicted values
np.savetxt("H_NNestimated_n"+str(ntimes)+".csv",test_predictions,delimiter=",")

end_time = time.time()
time_taken = end_time - start_time
minutes = time_taken / 60
print("Time taken: ", time_taken, "seconds")
print("Time taken: ", minutes, "minutes")