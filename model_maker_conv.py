"""
Author: Thomas Ross
Code that makes convolution model

With attention layer will have to load model in a different way once saved.
from tensorflow.keras.utils import custom_object_scope
with custom_object_scope({'AttentionLayer': AttentionLayer}):
    model = tf.keras.models.load_model("attention_layer_model.h5")
Also include class function in code.
"""
# import tensorflow and other necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN
from stochastic.processes.continuous import FractionalBrownianMotion
from tensorflow.keras.layers import GRU, Conv1D, Dropout
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
# Import the necessary libraries and layers
from tensorflow.keras.layers import Layer, LeakyReLU, MaxPooling1D
from tensorflow.keras.activations import swish
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, LeakyReLU, MaxPooling1D, Lambda
from tensorflow.keras.utils import custom_object_scope
import time
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
def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': swish})
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

traindata_conv = np.reshape(traindata, (traindata.shape[0], traindata.shape[1], 1))
testdata_conv = np.reshape(testdata, (testdata.shape[0], testdata.shape[1], 1))
np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")

# Replace the activation function strings with their respective function calls
model = tf.keras.Sequential([
    Conv1D(filters=53, kernel_size=10, dilation_rate=100, activation='elu', input_shape=(ntimes, 1), padding='same'),
    Dropout(0.0141),
    BatchNormalization(),
    Conv1D(filters=227, kernel_size=10, dilation_rate=100, padding='same'),
    Lambda(swish),
    Dropout(0.0933),
    BatchNormalization(),
    Conv1D(filters=250, kernel_size=10, dilation_rate=100, padding='same'),
    LeakyReLU(),
    Dropout(0.109),
    BatchNormalization(),
    MaxPooling1D(),
    Conv1D(filters=8, kernel_size=10, dilation_rate=100, padding='same'),
    Lambda(swish),
    Dropout(0.052),
    BatchNormalization(),
    Conv1D(filters=42, kernel_size=10, dilation_rate=100, activation='relu', padding='same'),
    Dropout(0.104),
    BatchNormalization(),
    Conv1D(filters=25, kernel_size=10, dilation_rate=100, padding='same'),
    LeakyReLU(alpha=0.01),
    Dropout(0.1798),
    BatchNormalization(),
    Conv1D(filters=57, kernel_size=10, dilation_rate=100, padding='same'),
    Lambda(swish),
    GlobalAveragePooling1D(),
    AttentionLayer(),
    Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00263, clipvalue=1.0)
model.compile(optimizer=optimizer,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error','mean_squared_error']
)
model.summary()
#train the model
EPOCHS = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000)
callbacks = [early_stopping, reduce_lr]
history = model.fit(traindata_conv, trainlabels, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=callbacks)

#Save model
print("Saving model")

model.save("model3dense_n100.h5")
del model
# model = tf.keras.models.load_model("model3dense_n100.h5")
with custom_object_scope({'AttentionLayer': AttentionLayer}):
    model = tf.keras.models.load_model("./model3dense_n"+str(ntimes)+".h5")
#evaluate the model generalizes by using the test data set
loss, mae, mse = model.evaluate(testdata_conv, testlabels, verbose=1)
print("Testing set Mean Abs Error: {:5.2f}".format(mae))
#predict values using data in the testing set
test_predictions = model.predict(testdata_conv)
#save predicted values
np.savetxt("H_NNestimated_n"+str(ntimes)+".csv", test_predictions, delimiter=",")
end_time = time.time()
time_taken = end_time - start_time
minutes = time_taken / 60
print("Time taken: ", time_taken, "seconds")
print("Time taken: ", minutes, "minutes")