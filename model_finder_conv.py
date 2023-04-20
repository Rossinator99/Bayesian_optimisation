# import tensorflow and other necessary libraries
import numpy as np
import tensorflow as tf
from stochastic.processes.continuous import FractionalBrownianMotion
from tensorflow.keras.layers import  Conv1D, Dropout, Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
import time
from tensorflow.keras.layers import Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import swish
np.random.seed(42)
tf.random.set_seed(42)
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
    
# Terminates upon validation MAE of NaN
class NaNStopping(Callback):
    def __init__(self):
        super(NaNStopping, self).__init__()
        self.nan_detected = False
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_mae = logs.get('val_mae')
        if val_mae is not None and np.isnan(val_mae):
            print(f"Stopping fold training at epoch {epoch} due to NaN MAE value.")
            self.model.stop_training = True
            self.nan_detected = True   
# Terminates upon validation MAE above 0.24
class TerminateOnMAE(Callback):
    def __init__(self, threshold=0.24, epoch_limit=0):
        super(TerminateOnMAE, self).__init__()
        self.threshold = threshold
        self.epoch_limit = epoch_limit
        self.early_stop = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_mae = logs.get('val_mae')
        if epoch >= self.epoch_limit:
            if val_mae is not None and val_mae > self.threshold:
                print(f"Stopping fold training at epoch {epoch} due to MAE above threshold ({self.threshold}).")
                self.model.stop_training = True
                self.early_stop = True
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

np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")
print('traindata_rnn shape:', traindata_conv.shape)
print('training data shape:',traindata.shape,'training labels shape:', trainlabels.shape,'test data shape:',testdata.shape,'test labels shape:',testlabels.shape)

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(traindata_conv, trainlabels, test_size=0.2, random_state=42)
EPOCHS = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.000)
# Add NaNStopping and TerminateOnMAE to your callbacks list
callbacks = [early_stopping, reduce_lr]

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
def create_and_evaluate_model(params, save_best_model=False):
    # Convert params to a dictionary
    params_dict = dict(zip([dim.name for dim in search_space], params))
    for i in range(7):
        num_filters_i = params_dict["num_filters" + str(i + 1)]
        activation_i = params_dict["activation" + str(i + 1)]
        dilation_rate_i = params_dict["dilation_rate" + str(i + 1)]
        kernel_size_i = params_dict["kernel_size" + str(i + 1)]
        if i < 6:
            dropout_rate_i = params_dict["dropout_rate" + str(i + 1)]
            print(f"Layer {i+1}: Conv1D, Filters: {num_filters_i}, Kernel size: {kernel_size_i}, Activation: {activation_i}, Dilation rate: {dilation_rate_i}, Dropout rate: {dropout_rate_i}")
        else:
            print(f"Layer {i+1}: Conv1D, Filters: {num_filters_i}, Kernel size: {kernel_size_i}, Activation: {activation_i}, Dilation rate: {dilation_rate_i}")
    
    print(f"Learning rate: {params_dict['learning_rate']}")
    print(f"Output activation: {params_dict['output_activation']}")
    num_filters = params[:7]
    kernel_size = params[7:14]
    dropout_rates = params[14:20]
    learning_rate = params[20]
    dilation_rates = params[21:28]
    activations = params[28:35]
    output_activation = params[35]

    # Create the model using the extracted hyperparameters
    model = tf.keras.Sequential()
    model.add(Input(shape=(ntimes, 1)))
    
    for i in range(7):
        model.add(Conv1D(filters=int(num_filters[i]), kernel_size=int(kernel_size[i]), dilation_rate=int(dilation_rates[i]), activation=activations[i], padding='same'))
        model.add(BatchNormalization())  # Add batch normalization after the Conv1D

        if activations[i] == 'leaky_relu':
            model.add(LeakyReLU(alpha=0.01))
        elif activations[i] == 'swish':
            model.add(Activation(swish))
        else:
            model.add(Activation(activations[i]))
        if i < 6:
            model.add(Dropout(dropout_rates[i]))
        if i == 3:  # Add MaxPooling1D after the 3rd Conv1D layer
            model.add(MaxPooling1D())
    # Add the AttentionLayer to your model
    model.add(AttentionLayer())
    # Add GlobalAveragePooling1D before the Dense layer
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dense(1, activation=output_activation))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Train the model using early stopping and learning rate reduction
    model.fit(traindata_conv, trainlabels, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=callbacks)
    if save_best_model:
        model.save("./best_model_n"+str(ntimes)+".h5")
    # Evaluate the model and return the mean absolute error
    _, test_mae, _ = model.evaluate(testdata_conv, testlabels)
    return test_mae


search_space = [
    Integer(2, 128, name='num_filters1'),
    Integer(8, 250, name='num_filters2'),
    Integer(8, 250, name='num_filters3'),
    Integer(8, 250, name='num_filters4'),
    Integer(8, 250, name='num_filters5'),
    Integer(8, 500, name='num_filters6'),
    Integer(8, 500, name='num_filters7'),
    Integer(1, 10, name='kernel_size1'),
    Integer(2, 10, name='kernel_size2'),
    Integer(2, 10, name='kernel_size3'),
    Integer(2, 10, name='kernel_size4'),
    Integer(2, 10, name='kernel_size5'),
    Integer(2, 10, name='kernel_size6'),
    Integer(2, 10, name='kernel_size7'),
    Real(0.013, 0.015, name='dropout_rate1'),
    Real(0.09, 0.095, name='dropout_rate2'),
    Real(0.105, 0.115, name='dropout_rate3'),
    Real(0.05, 0.055, name='dropout_rate4'),
    Real(0.1, 0.11, name='dropout_rate5'),
    Real(0.175, 0.185, name='dropout_rate6'),
    Real(0.001, 0.003, name='learning_rate'),
    Integer(1, 5, name='dilation_rate1'),
    Integer(2, 10, name='dilation_rate2'),
    Integer(2, 20, name='dilation_rate3'),
    Integer(2, 30, name='dilation_rate4'),
    Integer(2, 40, name='dilation_rate5'),
    Integer(2, 70, name='dilation_rate6'),
    Integer(2, 90, name='dilation_rate7'),
    Categorical(['elu'], name='activation1'),
    Categorical(['swish'], name='activation2'),
    Categorical(['leaky_relu'], name='activation3'),
    Categorical(['swish'], name='activation4'),
    Categorical(['relu'], name='activation5'),
    Categorical(['leaky_relu'], name='activation6'),
    Categorical(['leaky_relu'], name='activation7'),
    Categorical(['sigmoid'], name='output_activation')
]
def objective(params):
    return create_and_evaluate_model(params, save_best_model=False)
res = gp_minimize(lambda params: objective(params), search_space, n_calls=100, random_state=42, n_jobs=1, verbose=True)

best_hyperparameters = {
    'num_filters1': int(res.x[0]),
    'num_filters2': int(res.x[1]),
    'num_filters3': int(res.x[2]),
    'num_filters4': int(res.x[3]),
    'num_filters5': int(res.x[4]),
    'num_filters6': int(res.x[5]),
    'num_filters7': int(res.x[6]),
    'kernel_size1': int(res.x[7]),
    'kernel_size2': int(res.x[8]),
    'kernel_size3': int(res.x[9]),
    'kernel_size4': int(res.x[10]),
    'kernel_size5': int(res.x[11]),
    'kernel_size6': int(res.x[12]),
    'kernel_size7': int(res.x[13]),
    'dropout_rate1': res.x[14],
    'dropout_rate2': res.x[15],
    'dropout_rate3': res.x[16],
    'dropout_rate4': res.x[17],
    'dropout_rate5': res.x[18],
    'dropout_rate6': res.x[19],
    'learning_rate': res.x[20],
    'dilation_rate1': res.x[21],
    'dilation_rate2': res.x[22],
    'dilation_rate3': res.x[23],
    'dilation_rate4': res.x[24],
    'dilation_rate5': res.x[25],
    'dilation_rate6': res.x[26],
    'dilation_rate7': res.x[27],
    'activation1': res.x[28],
    'activation2': res.x[29],
    'activation3': res.x[30],
    'activation4': res.x[31],
    'activation5': res.x[32],
    'activation6': res.x[33],
    'activation7': res.x[34],
    'output_activation': res.x[35]
}

# Create, train, and evaluate the final model using the best hyperparameters
final_mae = create_and_evaluate_model(res.x, save_best_model=True)
print("Final Test MAE:", final_mae)

# Print the best hyperparameters
print("Best Hyperparameters:")
best_params = dict(zip([dim.name for dim in search_space], res.x))
for i in range(7):
    num_filters_i = best_params["num_filters" + str(i + 1)]
    activation_i = best_params["activation" + str(i + 1)]
    dilation_rate_i = best_params["dilation_rate" + str(i + 1)]
    kernel_size_i = best_params["kernel_size" + str(i + 1)]
    if i < 6:
        dropout_rate_i = best_params["dropout_rate" + str(i + 1)]
        print(f"Layer {i+1}: Conv1D, Filters: {num_filters_i}, Kernel size: {kernel_size_i}, Activation: {activation_i}, Dilation rate: {dilation_rate_i}, Dropout rate: {dropout_rate_i}")
    else:
        print(f"Layer {i+1}: Conv1D, Filters: {num_filters_i}, Kernel size: {kernel_size_i}, Activation: {activation_i}, Dilation rate: {dilation_rate_i}")

print(f"Learning rate: {best_params['learning_rate']}")
print(f"Output activation: {best_params['output_activation']}")


end_time = time.time()
time_taken = end_time - start_time
minutes = time_taken / 60
print("Time taken: ", time_taken, "seconds")
print("Time taken: ", minutes, "minutes") 
