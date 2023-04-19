"""
Author: Thomas Ross
Code for finding a optimised hyperparameters using Bayesian optimisation for 
RNN model that predicts the Hurst exponents for fbm tracks
"""
# import tensorflow and other necessary libraries
import numpy as np
import tensorflow as tf
from stochastic.processes.continuous import FractionalBrownianMotion
from tensorflow.keras.layers import GRU, Conv1D, Dropout, LSTM, Input, Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
# Set model maker to same seed. Important for repeatable model results
np.random.seed(42)
tf.random.set_seed(42)
# define the attention layer
# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         self.w = self.add_weight(shape=(input_shape[-1], 1),
#                                  initializer='uniform',
#                                  trainable=True)
#         super(AttentionLayer, self).build(input_shape)
    
#     def call(self, x):
#         e = K.dot(x, self.w)
#         a = K.softmax(e, axis=1)
#         output = x * a
#         return output
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

# Save data
# Use this data with model maker, for repeatable model results
np.save("traindata.npy", traindata)
np.save("trainlabels.npy", trainlabels)
np.save("testdata.npy", testdata)
np.save("testlabels.npy", testlabels)

# Reshape the training and testing data to have the correct shape
traindata_rnn = np.reshape(traindata, (traindata.shape[0], traindata.shape[1], 1))
testdata_rnn = np.reshape(testdata, (testdata.shape[0], testdata.shape[1], 1))

np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")
print('traindata_rnn shape:', traindata_rnn.shape)
print('training data shape:',traindata.shape,'training labels shape:', trainlabels.shape,'test data shape:',testdata.shape,'test labels shape:',testlabels.shape)

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(traindata_rnn, trainlabels, test_size=0.2, random_state=42)
EPOCHS = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000)
nan_stopping = NaNStopping()
terminate_on_mae = TerminateOnMAE(threshold=0.24, epoch_limit=0)
callbacks = [early_stopping, reduce_lr, nan_stopping, terminate_on_mae]
    
def remove_nan_rows(X, y):
    mask = ~np.isnan(y).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean

trial_counter = 0
def create_and_evaluate_model(params, trial_num=None, return_model=False):
    global best_mae, best_params
    num_units = params[:5]
    dropout_rates = params[5:10]
    learning_rate = params[10]
    activations = params[11:16]
    num_layers = int(params[16])
    layer_types = params[17:22]
    output_activation = params[22]
    batch_size = int(params[23])
    l1_reg = params[24]
    l2_reg = params[25]
    if trial_num is not None:
        print(f"\nTrial {trial_num}:")
        print(f"Batch size: {batch_size}")

    input_layer = Input(shape=(traindata_rnn.shape[1], traindata_rnn.shape[2]))
    rnn_layer = input_layer
    
    for i in range(num_layers):
        num_units_i = int(num_units[i])
        activation_i = activations[i]
        layer_type_i = layer_types[i]
    
        bidirectional = layer_type_i.startswith('Bi')
        if bidirectional:
            layer_type_i = layer_type_i[2:]
        rnn_cell = eval(layer_type_i)(num_units_i, return_sequences=(i < num_layers - 1), activation=activation_i) #, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
        if bidirectional:
            rnn_cell = Bidirectional(rnn_cell)
        rnn_layer = rnn_cell(rnn_layer)
        rnn_layer = Dropout(dropout_rates[i])(rnn_layer)
        print(f"Layer {i+1}: {'Bi' if bidirectional else ''}{layer_type_i}, Neurons: {num_units_i}, Activation: {activation_i}, Dropout rate: {dropout_rates[i]}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output activation: {output_activation}")
    print(f"L1 regularization: {l1_reg}")
    print(f"L2 regularization: {l2_reg}")
    # attention_layer = AttentionLayer()(rnn_layer)
    # to use attention layer replace rnn_layer with (attention_layer)
    output_layer = Dense(1, activation=output_activation, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(rnn_layer) 
    # Comment out kernel regularizer in seach space and l1_reg and l2_reg variables above for no kernel regularizer
    # Would recommend for finding initial model that works
    # output_layer = Dense(1, activation=output_activation))(rnn_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the specified learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0), loss="mse", metrics=["mae"])
    # Remove NaN values from the input data
    X_train_clean, y_train_clean = remove_nan_rows(X_train, y_train)
    X_val_clean, y_val_clean = remove_nan_rows(X_val, y_val)
    testdata_rnn_clean, testlabels_clean = remove_nan_rows(testdata_rnn, testlabels)
    terminate_on_mae.early_stop = False
    # Train the model using early stopping and learning rate reduction
    model.fit(X_train_clean, y_train_clean, batch_size=batch_size, epochs=EPOCHS, validation_data=(X_val_clean, y_val_clean), callbacks=callbacks, verbose=1)
    if terminate_on_mae.early_stop or nan_stopping.nan_detected:
        penalty_value = 1e10
        if return_model:
            return (penalty_value, None)
        else:
            return penalty_value
      
    # Evaluate the model on the validation set
    loss = model.evaluate(X_val_clean, y_val_clean, verbose=0)
    # Evaluate the model and return the mean absolute error
    _, test_mae = model.evaluate(testdata_rnn_clean, testlabels_clean)
    # If test_mae is NaN, return a large penalty value
    if np.isnan(test_mae) or terminate_on_mae.early_stop or nan_stopping.nan_detected:
        penalty_value = 1e10
        if return_model:
            return (penalty_value, None)
        else:
            return penalty_value
    else:
        if return_model:
            return (test_mae, model)
        else:
            return -loss[0] 

# Vary parameters how you wish in here
search_space = [
    Integer(10, 300, name='num_units1'),
    Integer(10, 300, name='num_units2'),
    Integer(10, 300, name='num_units3'),
    Integer(5, 300, name='num_units4'),
    Integer(1, 200, name='num_units5'),
    Real(0.0, 0.2, name='dropout_rate1'),
    Real(0.0, 0.2, name='dropout_rate2'),
    Real(0.0, 0.2, name='dropout_rate3'),
    Real(0.0, 0.2, name='dropout_rate4'),
    # Real(0.0, 0.1, name='dropout_rate5'),
    Categorical([0], name='dropout_rate5'),
    Real(0.001, 0.01, name='learning_rate'),
    Categorical(['softsign'], name='activation1'),
    Categorical(['softsign'], name='activation2'),
    Categorical(['relu'], name='activation3'),
    Categorical(['hard_sigmoid'], name='activation4'),
    Categorical(['elu'], name='activation5'),
    # Integer(1, 4, name='num_layers'), #  1, 4 testing for range of layers
    Categorical([5], name='num_layers'),
    Categorical(['relu', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid'], name='layer_type1'),
    Categorical(['relu', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid'], name='layer_type2'),
    Categorical(['relu', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid'], name='layer_type3'),
    Categorical(['relu', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid'], name='layer_type4'),
    Categorical(['relu', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid'], name='layer_type5'),
    Categorical(['linear', 'sigmoid'], name='output_activation'),
    Integer(1, 100, name='batch_size'),
    Real(0.069, 0.072, name='l1_reg'),
    Real(0.58, 0.6, name='l2_reg')
    ]
def create_and_evaluate_model_with_trial_num(params):
    global trial_counter
    trial_counter += 1
    mae = create_and_evaluate_model(params, trial_num=trial_counter, return_model=False)
    return mae
# n_calls is the number of trials you would like to run
res = gp_minimize(create_and_evaluate_model_with_trial_num, search_space, n_calls=1000, random_state=42)

final_mae, best_model = create_and_evaluate_model(res.x, return_model=True)

end_time = time.time()
time_taken = end_time - start_time
minutes = time_taken / 60
print("Time taken: ", time_taken, "seconds")
print("Time taken: ", minutes, "minutes")
