# neural net for mapping movement data like velocities and accelerations to muscle activations
import numpy as np
import tensorflow as tf
from operator import itemgetter
from numpy import savetxt
import matplotlib.pyplot as plt
from functions import *
import time
import os
import random

# we act like val_data is equal to raw_data
# raw_data_path = os.path.dirname(os.path.realpath(__file__)) + "\\tr_data.csv"
cross_validation_training = False
create_new_trData = False
train_new = True
use_lstm = True
number_of_steps_lstm = 1
trainingset_name = 'new_data_short'
raw_data_path = 'tr_data.npy'
model_name = 'L_best'
descending = False

def create_and_optimize(tr_x, tr_y, optional_model_name, use_lstm):
    if use_lstm:
        model = tf.keras.models.Sequential([tf.keras.layers.LSTM(40, input_shape=(tr_x.shape[1], tr_x.shape[2]), return_sequences=True),
                                            tf.keras.layers.LSTM(70, return_sequences=True),
                                            tf.keras.layers.LSTM(100, return_sequences=True),
                                            tf.keras.layers.LSTM(50, return_sequences=True),
                                            tf.keras.layers.LSTM(20, return_sequences=False),
                                            tf.keras.layers.Dense(5, activation='linear')])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
        model.summary()
        model.fit(tr_x, tr_y, epochs=50, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=2, min_delta=1e-5, verbose=2)])  # 50

    else:
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                                            tf.keras.layers.Dense(5, activation='linear')])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
        model.fit(tr_x, tr_y, epochs=50, batch_size=32,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=2, min_delta=1e-5, verbose=2)])


    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()

    # model.evaluate(val_x, val_y, verbose=2)  #2

    model.summary()

    if optional_model_name:
        model.save(optional_model_name)
    return model



if create_new_trData:
    tr_data, val_data, tr_x, tr_y, val_x, val_y = shuffle_and_separate_tr_data(raw_data_path, trainingset_name)  # copy and paste in other script
    print('done shuffle and separate')
    # model = create_and_optimize(tr_x, tr_y, val_x, val_y, model_name, use_lstm)
    # print('done training')
else:
    # load training data (shuffled sets)
    # tr_original = np.load('tr_data.npy')
    tr_data, val_data, tr_x, tr_y, val_x, val_y = get_tr_data(trainingset_name)
    val_split = 0.2
    print(tr_data.shape)
    print('validation split: {}'.format(val_split))
    print('(only used during evaluation, not training)')
    idx = len(val_data)
    print('number of validation samples: {}'.format(idx))
    # sort val data by weight, angle and speed
    val_data = sort_data(val_data, [2, 3, 0, 1], descending)
    val_x = val_data[:, :4]
    val_y = val_data[:, 4:]
    print('valshape: ' + str(val_data.shape))
    print('trshape: ' + str(tr_data.shape))
    if use_lstm:
        tr_x, tr_y, val_x, val_y = create_samples_ntimesteps(tr_data, val_data, number_of_steps_lstm)
        print('lstm_tr_shape: ' + str(tr_x.shape))

    if train_new:
        if cross_validation_training:
            # create cv model here
            if use_lstm:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.LSTM(10, input_shape=(tr_x.shape[1], tr_x.shape[2]), return_sequences=True),
                    tf.keras.layers.LSTM(15, return_sequences=True),
                    tf.keras.layers.LSTM(10, return_sequences=False),
                    tf.keras.layers.Dense(5, activation='linear')])
            else:
                model = tf.keras.models.Sequential([
                    # tf.keras.layers.Flatten(input_shape=(4,)),
                    tf.keras.layers.Dense(10, activation='relu'),
                    tf.keras.layers.Dense(15, activation='relu'),
                    tf.keras.layers.Dense(10, activation='relu'),
                    tf.keras.layers.Dense(5, activation='linear')])
            print('compiling')
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
            print('training')
            # partitioning data
            n_samples_total = tr_x.shape[0]  # tr_data.shape[0]
            n_folds = 5  # number of folds during cv
            idx_lst = part_data(n_folds, n_samples_total)
            # cross-validation
            batch_size = 32
            n_epochs = 50
            results, model = cv_train(model, tr_x, tr_y, idx_lst, n_epochs, batch_size)
            #model.save(model_name)
        else:
            model = create_and_optimize(tr_x, tr_y, model_name, use_lstm)
            model.evaluate(val_x, val_y)
            print('done training')
    else:
        model = load_model(model_name)
        model.evaluate(val_x, val_y)
        print('done loading model')


mae_local, mae_remote, mae_smoothed = evaluate_outputs(val_x, val_y, model.predict(val_x), model.predict(val_x), number_of_steps_lstm)

print('done eval')

#plot_sorted(mae_local, mae_remote, mae_smoothed, model_name)

'''        tr_data, val_data, tr_x, tr_y, val_x, val_y = shuffle_and_separate_tr_data(raw_data_path, model_name)
        print('done shuffle and separate')
        # tr_data, val_data, tr_x, tr_y, val_x, val_y = get_tr_data(model_name)
        # print('done loading')
        model = create_and_optimize(tr_x, tr_y, val_x, val_y, model_name)
        print('done training')
        # model = load_model(model_name)
        loss, mse, mae = compute_metrics(val_data, val_x, val_y, model, model_name)
        print('done eval')
        # loss, mse, mae = load_metrics(model_name)
'''

# sort MAE according to each input ones
#sort_priority = [1, 2, 0, 0]
#sorted0_input_and_MAE = sort_data(mae, sort_priority, descending)
#sort_priority = [2, 1, 0, 0]
#sorted1_input_and_MAE = sort_data(mae, sort_priority, descending)
#sort_priority = [2, 0, 1, 0]
#sorted2_input_and_MAE = sort_data(mae, sort_priority, descending)
#sort_priority = [2, 0, 0, 1]
#sorted3_input_and_MAE = sort_data(mae, sort_priority, descending)

'''sort_priority = [1, 2, 0, 0]
sorted0_input_and_MAE_smoothed = sort_data(mae_smoothed, sort_priority, descending)
sort_priority = [2, 1, 0, 0]
sorted1_input_and_MAE_smoothed = sort_data(mae_smoothed, sort_priority, descending)
sort_priority = [2, 0, 1, 0]
sorted2_input_and_MAE_smoothed = sort_data(mae_smoothed, sort_priority, descending)
sort_priority = [2, 0, 0, 1]
sorted3_input_and_MAE_smoothed = sort_data(mae_smoothed, sort_priority, descending)

sort_priority = [1, 2, 0, 0]
sorted0_input_and_MAE_remote = sort_data(mae_remote, sort_priority, descending)
sort_priority = [2, 1, 0, 0]
sorted1_input_and_MAE_remote = sort_data(mae_remote, sort_priority, descending)
sort_priority = [2, 0, 1, 0]
sorted2_input_and_MAE_remote = sort_data(mae_remote, sort_priority, descending)
sort_priority = [2, 0, 0, 1]
sorted3_input_and_MAE_remote = sort_data(mae_remote, sort_priority, descending)

sort_priority = [1, 2, 0, 0]
sorted0_input_and_MAE_local = sort_data(mae_local, sort_priority, descending)
sort_priority = [2, 1, 0, 0]
sorted1_input_and_MAE_local = sort_data(mae_local, sort_priority, descending)
sort_priority = [2, 0, 1, 0]
sorted2_input_and_MAE_local = sort_data(mae_local, sort_priority, descending)
sort_priority = [2, 0, 0, 1]
sorted3_input_and_MAE_local = sort_data(mae_local, sort_priority, descending)'''

#plot_everything(sorted0_input_and_MAE_local, sorted1_input_and_MAE_local, sorted2_input_and_MAE_local, sorted3_input_and_MAE_local, model_name + 'local')
#plot_everything(sorted0_input_and_MAE_remote, sorted1_input_and_MAE_remote, sorted2_input_and_MAE_remote, sorted3_input_and_MAE_remote, model_name + 'remote')
#plot_everything(sorted0_input_and_MAE_smoothed, sorted1_input_and_MAE_smoothed, sorted2_input_and_MAE_smoothed, sorted3_input_and_MAE_smoothed, model_name + 'smoothed')


print('done')