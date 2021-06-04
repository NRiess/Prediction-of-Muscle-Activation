# testsetting for evaluating multiple networks
import numpy as np
import tensorflow as tf
from operator import itemgetter
from numpy import savetxt
import matplotlib.pyplot as plt
import os
from functions import *
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from tensorflow.keras.utils import plot_model
import string


def plot_mae_for_angle_range(val_x, mae, angle_range, LSTM):
    val_x_angle_range = val_x
    if LSTM:
        mae_angle_range = mae[:, 1]
    else:
        mae_angle_range = mae[:, 0]
    if angle_range == 0:
        for row in range(len(val_x)-1, -1, -1):
            if val_x[row][0]>0.25:
                val_x_angle_range = np.delete(val_x_angle_range, row, 0)
                mae_angle_range = np.delete(mae_angle_range, row)
        if LSTM:
            plt.plot(mae_angle_range, label="LSTM, angle_range: 0-0.25")
        else:
            plt.plot(mae_angle_range, label="Dense, angle_range: 0-0.25")

    elif angle_range == 1:
        for row in range(len(val_x)-1, -1, -1):
            if val_x_angle_range[row][0]<=0.25 or val_x_angle_range[row][0]>=0.75:
                val_x_angle_range = np.delete(val_x_angle_range, row, 0)
                mae_angle_range = np.delete(mae_angle_range, row)
        if LSTM:
            plt.plot(mae_angle_range, label="LSTM, angle_range: 0.25-0.75")
        else:
            plt.plot(mae_angle_range, label="Dense, angle_range: 0.25-0.75")

    elif angle_range == 2:
        for row in range(len(val_x)-1, -1, -1):
            if val_x_angle_range[row][0]<0.75:
                val_x_angle_range = np.delete(val_x_angle_range, row, 0)
                mae_angle_range = np.delete(mae_angle_range, row)
        if LSTM:
            plt.plot(mae_angle_range, label="LSTM, angle_range: 0.75-1")
        else:
            plt.plot(mae_angle_range, label="Dense, angle_range: 0.75-1")

    else:
        raise ValueError("The angle range hast to be \n" +
                  " 0 for validation samples with an angle between 0 and 0.25 \n" +
                  " 1 for validation samples with an angle between 0.25 and 0.75 \n" +
                  " 2 for validation samples with an angle between 0.75 and 1")

    return np.mean(mae_angle_range)

def print_average_mae_for_each_angle_range(angle_range, mae_angle_range, LSTM):
    if angle_range == 0:
        if LSTM:
            print('Average MAE of LSTM NN in angle range 0 to 0.25: ' + str(np.mean(mae_angle_range)))
        else:
            print('Average MAE of Dense NN in angle range 0 to 0.25: ' + str(np.mean(mae_angle_range)))

    elif angle_range == 1:
        if LSTM:
            print('Average MAE of LSTM NN in angle range 0.25 to 0.75: ' + str(np.mean(mae_angle_range)))
        else:
            print('Average MAE of Dense NN in angle range 0.25 to 0.75: ' + str(np.mean(mae_angle_range)))

    elif angle_range == 2:
        if LSTM:
            print('Average MAE of LSTM NN in angle range 0.75 to 1: ' + str(np.mean(mae_angle_range)))
        else:
            print('Average MAE of Dense NN in angle range 0.75 to 1: ' + str(np.mean(mae_angle_range)))

def plot_outputs(model, val_x, LSTM, angle_range, muscle):
    fontsize = 10
    if LSTM:
        eval = model[0].predict(valn_x)
    else:
        eval = model[0].predict(val_x)

    if angle_range == 0:
        for row in range(len(val_x)-1, -1, -1):
            if val_x[row][0]>0.25:
                eval = np.delete(eval, row, 0)
        if LSTM:
            plt.plot(eval[:, muscle])
            plt.title("LSTM, angle_range: 0-0.25, muscle: "+str(muscle), fontsize=fontsize)
        else:
            plt.plot(eval[:, muscle])
            plt.title("Dense, angle_range: 0-0.25, muscle: "+str(muscle), fontsize=fontsize)

    elif angle_range == 1:
        for row in range(len(val_x)-1, -1, -1):
            if val_x[row][0]<=0.25 or val_x[row][0]>=0.75:
                eval = np.delete(eval, row, 0)
        if LSTM:
            plt.plot(eval[:, muscle])
            plt.title("LSTM, angle_range: 0.25-0.75, muscle: "+str(muscle), fontsize=fontsize)
        else:
            plt.plot(eval[:, muscle])
            plt.title("Dense, angle_range: 0.25-0.75, muscle: "+str(muscle), fontsize=fontsize)

    elif angle_range == 2:
        for row in range(len(val_x)-1, -1, -1):
            if val_x[row][0]<0.75:
                eval = np.delete(eval, row, 0)
        if LSTM:
            plt.plot(eval[:, muscle])
            plt.title("LSTM, angle_range: 0.75-1, muscle: "+str(muscle), fontsize=fontsize)
        else:
            plt.plot(eval[:, muscle])
            plt.title("Dense, angle_range: 0.75-1, muscle: "+str(muscle), fontsize=fontsize)

    else:
        raise ValueError("The angle range hast to be \n" +
                  " 0 for validation samples with an angle between 0 and 0.25 \n" +
                  " 1 for validation samples with an angle between 0.25 and 0.75 \n" +
                  " 2 for validation samples with an angle between 0.75 and 1")

if __name__ == "__main__":
    trainingset_name = 'tr_data_shuffled_and_separated'
    raw_data_path = 'tr_data.npy'
    tr_data, val_data, tr_x, tr_y, val_x, val_y = shuffle_and_separate_tr_data(raw_data_path, trainingset_name)

    # load data
    tr_data, val_data, tr_x, tr_y, val_x, val_y = get_tr_data('tr_data_shuffled_and_separated')  # ('low_data')
    # sort val data by weight, angle and speed
    val_data = sort_data(val_data, [2, 3, 0, 1], False)
    val_x = val_data[:, :4]
    val_y = val_data[:, 4:]
    print('val_shape: ' + str(val_x.shape))
    print('tr_shape: ' + str(tr_x.shape))

    # reshape for LSTM (not needed if using next block of code)
    # tr_x = np.reshape(tr_x, [tr_x.shape[0], 1, tr_x.shape[1]])
    # val_x = np.reshape(val_x, [val_x.shape[0], 1, val_x.shape[1]])

    # data for n timesteps LSTM (use these 2 lines to create LSTM-ready data)
    n = 1
    # data gets sorted by weight, angle, speed in 'create_samples_ntimesteps'
    trn_x, trn_y, valn_x, valn_y = create_samples_ntimesteps(tr_data, val_data, n)


    # 8 layers parameters for 50 nets
    layers_arr8 = np.zeros([50, 8])
    for i in range(1, 51):
        setup8 = [i, math.ceil(1.5 * i), 2 * i, 3 * i, 2 * i, i, math.ceil(i * 0.5), 5]
        layers_arr8[i - 1] = setup8

    # 7 layers parameters for 50 nets
    layers_arr7 = np.zeros([50, 7])
    for i in range(1, 51):
        setup7 = [i, math.ceil(1.5 * i), 2 * i, 2 * i, i, math.ceil(0.5 * i), 5]
        layers_arr7[i - 1] = setup7

    # 6 layers parameters for 50 nets
    layers_arr6 = np.zeros([50, 6])
    for i in range(1, 51):
        setup6 = [i, math.ceil(1.5 * i), 2 * i, i, math.ceil(0.5 * i), 5]
        layers_arr6[i - 1] = setup6

    # 5 layers parameters for 50 nets
    layers_arr5 = np.zeros([50, 5])
    for i in range(1, 51):
        setup5 = [i, math.ceil(1.5 * i), 2 * i, i, 5]
        layers_arr5[i - 1] = setup5

    # 4 layers parameters for 50 nets
    layers_arr4 = np.zeros([50, 4])
    for i in range(1, 51):
        setup4 = [i, 2 * i, math.ceil(1.5 * i), 5]
        layers_arr4[i - 1] = setup4

    # 3 layers parameters for 50 nets
    layers_arr3 = np.zeros([50, 3])
    for i in range(1, 51):
        setup3 = [math.ceil(0.5 * i), math.ceil(0.75 * i), 5]
        layers_arr3[i - 1] = setup3


    # testing networks
    # create networks with create_networksLSTM or create_networksDENSE
    # eval_models( ) saves and returns results as array with [mae, avg inference time] for each model and an array of the mae for all steps
    #
    layers = np.array([[10, 20, 40, 20, 10, 5]])  # 5, 20, 5
    models_LSTM = create_networksLSTM(trn_x, trn_y, layers)
    layers = np.array([[10, 20, 5]])  # 5, 20, 5
    # choose between: create_networksDENSE()        or      create_networksLSTM()
    models_DENSE = create_networksDENSE(tr_x, tr_y, layers)  # create_networksDENSE      create_networksLSTM
    resultsD_DENSE, mae_DENSE = eval_models(val_x, val_y, models_DENSE, '3', 'D')
    resultsD_LSTM, mae_LSTM = eval_models(valn_x, valn_y, models_LSTM, '3', 'D')
    mae = np.concatenate((mae_DENSE[0, :].reshape(mae_DENSE.shape[1], 1), mae_LSTM[0, :].reshape(mae_LSTM.shape[1], 1)), axis=1)
    plot_model(models_DENSE[0], to_file='model_DENSE.png', show_shapes=True, show_layer_names=True)
    plot_model(models_LSTM[0], to_file='model_LSTM.png', show_shapes=True, show_layer_names=True)

    fig_mae, axs_mae = plt.subplots(1, 1)
    plt.title('MAE')
    mae_angle_range = np.zeros((2, 3))
    for LSTM in range(0, 2):
        for angle_range in range(0, 3):
            average_mae = plot_mae_for_angle_range(val_x, mae, angle_range, LSTM)
            mae_angle_range[LSTM, angle_range] = average_mae
    print('Average MAE of dense network: ' + str(np.mean(mae_angle_range[0, :].flatten())))
    print('Average MAE of LSTM network: ' + str(np.mean(mae_angle_range[1, :].flatten())))
    plt.legend(['Dense, angle: 0-0.25', 'Dense, angle: 0.25-0.75', 'Dense, angle: 0.75-1',
                'LSTM, angle: 0-0.25', 'LSTM, angle: 0.25-0.75', 'LSTM, angle: 0.75-1'])
    plt.get_current_fig_manager().window.showMaximized()

    for LSTM in range(0, 2):
        for angle_range in range(0, 3):
            print_average_mae_for_each_angle_range(angle_range, mae_angle_range[LSTM, angle_range], LSTM)
    print_average_mae_for_each_angle_range(mae_angle_range[LSTM, angle_range], LSTM)


    fig_output, axs_output = plt.subplots(3, 5)
    fig_output.suptitle('Muscle activation')
    counter = 1
    for angle_range in range(0, 3):
        for muscle in range(0, 5):
            plt.subplot(3, 5, counter)
            plot_outputs(models_DENSE, val_x, False, angle_range, muscle)
            plot_outputs(models_LSTM, val_x, True, angle_range, muscle)
            plt.legend(['Dense', 'LSTM'])
            plt.ylim(-0.2, 1.2)
            counter += 1
    plt.get_current_fig_manager().window.showMaximized()


    plt.show()




