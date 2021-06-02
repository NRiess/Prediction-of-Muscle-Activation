# neural net for mapping movement data like velocities and accelerations to muscle activations
import numpy as np
import tensorflow as tf
from numpy import savetxt
from operator import itemgetter, attrgetter
import string
import matplotlib.pyplot as plt

def shuffle_and_separate_tr_data(raw_data_path, save_tr_val_as):
    raw_data = np.loadtxt(raw_data_path, delimiter=";")

    np.random.shuffle(raw_data)
    tr_data = raw_data[0:int(0.8 * len(raw_data)), :]
    val_data = raw_data[int(0.8 * len(raw_data)):len(raw_data), :]

    #  save as npy
    np.save('tr_' + save_tr_val_as + '_npy', tr_data)
    np.save('val_' + save_tr_val_as + '_npy', val_data)
    #  save as csv
    savetxt('tr_' + save_tr_val_as + '.csv', tr_data, delimiter=';')
    savetxt('val_' + save_tr_val_as + '.csv', val_data, delimiter=';')

    tr_x = tr_data[:, 0:4]
    tr_y = tr_data[:, 4:9]
    val_x = val_data[:, 0:4]
    val_y = val_data[:, 4:9]

    return tr_data, val_data, tr_x, tr_y, val_x, val_y


def get_tr_data(raw_data_path, save_tr_val_as):
    tr_data = np.loadtxt('tr_' + save_tr_val_as + '.csv', delimiter=";")    # training data
    val_data = np.loadtxt('val_' + save_tr_val_as + '.csv', delimiter=";")

    tr_x = tr_data[:, 0:4]
    tr_y = tr_data[:, 4:9]
    val_x = val_data[:, 0:4]
    val_y = val_data[:, 4:9]
    print(tr_x.shape)
    print(tr_y.shape)
    return tr_data, val_data, tr_x, tr_y, val_x, val_y


def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model


def compute_metrics(val_data, val_x, val_y, model, metrics_name):
    # get 2darray with first 4 columns being the input and the last column being the MEA of the neural network
    # only consider the validation set
    loss_val = np.ones([len(val_data), 5]) * np.nan
    mse_val = np.ones([len(val_data), 5]) * np.nan
    mae_val = np.ones([len(val_data), 5]) * np.nan
    for idx in range(len(val_data)):  # range(100):
        scores = model.evaluate(val_x[idx:idx + 1, :], val_y[idx:idx + 1, :], verbose=0)
        loss_val[idx, :] = np.hstack([val_x[idx, :], scores[0]])
        mse_val[idx, :] = np.hstack([val_x[idx, :], scores[1]])
        mae_val[idx, :] = np.hstack([val_x[idx, :], scores[2]])
        if idx % 100 == 0:
            print(str(idx) + '/' + str(len(val_data)))
    loss = loss_val
    mse = mse_val
    mae = mae_val
    savetxt(metrics_name + '_loss.csv', loss_val, delimiter=';')
    savetxt(metrics_name + '_mse.csv', mse_val, delimiter=';')
    savetxt(metrics_name + '_mae.csv', mae_val, delimiter=';')
    return loss, mse, mae


def load_metrics(metrics_name):
    loss = np.loadtxt(metrics_name + '_loss.csv', delimiter=';')
    mse = np.loadtxt(metrics_name + '_mse.csv', delimiter=';')
    mae = np.loadtxt(metrics_name + '_mae.csv', delimiter=';')
    return loss, mse, mae


def sort_data(data, sort_priority, descending):
    # tr_data = np.load('tr_data_npy.npy')
    # tr_data = sorted(tr_data, key=itemgetter(idx_feature), reverse=descending)

    # eval_data = np.load('eval_data_npy.npy')
    # eval_data = sorted(eval_data, key=itemgetter(idx_feature), reverse=descending)

    number_of_zeros = 0
    for i in range(len(sort_priority)):
        if sort_priority[i] == 0:
            number_of_zeros += 1
    number_of_nonzero_entries = len(sort_priority) - number_of_zeros

    idx_feature = np.argsort(np.absolute(sort_priority))
    idx_feature = idx_feature[number_of_zeros:len(idx_feature)]


    if number_of_nonzero_entries == 0:
        result = data
    if number_of_nonzero_entries == 1:
        result = sorted(data, key=itemgetter(idx_feature[0]),
                        reverse=descending)
    if number_of_nonzero_entries == 2:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1]),
                        reverse=descending)
    if number_of_nonzero_entries == 3:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2]), reverse=descending)
    if number_of_nonzero_entries == 4:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3]),
                        reverse=descending)
    if number_of_nonzero_entries == 5:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                        idx_feature[4], idx_feature[5]), reverse=descending)
    if number_of_nonzero_entries == 6:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                        idx_feature[4], idx_feature[5]), reverse=descending)
    if number_of_nonzero_entries == 7:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                        idx_feature[4], idx_feature[5], idx_feature[6]), reverse=descending)
    if number_of_nonzero_entries == 8:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                        idx_feature[4], idx_feature[5], idx_feature[6], idx_feature[7]), reverse=descending)
    if number_of_nonzero_entries == 9:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                        idx_feature[4], idx_feature[5], idx_feature[6], idx_feature[7], idx_feature[8]),
                        reverse=descending)
    result = np.array(result)
    return result


def plot_sorted_input(sorted_input_and_MAE):
    fig, ax = plt.subplot(1, 1, 1)
    ax[0, 0] = plt.scatter(np.arange(0, len(sorted_input_and_MAE), 1), sorted_input_and_MAE[:, 4], s=0.1)
    sorted_input_and_MAE_string = [k for k, v in locals().iteritems() if v == sorted_input_and_MAE][0]
    all = string.maketrans('', '')
    nodigs = all.translate(all, string.digits)
    sorted_input_and_MAE_string.translate(all, nodigs)
    fig.suptitle('Sorted by input ' + sorted_input_and_MAE_string)
    plt.show()

def plot_everything(sorted0_input_and_MAE, sorted1_input_and_MAE, sorted2_input_and_MAE, sorted3_input_and_MAE,
                    sorted30_input_and_MAE, sorted31_input_and_MAE, sorted32_input_and_MAE):
    # plot results
    figure, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].scatter(np.arange(0, len(sorted0_input_and_MAE), 1), sorted0_input_and_MAE[:, 4], s=0.1)
    axes[0, 0].set_title('Sorted by input 0')

    axes[0, 1].scatter(np.arange(0, len(sorted1_input_and_MAE), 1), sorted1_input_and_MAE[:, 4], s=0.1)
    axes[0, 1].set_title('Sorted by input 1')

    axes[1, 0].scatter(np.arange(0, len(sorted2_input_and_MAE), 1), sorted2_input_and_MAE[:, 4], s=0.1)
    axes[1, 0].set_title('Sorted by input 2')

    axes[1, 1].scatter(np.arange(0, len(sorted3_input_and_MAE), 1), sorted3_input_and_MAE[:, 4], s=0.1)
    axes[1, 1].set_title('Sorted by input 3')

    plt.get_current_fig_manager().window.showMaximized()

    # plot results

    figure2, axes2 = plt.subplots(nrows=2, ncols=2)

    axes2[0, 0].scatter(np.arange(0, len(sorted30_input_and_MAE), 1), sorted30_input_and_MAE[:, 4], s=0.1)
    axes2[0, 0].set_title('Sorted by input 0')

    axes2[0, 1].scatter(np.arange(0, len(sorted31_input_and_MAE), 1), sorted31_input_and_MAE[:, 4], s=0.1)
    axes2[0, 1].set_title('Sorted by input 1')

    axes2[1, 0].scatter(np.arange(0, len(sorted32_input_and_MAE), 1), sorted32_input_and_MAE[:, 4], s=0.1)
    axes2[1, 0].set_title('Sorted by input 2')

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()



