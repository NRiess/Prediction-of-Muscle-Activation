# neural net for mapping movement data like velocities and accelerations to muscle activations
import numpy as np
import tensorflow as tf
from numpy import savetxt
from operator import itemgetter, attrgetter
import string
import time

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.legend_handler import HandlerLine2D

def shuffle_and_separate_tr_data(raw_data_path, save_tr_val_as):
    raw_data = np.load(raw_data_path)
    raw_data[:, 0] /= 150
    raw_data[:, 1] /= 150
    raw_data[:, 2] /= 250
    raw_data[:, 3] /= 2.0
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


def get_tr_data(save_tr_val_as):
    tr_data = np.loadtxt('tr_' + save_tr_val_as + '.csv', delimiter=";")  # training data
    val_data = np.loadtxt('val_' + save_tr_val_as + '.csv', delimiter=";")

    tr_x = tr_data[:, 0:4]
    tr_y = tr_data[:, 4:9]
    val_x = val_data[:, 0:4]
    val_y = val_data[:, 4:9]
    return tr_data, val_data, tr_x, tr_y, val_x, val_y


def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model


def mae_quick(predicted, real):
    absolutes = 0.0
    for i in range(5):
        absolutes = absolutes + abs(predicted[i] - real[i])
    return absolutes / 5.0


def compute_mae(val_x, val_y, model):
    # compute mae based on validation set
    result = model.predict(val_x)
    mae = np.zeros([len(val_y), 5])
    for i in range(len(result)):
        absolutes = 0.0
        for j in range(5):
            absolutes = absolutes + abs(result[i, j] - val_y[i, j])
        mae[i, :] = np.hstack([val_x[i, :], absolutes / 5.0])

    return mae


def compute_mae_LSTM(val_x, val_y, model):
    # compute mae based on validation set
    result = model.predict(val_x)
    mae = np.zeros(len(val_y))
    for i in range(len(result)):
        absolutes = 0.0
        for j in range(5):
            absolutes = absolutes + abs(result[i, j] - val_y[i, j])
        mae[i] = absolutes / 5.0

    return mae


def compute_metrics(val_data, val_x, val_y, model, metrics_name):
    # get 2darray with first 4 columns being the input and the last column being the MEA of the neural network
    # only consider the validation set
    loss_val = np.ones([len(val_data), 5]) * np.nan
    mse_val = np.ones([len(val_data), 5]) * np.nan
    mae_val = np.ones([len(val_data), 5]) * np.nan
    for idx in range(len(val_data)):  # range(100):
        scores = model.evaluate(val_x[idx:idx + 1, :], val_y[idx:idx + 1, :], verbose=0)
        tf.keras.backend.clear_session()
        # loss_val[idx, :] = np.hstack([val_x[idx, :], scores[0]])
        # mse_val[idx, :] = np.hstack([val_x[idx, :], scores[1]])
        mae_val[idx, :] = np.hstack([val_x[idx, :], scores[2]])
        if idx % 100 == 0:
            print(str(idx) + '/' + str(len(val_data)))

    # savetxt(metrics_name + '_metrics_loss.csv', loss_val, delimiter=';')
    # savetxt(metrics_name + '_metrics_mse.csv', mse_val, delimiter=';')
    savetxt(metrics_name + '_metrics_mae.csv', mae_val, delimiter=';')
    return loss_val, mse_val, mae_val


def load_metrics(metrics_name):
    # loss = np.loadtxt(metrics_name + '_metrics_loss.csv', delimiter=';')
    # mse = np.loadtxt(metrics_name + '_metrics_mse.csv', delimiter=';')
    mae = np.loadtxt(metrics_name + '_metrics_mae.csv', delimiter=';')
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
                                             idx_feature[4], idx_feature[5], idx_feature[6], idx_feature[7]),
                        reverse=descending)
    if number_of_nonzero_entries == 9:
        result = sorted(data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3],
                                             idx_feature[4], idx_feature[5], idx_feature[6], idx_feature[7],
                                             idx_feature[8]),
                        reverse=descending)
    result = np.array(result)
    return result


def plot_everything(sorted0_input_and_MAE, sorted1_input_and_MAE, sorted2_input_and_MAE, sorted3_input_and_MAE,
                    save_name):
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

    # plt.get_current_fig_manager().window.showMaximized()
    plt.savefig(save_name + '_plot.pdf')

    plt.show()

    # plot results


def plot_sorted(mae_local, mae_smoothed, save_name):
    x = np.arange(0.0, len(mae_local), 1)
    # local
    plt.ylim([0, 0.25])
    plt.scatter(x, mae_local, color='b', s=0.1)
    plt.ylabel('mae')
    plt.xlabel('validation step')
    plt.title('Local')
    plt.savefig(save_name + '_local.pdf')
    plt.show()
    # smoothed
    plt.ylim([0, 0.25])
    plt.scatter(x, mae_smoothed, color='r', s=0.1)
    plt.title('Heuristic Merge')
    plt.ylabel('mae')
    plt.xlabel('validation step')
    plt.savefig(save_name + '_smoothed.pdf')
    plt.show()


def local_smoothing(current_output, remote_output, alpha):
    # compute local smoothing of outputs
    smoothed_output = np.zeros([5])
    for i in range(len(remote_output)):
        smoothed_output[i] = (1 - alpha) * current_output[i] + alpha * remote_output[i]
    return smoothed_output


def compute_change(request, current):
    # compute change between output at request timestep and current timestep
    change = np.zeros([5])
    for i in range(5):
        change[i] = current[i] - request[i]
    return change


def adjust_remote(remote, change):
    # compute output that is matching current timestep with change calculated between request timestep and current timestep
    newremote = np.zeros([5])
    for i in range(5):
        newremote[i] = remote[i] + change[i]
    return newremote

def prepare_training(data, timesteps):
    x, y = list(), list()
    for i in range(len(data)):
        end = i+timesteps
        if end > len(data):
            break
        seq_x, seq_y = data[i:end, 0:4], data[end-1, 4:9]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def shuffle_data(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def create_samples_ntimesteps(tr_data, val_data, n):
    # returns training samples for n timesteps and validation data sorted by weight, velocity and angle

    # sort training data by weight, velocity and angle
    tr_sorted = sort_data(tr_data, [2, 3, 0, 1], False)
    # trainingset for n timesteps
    trn_x, trn_y = prepare_training(tr_sorted, n)
    # shuffle training data
    trn_x, trn_y = shuffle_data(trn_x, trn_y)
    # sorted validation for n timesteps
    val_sorted = sort_data(val_data, [2, 3, 0, 1], False)
    # validationset for n timesteps at a time
    valn_x, valn_y = prepare_training(val_sorted, n)
    return trn_x, trn_y, valn_x, valn_y


def alpha_small_angle(request_step, current_step):
    alpha = 1 - (0.000065*(current_step - request_step) ** 2)
    if current_step - request_step < 100:
        alpha = 1
    else:
        alpha = 1 - (1 / 25 * (current_step - request_step - 100))
    return max(alpha, 0)


def alpha_medium_angle(request_step, current_step):
    alpha = 1 - (0.000178*(current_step - request_step) ** 2)
    if current_step - request_step < 50:
        alpha = 1
    else:
        alpha = 1 - (1/25 * (current_step-request_step-50))
    return max(alpha, 0)


def alpha_large_angle(request_step, current_step):
    alpha = 1 - (1/175 * (current_step-request_step))
    if current_step - request_step < 150:
        alpha = 1
    else:
        alpha = 1 - (1 / 25 * (current_step - request_step - 150))
    return max(alpha, 0)


# evaluation with all outputs available
def evaluate_outputs(val_y, local_result, remote_result, timesteps, k, d):
    # evaluate mae of local and remote output (for testing remote output can be exact output)
    # remote_result = val_y
    totallocal = 0
    totalremote = 0
    totalsmoothed = 0
    continued = np.zeros([len(val_y), 5])
    delayed = local_result
    local = np.zeros([len(val_y)])
    smoothed = np.zeros([len(val_y)])
    remote = np.zeros([len(val_y)])
    number_of_steps_weight_zero = 2426
    number_of_steps_weight_zerotwofive = 2500
    number_of_steps_weight_zerofive = 2565
    number_of_steps_weight_one = 2509

    # only local mae
    for i in range(len(val_y)):
        local[i] = mae_quick(local_result[i], val_y[i])
        totallocal += local[i]
    print('local ', totallocal/len(local))


    # continued remote request every k steps computed only with the changes of the local network
    for h in range(round(len(local_result)/k)):
        for m in range(d, k+d):
            if h*k+m < len(local_result):
                continued_output = adjust_remote(remote_result[h * k],
                                                 compute_change(local_result[h * k], local_result[h * k + m]))
                # save continued outputs of each time step for later testing
                continued[h * k + m] = continued_output
                # compute mae of continued output
                remote[h * k + m] = mae_quick(continued_output, val_y[h * k + m])
                totalremote += remote[h * k + m]
    #np.save('continued_results.npy', continued)
    print('remote ', totalremote/len(remote))
    # smoothed result if remote result arrives d time steps after requesting (merge of remote and local outputs)
    for h in range(round(len(local_result)/k)):
        for m in range(d, k+d):
            if (h*k+m) < len(local_result):
                #alpha = .5
                if h*k+m < number_of_steps_weight_zero:
                    # weight 0.0
                    if h*k+m < 607:
                        # alpha is percentage of remote result taken into account for computed results on the local machine
                        alpha = alpha_small_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 607 <= h*k+m < 1820:
                        alpha = alpha_medium_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 1820 <= h*k+m < 2426:
                        alpha = alpha_large_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)


                if 2426 <= h*k+m < 4926:
                    # weight 0.25
                    if h*k+m < 3051:
                        alpha = alpha_small_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 3051 <= h*k+m < 4301:
                        alpha = alpha_medium_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 4301 <= h * k + m < 4926:
                        alpha = alpha_large_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)

                if 4926 <= h*k+m < 7491:
                    # weight 0.5
                    if h*k+m < 5568:
                        alpha = alpha_small_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 5568 <= h*k+m < 6849:
                        alpha = alpha_medium_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 6849 <= h * k + m < 7491:
                        alpha = alpha_large_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)

                if 7491 <= h*k+m:
                    # weight 1.0
                    if h*k+m < 8119:
                        alpha = alpha_small_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 8119 <= h * k + m < 9373:
                        alpha = alpha_medium_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)
                    if 9373 <= h * k + m < 10000:
                        alpha = alpha_large_angle(h * k, h * k + m)
                        delayed[h * k + m] = local_smoothing(local_result[h * k + m], continued[h * k + m], alpha)


    for k in range(len(val_y)):
        smoothed[k] = mae_quick(delayed[k], val_y[k])
        totalsmoothed += smoothed[k]
    print('smoothed ', totalsmoothed/len(smoothed))

    return local, smoothed


# cv train methods
def cv_train(model, X, Y, idx_lst, n_epochs, batch_size):
    # cross validation training setup
    # goes through the index list idx_lst to choose
    # the current validation data
    results = [None] * len(idx_lst)
    indices = np.array(range(X.shape[0]))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    for n, idx in enumerate(idx_lst):
        print('processing fold %d/%d' % (n + 1, len(idx_lst)))
        val_data = (X[idx[0]:idx[1], :], Y[idx[0]:idx[1], :])
        tr_x = np.vstack([X[:idx[0]], X[idx[1]:]])
        tr_y = np.vstack([Y[:idx[0]], Y[idx[1]:]])
        results[n] = model.fit(tr_x, tr_y, epochs=n_epochs, validation_data=val_data, batch_size=batch_size,
                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=2,
                                                                           min_delta=1e-2, verbose=1)])
    return results, model


def part_data(n_folds, n_samples):
    # function to partition the dataset into n folds by defining
    # the beginning and end index of the validation data within
    # the whole data set
    step_size = int(np.round(n_samples/n_folds))
    idx_lst = [None] * n_folds
    for idx in range(n_folds):
        idx_lst[idx] = [step_size * idx, min(step_size * (idx + 1), n_samples - 1)]
    return idx_lst


# function to create LSTM networks
def create_networksLSTM(tr_x, tr_y, layers_arr):
    # training_x, training_y
    # layers_arr is np arrayof shape [[x, y, 5], [a, b, c, 5]] creates 1 NN with layer1LSTM: x nodes,layer2LSTM: y nodes, layer3Dense: 5
    # and 1 NN with layer1LSTM: a nodes, layer2LSTM: b nodes, layer3LSTM: c nodes, layer4Dense: 5 nodes
    # always use 5 nodes as output layer
    models = []
    for idx in range(len(layers_arr)):
        print('current model: ' + str(idx))
        model = tf.keras.Sequential()
        # input layer with return_sequences=True
        model.add(tf.keras.layers.LSTM(int(layers_arr[idx, 0]), input_shape=(tr_x.shape[1], tr_x.shape[2]),
                                       return_sequences=True))
        # layer with return_sequences=True
        for entry in layers_arr[idx, 1:len(layers_arr[0]) - 2]:
            model.add(tf.keras.layers.LSTM(int(entry), return_sequences=True))
        # layer with return_sequences=False
        if len(layers_arr[idx]) > 2:
            model.add(tf.keras.layers.LSTM(int(layers_arr[idx, len(layers_arr[0]) - 2]), return_sequences=False))
        # output layer
        model.add(tf.keras.layers.Dense(layers_arr[idx, len(layers_arr[0]) - 1], activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
        model.fit(tr_x, tr_y, epochs=50, batch_size=32,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=2, min_delta=1e-4, verbose=1)])
        models.append(model)
    # returns trained models in array
    return models


# function to create DENSE networks
def create_networksDENSE(tr_x, tr_y, layers_arr):
    # training_x, training_y
    # layers_arr is np array of shape [[x, y, 5], [a, b, c, 5]] creates 1 NN with layer1Dense: x nodes,layer2Dense: y nodes, layer3Dense: 5
    # and 1 NN with layer1Dense: a nodes, layer2Dense: b nodes, layer3Dense: c nodes, layer4Dense: 5 nodes
    # always use 5 nodes as output layer
    models = []
    for idx in range(len(layers_arr)):
        print('current model: ' + str(idx))
        model = tf.keras.Sequential()
        # input layer
        model.add(tf.keras.layers.Dense(layers_arr[idx, 0], activation='relu', input_shape=(4,)))
        # middle layers
        for entry in layers_arr[idx, 1:len(layers_arr[0]) - 1]:
            model.add(tf.keras.layers.Dense(entry, activation='relu'))
        # output layer
        model.add(tf.keras.layers.Dense(layers_arr[idx, len(layers_arr[0]) - 1], activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
        model.fit(tr_x, tr_y, epochs=50, batch_size=32,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=2, min_delta=1e-4, verbose=1)])
        models.append(model)
    # returns trained models in array
    return models


# function to evaluate the models with respect to avg inference time and mae
def eval_models(val_x, val_y, models, layers, type):
    results = np.zeros([50, 2])
    mae = np.zeros([len(models), len(val_y)])
    k = 0
    m = 0
    for model in models:
        results_local = model.predict(val_x)
        # all mae for all steps
        for i in range(len(val_y)):
            mae[m, i] = mae_quick(results_local[i], val_y[i])
        start_time = time.process_time()
        model.predict(val_x)
        end_time = time.process_time()
        delta = end_time - start_time
        # one mae for all steps
        eval = model.evaluate(val_x, val_y)  # ['loss', 'mean_squared_error', 'mean_absolute_error']
        results[k, 0] = eval[2]
        results[k, 1] = delta / len(val_x)
        k += 1
        m += 1
    savetxt('results' + layers + type + '.csv', results, delimiter=';')
    savetxt('results' + layers + type + 'mae_only.csv', mae, delimiter=';')
    # return values for each network
    return results, mae


# function to create 3d plots for results of all models of one type (amount of layers)
def plot_results(results, layers, type):
    x = np.arange(0.0, len(results), 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, results[:, 0], results[:, 1], color='b', marker='o')
    ax.set_xlabel('#Network to test')
    ax.set_ylabel('mae')
    ax.set_zlabel('avg. inference time')
    ax.set_title(layers + ' layer test')
    plt.savefig(layers + 'layer' + type + '.pdf')
    plt.show()


def plot_mae(mae_values):
    # use this for
    x = np.arange(len(mae_values))
    plt.scatter(x, mae_values, s=.2)
    plt.xlabel('validation step')
    plt.ylabel('mae')
    #plt.savefig('mae_singlenet.png')
    plt.show()


def networktest_plot():
    # plot single values of mae and inference time for one type
    for i in range(4, 8):
        resLSTM = np.loadtxt('results' + str(i) + 'L.csv', delimiter=';')
        resDENSE = np.loadtxt('results' + str(i) + 'D.csv', delimiter=';')
        #resLSTM = np.loadtxt('results7L.csv', delimiter=';')
        #resDENSE = np.loadtxt('results7D.csv', delimiter=';')
        x = np.arange(0.0, len(resDENSE), 1)
        fig, ax = plt.subplots
        avgD = 0
        avgL = 0
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(x, resLSTM[:, 0], resLSTM[:, 1], color='b', marker='o', label='LSTM')
        #ax.scatter(x, resDENSE[:, 0], resDENSE[:, 1], color='r', marker='o', label='Dense')
        plt.scatter(resLSTM[:, 0], resLSTM[:, 1], color='b', marker= '^', s=10, alpha=0.5, label='LSTM')
        plt.scatter(resDENSE[:, 0], resDENSE[:, 1], color='r', s=10, alpha=0.5, label='DENSE')

        #ax.set_xlabel('Validation data (sorted by angle)')
        #ax.set_ylabel('mae')
        #ax.set_zlabel('avg. inference time')
        '''for j in range(len(resDENSE)-1):
            avgD += resDENSE[j, 0]
            avgL += resLSTM[j, 0]
        avgD = avgD / len(resDENSE)
        avgL = avgL / len(resLSTM)
        #ax.boxplot(resDENSE[:, 0], positions=[resDENSE[0, 1]*10000], vert=False)
        #ax.boxplot(resLSTM[:, 0], positions=[resLSTM[0, 1]*10000], vert=False)'''
        ax.set_xlabel('mae')
        ax.set_ylabel('avg. inference time ($10^{-5}s$)')
        leg = plt.legend()
        #ax.set_title('LSTM (top, mae=' + str(avgL)[:6] + ') vs DENSE (bottom, mae=' + str(avgD)[:6] + ') (5layers)')
        ax.set_title('LSTM vs DENSE (' + str(i) + 'layers)')
        #plt.savefig(str(i) + 'layercomparison.pdf')

        #fig.savefig('7layercomparison')
        #plt.show()