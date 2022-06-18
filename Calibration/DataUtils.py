import os
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.utils as sklu
import inspect
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def get_raw_data_from_csv(filepath, selected_features, shuffle=True):
    dataframes = pd.read_csv(filepath)
    x = []
    y = []
    for i in dataframes.index:
        data = dataframes.loc[i]
        data = np.asarray(list(map(float, data)))
        item = data[1:selected_features + 1]
        x.append(item.astype(np.float32))
        y.append(data[-1])
    x = np.array(x).astype(np.float32)
    y = np.array(y)
    if shuffle:
        x, y = sklu.shuffle(x, y, random_state=0)
    return x, y


def get_train_validation_test_data_from_folder(folder_path, selected_features):
    X = np.array([]).reshape((0, selected_features))
    Y = np.array([])
    data_files_path = os.listdir(folder_path)
    for file_name in data_files_path:
        if file_name.__contains__(".csv"):
            temp_X, temp_Y = get_raw_data_from_csv(folder_path + "/" + file_name, selected_features)
            X = np.append(X, temp_X, axis=0)
            Y = np.append(Y, temp_Y, axis=0)
    X, Y = sklu.shuffle(X, Y, random_state=2)
    X, Y = sklu.shuffle(X, Y, random_state=42)
    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.3)
    X_train, X_validation, y_train, y_validation = ms.train_test_split(X_train, y_train, test_size=0.5)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def get_train_unseen_domains(training_domain_idx, testing_domain_idx, folder_path, selected_features,
                             separate_for_training, separate_for_unseen):
    data_files_path = os.listdir(folder_path)
    training_domain = []
    unseen_domain = []
    if separate_for_training:
        for idx in training_domain_idx:
            for file_name in data_files_path:
                if file_name.__contains__("E" + str(idx) + ".csv"):
                    if file_name.__contains__(str(idx)):
                        train_x, train_y = \
                            get_raw_data_from_csv(folder_path + "/" + file_name, selected_features, shuffle=True)
                        training_domain.append([train_x, train_y])
                        data_files_path.remove(file_name)
    else:
        training_domain_X = np.array([]).reshape((0, selected_features))
        training_domain_Y = np.array([])
        for idx in training_domain_idx:
            for file_name in data_files_path:
                if file_name.__contains__("E" + str(idx) + ".csv"):
                    if file_name.__contains__(str(idx)):
                        train_x, train_y = \
                            get_raw_data_from_csv(folder_path + "/" + file_name, selected_features, shuffle=True)
                        training_domain_X = np.append(training_domain_X, train_x, axis=0)
                        training_domain_Y = np.append(training_domain_Y, train_y, axis=0)
                        data_files_path.remove(file_name)
        training_domain.append([training_domain_X, training_domain_Y])

    if separate_for_unseen:
        for idx_2 in testing_domain_idx:
            for file_name_left in data_files_path:
                if file_name_left.__contains__("E" + str(idx_2) + ".csv"):
                    if file_name_left.__contains__(str(idx_2)):
                        unseen_x, unseen_y = \
                            get_raw_data_from_csv(folder_path + "/" + file_name_left, selected_features, shuffle=True)
                        unseen_domain.append([unseen_x, unseen_y])
                        data_files_path.remove(file_name_left)

    else:
        unseen_domain_X = np.array([]).reshape((0, selected_features))
        unseen_domain_Y = np.array([])
        for idx_2 in testing_domain_idx:
            for file_name_left in data_files_path:
                if file_name_left.__contains__("E" + str(idx_2) + ".csv"):
                    if file_name_left.__contains__(str(idx_2)):
                        unseen_x, unseen_y = \
                            get_raw_data_from_csv(folder_path + "/" + file_name_left, selected_features, shuffle=True)
                        unseen_domain_X = np.append(unseen_domain_X, unseen_x, axis=0)
                        unseen_domain_Y = np.append(unseen_domain_Y, unseen_y, axis=0)
                        data_files_path.remove(file_name_left)
        unseen_domain.append([unseen_domain_X, unseen_domain_Y])
    return training_domain, unseen_domain


def train_test_validation_split(X, Y, validation_size, test_size):
    X, Y = sklu.shuffle(X, Y, random_state=2)
    X_train, X_validation, y_train, y_validation = ms.train_test_split(X, Y, test_size=validation_size)
    X_train, X_test, y_train, y_test = ms.train_test_split(X_train, y_train, test_size=test_size)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def plot_model_numEnv_ACC(model_name, xlabel, ylabel, numEnv, ACC, path):
    plt.rcParams.update({'font.size': 14})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(model_name)
    plt.scatter(numEnv, ACC)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + model_name + ".jpg")
    plt.close()

def create_dic_based_on_name(names):
    dict = {}
    for name in names:
        l = []
        dict.update(({name: l}))
    return dict

def plot_ECE_VS_ACC(ECE, ACC, path):
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Expected Calibration Error")
    plt.ylabel("Average accuracy on the unseen domains")
    plt.title("ACC VS ECE")
    plt.scatter(ECE, ACC)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "ECC_VS_ACC.jpg")
    plt.close()

def plot_ValidationACC_VS_UnseenACC(ValidationACC, UnseenACC, path):
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Average accuracy on the validation set of all training domains")
    plt.ylabel("Average accuracy on the unseen domains")
    plt.title("Validation ACC VS Unseen ACC")
    plt.scatter(ValidationACC, UnseenACC)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "Validation_ACC_VS_Unseen_ACC.jpg")
    plt.close()

def get_train_unseen_domains_inone(training_domain_idx, testing_domain_idx, folder_path, selected_features, validation_size, test_size):
    data_files_path = os.listdir(folder_path)
    training_domain = {"training_set": [np.array([]).reshape((0, selected_features)),  np.array([])],
                       "test_set": [np.array([]).reshape((0, selected_features)),  np.array([])],
                       "validation_set": [np.array([]).reshape((0, selected_features)),  np.array([])]}
    unseen_domain = []
    for idx in training_domain_idx:
        for file_name in data_files_path:
            if file_name.__contains__("E" + str(idx) + ".csv"):
                if file_name.__contains__(str(idx)):
                    train_x, train_y = \
                        get_raw_data_from_csv(folder_path + "/" + file_name, selected_features, shuffle=True)
                    X_train, X_validation, X_test, y_train, y_validation, y_test = train_test_validation_split(train_x, train_y, validation_size, test_size)
                    training_data = training_domain.get("training_set")
                    training_data[0] = np.append(training_data[0], X_train, axis=0)
                    training_data[1] = np.append(training_data[1], y_train, axis=0)

                    validation_data = training_domain.get("validation_set")
                    validation_data[0] = np.append(validation_data[0], X_validation, axis=0)
                    validation_data[1] = np.append(validation_data[1],y_validation, axis=0)

                    test_data = training_domain.get("test_set")
                    test_data[0] = np.append(test_data[0], X_test, axis=0)
                    test_data[1] = np.append(test_data[1], y_test, axis=0)

                    data_files_path.remove(file_name)

    unseen_domain_X = np.array([]).reshape((0, selected_features))
    unseen_domain_Y = np.array([])
    for idx_2 in testing_domain_idx:
        for file_name_left in data_files_path:
            if file_name_left.__contains__("E" + str(idx_2) + ".csv"):
                if file_name_left.__contains__(str(idx_2)):
                    unseen_x, unseen_y = \
                        get_raw_data_from_csv(folder_path + "/" + file_name_left, selected_features, shuffle=True)
                    unseen_domain_X = np.append(unseen_domain_X, unseen_x, axis=0)
                    unseen_domain_Y = np.append(unseen_domain_Y, unseen_y, axis=0)
                    data_files_path.remove(file_name_left)
    unseen_domain.append([unseen_domain_X, unseen_domain_Y])

    return training_domain, unseen_domain
