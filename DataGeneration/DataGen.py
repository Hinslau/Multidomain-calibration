import os
import random
import numpy as np
import sklearn
from DataGeneration import MatrixGen
import pandas as pd
from sklearn import datasets
import numpy.random as rand


def spurious_features_gen(mean, cov, label_0_num, label_1_num):
    """
    generate the spurious features
    :param mean: the mean of the spurious features
    :param cov: the covariance of the spurious features
    :param label_0_num: the number of data of label 0
    :param label_1_num: the number of data of label 1
    :return: spurious features for label 0, spurious features for label 1
    """
    # spurious features for label 0

    X_spurious_label0 = np.random.multivariate_normal(list(map(lambda x: x * (-0.5), mean)), cov,
                                                      label_0_num)

    # spurious features for label 1
    X_spurious_label1 = np.random.multivariate_normal(list(map(lambda x: x * 0.5, mean)), cov,
                                                      label_1_num)

    return X_spurious_label0, X_spurious_label1


def env_gen(X, y, mean, cov):
    """
    generate the data for the environment
    :param X: X (informative features + redundant features)
    :param y: Y [0, 1]
    :param mean: mean of the spurious features
    :param cov: covariance of the spurious features
    :return: [X, Y] for the environment
    """
    # count the number of data-points label 0
    label_0_num = np.count_nonzero(y == 0)
    # count the number of data-points label 1
    label_1_num = np.count_nonzero(y == 1)

    # generate the spurious features for label 0 and label 1
    X_spurious_label0, X_spurious_label1 = spurious_features_gen(mean, cov, label_0_num, label_1_num)

    # divide the data feature based on different labels
    informative_feature_label0 = []
    informative_feature_label1 = []
    for feature, label in zip(X, y):
        if label == 0:
            informative_feature_label0.append(feature)
        else:
            informative_feature_label1.append(feature)
    informative_feature_label0 = np.array(informative_feature_label0)
    informative_feature_label1 = np.array(informative_feature_label1)

    # put the informative features, redundant features and spurious features together
    # such as x = [x-informative, x-redundant, x-spurious]
    combine_feature_label0 = np.hstack((informative_feature_label0, X_spurious_label0))
    combine_feature_label0 = np.hstack(
        (combine_feature_label0, np.zeros(label_0_num).reshape((label_0_num, 1))))
    combine_feature_label1 = np.hstack((informative_feature_label1, X_spurious_label1))
    combine_feature_label1 = np.hstack(
        (combine_feature_label1, np.ones(label_1_num).reshape((label_1_num, 1))))

    # put all X together
    X = np.append(combine_feature_label0, combine_feature_label1, axis=0)

    return X


def generate_raw_data(num_envs, num_data_each_env, num_informative_features, num_spurious_features,
                      num_redundant_features, low_mean, high_mean, low_cov, high_cov, save_path, ALL_idx, unseen_domains):
    # the number of the data-points over all the environments
    total_num_data = (num_data_each_env * num_envs) + (num_data_each_env * (num_envs - len(unseen_domains)))

    # the number of the features
    n_features = num_informative_features + num_redundant_features

    # create the features for all environments
    flip_y = random.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    class_sep = random.choice([1, 0.95, 0.9, 0.85, 0.8, 0.75])
    scale = random.choice([10, 50, 100, 150, 200, 250, 300, 350, 500, 600, 900, 1000, 1100, 2000, 3000, 4000])
    random_state = random.choice([i for i in range(0, 2000)] )
    X, y = datasets.make_classification(n_samples=total_num_data, n_features=n_features,
                                        n_informative=num_informative_features, n_redundant=num_redundant_features,
                                        random_state= random_state,
                                        weights=(0.5,), flip_y=flip_y, class_sep= class_sep, scale=scale, shuffle=True)

    # all the means
    all_means = []
    all_covar = []
    # create the data for different environments and save it
    for idx in range(1, num_envs + 1):
        # the mean of the environment
        mean = []
        for i in range(0, num_spurious_features):
            mean.append(rand.uniform(low_mean, high_mean))
        print(mean)

        all_means.append(mean)
        # the scale of the covar for the environment
        scale_cov = rand.uniform(low_cov, high_cov)

        # the covariance matrices of the environment
        covar_env = MatrixGen.matGen(num_spurious_features, scale_cov)
        all_covar.append(covar_env)

        # Add the spurious features
        data_env = env_gen(X[(idx - 1) * num_data_each_env: idx * num_data_each_env, :],
                           y[(idx - 1) * num_data_each_env: idx * num_data_each_env],
                           mean, covar_env)

        os.makedirs(save_path, exist_ok=True)
        dataframe = pd.DataFrame(data_env)
        dataframe.to_csv(save_path + "/E" + str(idx) + ".csv")

        if idx in unseen_domains:
            ALL_PTAH = "../Data/ALL_SP" + str(num_spurious_features) + \
                       "INF" + str(num_informative_features) + "RED" + \
                       str(num_redundant_features) + "ENVS" + str(num_envs) + \
                       "/DataSet" + str(ALL_idx) + "/"
            os.makedirs(ALL_PTAH, exist_ok=True)
            dataframe.to_csv(ALL_PTAH +  "/E" + str(idx) + ".csv")

    m1 = all_means[0]
    c1 = all_covar[0]
    d1 = env_gen(X[num_data_each_env * num_envs:, :],
                 y[num_data_each_env * num_envs:],
                 m1, c1)

    ALL_PTAH = "../Data/ALL_SP" + str(num_spurious_features) + \
               "INF" + str(num_informative_features) + "RED" + \
               str(num_redundant_features) + "ENVS" + str(num_envs) + \
               "/DataSet" + str(ALL_idx) + "/"

    os.makedirs(ALL_PTAH, exist_ok=True)
    dataframe = pd.DataFrame(d1)
    dataframe.to_csv(ALL_PTAH + "/E1.csv")

def generate_multi_datasets(number_of_dataset, num_envs, num_data_each_env, num_informative_features,
                            num_spurious_features, num_redundant_features,
                            unseen_domains,low_mean, high_mean, low_cov, high_cov):
    for i in range(1, number_of_dataset + 1):
        save_path = "../Data/sp" + str(num_spurious_features) + "inf" + str(
            num_informative_features) \
                    + "redun" + str(num_redundant_features) + "NumEnvs" + str(num_envs) + "/DataSet" + str(i)
        generate_raw_data(num_envs, num_data_each_env, num_informative_features, num_spurious_features,
                          num_redundant_features, low_mean, high_mean, low_cov, high_cov, save_path, i, unseen_domains)
