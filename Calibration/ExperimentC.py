import os
import pandas as pd
import pingouin as pg
from Calibration.Performance import average_ECE_over_domains_TF
import numpy as np
from keras import models
from keras import layers
import tensorflow as tf

from Calibration import DataUtils


def build_model():
    model = models.Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.LayerNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer= "RMSprop", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def acc(pre, true):
    total = 0
    count = 0
    for i,j in zip(pre, true):
        if i == j:
            count = count + 1
            total = total + 1
        else:
            total = total + 1
    return count/total

def pred_class(confident):
    pred = []
    for i in confident:
        if i > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred


num_envs = 20
all_domain_idx = [i for i in range(1, num_envs + 1)]
unseen_domain_idx = [j for j in range(11, 21)]
training_domain_idx = [i for i in range(1, 11)]
folder_path = "../Data/Model_selection/sp7inf4redun2NumEnvs20"
data_files_paths = os.listdir(folder_path)
selected_features = 13
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

acc_over_validation_set = []
acc_over_unseen_domain = []
average_ece_over_training_domain = []

dataset_path = folder_path + "/DataSet1"
training_domains, unseen_domains = DataUtils.get_train_unseen_domains(training_domain_idx, unseen_domain_idx,
                                                                      dataset_path,
                                                                      selected_features, separate_for_training=True,
                                                                      separate_for_unseen=True)

train_domain_in_one, unseen_domains_in_one = DataUtils.get_train_unseen_domains(training_domain_idx, unseen_domain_idx,
                                                                                dataset_path,
                                                                                selected_features,
                                                                                separate_for_training=False,
                                                                                separate_for_unseen=False)

for i in range(0, 400):
    print("round " + str(i))
    Model = build_model()
    # loop all the dataset (train and calibrate) and get the ECE ACC etc for each dataset
    acc_on_validation_one_dataset = []
    acc_unseen_one_dataset = []
    average_ece_one_dataset = []
    for dataset in data_files_paths:
        if not dataset.__contains__("result.txt"):
            print("Running on: " + dataset)

            # load the dataset
            train_x = train_domain_in_one[0][0]
            train_y = train_domain_in_one[0][1]
            unseen_x = unseen_domains_in_one[0][0]
            unseen_y = unseen_domains_in_one[0][1]
            X_train, X_validation, X_test, y_train, y_validation, y_test = DataUtils.train_test_validation_split(train_x,train_y)
            #
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

            # set up the model
            history = Model.fit(X_train,
                                y_train,
                                epochs=50, batch_size =512,
                                validation_data=(X_validation, y_validation), callbacks=[callback])


            confident_unseen = Model.predict(unseen_x, verbose=1)
            pred_class_unseen = pred_class(confident_unseen)
            acc_unseen = acc(pred_class_unseen, unseen_y)

            ECE = average_ECE_over_domains_TF(Model, training_domains, training_domain_idx)
            confident_val = Model.predict(X_validation, verbose=1)
            pred_class_val = pred_class(confident_val)
            acc_val = acc(pred_class_val, y_validation)

            acc_on_validation_one_dataset.append(acc_val)
            acc_unseen_one_dataset.append(acc_unseen)
            average_ece_one_dataset.append(ECE)
            print("acc_val: " + str(acc_val))
            print("acc_unseen: " + str(acc_unseen))
            print("ECE: " + str(ECE))

    acc_over_validation_set.append(np.average(acc_on_validation_one_dataset))
    acc_over_unseen_domain.append(np.average(acc_unseen_one_dataset))
    average_ece_over_training_domain.append(np.average(average_ece_one_dataset))

f_res = open(folder_path + 'model_selection.txt', 'a')
f_res.write("acc_over_validation_set:" + str(acc_over_validation_set) + '\n')
f_res.write("average_ece_over_training_domain:" + str(average_ece_over_training_domain) + '\n')
f_res.write("acc_over_unseen_domain:" + str(acc_over_unseen_domain) + '\n')
data = np.array([acc_over_validation_set, acc_over_unseen_domain, average_ece_over_training_domain])
DataUtils.plot_ECE_VS_ACC(average_ece_over_training_domain, acc_over_unseen_domain, folder_path + "/graph/")
DataUtils.plot_ValidationACC_VS_UnseenACC(acc_over_validation_set, acc_over_unseen_domain, folder_path + "/graph/")
pccs1 = np.corrcoef(average_ece_over_training_domain, acc_over_unseen_domain)
pccs2 = np.corrcoef(acc_over_validation_set, acc_over_unseen_domain)
print(pccs1)
print(pccs2)
test = {
"acc_over_validation_set": acc_over_validation_set,
"average_ece_over_training_domain":average_ece_over_training_domain,
"acc_over_unseen_domain": acc_over_unseen_domain
}
df = pd.DataFrame(test, columns = ['acc_over_validation_set','average_ece_over_training_domain', 'acc_over_unseen_domain'])
print(pg.partial_corr(data=df, x='average_ece_over_training_domain', y='acc_over_unseen_domain', covar='acc_over_validation_set'))
a = df.pcorr().round(3)
a.to_csv(folder_path + 'model_selection_partial_corr.csv')

