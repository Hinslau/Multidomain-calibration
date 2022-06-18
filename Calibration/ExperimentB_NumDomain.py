import numpy as np
import DataUtils
from sklearn.calibration import CalibratedClassifierCV
from Calibration import Performance, Util_EXB
import os
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

num_envs = 15
all_domain_idx = [i for i in range(1, num_envs + 1)]
unseen_domain_idx = [ 11, 12, 13, 14, 15]
all_training_domain_idx = [item for item in all_domain_idx if item not in unseen_domain_idx]
folder_path = "../Data/ExperimentB_Data/sp4inf3redun0NumEnvs15"
data_files_paths = os.listdir(folder_path)
selected_features = 8

# model setup
names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "Linear SVM",
]

classifiers = [
    linear_model.LogisticRegression(class_weight ='balanced'),
    DecisionTreeClassifier(criterion="gini",  splitter="best", max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
    MLPClassifier(alpha=1, max_iter=100, learning_rate='adaptive',early_stopping=True),
    AdaBoostClassifier( linear_model.LogisticRegression(class_weight ='balanced'), n_estimators=2),
    GaussianNB(),
    SVC(kernel='linear', C=1, probability=True, gamma='auto', class_weight="balanced", max_iter=500),
]

the_num_training_domain = []
acc_of_all_model_over_unseen_based_on_nTrain_after_calibration = DataUtils.create_dic_based_on_name(names)
acc_of_all_model_over_unseen_based_on_nTrain_before_calibration = DataUtils.create_dic_based_on_name(names)
ECE_of_all_model_over_unseen_based_on_nTrain_before_calibration = DataUtils.create_dic_based_on_name(names)
ECE_of_all_model_over_unseen_based_on_nTrain_after_calibration = DataUtils.create_dic_based_on_name(names)
roc_of_all_model_over_unseen_based_on_nTrain_after_calibration = DataUtils.create_dic_based_on_name(names)
roc_of_all_model_over_unseen_based_on_nTrain_before_calibration = DataUtils.create_dic_based_on_name(names)


for numOfTrainingDomain in range(1, len(all_training_domain_idx) + 1):
    # the index of each training domain
    training_domain_idx = all_training_domain_idx[0: numOfTrainingDomain]
    print(training_domain_idx)

    # the number of training domain, put it in the array as X
    the_num_training_domain.append(len(training_domain_idx))

    # save the average ece and acc over all the datasets for each model and put it together
    f_res_all_model = open(folder_path + '/ALL_Result_ + num_training_domain_' +
                           str(len(training_domain_idx)) + 'all_model.txt', 'a')

    for name, original_model in zip(names, classifiers):
        print(name)
        # save the result for each model (each dataset) and put it in a txt file
        f_each_model = open(folder_path + '/num_training_domain' +
                            str(len(training_domain_idx)) + name + '.txt', 'a')

        # List to store the ece and acc before and after calibration for each dataset
        # Later we will calculate the average ECE and ACC over all the datasets
        ECE_before_calibration = []
        ECE_after_calibration = []
        ACC_before_calibration = []
        ACC_after_calibration = []
        ROC_before_calibration = []
        ROC_after_calibration = []

        # loop all the dataset (train and calibrate) and get the ECE ACC etc for each dataset
        for dataset in data_files_paths:
            if not dataset.__contains__(".txt"):
                print("Running on: " + dataset)

                # load the dataset
                dataset_path = folder_path + "/" + dataset
                training_domains, unseen_domains = DataUtils.get_train_unseen_domains(training_domain_idx,
                                                                                      unseen_domain_idx, dataset_path,
                                                                                      selected_features,
                                                                                      separate_for_training=True,
                                                                                      separate_for_unseen=True)

                train_domain_in_one, unseen_domains_in_one = DataUtils.get_train_unseen_domains_inone(
                    training_domain_idx, unseen_domain_idx, dataset_path,
                    selected_features, 0.2, 0.2)

                unseen_x = unseen_domains_in_one[0][0]
                unseen_y = unseen_domains_in_one[0][1]

                training_set = train_domain_in_one.get("training_set")
                test_set = train_domain_in_one.get("test_set")
                validation_set = train_domain_in_one.get("validation_set")
                X_train = training_set[0]
                X_validation = validation_set[0]
                X_test = test_set[0]
                y_train = training_set[1]
                y_validation = validation_set[1]
                y_test = test_set[1]

                # prepossessing
                scaler = None
                if name == "None":
                    scaler = preprocessing.StandardScaler()
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_val_scaled = scaler.transform(X_validation)
                else:
                    X_train_scaled = preprocessing.normalize(X_train)
                    X_test_scaled = preprocessing.normalize(X_test)
                    X_val_scaled = preprocessing.normalize(X_validation)

                # set up the model
                original_model.fit(X_train_scaled, y_train)

                # apply calibration
                calibrated_model = CalibratedClassifierCV(original_model, method='isotonic')
                calibrated_model.fit(X_val_scaled, y_validation)

                # performance
                performance = Performance.summary_performance(original_model, calibrated_model, training_domains,
                                                              training_domain_idx, unseen_domains, unseen_domain_idx,
                                                              X_test_scaled, y_test, scaler)

                Average_ECE_Over_training_domains_before_calibration = performance.get('Average_ECE_Over_training_domains_before_calibration')
                Average_ECE_Over_training_domains_after_calibration = performance.get('Average_ECE_Over_training_domains_after_calibration')
                Average_ACC_Over_unseen_domains_before_calibration = performance.get('Average_ACC_Over_unseen_domains_before_calibration')
                Average_ACC_Over_unseen_domains_after_calibration = performance.get('Average_ACC_Over_unseen_domains_after_calibration')
                Average_ROC_Over_unseen_domains_before_calibration = performance.get('Average_ROC_Over_unseen_domains_before_calibration')
                Average_ROC_Over_unseen_domains_after_calibration = performance.get('Average_ROC_Over_unseen_domains_after_calibration')

                ECE_before_calibration.append(Average_ECE_Over_training_domains_before_calibration)
                ECE_after_calibration.append(Average_ECE_Over_training_domains_after_calibration)
                ACC_before_calibration.append(Average_ACC_Over_unseen_domains_before_calibration)
                ACC_after_calibration.append(Average_ACC_Over_unseen_domains_after_calibration)
                ROC_before_calibration.append(Average_ROC_Over_unseen_domains_before_calibration)
                ROC_after_calibration.append(Average_ROC_Over_unseen_domains_after_calibration)

                f_each_model.write(dataset + '\n')
                for key, value in performance.items():
                    print(key + ': ' + str(value))
                    f_each_model.write(key + ': ' + str(value))
                    f_each_model.write('\n')
        f_each_model.close()

        Average_ECE_across_all_datasets_before_calibration = np.average(ECE_before_calibration)
        Average_ECE_across_all_datasets_after_calibration = np.average(ECE_after_calibration)
        Average_ACC_across_all_datasets_before_calibration = np.average(ACC_before_calibration)
        Average_ACC_across_all_datasets_after_calibration = np.average(ACC_after_calibration)
        Average_ROC_across_all_datasets_before_calibration = np.average(ROC_before_calibration)
        Average_ROC_across_all_datasets_after_calibration = np.average(ROC_after_calibration)

        res = [Average_ECE_across_all_datasets_before_calibration, Average_ECE_across_all_datasets_after_calibration,
               Average_ACC_across_all_datasets_before_calibration, Average_ACC_across_all_datasets_after_calibration,
               Average_ROC_across_all_datasets_before_calibration, Average_ROC_across_all_datasets_after_calibration
               ]

        resList = acc_of_all_model_over_unseen_based_on_nTrain_after_calibration.get(name)
        resList.append(Average_ACC_across_all_datasets_after_calibration)

        resList = acc_of_all_model_over_unseen_based_on_nTrain_before_calibration.get(name)
        resList.append(Average_ACC_across_all_datasets_before_calibration)

        resList = roc_of_all_model_over_unseen_based_on_nTrain_after_calibration.get(name)
        resList.append(Average_ROC_across_all_datasets_after_calibration)

        resList = roc_of_all_model_over_unseen_based_on_nTrain_before_calibration.get(name)
        resList.append(Average_ROC_across_all_datasets_before_calibration)

        resList = ECE_of_all_model_over_unseen_based_on_nTrain_before_calibration.get(name)
        resList.append(Average_ECE_across_all_datasets_before_calibration)

        resList = ECE_of_all_model_over_unseen_based_on_nTrain_after_calibration.get(name)
        resList.append(Average_ECE_across_all_datasets_after_calibration)



        #####################################################
        f_res_all_model.write(name + '\n')
        for re in res:
            print(str(DataUtils.retrieve_name(re)[0]) + ": " + str(re))
            f_res_all_model.write(str(DataUtils.retrieve_name(re)[0]) + ": " + str(re))
            f_res_all_model.write('\n')
    f_res_all_model.close()

f_num_envs_acc_ece = open(folder_path + "/numOfenvironment_acc_ece.txt", 'a')

for k, v in acc_of_all_model_over_unseen_based_on_nTrain_after_calibration.items():
    correlation = Util_EXB.get_correlation(the_num_training_domain, v)
    f_num_envs_acc_ece.write(k + "_ACC_after_calibration = " + str(v))
    f_num_envs_acc_ece.write('\n')
    f_num_envs_acc_ece.write(k + "_correlation_acc_num_data_after = " + str(correlation))
    f_num_envs_acc_ece.write('\n')

for k, v in roc_of_all_model_over_unseen_based_on_nTrain_after_calibration.items():
    correlation = Util_EXB.get_correlation(the_num_training_domain, v)
    f_num_envs_acc_ece.write(k + "ROC_after_calibration = " + str(v))
    f_num_envs_acc_ece.write('\n')
    f_num_envs_acc_ece.write(k + "_correlation_roc_num_data_after = " + str(correlation))
    f_num_envs_acc_ece.write('\n')

for k, v in acc_of_all_model_over_unseen_based_on_nTrain_before_calibration.items():
    correlation = Util_EXB.get_correlation(the_num_training_domain, v)
    f_num_envs_acc_ece.write(k + "_ACC_before_calibration = " + str(v))
    f_num_envs_acc_ece.write('\n')
    f_num_envs_acc_ece.write(k + "_correlation_acc_num_data_before = " + str(correlation))
    f_num_envs_acc_ece.write('\n')

for k, v in roc_of_all_model_over_unseen_based_on_nTrain_before_calibration.items():
    correlation = Util_EXB.get_correlation(the_num_training_domain, v)
    f_num_envs_acc_ece.write(k + "ROC_before_calibration = " + str(v))
    f_num_envs_acc_ece.write('\n')
    f_num_envs_acc_ece.write(k + "_correlation_roc_num_data_before = " + str(correlation))
    f_num_envs_acc_ece.write('\n')

f_num_envs_acc_ece.close()
