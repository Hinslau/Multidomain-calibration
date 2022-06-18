import numpy as np
import DataUtils
from sklearn.calibration import CalibratedClassifierCV
from Calibration import Performance
import os
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from scipy.stats import kstest
import scipy.stats as ts

num_envs = 8
all_domain_idx = [i for i in range(1, num_envs + 1)]
selected_num_training_domain = 4
training_domain_idx = [i for i in range(1, selected_num_training_domain+1)]
unseen_domain_idx = [5, 6, 7, 8]
folder_path = "../Data/ExperimentA_Data/sp4inf4redun0NumEnvs8"
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
    DecisionTreeClassifier(criterion="gini",  splitter="best"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
    MLPClassifier(alpha=1, learning_rate='adaptive',early_stopping=True),
    AdaBoostClassifier(linear_model.LogisticRegression(class_weight ='balanced'), n_estimators=2),
    GaussianNB(),
    SVC(kernel='linear', C=1, probability=True, gamma='auto', class_weight="balanced"),
]



# save the average ece and acc over all the datasets for each model and put it together
f_res_all_model = open(folder_path + '/ALL_Result_ + num_training_domain_' +
                       str(selected_num_training_domain) + 'all_model.txt', 'a')

for name, original_model in zip(names, classifiers):
    print(name)
    # save the result for each model (each dataset) and put it in a txt file
    f_each_model = open(folder_path + '/num_training_domain' +
                        str(selected_num_training_domain) + name + '.txt', 'a')

    # List to store the ece and acc before and after calibration for each dataset
    # Later we will calculate the average ECE and ACC over all the datasets
    ECE_before_calibration = []
    ECE_after_calibration = []
    ACC_before_calibration = []
    ACC_after_calibration = []
    ECE_before_calibration_Test = []
    ECE_after_calibration_Test = []
    ROC_before_calibration = []
    ROC_after_calibration = []


    # loop all the dataset (train and calibrate) and get the ECE ACC etc for each dataset
    COUNT = 0
    WORKING_TIME = 0
    DIFF_ACC = []
    DIFF_ROC = []
    DIFF_ECE = []
    DIFF_ECE_TEST = []
    for dataset in data_files_paths:
        if not dataset.__contains__(".txt"):
            print("Running on: " + dataset)

            # load the dataset
            dataset_path = folder_path + "/" + dataset
            training_domains, unseen_domains = DataUtils.get_train_unseen_domains(training_domain_idx, unseen_domain_idx, dataset_path,
                                                                                  selected_features,
                                                                                  separate_for_training=True,
                                                                                  separate_for_unseen=True)

            train_domain_in_one, unseen_domains_in_one = DataUtils.get_train_unseen_domains_inone(training_domain_idx, unseen_domain_idx, dataset_path,
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
            ECE_over_test_set_before_calibration = performance.get("ECE_over_test_set_before_calibration")
            ECE_over_test_set_after_calibration = performance.get("ECE_over_test_set_after_calibration")
            Average_ROC_Over_unseen_domains_before_calibration =performance.get('Average_ROC_Over_unseen_domains_before_calibration')
            Average_ROC_Over_unseen_domains_after_calibration = performance.get('Average_ROC_Over_unseen_domains_after_calibration')

            ECE_before_calibration.append(Average_ECE_Over_training_domains_before_calibration)
            ECE_after_calibration.append(Average_ECE_Over_training_domains_after_calibration)
            ACC_before_calibration.append(Average_ACC_Over_unseen_domains_before_calibration)
            ACC_after_calibration.append(Average_ACC_Over_unseen_domains_after_calibration)
            ROC_before_calibration.append(Average_ROC_Over_unseen_domains_before_calibration)
            ROC_after_calibration.append(Average_ROC_Over_unseen_domains_after_calibration)
            ECE_before_calibration_Test.append(ECE_over_test_set_before_calibration)
            ECE_after_calibration_Test.append(ECE_over_test_set_after_calibration)


            diff_acc = Average_ACC_Over_unseen_domains_after_calibration - Average_ACC_Over_unseen_domains_before_calibration
            DIFF_ACC.append(diff_acc)
            diff_roc = Average_ROC_Over_unseen_domains_after_calibration - Average_ROC_Over_unseen_domains_before_calibration
            DIFF_ROC.append(diff_roc)

            diff_ece = Average_ECE_Over_training_domains_after_calibration - Average_ECE_Over_training_domains_before_calibration
            DIFF_ECE.append(diff_ece)
            diff_ece_test = ECE_over_test_set_after_calibration - ECE_over_test_set_before_calibration
            DIFF_ECE_TEST.append(diff_ece_test)

            f_each_model.write(dataset + '\n')
            for key, value in performance.items():
                print(key + ': ' + str(value))
                f_each_model.write(key + ': ' + str(value))
                f_each_model.write('\n')
    f_each_model.close()

    if DIFF_ACC:
        Average_DIFF_ACC = np.average(DIFF_ACC)
    else:
        Average_DIFF_ACC = -100
    if DIFF_ROC:
        Average_DIFF_ROC = np.average(DIFF_ROC)
    else:
        Average_DIFF_ROC = -100
    if DIFF_ECE:
        Average_DIFF_ECE = np.average(DIFF_ECE)
    else:
        Average_DIFF_ECE = -100
    if DIFF_ECE_TEST:
        Average_DIFF_ECE_Test = np.average(DIFF_ECE_TEST)
    else:
        Average_DIFF_ECE_Test = -100

    Ttest_ACC = stats.ttest_1samp(DIFF_ACC, 0, axis=0)
    DIFF_ACC_Normal = kstest(DIFF_ACC, 'norm')
    P_value_ACC = Performance.P_value_ACC(DIFF_ACC)
    data_Acc = (DIFF_ACC,)
    CI_ACC = ts.bootstrap(data_Acc, np.mean, confidence_level=0.95, n_resamples=5000)

    Ttest_ECE = stats.ttest_1samp(DIFF_ECE, 0, axis=0)
    DIFF_ECE_Normal = kstest(DIFF_ECE, 'norm')
    P_value_ECE_Training = Performance.P_value_ECE(DIFF_ECE)

    Ttest_ECE_TEST = stats.ttest_1samp(DIFF_ECE_TEST, 0, axis=0)
    DIFF_ECE_TEST_Normal = kstest(DIFF_ECE_TEST, 'norm')
    P_value_ECE_Test = Performance.P_value_ECE(DIFF_ECE_TEST)

    Ttest_ROC = stats.ttest_1samp(DIFF_ROC, 0, axis=0)
    DIFF_ROC_Normal = kstest(DIFF_ROC, 'norm')
    P_value_ROC = Performance.P_value_ACC(DIFF_ROC)
    data_roc = (DIFF_ROC,)
    CI_ROC = ts.bootstrap(data_roc, np.mean, confidence_level=0.95, n_resamples=5000)

    Average_ECE_across_all_datasets_before_calibration = np.average(ECE_before_calibration)
    Average_ECE_across_all_datasets_after_calibration = np.average(ECE_after_calibration)
    Average_ECE_ON_TEST_before_calibration = np.average(ECE_before_calibration_Test)
    Average_ECE_ON_TEST_after_calibration = np.average(ECE_after_calibration_Test)
    Average_ACC_across_all_datasets_before_calibration = np.average(ACC_before_calibration)
    Average_ACC_across_all_datasets_after_calibration = np.average(ACC_after_calibration)
    Average_ROC_across_all_datasets_before_calibration = np.average(ROC_before_calibration)
    Average_ROC_across_all_datasets_after_calibration = np.average(ROC_after_calibration)

    res = [Average_ECE_across_all_datasets_before_calibration, Average_ECE_across_all_datasets_after_calibration,
           Average_ECE_ON_TEST_before_calibration, Average_ECE_ON_TEST_after_calibration,
           DIFF_ECE, Average_DIFF_ECE, Ttest_ECE, DIFF_ECE_Normal, P_value_ECE_Training,
           DIFF_ECE_TEST, Average_DIFF_ECE_Test, Ttest_ECE_TEST, DIFF_ECE_TEST_Normal, P_value_ECE_Test,
           Average_ACC_across_all_datasets_before_calibration, Average_ACC_across_all_datasets_after_calibration,
           DIFF_ACC, Average_DIFF_ACC, Ttest_ACC, DIFF_ACC_Normal, P_value_ACC, CI_ACC,
           Average_ROC_across_all_datasets_before_calibration, Average_ROC_across_all_datasets_after_calibration,
           DIFF_ROC, Average_DIFF_ROC, Ttest_ROC, DIFF_ROC_Normal, P_value_ROC, CI_ROC
        ]

    f_res_all_model.write(name + '\n')
    for re in res:
        print(str(DataUtils.retrieve_name(re)[0]) + ": " + str(re))
        f_res_all_model.write(str(DataUtils.retrieve_name(re)[0]) + ": " + str(re))
        f_res_all_model.write('\n')
    f_res_all_model.write('\n')
f_res_all_model.close()
