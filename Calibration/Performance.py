import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from calibration_module.utils import compute_calibration_error

def ECE_per_domain(model, domains, idx, scaler):
    res = {}
    for (domain, i) in zip(domains, idx):
        x = domain[0]
        y = domain[1]
        if scaler != None:
            x = scaler.transform(x)
        else:
            x = preprocessing.normalize(x)
        pred_prob = model.predict_proba(x)[:, 1]
        ECE = compute_calibration_error(y, pred_prob, 15, 4)
        res.update(({'ECE on domain' + str(i): ECE}))
    return res

def ECE_per_domain_TF(model, domains, idx):
    res = {}
    for (domain, i) in zip(domains, idx):
        x = domain[0]
        y = domain[1]
        pred_prob = model.predict(x)
        ECE = compute_calibration_error(y, pred_prob, 25, 5)
        print(ECE)
        res.update(({'ECE on domain' + str(i): ECE}))
    return res

def average_ECE_over_domains_TF(model, domains, idx):
    res = ECE_per_domain_TF(model, domains, idx)
    print(dict_Avg(res))
    return dict_Avg(res)


def ACC_per_domain(model, domains, idx, scaler):
    res = {}
    for (domain, i) in zip(domains, idx):
        x = domain[0]
        y = domain[1]
        if scaler != None:
            x = scaler.transform(x)
        else:
            x = preprocessing.normalize(x)
        pred = model.predict(x)
        acc = metrics.accuracy_score(y, pred)
        res.update(({'ACC on domain' + str(i): acc}))
    return res

def ACC_on_test_set(model, x, y):
    res = {}
    pred = model.predict(x)
    acc = metrics.accuracy_score(y, pred)
    res.update(({'ACC on test set' : acc}))
    return res

def ROC_per_domain(model, domains, idx, scaler):
    res = {}
    for (domain, i) in zip(domains, idx):
        x = domain[0]
        y = domain[1]
        if scaler != None:
            x = scaler.transform(x)
        else:
            x = preprocessing.normalize(x)
        pred = model.predict_proba(x)[:, 1]
        # print(pred)
        acc = metrics.roc_auc_score(y, pred)
        res.update(({'AUC on domain' + str(i): acc}))
    return res


def average_ECE_over_domains(model, domains, idx, scaler):
    res = ECE_per_domain(model, domains, idx, scaler)
    return dict_Avg(res)


def average_ACC_over_domains(model, domains, idx, scaler):
    res = ACC_per_domain(model, domains, idx, scaler)
    return dict_Avg(res)

def average_ROC_over_domains(model, domains, idx, scaler):
    res = ROC_per_domain(model, domains, idx, scaler)
    return dict_Avg(res)


def ECE_over_test_set(model, test_x, test_y):
    pred_prob = model.predict_proba(test_x)[:, 1]
    ECE = compute_calibration_error(test_y, pred_prob, 15, 4)
    return ECE

def dict_Avg(Dict):
    L = len(Dict)
    S = sum(Dict.values())
    if S == 0:
        A = 0
    else:
        A = S / L
    return A

def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

def P_value_ACC(force_b):
    translated_force_b = force_b - np.mean(force_b) - 0

    # Take bootstrap replicates of Frog B`s translated impact forces: bs_replicates
    bs_replicates = draw_bs_reps(translated_force_b, np.mean, 5000)

    # Compute fraction of replicates that are larger than the observed Frog B force: p
    p = np.sum(bs_replicates > np.mean(force_b)) / 5000
    return p

def P_value_ECE(force_b):
    translated_force_b = force_b - np.mean(force_b) - 0

    # Take bootstrap replicates of Frog B`s translated impact forces: bs_replicates
    bs_replicates = draw_bs_reps(translated_force_b, np.mean, 5000)

    # Compute fraction of replicates that are larger than the observed Frog B force: p
    p = np.sum(bs_replicates < np.mean(force_b))  / 5000
    return p



def summary_performance(original_model, calibrated_model, training_domains, training_domain_idx,
                        unseen_domains, unseen_domain_idx, X_test_scaled, y_test, scaler=None):
    ECE_Over_training_domains_before_calibration = \
        ECE_per_domain(original_model, training_domains, training_domain_idx, scaler)

    ECE_Over_training_domains_after_calibration = \
        ECE_per_domain(calibrated_model, training_domains,
                       training_domain_idx, scaler)

    ACC_Over_unseen_domain_before_calibration = \
        ACC_per_domain(original_model, unseen_domains,
                       unseen_domain_idx, scaler)

    ACC_Over_unseen_domain_after_calibration = \
        ACC_per_domain(calibrated_model, unseen_domains,
                       unseen_domain_idx, scaler)

    Average_ECE_Over_training_domains_before_calibration = \
        average_ECE_over_domains(original_model, training_domains,
                                 training_domain_idx, scaler)

    Average_ECE_Over_training_domains_after_calibration = \
        average_ECE_over_domains(calibrated_model, training_domains,
                                 training_domain_idx, scaler)

    Average_ACC_Over_unseen_domains_before_calibration = \
        average_ACC_over_domains(original_model, unseen_domains,
                                 unseen_domain_idx, scaler)

    Average_ACC_Over_unseen_domains_after_calibration = \
        average_ACC_over_domains(calibrated_model, unseen_domains,
                                 unseen_domain_idx, scaler)

    ECE_over_test_set_before_calibration = \
        ECE_over_test_set(original_model, X_test_scaled, y_test)

    ECE_over_test_set_after_calibration = \
        ECE_over_test_set(calibrated_model, X_test_scaled, y_test)

    Average_ROC_Over_unseen_domains_before_calibration = \
        average_ROC_over_domains(original_model, unseen_domains,
                                 unseen_domain_idx, scaler)

    Average_ROC_Over_unseen_domains_after_calibration = \
        average_ROC_over_domains(calibrated_model, unseen_domains,
                                 unseen_domain_idx, scaler)

    ACC_ON_TEST_SET = ACC_on_test_set(calibrated_model, X_test_scaled, y_test)
    res = {'ACC_ON_TEST_SET' : ACC_ON_TEST_SET,
           'ECE_Over_training_domains_before_calibration': ECE_Over_training_domains_before_calibration,
           'ECE_Over_training_domains_after_calibration': ECE_Over_training_domains_after_calibration,
           'ACC_Over_unseen_domain_before_calibration': ACC_Over_unseen_domain_before_calibration,
           'ACC_Over_unseen_domain_after_calibration': ACC_Over_unseen_domain_after_calibration,
           'Average_ECE_Over_training_domains_before_calibration': Average_ECE_Over_training_domains_before_calibration,
           'Average_ECE_Over_training_domains_after_calibration': Average_ECE_Over_training_domains_after_calibration,
           'Average_ACC_Over_unseen_domains_before_calibration': Average_ACC_Over_unseen_domains_before_calibration,
           'Average_ACC_Over_unseen_domains_after_calibration': Average_ACC_Over_unseen_domains_after_calibration,
           'ECE_over_test_set_before_calibration': ECE_over_test_set_before_calibration,
           'ECE_over_test_set_after_calibration': ECE_over_test_set_after_calibration,
           'Average_ROC_Over_unseen_domains_before_calibration' : Average_ROC_Over_unseen_domains_before_calibration,
           'Average_ROC_Over_unseen_domains_after_calibration' : Average_ROC_Over_unseen_domains_after_calibration
           }

    return res
