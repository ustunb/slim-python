import cplex
import os
import sys
import time
import numpy as np

#PRINTING AND LOGGING
def print_log(msg, print_flag = True):
    if print_flag:
        if type(msg) is str:
            print ('%s | ' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()))) + msg
        else:
            print '%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg)
        sys.stdout.flush()

def get_rho_string(rho, vtypes = 'I'):
    if len(vtypes) == 1:
        if vtypes == 'I':
            rho_string = ' '.join(map(lambda x: str(int(x)), rho))
        else:
            rho_string = ' '.join(map(lambda x: str(x), rho))
    else:
        rho_string = ''
        for j in range(0, len(rho)):
            if vtypes[j] == 'I':
                rho_string += ' ' + str(int(rho[j]))
            else:
                rho_string += (' %1.6f' % rho[j])

    return rho_string

#LOADING SETTINGS FROM DISK
def easy_type(data_value):
    type_name = type(data_value).__name__
    if type_name in {"list", "set"}:
        types = {easy_type(item) for item in data_value}
        if len(types) == 1:
            return next(iter(types))
        elif types.issubset({"int", "float"}):
            return "float"
        else:
            return "multiple"
    elif type_name == "str":
        if data_value in {'True', 'TRUE'}:
            return "bool"
        elif data_value in {'False', 'FALSE'}:
            return "bool"
        else:
            return "str"
    elif type_name == "int":
        return "int"
    elif type_name == "float":
        return "float"
    elif type_name == "bool":
        return "bool"
    else:
        return "unknown"

def convert_str_to_bool(val):
    val = val.lower().strip()
    if val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        return None


def get_or_set_default(settings, setting_name, default_value, type_check = False, print_flag = False):

    if setting_name in settings:
        if type_check:
            #check type match
            default_type = type(default_value)
            user_type = type(settings[setting_name])
            if user_type == default_type:
                settings[setting_name] = default_value
            else:
                print_log("type mismatch on %s: user provided type: %s and but expected type: %s" % (setting_name, user_type, default_type), print_flag)
                print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
                settings[setting_name] = default_value
                #else: do nothing
    else:
        print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
        settings[setting_name] = default_value

    return settings


#PROCESSING
def get_prediction(x, rho):
    return np.sign(x.dot(rho))

def get_true_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] == 1)

def get_false_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] == 1)

def get_true_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] != 1)

def get_false_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] != 1)

def get_accuracy_stats(model, data, error_checking = True):

    # old functions (inefficient)
    # get_true_positives = lambda x, y, rho: np.sum(get_prediction(x[y == 1], rho) == 1)
    # get_true_negatives = lambda x, y, rho: np.sum(get_prediction(x[y != 1], rho) != 1)
    # get_false_positives = lambda x, y, rho: np.sum(get_prediction(x[y != 1], rho) == 1)
    # get_false_negatives = lambda x, y, rho: np.sum(get_prediction(x[y == 1], rho) != 1)

    accuracy_stats = {
        'train_true_positives': np.nan,
        'train_true_negatives':  np.nan,
        'train_false_positives':  np.nan,
        'train_false_negatives':  np.nan,
        'valid_true_positives': np.nan,
        'valid_true_negatives': np.nan,
        'valid_false_positives': np.nan,
        'valid_false_negatives': np.nan,
        'test_true_positives': np.nan,
        'test_true_negatives': np.nan,
        'test_false_positives': np.nan,
        'test_false_negatives': np.nan,
    }

    model = np.array(model).reshape(data['X'].shape[1], 1)

    # training set
    data_prefix = 'train'
    X_field_name = 'X'
    Y_field_name = 'Y'
    Yhat = get_prediction(data['X'], model)
    pos_ind = data[Y_field_name] == 1

    accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

    if error_checking:
        N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                   accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                   accuracy_stats[data_prefix + '_' + 'false_positives'] +
                   accuracy_stats[data_prefix + '_' + 'false_negatives'])
        assert data[X_field_name].shape[0] == N_check

    # validation set
    data_prefix = 'valid'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_validation_set = (X_field_name in data and
                          Y_field_name in data and
                          data[X_field_name].shape[0] > 0 and
                          data[Y_field_name].shape[0] > 0)

    if has_validation_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    # test set
    data_prefix = 'test'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_test_set = (X_field_name in data and
                    Y_field_name in data and
                    data[X_field_name].shape[0] > 0 and
                    data[Y_field_name].shape[0] > 0)

    if has_test_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    return accuracy_stats

