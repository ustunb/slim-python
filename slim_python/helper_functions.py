import os
import sys
import time
import numpy as np
import cplex
import warnings
from prettytable import PrettyTable

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

#DATA CHECKING
def check_data(X, X_names, Y):

    #type checks
    assert type(X) is np.ndarray, "type(X) should be numpy.ndarray"
    assert type(Y) is np.ndarray, "type(Y) should be numpy.ndarray"
    assert type(X_names) is list, "X_names should be a list"

    #sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'len(Y) should be same as # of rows in X'
    assert len(list(set(X_names))) == len(X_names), 'X_names is not unique'
    assert len(X_names) == P, 'len(X_names) should be same as # of cols in X'

    #X_matrix values
    if '(Intercept)' in X_names:
        assert all(X[:, X_names.index('(Intercept)')] == 1.0), "'(Intercept)' column should only be composed of 1s"
    else:
        warnings.warn("there is no column named '(Intercept)' in X_names")
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    #Y vector values
    assert all((Y == 1)|(Y == -1)), 'Y[i] should = [-1,1] for all i'
    if all(Y == 1):
        warnings.warn("all Y_i == 1 for all i")
    if all(Y == -1):
        warnings.warn("all Y_i == -1 for all i")

        #TODO (optional) collect warnings and return those?

def check_slim_IP_output(slim_IP, slim_info, X, Y, coef_constraints):

    #TODO skip tests if there is no solution
    #TODO return true to prove that it's passed tests
    #TODO (optional) collect warnings and return those?

    #MIP related sanity checks
    assert len(slim_IP.solution.get_values()) == slim_info['n_variables']

    #setup function handles for convenient checking
    get_L0_norm = lambda x: np.sum(np.count_nonzero(x[slim_info['L0_reg_ind']]))

    #key variables
    rho = np.array(slim_IP.solution.get_values(slim_info['rho_idx']))
    alpha = np.array(slim_IP.solution.get_values(slim_info['alpha_idx']))
    beta = np.array(slim_IP.solution.get_values(slim_info['beta_idx']))
    err = np.array(slim_IP.solution.get_values(slim_info['error_idx']))

    #auxiliary variables
    total_error = np.array(slim_IP.solution.get_values(slim_info['total_error_idx']))
    total_error_pos = np.array(slim_IP.solution.get_values(slim_info['total_error_pos_idx']))
    total_error_neg = np.array(slim_IP.solution.get_values(slim_info['total_error_neg_idx']))
    total_l0_norm = np.array(slim_IP.solution.get_values(slim_info['total_l0_norm_idx']))

    # helper parameters
    L0_reg_ind = slim_info['L0_reg_ind']
    L1_reg_ind = slim_info['L1_reg_ind']
    rho_L0_reg = rho[L0_reg_ind]
    rho_L1_reg = rho[L1_reg_ind]
    beta_ub_reg = np.maximum(abs(coef_constraints.ub[L1_reg_ind]), coef_constraints.lb[L1_reg_ind])
    beta_lb_reg = np.zeros_like(beta_ub_reg)
    beta_lb_reg = np.maximum(beta_lb_reg, slim_info['rho_lb'][L1_reg_ind])
    beta_lb_reg = -np.minimum(beta_lb_reg, slim_info['rho_ub'][L1_reg_ind])
    beta_lb_reg = abs(beta_lb_reg)

    # test on coefficient vector
    assert len(rho) == len(coef_constraints), 'rho has the wrong length'
    assert all(rho <= slim_info['rho_ub']), 'rho exceeds upper bounds'
    assert all(rho >= slim_info['rho_lb']), 'rho exceeds lower bounds'

    # tests on L0 indicator variables
    assert all((alpha == 0)|(alpha == 1)), 'alpha should be binary'
    assert all(abs(rho_L0_reg[alpha == 0]) == 0.0), 'alpha = 0 should => that rho == 0'
    assert all(abs(rho_L0_reg[alpha == 1]) > 0.0), 'alpha = 1 should => that rho != 0'

    # tests on L1 helper variables
    assert all(abs(rho_L1_reg) == beta), 'beta != abs(rho)'
    assert all(beta >= beta_lb_reg), 'beta should be <= beta_ub'
    assert all(beta <= beta_ub_reg), 'beta should be >= beta_lb'

    # L0-norm bounds
    expected_l0_norm = get_L0_norm(rho)
    assert sum(alpha) == expected_l0_norm, 'alpha should := 1[rho != 0]'
    assert total_l0_norm == expected_l0_norm
    assert sum(alpha) >= slim_info['L0_min']
    assert sum(alpha) <= slim_info['L0_max']
    assert total_l0_norm >= slim_info['L0_min']
    assert total_l0_norm <= slim_info['L0_max']
    assert expected_l0_norm >= slim_info['L0_min']
    assert expected_l0_norm <= slim_info['L0_max']

    # aggregate error measure tests
    expected_scores = (Y*X).dot(rho)
    expected_err_values = expected_scores <= slim_info['epsilon']
    assert all((err == 0) | (err == 1)), 'err should be binary'
    assert all(err == expected_err_values), 'error vector is not == sign(XY.dot(rho) + epsilon)'
    assert total_error == sum(err), 'total_error should == sum(error(i))'
    assert total_error == total_error_pos + total_error_neg, 'total_error should == total_error_pos + total_error_neg'
    assert total_error_pos == sum(err[slim_info['pos_ind']])
    assert total_error_neg == sum(err[slim_info['neg_ind']])
    assert all(-expected_scores <= slim_info['M']), 'Big M is not big enough'

    # extra sanity check tests
    assert total_error <= min(slim_info['N_pos'], slim_info['N_neg']), 'total_error should be less than total_error_pos + total_error_neg'

def print_slim_model(rho, X_names, Y_name, show_omitted_variables = False):

    rho_values = np.copy(rho)
    rho_names = list(X_names)

    if '(Intercept)' in rho_names:
        intercept_ind = X_names.index('(Intercept)')
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove('(Intercept)' )
    else:
        intercept_val = 0

    if Y_name is None:
        predict_string = "PREDICT Y = +1 IF SCORE >= %d" % intercept_val
    else:
        predict_string = "PREDICT %s IF SCORE >= %d" % (Y_name[0].upper(), intercept_val)

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

        #sort by most positive to most negative
        sort_ind = np.argsort(-np.array(rho_values))
        rho_values = [rho_values[j] for j in sort_ind]
        rho_names = [rho_names[j] for j in sort_ind]
        rho_values = np.array(rho_values)

    rho_values_string = [str(int(i)) + " points" for i in rho_values]
    n_variable_rows = len(rho_values)
    total_string = "ADD POINTS FROM ROWS %d to %d" % (1, n_variable_rows)

    max_name_col_length = max(len(predict_string), len(total_string), max([len(s) for s in rho_names])) + 2
    max_value_col_length = max(7, max([len(s) for s in rho_values_string]) + len("points")) + 2


    m = PrettyTable()
    m.field_names = ["Variable", "Points", "Tally"]

    m.add_row([predict_string, "", ""])
    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])

    for v in range(0, n_variable_rows):
        m.add_row([rho_names[v], rho_values_string[v], "+ ....."])

    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])
    m.add_row([total_string, "SCORE", "= ....."])
    m.header = False
    m.align["Variable"] = "l"
    m.align["Points"] = "r"
    m.align["Tally"] = "r"
    return(m)

def get_rho_summary(rho, slim_info, X, Y):

    #build a pretty table model
    printed_model = print_slim_model(rho, X_names = slim_info['X_names'], Y_name = slim_info['Y_name'], show_omitted_variables = False)

    #transform Y
    y = np.array(Y.flatten(), dtype = np.float)
    pos_ind = y == 1
    neg_ind = ~pos_ind
    N = len(Y)
    N_pos = np.sum(pos_ind)
    N_neg = N - N_pos

    #get predictions
    yhat = X.dot(rho) > 0
    yhat = np.array(yhat, dtype = np.float)
    yhat[yhat == 0] = -1

    true_positives = np.sum(yhat[pos_ind] == 1)
    false_positives = np.sum(yhat[neg_ind] == 1)
    true_negatives= np.sum(yhat[neg_ind] == -1)
    false_negatives = np.sum(yhat[pos_ind] == -1)

    rho_summary = {
        'rho': rho,
        'pretty_model': printed_model,
        'string_model': printed_model.get_string(),
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mistakes': np.sum(y != yhat),
        'error_rate': (false_positives + false_negatives) / N,
        'true_positive_rate': true_positives / N_pos,
        'false_positive_rate': false_positives / N_neg,
        'L0_norm': np.sum(rho[slim_info['L0_reg_ind']]),
    }

    return(rho_summary)

def get_slim_summary(slim_IP, slim_info, X, Y):

    #TODO: pull add rho_summary for each solution in solution pool

    #MIP Related Items
    slim_summary = {
        #
        # IP related information
        #
        'solution_status_code': slim_IP.solution.get_status(),
        'solution_status': slim_IP.solution.get_status_string(slim_IP.solution.get_status()),
        'objective_value': slim_IP.solution.get_objective_value(),
        'optimality_gap': slim_IP.solution.MIP.get_best_objective(),
        'objval_lowerbound': slim_IP.solution.MIP.get_mip_relative_gap(),
        'simplex_iterations': slim_IP.solution.progress.get_num_iterations(),
        'nodes_processed': slim_IP.solution.progress.get_num_nodes_processed(),
        'nodes_remaining': slim_IP.solution.progress.get_num_nodes_remaining(),
        #
        # Solution based information (default values)
        #
        'rho': np.nan,
        'pretty_model': np.nan,
        'string_model': np.nan,
        'true_positives': np.nan,
        'true_negatives': np.nan,
        'false_positives': np.nan,
        'false_negatives': np.nan,
        'mistakes': np.nan,
        'error_rate': np.nan,
        'true_positive_rate': np.nan,
        'false_positive_rate': np.nan,
        'L0_norm': np.nan,
    }

    #Update with Solution-Based Stats
    try:
        rho = np.array(slim_IP.solution.get_values(slim_info['rho_idx']))
        slim_summary.update(get_rho_summary(rho, slim_info, X, Y))
    except CplexError as e:
        print_log(e)

    return(slim_summary)