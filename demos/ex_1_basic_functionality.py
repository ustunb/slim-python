import os
import cplex
import numpy as np
import pandas as pd

from slim_python.SLIMCoefficientConstraints import SLIMCoefficientConstraints
from slim_python.create_slim_IP import create_slim_IP

#for SLIM helper functions
import warnings
from prettytable import PrettyTable

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

    true_positives = np.sum(yhat[pos_ind] == 1),
    true_negatives= np.sum(yhat[neg_ind] == -1),
    false_positives = np.sum(yhat[neg_ind] == -1),
    false_negatives = np.sum(yhat[neg_ind] == -1),

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

def get_slim_summary(slim_IP, slim_info, X, Y, coef_constraints):

    #TODO: check data with flag
    #TODO: check slim_IP solution quality with flag
    #TODO: default values if there is no solution
    #TODO: pull rho for each solution pool

    #get actual solution
    rho = np.array(slim_IP.solution.get_values(slim_info['rho_idx']))
    objval_upperbound = slim_IP.solution.get_objective_value()
    objval_lowerbound = slim_IP.solution.get_best_objective()
    objval_relgap = slim_IP.solution.get_mip_relative_gap

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

    true_positives = np.sum(yhat[pos_ind] == 1),
    true_negatives= np.sum(yhat[neg_ind] == -1),
    false_positives = np.sum(yhat[neg_ind] == -1),
    false_negatives = np.sum(yhat[neg_ind] == -1),

    #build a pretty table model
    printed_model = print_slim_model(rho, X_names = slim_info['X_names'], Y_name = slim_info['Y_name'], show_omitted_variables = False)

    #MIP related sanity checks
    slim_summary = {
        'solution_status_code': slim_IP.solution.get_status(),
        'solution_status': slim_IP.solution.get_status_string(slim_IP.solution.get_status()),
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
        #'solution_pool':
        #
        # MIP-related
        'objective_value': objval_upperbound,
        'optimality_gap': objval_relgap,
        'objval_lowerbound': objval_lowerbound,
        'simplex_iterations' slim_IP.solution.progress.get_num_iterations(),
        'nodes_processed': slim_IP.solution.progress.get_num_nodes_processed(),
        'nodes_remaining': slim_IP.solution.progress.get_num_nodes_remaining(),
    }

    return(slim_summary)


# requirements for CSV data file
# - outcome variable in first column
# - outcome variable values should be [-1, 1] or [0, 1]
# - first row contains names for the outcome variable + input variables
# - no empty cells

data_name = 'breastcancer'
data_dir = os.getcwd() + '/data/'
data_csv_file = data_dir + data_name + '_processed.csv'

# load data file from csv
df = pd.read_csv(data_csv_file, sep = ',')
data = df.as_matrix()
data_headers = list(df.columns.values)
N = data.shape[0]

# setup Y vector and Y_name
Y_col_idx = [0]
Y = data[:, Y_col_idx]
Y_name = [data_headers[j] for j in Y_col_idx]
Y[Y == 0] = -1

# setup X and X_names
X_col_idx = [j for j in range(data.shape[1]) if j not in Y_col_idx]
X = data[:, X_col_idx]
X_names = [data_headers[j] for j in X_col_idx]

# insert a column of ones to X for the intercept
X = np.insert(arr = X, obj = 0, values = np.ones(N), axis = 1)
X_names.insert(0, '(Intercept)')

# TODO: write a function to run sanity checks on X, Y
check_data(X = X, Y = Y, X_names = X_names)

# setup SLIM coefficient set
coef_constraints = SLIMCoefficientConstraints(variable_names = X_names, ub = 5, lb = -5)
coef_constraints.view()

#create SLIM IP
slim_input = {
    'X': X,
    'X_names': X_names,
    'Y': Y,
    'C_0': 0.01,
    'w_pos': 1.0,
    'w_neg': 1.0,
    'L0_min': 0,
    'L0_max': float('inf'),
    'err_min': 0,
    'err_max': 1.0,
    'pos_err_min': 0,
    'pos_err_max': 1.0,
    'neg_err_min': 0,
    'neg_err_max': 1.0,
    'coef_constraints': coef_constraints
}

# input = slim_input
# print_flag = True
# from slim_python.helper_functions import *
# from math import ceil, floor
slim_IP, slim_info = create_slim_IP(slim_input)

# setup SLIM IP parameters
# see docs/usrccplex.pdf for more about these parameters
#TODO: add these default settings to create_slim_IP
slim_IP.parameters.randomseed.set(0)
slim_IP.parameters.threads.set(1)
slim_IP.parameters.parallel.set(1)
slim_IP.parameters.output.clonelog.set(0)
slim_IP.parameters.mip.tolerances.mipgap.set(np.finfo(np.float).eps)
slim_IP.parameters.mip.tolerances.absmipgap.set(np.finfo(np.float).eps)
slim_IP.parameters.mip.tolerances.integrality.set(np.finfo(np.float).eps)
slim_IP.parameters.emphasis.mip.set(1)
slim_IP.parameters.timelimit.set(60.0)

# solve SLIM IP
slim_IP.solve()
check_slim_IP_output(slim_IP, slim_info, X, Y, coef_constraints)


