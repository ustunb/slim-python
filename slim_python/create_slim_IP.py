import cplex
import numpy as np
from math import ceil, floor
from helper_functions import *
from SLIMCoefficientConstraints import SLIMCoefficientConstraints

def create_slim_IP(input, print_flag = False):
    """
    :param input: dictionary with the following keys
    %Y          N x 1 np.array of labels (-1 or 1 only)
    %X          N x P np.matrix of feature values (should include a column of 1s to act as an intercept
    %X_names    P x 1 list of strings with names of the feature values (all unique and Intercept name)

    :return:
    %slim_IP
    %slim_info
    """

    #setup printing
    if print_flag:
        def print_handle(msg):
            print_log(msg)
    else:
        def print_handle(msg):
            pass

    #check preconditions
    assert 'X' in input, 'no field named X  in input'
    assert 'X_names' in input, 'no field named X_names in input'
    assert 'Y' in input, 'no field named Y in input'
    assert input['X'].shape[0] == input['Y'].shape[0]
    assert input['X'].shape[1] == len(input['X_names'])
    assert all((input['Y'] == 1) | (input['Y'] == -1))

    XY = input['X'] * input['Y']

    #sizes
    N = input['X'].shape[0]
    P = input['X'].shape[1]
    pos_ind = np.flatnonzero(input['Y'] == 1)
    neg_ind = np.flatnonzero(input['Y'] == -1)
    N_pos = len(pos_ind)
    N_neg = len(neg_ind)
    binary_data_flag = np.all((input['X'] == 0) | (input['X'] == 1))

    #outcome variable name
    if ('Y_name' in input) and (type(input['Y_name']) is list):
        input['Y_name'] = input['Y_name'][0]
    elif ('Y_name' in input) and (type(input['Y_name']) is str):
        pass
    else:
        input['Y_name'] = 'Outcome'

    #TODO: check intercept conditions
    ## first column of X should be all 1s
    ## first element of X_name should be '(Intercept)'

    #set default parameters
    input = get_or_set_default(input, 'C_0', 0.01, print_flag = print_flag)
    input = get_or_set_default(input, 'w_pos', 1.0, print_flag = print_flag)
    input = get_or_set_default(input, 'w_neg', 2.0 - input['w_pos'], print_flag = print_flag)
    input = get_or_set_default(input, 'L0_min', 0, print_flag = print_flag)
    input = get_or_set_default(input, 'L0_max', P, print_flag = print_flag)
    input = get_or_set_default(input, 'err_min', 0.00, print_flag = print_flag)
    input = get_or_set_default(input, 'err_max', 1.00, print_flag = print_flag)
    input = get_or_set_default(input, 'pos_err_min', 0.00, print_flag = print_flag)
    input = get_or_set_default(input, 'pos_err_max', 1.00, print_flag = print_flag)
    input = get_or_set_default(input, 'neg_err_min', 0.00, print_flag = print_flag)
    input = get_or_set_default(input, 'neg_err_max', 1.00, print_flag = print_flag)

    #internal parameters
    input = get_or_set_default(input, 'C_1', float('nan'), print_flag = print_flag)
    input = get_or_set_default(input, 'M', float('nan'), print_flag = print_flag)
    input = get_or_set_default(input, 'epsilon', 0.001, print_flag = print_flag)

    #coefficient constraints
    if 'coef_constraints' in input:
        coef_constraints = input['coef_constraints']
    else:
        coef_constraints = SLIMCoefficientConstraints(variable_names = input['X_names'])


    assert len(coef_constraints) == P

    # bounds
    rho_lb = np.array(coef_constraints.lb)
    rho_ub = np.array(coef_constraints.ub)
    rho_max = np.maximum(np.abs(rho_lb), np.abs(rho_ub))
    beta_ub = rho_max
    beta_lb = np.zeros_like(rho_max)
    beta_lb[rho_lb > 0] = rho_lb[rho_lb > 0]
    beta_lb[rho_ub < 0] = rho_ub[rho_ub < 0]

    # signs
    signs = coef_constraints.sign
    sign_pos = signs == 1
    sign_neg = signs == -1

    #types
    types = coef_constraints.get_field_as_list('vtype')
    rho_type = ''.join(types)
    #TODO: add support for custom variable types

    #class-based weights
    w_pos = input['w_pos']
    w_neg = input['w_neg']
    w_total = w_pos + w_neg
    w_pos = 2.0 * (w_pos/w_total)
    w_neg = 2.0 * (w_neg/w_total)
    assert w_pos > 0.0
    assert w_neg > 0.0
    assert w_pos + w_neg == 2.0

    #L0 regularization penalty
    C_0j = np.copy(coef_constraints.C_0j)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input['C_0']
    C_0 = C_0j
    assert(all(C_0[L0_reg_ind] > 0.0))

    #L1 regularization penalty
    L1_reg_ind = L0_reg_ind
    if not np.isnan(input['C_1']):
        C_1 = input['C_1']
    else:
        C_1 = 0.5 * min(w_pos/N, w_neg/N, min(C_0[L1_reg_ind] / np.sum(rho_max)))
    C_1 = C_1 * np.ones(shape = (P,))
    C_1[~L1_reg_ind] = 0.0
    assert(all(C_1[L1_reg_ind] > 0.0))

    # model size bounds
    L0_min = max(input['L0_min'], 0.0)
    L0_max = min(input['L0_max'], np.sum(L0_reg_ind))
    L0_min = ceil(L0_min)
    L0_max = floor(L0_max)
    assert(L0_min <= L0_max)

    # total positive error bounds
    pos_err_min = 0.0 if np.isnan(input['pos_err_min']) else input['pos_err_min']
    pos_err_max = 1.0 if np.isnan(input['pos_err_max']) else input['pos_err_max']
    pos_err_min = max(ceil(N_pos*pos_err_min), 0)
    pos_err_max = min(floor(N_pos*pos_err_max), N_pos)

    # total negative error bounds
    neg_err_min = 0.0 if np.isnan(input['neg_err_min']) else input['neg_err_min']
    neg_err_max = 1.0 if np.isnan(input['neg_err_max']) else input['neg_err_max']
    neg_err_min = max(ceil(N_neg*neg_err_min), 0)
    neg_err_max = min(floor(N_neg*neg_err_max), N_neg)

    # total error bounds
    err_min = 0.0 if np.isnan(input['err_min']) else input['err_min']
    err_max = 1.0 if np.isnan(input['err_max']) else input['err_max']
    err_min = max(ceil(N*err_min), 0)
    err_max = min(floor(N*err_max), N)

    # sanity checks for error bounds
    assert(err_min <= err_max)
    assert(pos_err_min <= pos_err_max)
    assert(neg_err_min <= neg_err_max)
    assert(err_min >= 0)
    assert(pos_err_min >= 0)
    assert(neg_err_min >= 0)
    assert(err_max <= N)
    assert(pos_err_max <= N_pos)
    assert(neg_err_max <= N_neg)

    #TODO: strengthen bounds
    #loss constraint parameters
    epsilon  = input['epsilon']
    if np.isnan(input['M']):
        max_points = np.maximum(XY * rho_lb, XY * rho_ub)
        max_score_reg = np.sum(-np.sort(-max_points[:, L0_reg_ind])[:, 0:int(L0_max)], axis = 1)
        max_score_no_reg = np.sum(max_points[:, ~L0_reg_ind], axis = 1)
        max_score = max_score_reg + max_score_no_reg
        M = max_score + 1.05 * epsilon
    else:
        M = input['M']

    #sanity checks for loss constraint parameters
    M = M * np.ones(shape = (N,))
    M_max = max(np.sum(abs(XY) * rho_max, axis = 1)) + 1.05 * input['epsilon']
    assert(len(M) == N)
    assert(all(M > 0))
    assert(all(M <= M_max))
    assert(epsilon > 0.0)
    assert(epsilon < 1.0)


    #### CREATE CPLEX IP

    #TODO: describe IP

    # x = [loss_pos, loss_neg, rho_j, alpha_j]

    #optional constraints:
    # objval = w_pos * loss_pos + w_neg * loss_min + sum(C_0j * alpha_j) (required for callback)
    # L0_norm = sum(alpha_j) (required for callback)

    #rho = P x 1 vector of coefficient values
    #alpha  = P x 1 vector of L0-norm variables, alpha(j) = 1 if lambda_j != 0
    #beta   = P x 1 vector of L1-norm variables, beta(j) = abs(lambda_j)
    #error  = N x 1 vector of loss variables, error(i) = 1 if error on X(i)
    #pos_err = auxiliary variable := sum(error[i]) for i: y_i = +1
    #neg_err = auxiliary variable := sum(error[i]) for i: y_i = +1
    #l0_norm = auxiliary variable := L0_norm = sum(alpha[j])

    ## IP VARIABLES

    #objective costs (we solve min total_error + N * C_0 * L0_norm + N
    err_cost = np.ones(shape = (N,))
    err_cost[pos_ind] = w_pos
    err_cost[neg_ind] = w_neg
    C_0 = N * C_0
    C_1 = N * C_1

    #variable-related values
    obj = [0.0] * P + C_0.tolist() + C_1.tolist() + err_cost.tolist()
    ub = rho_ub.tolist() + [1] * P + beta_ub.tolist() + [1] * N
    lb = rho_lb.tolist() + [0] * P + beta_lb.tolist() + [0] * N
    ctype  = rho_type + 'B'*P + 'C'*P + 'B'*N

    #variable-related names
    rho_names   = ['rho_' + str(j) for j in range(0, P)]
    alpha_names = ['alpha_' + str(j) for j in range(0, P)]
    beta_names = ['beta_' + str(j) for j in range(0, P)]
    error_names = ['error_' + str(i) for i in range(0, N)]
    var_names = rho_names + alpha_names + beta_names + error_names

    #variable-related error checking
    n_var = 3 * P + N
    assert(len(obj) == n_var)
    assert(len(ub) == n_var)
    assert(len(lb) == n_var)
    assert(len(ctype) == n_var)
    assert(len(var_names) == n_var)

    #add variables
    slim_IP = cplex.Cplex()
    slim_IP.objective.set_sense(slim_IP.objective.sense.minimize)
    slim_IP.variables.add(obj = obj, lb = lb, ub = ub, types = ctype, names=var_names)

    #Loss Constraints
    #Enforce z_i = 1 if incorrect classification)
    #M_i * z_i >= XY[i,].dot(rho) + epsilon
    for i in range(0, N):
        slim_IP.linear_constraints.add(names = ["error_" + str(i)],
                                       lin_expr = [cplex.SparsePair(ind = rho_names + [error_names[i]],
                                                                    val = XY[i,].tolist() + [M[i]])],
                                       senses = "G",
                                       rhs = [epsilon])

    # 0-Norm LB Constraints:
    # lambda_j,lb * alpha_j <= lambda_j <= Inf
    # 0 <= lambda_j - lambda_j,lb * alpha_j < Inf
    for j in range(0, P):
        slim_IP.linear_constraints.add(names = ["L0_norm_lb_" + str(j)],
                                       lin_expr = [cplex.SparsePair(ind = [rho_names[j], alpha_names[j]],
                                                                    val = [1.0, -rho_lb[j]])],
                                       senses = "G",
                                       rhs = [0.0])

    # 0-Norm UB Constraints:
    # lambda_j <= lambda_j,ub * alpha_j
    # 0 <= -lambda_j + lambda_j,ub * alpha_j
    for j in range(0, P):
        slim_IP.linear_constraints.add(names = ["L0_norm_ub_" + str(j)],
                                       lin_expr = [cplex.SparsePair(ind = [rho_names[j], alpha_names[j]],
                                                                    val = [-1.0, rho_ub[j]])],
                                       senses = "G",
                                       rhs = [0.0])


    # 1-Norm Positive Constraints:
    #actual constraint: lambda_j <= beta_j
    #cplex constraint:  0 <= -lambda_j + beta_j <= Inf
    for j in range(0, P):
        slim_IP.linear_constraints.add(names = ["L1_norm_pos_" + str(j)],
                                       lin_expr = [cplex.SparsePair(ind = [rho_names[j], beta_names[j]],
                                                                    val = [-1.0, 1.0])],
                                       senses = "G",
                                       rhs = [0.0])


    # 1-Norm Negative Constraints:
    #actual constraint: -lambda_j <= beta_j
    #cplex constraint:  0 <= lambda_j + beta_j <= Inf
    for j in range(0, P):
        slim_IP.linear_constraints.add(names = ["L1_norm_neg_" + str(j)],
                                       lin_expr = [cplex.SparsePair(ind = [rho_names[j], beta_names[j]],
                                                                    val = [1.0, 1.0])],
                                       senses = "G",
                                       rhs = [0.0])

    # flags for whether or not we will add contraints
    add_L0_norm_constraint = (L0_min > 0) or (L0_max < P)
    add_total_error_constraint = (err_min > 0) or (err_max < N)
    add_pos_error_constraint = (pos_err_min > 0) or (pos_err_max < N_pos) or (add_total_error_constraint)
    add_neg_error_constraint = (neg_err_min > 0) or (neg_err_max < N_neg) or (add_total_error_constraint)

    ### auxiliary variables and bounds
    total_l0_norm_name = ['total_l0_norm']
    total_error_name = ['total_error']
    total_error_pos_name = ['total_error_pos_name']
    total_error_neg_name = ['total_error_neg_name']

    slim_IP.variables.add(names = total_l0_norm_name,
                          obj = [0.0],
                          lb = [L0_min],
                          ub = [L0_max],
                          types = 'I')

    slim_IP.variables.add(names = total_error_name,
                          obj = [0.0],
                          lb = [err_min],
                          ub = [err_max],
                          types = 'I')

    slim_IP.variables.add(names = total_error_pos_name,
                          obj = [0.0],
                          lb = [pos_err_min],
                          ub = [pos_err_max],
                          types = 'I')

    slim_IP.variables.add(names = total_error_neg_name,
                          obj = [0.0],
                          lb = [neg_err_min],
                          ub = [neg_err_max],
                          types = 'I')


    # L0_norm constraint
    #if add_L0_norm_constraint:
    slim_IP.linear_constraints.add(names = ["total_L0_norm"],
                                   lin_expr = [cplex.SparsePair(ind = alpha_names + total_l0_norm_name,
                                                                val = [-1.0] * P + [1.0])],
                                   senses = "E",
                                   rhs = [0.0])

    # total_pos_error variable definition constraint
    #err_pos = sum(error[i]) for i in pos_ind
    #if add_pos_error_constraint:
    slim_IP.linear_constraints.add(names = ["total_pos_error"],
                                   lin_expr = [cplex.SparsePair(ind = [error_names[i] for i in pos_ind] + total_error_pos_name,
                                                                val = [-1.0] * N_pos + [1.0])],
                                   senses = "E",
                                   rhs = [0.0])


    # total_neg_error variable definition constraint
    #err_neg = sum(error[i]) for i in neg_ind
    #if add_neg_error_constraint:
    slim_IP.linear_constraints.add(names = ["total_neg_error"],
                                   lin_expr = [cplex.SparsePair(ind = [error_names[i] for i in neg_ind] + total_error_neg_name,
                                                                val = [-1.0] * N_neg + [1.0])],
                                   senses = "E",
                                   rhs = [0.0])

    # total_error variable definition constraint
    #if add_total_error_constraint:
    slim_IP.linear_constraints.add(names = ["total_error"],
                                   lin_expr = [cplex.SparsePair(ind = total_error_name + total_error_pos_name + total_error_neg_name,
                                                                val = [1.0, -1.0, -1.0])],
                                   senses = "E",
                                   rhs = [0.0])


    #### Drop Variables and Constraints
    variables_to_drop = []
    constraints_to_drop = []

    # drop alpha[j] and beta[j] if L0_reg_ind == False
    no_L0_reg_ind = np.flatnonzero(~L0_reg_ind).tolist()
    variables_to_drop += ["alpha_" + str(j) for j in no_L0_reg_ind]
    variables_to_drop += ["beta_" + str(j) for j in no_L0_reg_ind]
    constraints_to_drop += ["L0_norm_lb_" + str(j) for j in no_L0_reg_ind]
    constraints_to_drop += ["L0_norm_ub_" + str(j) for j in no_L0_reg_ind]
    constraints_to_drop += ["L1_norm_pos_" + str(j) for j in no_L0_reg_ind]
    constraints_to_drop += ["L1_norm_neg_" + str(j) for j in no_L0_reg_ind]

    # drop beta[j] if L0_reg_ind == False or L1_reg_ind == False
    no_L1_reg_ind = np.flatnonzero(~L1_reg_ind).tolist()
    variables_to_drop += ["beta_" + str(j) for j in no_L1_reg_ind]
    constraints_to_drop += ["L1_norm_pos_" + str(j) for j in no_L1_reg_ind]
    constraints_to_drop += ["L1_norm_neg_" + str(j) for j in no_L1_reg_ind]

    # drop alpha[j] and beta[j] if values are fixed at 0
    fixed_value_ind = np.flatnonzero(rho_lb == rho_ub).tolist()
    variables_to_drop += ["alpha_" + str(j) for j in fixed_value_ind]
    variables_to_drop += ["beta_" + str(j) for j in fixed_value_ind]
    constraints_to_drop += ["L0_norm_lb_" + str(j) for j in fixed_value_ind]
    constraints_to_drop += ["L0_norm_ub_" + str(j) for j in fixed_value_ind]
    constraints_to_drop += ["L1_norm_pos_" + str(j) for j in fixed_value_ind]
    constraints_to_drop += ["L1_norm_neg_" + str(j) for j in fixed_value_ind]

    # drop constraints based on signs
    sign_pos_ind = np.flatnonzero(sign_pos).tolist()
    sign_neg_ind = np.flatnonzero(sign_neg).tolist()

    # drop L1_norm_neg constraint for any variable rho[j] >= 0
    constraints_to_drop += ["L1_norm_neg_" + str(j) for j in sign_pos_ind]

    # drop L0_norm_lb constraint for any variable rho[j] >= 0
    constraints_to_drop += ["L0_norm_lb_" + str(j) for j in sign_pos_ind]

    # drop L1_norm_pos constraint for any variable rho[j] <= 0
    constraints_to_drop += ["L1_norm_pos_" + str(j) for j in sign_neg_ind]

    # drop L0_norm_ub constraint for any variable rho[j] <= 0
    constraints_to_drop += ["L0_norm_ub_" + str(j) for j in sign_neg_ind]

    if len(constraints_to_drop) > 0:
        constraints_to_drop = list(set(constraints_to_drop))
        slim_IP.linear_constraints.delete(constraints_to_drop)

    if len(variables_to_drop) > 0:
        variables_to_drop = list(set(variables_to_drop))
        slim_IP.variables.delete(variables_to_drop)

    #create info dictionary for debugging
    rho_names = [n for n in rho_names if n not in variables_to_drop]
    alpha_names = [n for n in alpha_names if n not in variables_to_drop]
    beta_names = [n for n in beta_names if n not in variables_to_drop]

    slim_info = {
        #
        # key parameters
        #
        "C_0": C_0,
        "C_1": C_1,
        "w_pos": w_pos,
        "w_neg": w_neg,
        "err_min": err_min,
        "err_max": err_max,
        "pos_err_min": pos_err_min,
        "pos_err_max": pos_err_max,
        "neg_err_min": neg_err_min,
        "neg_err_max": neg_err_max,
        "L0_min": L0_min,
        "L0_max": L0_max,
        "N": N,
        "P": P,
        "N_pos": N_pos,
        "N_neg": N_neg,
        "rho_ub": rho_ub,
        "rho_lb": rho_lb,
        "M": M,
        "epsilon": epsilon,
        #
        "binary_data_flag": binary_data_flag,
        "pos_ind": pos_ind,
        "neg_ind": neg_ind,
        "L0_reg_ind": L0_reg_ind,
        "L1_reg_ind": L1_reg_ind,
        #
        "n_variables": slim_IP.variables.get_num(),
        "n_constraints": slim_IP.linear_constraints.get_num(),
        "names": slim_IP.variables.get_names(),
        #
        # MIP variables indices
        #
        "rho_idx":  slim_IP.variables.get_indices(rho_names),
        "alpha_idx": slim_IP.variables.get_indices(alpha_names),
        "beta_idx": slim_IP.variables.get_indices(beta_names),
        "error_idx":  slim_IP.variables.get_indices(error_names),
        "total_l0_norm_idx": slim_IP.variables.get_indices(total_l0_norm_name),
        "total_error_idx": slim_IP.variables.get_indices(total_error_name),
        "total_error_pos_idx": slim_IP.variables.get_indices(total_error_pos_name),
        "total_error_neg_idx": slim_IP.variables.get_indices(total_error_neg_name),
        #
        # MIP variables names
        #
        "rho_names": rho_names,
        "alpha_names": alpha_names,
        "beta_names": beta_names,
        "rho_names": rho_names,
        "error_names": error_names,
        "total_error_name": total_error_name,
        "total_error_pos_name": total_error_pos_name,
        "total_error_neg_name": total_error_neg_name,
        #
        "X_names": input['X_names'],
        "Y_name": input['Y_name'],
        #
        # dropped
        "variables_to_drop": variables_to_drop,
        "constraints_to_drop": constraints_to_drop,
    }


    return slim_IP, slim_info
