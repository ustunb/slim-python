import cplex
import numpy as np
from cplex import infinity as CPX_INFINITY
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
    assert 'X' in input
    assert 'X_names' in input
    assert 'Y' in input
    assert input['X'].shape[0] == input['Y'].shape[0]
    assert input['X'].shape[1] == len(input['X_names'])
    assert all((input['Y'] == 1) | (input['Y'] == -1))

    #sizes
    N = input['X'].shape[0]
    P = input['X'].shape[1]
    N_pos = input['Y'] == 1
    N_neg = input['Y'] == -1
    pos_ind = np.flatnonzero(input['Y'] == 1)
    neg_ind = np.flatnonzero(input['Y'] == -1)

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
    input = get_or_set_default(input, 'neg_err_max', 0.00, print_flag = print_flag)

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

    # signs
    signs = coef_constraints.sign
    sign_pos = signs == 1
    sign_neg = signs == 0
    sign_fixed = sign_pos | sign_neg

    #types
    types = coef_constraints.get_field_as_list('vtype')
    rho_type = ''.join(types)
    #TODO: support for custom variable types

    #class-based weights
    assert  input['w_pos'] > 0.0
    assert  input['w_neg'] > 0.0
    w_pos = input['w_pos']
    w_neg = input['w_neg']
    w_total = w_pos + w_neg
    w_pos = 2.0*(w_pos/w_total)
    w_neg = 2.0*(w_pos/w_total)

    #L0 regularization penalty
    C_0j = np.copy(coef_constraints.C_0j)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input['C_0']
    C_0 = C_0j

    #L1 regularization penalty
    L1_reg_ind = L0_reg_ind
    if not np.isnan(input['C_1']):
        C_1 = input['C_1']
    else:
        C_1 = 0.5 * min(w_pos/N, w_neg/N, min(C_0[L1_reg_ind] / np.sum(rho_max)))
    C_1 = C_1 * np.ones(shape = (P,))
    C_1[~L1_reg_ind] = 0.0

    #loss constraint parameters
    epsilon  = input['epsilon']
    M        = input['M']
    if np.isnan(input['M']):
        M = sum(abs(input['X'].dot(rho_max)), 2) + 1.1*epsilon
    else:
        #TODO check that M has the correct format
        #must be N x 1 vector with only positive entries
        M = input['M']

    M = M * np.ones(shape = (N,))
    #data
    XY = input['X'] * input['Y']

    # model size bounds
    L0_min = max(input['L0_min'], 0.0)
    L0_min = ceil(L0_min)
    L0_max = min(input['L0_max'], np.sum(L0_reg_ind))
    L0_max = floor(L0_max)
    if L0_min > L0_max:
        print_handle("warning: L0_min > L0_max, setting both to trivial values")
        L0_min = 0
        L0_max = np.sum(L0_reg_ind)

    # error bounds
    if np.isnan(input['err_min']):
        err_min = 0.0
    else:
        err_min = input['err_min']

    if np.isnan(input['err_max']):
        err_max = 1.0
    else:
        err_max = input['err_max']

    err_min = max(ceil(N*err_min), 0)
    err_max = min(floor(N*err_max), N)

    # total positive error bounds
    if np.isnan(input['pos_err_min']):
        pos_err_min = 0.0
    else:
        pos_err_min = input['pos_err_min']

    if np.isnan(input['pos_err_max']):
        pos_err_max = 1.0
    else:
        pos_err_max = input['pos_err_max']

    pos_err_min = max(ceil(N_pos*pos_err_min), 0)
    pos_err_max = min(floor(N_pos*pos_err_max), N_pos)

    # total negative error bounds
    if np.isnan(input['neg_err_min']):
        neg_err_min = 0.0
    else:
        neg_err_min = input['neg_err_min']

    if np.isnan(input['neg_err_max']):
        neg_err_max = 1.0
    else:
        neg_err_max = input['neg_err_max']

    pos_err_min = max(ceil(N_neg*pos_err_min), 0)
    neg_err_max = min(floor(N_neg*neg_err_max), N_neg)

    assert(err_min <= err_max)
    assert(neg_err_min <= neg_err_max)
    assert(pos_err_min <= pos_err_max)

    # flags for whether or not we will add contraints
    add_L0_norm_constraint        = (L0_min > 0) or (L0_max < P)
    add_total_error_constraint    = (err_min > 0) or (err_max < N)
    add_pos_error_constraint      = (pos_err_min > 0) or (pos_err_max < N_pos)
    add_neg_error_constraint      = (neg_err_min > 0) or (neg_err_max < N_neg)

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

    ## IP VARIABLES

    #objective costs (we solve min total_error + N * C_0 * L0_norm + N
    err_cost = np.ones(shape = (N,))
    err_cost[pos_ind] = w_pos
    err_cost[neg_ind] = w_neg
    C_0 = N * C_0
    C_1 = N * C_1

    #variable-related values
    obj = [0.0] * P + C_0.tolist() + C_1.tolist() + err_cost.tolist()
    ub = rho_ub.tolist() + [1] * P + rho_max.tolist() + [1] * N
    lb = rho_lb.tolist() + [0] * P + rho_max.tolist() + [0] * N
    ctype  = 'I'*P + 'B'*P + 'C'*P + 'B'*P

    #variable-related names
    rho_names   = ['rho_' + str(j) for j in range(0, P)]
    alpha_names = ['alpha_' + str(j) for j in range(0, P)]
    beta_names = ['beta_' + str(j) for j in range(0, P)]
    error_names = ['error_' + str(i) for i in range(0, N)]
    var_names = rho_names + alpha_names + beta_names + error_names

    #variable-related error checking
    n_var = 3*P + N
    assert(len(obj) == n_var)
    assert(len(ub) == n_var)
    assert(len(lb) == n_var)

    #add variables
    slim_IP = cplex.Cplex()
    slim_IP.objective.set_sense(slim_IP.objective.sense.minimize)
    slim_IP.variables.add(obj = obj, lb = lb, ub = ub, types = ctype, names=var_names)

    # ## IP CONSTRAINTS
    #
    # #Loss Constraints
    # #Enforce z_i = 1 if incorrect classification)
    # for i in range(0, N):
    #     constraint_name = ["error_" + str(i)]
    #     constraint_expr = [cplex.sparsePair(ind = [rho_names, error_names[i]],
    #                                         val = [XY[i,].tolist(), M[i])]
    #
    #
    # A_loss         = [sparse(XY),sparse(N,P+P),spdiags(M,0,N,N)];
    # lhs_loss       = epsilon.*sparse_N_x_1_ones;
    # rhs_loss       = sparse_N_x_1_Inf;
    #
    # # 0-Norm LB Constraints:
    # # lambda_j,lb * alpha_j <= lambda_j <= Inf
    # # 0 <= lambda_j - lambda_j,lb * alpha_j < Inf
    # for j in range(0, P):
    #     constraint_name = ["L0_norm_lb_" + str(j)]
    #     constraint_expr = [cplex.SparsePair(ind =[rho_names[j], alpha_names[j]], val = [1.0, -rho_lb[j]])]
    #     constraint_rhs  = [0.0]
    #     constraint_sense = "G"
    #     slim_IP.linear_constraints.add(lin_expr = constraint_expr,
    #                                    senses = constraint_sense,
    #                                    rhs = constraint_rhs,
    #                                    names = constraint_name)
    #
    # # 0-Norm UB Constraints:
    # # lambda_j <= lambda_j,ub * alpha_j
    # # 0 <= -lambda_j + lambda_j,ub * alpha_j
    # for j in range(0, P):
    #     constraint_name = ["L0_norm_ub_" + str(j)]
    #     constraint_expr = [cplex.SparsePair(ind = [rho_names[j], alpha_names[j]], val = [-1.0, rho_ub[j]])]
    #     constraint_rhs  = [0.0]
    #     constraint_sense = "G"
    #     slim_IP.linear_constraints.add(lin_expr = constraint_expr,
    #                                    senses = constraint_sense,
    #                                    rhs = constraint_rhs,
    #                                    names = constraint_name)
    #
    #
    #
    # dropped_variables = []
    # sign_pos_ind = np.where(input['coef_set'].sign > 0)[0].tolist()
    # sign_neg_ind = np.where(input['coef_set'].sign < 0)[0].tolist()
    # fixed_value_ind = np.where(input['coef_set'].ub == input['coef_set'].lb)[0].tolist()
    #
    # # drop L0_norm_lb constraint for any variable with rho_lb >= 0
    # constraints_to_drop = ["L0_norm_lb_" + str(j) for j in sign_pos_ind]
    # slim_IP.linear_constraints.delete(constraints_to_drop)
    #
    # # drop L0_norm_ub constraint for any variable with rho_ub >= 0
    # constraints_to_drop = ["L0_norm_ub_" + str(j) for j in sign_neg_ind]
    # slim_IP.linear_constraints.delete(constraints_to_drop)
    #
    # # drop alpha for any variable where rho_ub = rho_lb = 0
    # variables_to_drop = ["alpha_" + str(j) for j in fixed_value_ind]
    # slim_IP.variables.delete(variables_to_drop)
    # dropped_variables += variables_to_drop
    # alpha_names = [alpha_names[j] for j in range(0, P) if alpha_names[j] not in dropped_variables]
    #
    # # drop alpha, L0_norm_ub and L0_norm_lb for ('Intercept')
    # try:
    #     intercept_ind = input['coef_set'].get_field_as_list('variable_names').index('(Intercept)')
    #     variables_to_drop = ['alpha_' + str(intercept_ind)]
    #     slim_IP.variables.delete(variables_to_drop)
    #     slim_IP.linear_constraints.delete(["L0_norm_lb_" + str(intercept_ind),
    #                                        "L0_norm_ub_" + str(intercept_ind)])
    #     alpha_names.pop(alpha_names.index('alpha_'+str(intercept_ind)))
    #     dropped_variables += ['alpha_'+str(intercept_ind)]
    #     print_handle("dropped L0-related constraints and variables for intercept")
    # except:
    #     raise

    #create info dictionary for debugging
    rho_idx = slim_IP.variables.get_indices(rho_names)
    alpha_idx = slim_IP.variables.get_indices(alpha_names)
    beta_idx = slim_IP.variables.get_indices(beta_names)
    error_idx = slim_IP.variables.get_indices(error_names)

    slim_info = {
        "n_variables": slim_IP.variables.get_num(),
        "n_constraints": slim_IP.linear_constraints.get_num(),
        "names": slim_IP.variables.get_names(),
        "rho_names": rho_names,
        "rho_idx": slim_IP.variables.get_indices(rho_names),
        "alpha_idx": alpha_idx,
        "beta_idx": beta_idx,
        "error_idx": error_idx,
        "rho_idx": rho_idx,
        "alpha_names": alpha_names,
        "beta_names": beta_names,
        "rho_names": rho_names,
        "error_names": error_names,
        "L0_reg_ind": L0_reg_ind,
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
    }


    return slim_IP, slim_info
