import math
from scipy.stats import hypergeom

def calculate_n_hyper(num_errors, alpha, tolerable_error, account_value):
    max_error_rate = tolerable_error / account_value
    
    def calculate_deviance(n_stichprobe):
        correct_mu = round((1 - max_error_rate) * account_value)
        return hypergeom.sf(num_errors - 1, account_value, correct_mu, n_stichprobe) - alpha
    
    correct_mu = round((1 - max_error_rate) * account_value)
    
    return math.ceil(optimize.root_scalar(calculate_deviance, bracket=[0, min(correct_mu + num_errors, account_value)]).root)


import math

def MUS_factor(confidence_level, pct_ratio):
    erro = -1
    resp = erro
    max_iter = 1000
    solved = 0.000001
    if confidence_level <= 0 or confidence_level >= 1 or pct_ratio < 0 or pct_ratio >= 1:
        raise ValueError("Parameters must be between 0 and 1.")
    else:
        F = math.gamma(confidence_level, 1, 1)
        if pct_ratio == 0:
            resp = F
        else:
            F1 = 0
            i = 0
            while abs(F1 - F) > solved and i <= max_iter:
                F1 = F
                F = math.gamma(confidence_level, 1 + pct_ratio * F1, 1)
                i = i + 1
            resp = F if abs(F1 - F) <= solved else erro
    return resp

import math

def MUS_calc_n_conservative(confidence_level, tolerable_error, expected_error, book_value):
    pct_ratio = expected_error / tolerable_error
    conf_factor = math.ceil(MUS_factor(confidence_level, pct_ratio) * 100) / 100
    return math.ceil(conf_factor / tolerable_error * book_value)


