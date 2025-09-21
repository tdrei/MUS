import numpy as np
from scipy.stats import hypergeom, gamma
from math import ceil

def calculate_n_hyper(num_errors: int, alpha: float, tolerable_error: float, account_value: float) -> int:
    """
    Calculate necessary observations for a given number of wrong units (num_errors).
    This function does not work for very small account values due to the discrete
    nature of the hypergeometric distribution.
    """
    max_error_rate = tolerable_error / account_value
    correct_mu = round((1 - max_error_rate) * account_value)
    
    def calculate_deviance(n_stichprobe: float) -> float:
        return hypergeom.cdf(num_errors, account_value, round(max_error_rate * account_value), n_stichprobe) - alpha
    
    from scipy.optimize import root_scalar
    interval = (0, min(correct_mu + num_errors, account_value))
    sol = root_scalar(
        calculate_deviance,
        bracket=interval,
        method='bisect'
    )
    if not sol.converged:
        raise ValueError("Root finding did not converge in calculate_n_hyper.")
    return int(np.ceil(sol.root))

def mus_factor(confidence_level: float, pct_ratio: float, max_iter: int = 1000, solved: float = 1e-6) -> float:
    """
    Calculate MUS Factor.
    Based on Technical Notes on the AICPA Audit Guide Audit Sampling, Trevor Stewart, AICPA, 2012.
    """
    if not (0 < confidence_level < 1) or not (0 <= pct_ratio < 1):
        raise ValueError("Parameters must be between 0 and 1.")
    F = gamma.ppf(confidence_level, 1, scale=1)
    if pct_ratio == 0:
        return F
    F1 = 0
    for _ in range(max_iter):
        if abs(F1 - F) <= solved:
            return F
        F1 = F
        F = gamma.ppf(confidence_level, 1 + pct_ratio * F1, scale=1)
    return -1  # failed to solve

def mus_calc_n_conservative(confidence_level: float, tolerable_error: float, expected_error: float, book_value: float) -> int:
    """
    Calculate n conservatively, as per AICPA audit guide.
    """
    pct_ratio = expected_error / tolerable_error
    conf_factor = np.ceil(mus_factor(confidence_level, pct_ratio) * 100) / 100
    return int(np.ceil(conf_factor / tolerable_error * book_value))

def mus_planning(
    data,
    col_name_book_values: str = "book.value",
    confidence_level: float = 0.95,
    tolerable_error: float = None,
    expected_error: float = None,
    n_min: int = 0,
    errors_as_pct: bool = False,
    conservative: bool = False,
    combined: bool = False  # Not used in function
):
    """
    Main MUS planning function.
    data: pandas DataFrame or numpy structured array
    col_name_book_values: column name for book values
    """
    import pandas as pd

    # Check parameters
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("Data needs to be a DataFrame or a numpy structured array.")
    if col_name_book_values not in data:
        raise ValueError(f"The data frame requires a column '{col_name_book_values}' with the book values.")

    book_values = np.array(data[col_name_book_values])
    if np.any(np.isinf(book_values)):
        print("Warning: There are missing or infinite values as book values in your data. Those elements have no chance for selection. You have to audit them separately.")
    if np.any(book_values == 0):
        print("Warning: There are zeros as book values in your data. Those elements have no chance for selection. You have to audit them separately.")
    if np.any(book_values < 0):
        print("Warning: There are negative values as book values in your data. Those elements have no chance for selection. You have to audit them separately.")

    book_value = np.sum(np.maximum(book_values, 0))
    num_items = len(book_values)

    if errors_as_pct and isinstance(tolerable_error, (int, float)) and isinstance(expected_error, (int, float)):
        tolerable_error = tolerable_error * book_value
        expected_error  = expected_error * book_value

    if not (isinstance(confidence_level, (float, int)) and 0 < confidence_level < 1):
        raise ValueError("Confidence level has to be a numeric value between 0 and 1 (exclusive).")
    if not (isinstance(tolerable_error, (float, int)) and tolerable_error > 0):
        raise ValueError("Tolerable Error has to be a numeric value between 0 and book value (exclusive).")
    if not (isinstance(expected_error, (float, int)) and expected_error >= 0):
        raise ValueError("Expected error has to be a numeric value greater or equal to 0.")
    if not (isinstance(n_min, int) and 0 <= n_min < num_items):
        raise ValueError("Minimum number of sample size has to be a numeric value between 0 and the number of items in the population (last exclusive).")

    too_large = (tolerable_error/book_value) * (1 - confidence_level) * np.sqrt(tolerable_error - expected_error) < 0.07
    if too_large:
        print("Warning: Combination of parameters leads to impractically large sample.")

    if tolerable_error >= book_value:
        print("Warning: Tolerable Error is >= book value. No sampling is necessary. Planning will return 0.")
        n_optimal = 0
    elif calculate_n_hyper(0, alpha=1-confidence_level, tolerable_error=tolerable_error, account_value=book_value) < 0:
        raise ValueError("Undefined situation: If 0 errors in the sample occur, the sample size needs to be positive!")
    elif calculate_n_hyper(num_items, alpha=1-confidence_level, tolerable_error=tolerable_error, account_value=book_value) * expected_error / book_value - num_items > 0:
        print("Warning: MUS makes no sense for your sampling problem - your sample size needs to be bigger than the number of items in your population.")
        if expected_error >= tolerable_error:
            print("Just for information: If the expected error is equal or larger than the tolerable error, MUS is not applicable.")
        n_optimal = num_items
    else:
        i = 0
        while calculate_n_hyper(i, alpha=1-confidence_level, tolerable_error=tolerable_error, account_value=book_value) * expected_error / book_value > i:
            i += 1
        ni   = calculate_n_hyper(i-1, alpha=1-confidence_level, tolerable_error=tolerable_error, account_value=book_value)
        nip1 = calculate_n_hyper(i,   alpha=1-confidence_level, tolerable_error=tolerable_error, account_value=book_value)
        try:
            n_optimal = ceil((ni/(nip1-ni)-(i-1))/(1/(nip1-ni)-expected_error/book_value))
        except ZeroDivisionError:
            raise ValueError("Zero division error in n_optimal calculation. Please check input values.")

        if n_optimal > num_items:
            print("Warning: MUS makes no sense for your sampling problem - your sample size needs to be bigger than the number of items in your population.")
            n_optimal = num_items
        elif n_optimal == nip1 + 1:
            n_optimal -= 1
        elif n_optimal < 0 or n_optimal < ni or n_optimal > nip1:
            raise ValueError(f"n_optimal={n_optimal} is not plausible. Internal error.")

    n_final = max(n_optimal, n_min)

    if conservative:
        n_final = max(n_final, mus_calc_n_conservative(confidence_level, tolerable_error, expected_error, book_value))
    
    interval = book_value / n_final if n_final != 0 else None
    tol_taint = expected_error / book_value * n_final if n_final != 0 else None

    result = {
        'data': data,
        'col_name_book_values': col_name_book_values,
        'confidence_level': confidence_level,
        'tolerable_error': tolerable_error,
        'expected_error': expected_error,
        'book_value': book_value,
        'n': n_final,
        'interval': interval,
        'tol_taint': tol_taint,
        'n_min': n_min,
        'conservative': conservative
    }
    return result
