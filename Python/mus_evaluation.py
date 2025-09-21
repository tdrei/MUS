import numpy as np
import pandas as pd
from scipy.stats import hypergeom, t

def calculate_m_hyper(error_taintings, sample_size, alpha, account_value):
    """
    Calculate maximal number of wrong items in the population, given the sample.
    """
    if error_taintings == sample_size:
        return np.inf

    def calculate_deviance(wrong_units):
        # phyper in R: cumulative hypergeometric probability
        # scipy: hypergeom.cdf
        return hypergeom.cdf(error_taintings, account_value, wrong_units, sample_size) - alpha

    # Find the root numerically
    from scipy.optimize import root_scalar
    res = root_scalar(
        lambda m: calculate_deviance(m),
        bracket=[0, account_value],
        method='bisect'
    )
    return int(np.ceil(res.root))

def mus_precision_gap_widening_table(ds, idx, population_amount, confidence_level, filled_sample):
    """
    Create a table based on the ideas of precision gap widening and cell evaluation.
    """
    sample_size = filled_sample.shape[0]
    UEL_factor = np.round([
        calculate_m_hyper(error_taintings, sample_size, 1-confidence_level, population_amount) * sample_size / population_amount
        for error_taintings in range(len(ds)+1)
    ], 4)
    average_ds = np.round(np.cumsum(ds) / np.arange(1, len(ds)+1), 4)

    max_sampling_interval = np.max(filled_sample['sampling_interval']) if 'sampling_interval' in filled_sample else None
    # For each idx, get sampling_interval or fill with max
    sampling_intervals = [max_sampling_interval] + list(filled_sample.loc[idx, 'sampling_interval']) if 'sampling_interval' in filled_sample else [None] * (len(ds)+1)

    result_table = pd.DataFrame({
        'Error_Stage': np.arange(len(ds)+1),
        'UEL_Factor': UEL_factor,
        'Tainting': np.concatenate(([1], ds)),
        'Average_Taintings': np.concatenate(([0], average_ds)),
        'UEL_previous_Stage': np.zeros(len(UEL_factor)),
        'Load_and_Spread': np.zeros(len(UEL_factor)),
        'Simple_Spread': np.concatenate(([UEL_factor[0]], np.full(len(UEL_factor)-1, np.nan))),
        'Stage_UEL_max': np.concatenate(([UEL_factor[0]], np.full(len(UEL_factor)-1, np.nan))),
        'sampling_interval': sampling_intervals
    })

    if len(ds) == 0:
        return result_table

    for row in range(1, len(ds)+1):
        result_table.loc[row, 'UEL_previous_Stage'] = result_table.loc[row-1, 'Stage_UEL_max']
        result_table.loc[row, 'Load_and_Spread'] = result_table.loc[row, 'UEL_previous_Stage'] + result_table.loc[row, 'Tainting']
        result_table.loc[row, 'Simple_Spread'] = result_table.loc[row, 'UEL_Factor'] * result_table.loc[row, 'Average_Taintings']
        result_table.loc[row, 'Stage_UEL_max'] = max(result_table.loc[row, 'Load_and_Spread'], result_table.loc[row, 'Simple_Spread'])
    return result_table

def mus_evaluation(
    extract, filled_sample, filled_high_values, 
    col_name_audit_values="audit_value", col_name_riskweights=None,
    interval_type="one-sided", print_advice=True, tainting_order="decreasing", experimental=False, combined=False
):
    # --- Parameter checks ---
    if extract.get('class', None) != "MUS.extraction.result":
        raise ValueError("extract must be an object of type MUS.extraction.result")

    if not (isinstance(col_name_audit_values, str) and len(col_name_audit_values) == 1):
        raise ValueError("col_name_audit_values must be a single string")
    if col_name_riskweights is not None and not (isinstance(col_name_riskweights, str) and len(col_name_riskweights) == 1):
        raise ValueError("col_name_riskweights must be None or a string")

    # --- Sample evaluation ---
    if filled_sample is not None and len(filled_sample) > 0:
        if not isinstance(filled_sample, (pd.DataFrame, np.ndarray)):
            raise ValueError("filled_sample must be a DataFrame or matrix")
        if extract['col_name_book_values'] not in filled_sample.columns:
            raise ValueError("filled_sample must include column with book values named by extract.col_name_book_values")
        if col_name_audit_values not in filled_sample.columns:
            raise ValueError("filled_sample must include column with audit values named by col_name_audit_values")
        if col_name_riskweights is not None and col_name_riskweights not in filled_sample.columns:
            raise ValueError("If col_name_riskweights is not None, filled_sample must include that column")

        population_amount = extract['sample_population'][extract['col_name_book_values']].sum()
        # Prevent name collision
        if 'd' in filled_sample.columns or 'tord' in filled_sample.columns:
            raise ValueError("filled_sample must not have columns named 'd' or 'tord'")

        # Add sampling interval if not present
        if 'sampling_interval' not in filled_sample.columns:
            filled_sample['sampling_interval'] = extract['sampling_interval']

        # Calculate d's and order for overstatements
        tmp = 1 - filled_sample[col_name_audit_values] / filled_sample[extract['col_name_book_values']]
        if tainting_order == "increasing":
            tord = 1 - tmp
        elif tainting_order == "absolute":
            tord = filled_sample[extract['col_name_book_values']] - filled_sample[col_name_audit_values]
        elif tainting_order == "random":
            tord = np.random.permutation(tmp)
        else:
            tord = tmp

        ds = filled_sample.copy()
        ds['d'] = tmp
        ds['tord'] = tord

        if col_name_riskweights is None:
            errors = ds[extract['col_name_book_values']] - ds[col_name_audit_values]
        else:
            errors = (ds[extract['col_name_book_values']] - ds[col_name_audit_values]) / ds[col_name_riskweights]

        ds_over = ds[ds['d'] > 0].sort_values('tord', ascending=False)
        idx_over = ds_over.index
        ds_over_vals = ds_over['d'].values
        if col_name_riskweights is not None:
            ds_over_vals = ds_over_vals / ds_over[col_name_riskweights].values
        ds_over_vals = np.round(ds_over_vals, 4)

        over = mus_precision_gap_widening_table(ds_over_vals, idx_over, population_amount, extract['confidence_level'], filled_sample)

        # Understatements
        tmp = 1 - filled_sample[col_name_audit_values] / filled_sample[extract['col_name_book_values']]
        if tainting_order == "increasing":
            tord = tmp
        elif tainting_order == "absolute":
            tord = filled_sample[extract['col_name_book_values']] - filled_sample[col_name_audit_values]
        elif tainting_order == "random":
            tord = np.random.permutation(tmp)
        else:
            tord = tmp

        ds = filled_sample.copy()
        ds['d'] = tmp
        ds['tord'] = tord

        ds_under = ds[ds['d'] < 0].sort_values('tord', ascending=True)
        idx_under = ds_under.index
        ds_under_vals = -ds_under['d'].values
        if col_name_riskweights is not None:
            ds_under_vals = ds_under_vals / ds_under[col_name_riskweights].values
        ds_under_vals = np.round(ds_under_vals, 4)

        under = mus_precision_gap_widening_table(ds_under_vals, idx_under, population_amount, extract['confidence_level'], filled_sample)

        # Calculate results
        sampling_interval = extract['sampling_interval']
        gross_most_likely_error = (np.array([over['Tainting'].sum() - 1, under['Tainting'].sum() - 1]) * sampling_interval)
        gross_upper_error_limit = np.array([over['Stage_UEL_max'].max(), under['Stage_UEL_max'].max()]) * sampling_interval
        basic_precision = calculate_m_hyper(0, filled_sample.shape[0], 1-extract['confidence_level'], population_amount)
        Results_Sample = {
            "Sample_Size": filled_sample.shape[0],
            "Number_of_Errors": {
                "overstatements": int(over['Error_Stage'].max()), 
                "understatements": int(under['Error_Stage'].max())
            },
            "Gross_most_likely_error": gross_most_likely_error,
            "Net_most_likely_error": np.array([1, -1]) * np.sum(gross_most_likely_error * np.array([1, -1])),
            "Basic_Precision": basic_precision,
            "Precision_Gap_widening": gross_upper_error_limit - gross_most_likely_error - basic_precision,
            "Total_Precision": gross_upper_error_limit - gross_most_likely_error,
            "Gross_upper_error_limit": gross_upper_error_limit,
            "Net_upper_error_limit": gross_upper_error_limit - gross_most_likely_error + np.array([1,-1]) * np.sum(gross_most_likely_error * np.array([1, -1])),
            "Gross_Value_of_Errors": {
                "overstatements": float(errors[errors > 0].sum()), 
                "understatements": float(errors[errors < 0].sum())
            }
        }
    else:
        Results_Sample = {
            "Sample_Size": 0,
            "Number_of_Errors": {"overstatements": 0, "understatements": 0},
            "Gross_most_likely_error": 0,
            "Net_most_likely_error": {"overstatements": 0, "understatements": 0},
            "Basic_Precision": 0,
            "Precision_Gap_widening": {"overstatements": 0, "understatements": 0},
            "Total_Precision": {"overstatements": 0, "understatements": 0},
            "Gross_upper_error_limit": {"overstatements": 0, "understatements": 0},
            "Net_upper_error_limit": 0,
            "Gross_Value_of_Errors": {"overstatements": 0, "understatements": 0}
        }
        filled_sample = "Not required because no sample items were selected during extraction"
        over = under = "Not applicable because no sample items were selected during extraction"

    # --- High value items evaluation ---
    if filled_high_values is not None and len(filled_high_values) > 0:
        if not isinstance(filled_high_values, (pd.DataFrame, np.ndarray)):
            raise ValueError("filled_high_values must be a DataFrame or matrix")
        if extract['col_name_book_values'] not in filled_high_values.columns:
            raise ValueError("filled_high_values must include column with book values named by extract.col_name_book_values")
        if col_name_audit_values not in filled_high_values.columns:
            raise ValueError("filled_high_values must include column with audit values named by col_name_audit_values")
        if col_name_riskweights is not None and col_name_riskweights not in filled_high_values.columns:
            raise ValueError("If col_name_riskweights is not None, filled_high_values must include that column")
        if col_name_riskweights is None:
            errors_high = filled_high_values[extract['col_name_book_values']] - filled_high_values[col_name_audit_values]
        else:
            errors_high = (filled_high_values[extract['col_name_book_values']] - filled_high_values[col_name_audit_values]) / filled_high_values[col_name_riskweights]
        Results_High_values = {
            "Number_of_high_value_items": filled_high_values.shape[0],
            "Number_of_Errors": {
                "overstatements": int((errors_high > 0).sum()),
                "understatements": int((errors_high < 0).sum())
            },
            "Gross_Value_of_Errors": {
                "overstatements": float(errors_high[errors_high > 0].sum()),
                "understatements": float(errors_high[errors_high < 0].sum())
            },
            "Net_Value_of_Errors": float(errors_high.sum())
        }
    else:
        Results_High_values = {
            "Number_of_high_value_items": 0,
            "Number_of_Errors": {"overstatements": 0, "understatements": 0},
            "Gross_Value_of_Errors": {"overstatements": 0, "understatements": 0},
            "Net_Value_of_Errors": 0
        }
        filled_high_values = "Not required because no high value items were selected during extraction."

    # --- Combined results ---
    Results_Total = {
        "Total_number_of_items_examined": Results_Sample['Sample_Size'] + Results_High_values['Number_of_high_value_items'],
        "Number_of_Errors": {
            "overstatements": Results_Sample['Number_of_Errors']['overstatements'] + Results_High_values['Number_of_Errors']['overstatements'],
            "understatements": Results_Sample['Number_of_Errors']['understatements'] + Results_High_values['Number_of_Errors']['understatements']
        },
        "Gross_most_likely_error": np.array(Results_Sample['Gross_most_likely_error']) + np.array([
            Results_High_values['Gross_Value_of_Errors']['overstatements'],
            Results_High_values['Gross_Value_of_Errors']['understatements']
        ]),
        "Gross_Value_of_Errors": {
            "overstatements": Results_Sample['Gross_Value_of_Errors']['overstatements'] + Results_High_values['Gross_Value_of_Errors']['overstatements'],
            "understatements": Results_Sample['Gross_Value_of_Errors']['understatements'] + Results_High_values['Gross_Value_of_Errors']['understatements']
        },
        "Net_most_likely_error": np.array([1, -1]) * np.sum(Results_Sample['Gross_most_likely_error']) + Results_High_values['Net_Value_of_Errors'] * np.array([1, -1]),
        "Gross_upper_error_limit": np.array(Results_Sample['Gross_upper_error_limit']) + np.array([
            Results_High_values['Gross_Value_of_Errors']['overstatements'],
            Results_High_values['Gross_Value_of_Errors']['understatements']
        ]),
        # For Net_upper_error_limit, as in R: Results.Sample$Gross.upper.error.limit - Results.Sample$Gross.most.likely.error + ...
        "Net_upper_error_limit": np.array(Results_Sample['Gross_upper_error_limit']) - np.array(Results_Sample['Gross_most_likely_error']) + np.array([1, -1]) * np.sum(Results_Sample['Gross_most_likely_error']) + np.array([
            Results_High_values['Gross_Value_of_Errors']['overstatements'],
            Results_High_values['Gross_Value_of_Errors']['understatements']
        ])
    }

    # --- Acceptability evaluation ---
    UEL_low_error_rate = np.max(Results_Total['Net_upper_error_limit'] * np.array([1, -1]))
    acceptable_low_error_rate = UEL_low_error_rate < extract['tolerable_error']
    acceptable = acceptable_low_error_rate

    # --- High error rate evaluation ---
    ratios = 1 - filled_sample[col_name_audit_values] / filled_sample[extract['col_name_book_values']]
    qty_errors = (ratios != 0).sum()
    ratios_mean = ratios.mean()
    ratios_sd = ratios.std(ddof=0)
    N = extract['data'].shape[0] - (filled_high_values.shape[0] if isinstance(filled_high_values, pd.DataFrame) else 0)
    R = 1 - (1 - extract['confidence_level']) / 2 if interval_type == "two-sided" else extract['confidence_level']
    U = t.ppf(R, max(R, qty_errors - 1) if qty_errors > 1 else 1)
    if isinstance(filled_high_values, pd.DataFrame):
        Y = extract['data'][extract['col_name_book_values']].sum() - filled_high_values[extract['col_name_book_values']].sum()
        high_values_error = (filled_high_values[extract['col_name_book_values']] - filled_high_values[col_name_audit_values]).sum()
    else:
        Y = extract['data'][extract['col_name_book_values']].sum()
        high_values_error = 0

    most_likely_error = ratios_mean * Y
    precision = U * Y * ratios_sd / np.sqrt(filled_sample.shape[0])
    UEL_high_error_rate = most_likely_error + precision * np.sign(most_likely_error) + high_values_error
    acceptable_high_error_rate = (UEL_high_error_rate <= extract['tolerable_error'])

    high_error_rate = {
        "most_likely_error": most_likely_error + high_values_error,
        "upper_error_limit": UEL_high_error_rate,
        "acceptable": acceptable_high_error_rate
    }

    # Advice for high error rate
    MLE_low_error_rate = Results_Total['Net_most_likely_error'][0]
    MLE_high_error_rate = high_error_rate['most_likely_error']
    MLE_final = MLE_low_error_rate
    if max(Results_Sample['Number_of_Errors'].values()) >= 20:
        if print_advice:
            print("** You had at least 20 errors in the sample. High Error Rate evaluation recommended.")
        acceptable = acceptable_high_error_rate
        MLE_final = MLE_high_error_rate

    # --- Result assembly ---
    result = {
        **extract,
        "filled_sample": filled_sample,
        "filled_high_values": filled_high_values,
        "col_name_audit_values": col_name_audit_values,
        "Overstatements_Result_Details": over,
        "Understatements_Result_Details": under,
        "Results_Sample": Results_Sample,
        "Results_High_values": Results_High_values,
        "Results_Total": Results_Total,
        "UEL_low_error_rate": UEL_low_error_rate,
        "UEL_high_error_rate": UEL_high_error_rate,
        "MLE_low_error_rate": MLE_low_error_rate,
        "MLE_high_error_rate": MLE_high_error_rate,
        "MLE_final": MLE_final,
        "acceptable_low_error_rate": acceptable_low_error_rate,
        "acceptable_high_error_rate": acceptable_high_error_rate,
        "high_error_rate": high_error_rate,
        "combined": combined
    }

    if experimental:
        # Placeholders for additional methods (moment, binomial, multinomial bounds)
        result['moment_bound'] = None # Implement if needed
        result['acceptable_moment_bound'] = None
        result['binomial_bound'] = None
        result['acceptable_binomial_bound'] = None
        result['multinomial_bound'] = None
        result['acceptable_multinomial_bound'] = None

    return result
