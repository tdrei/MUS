import numpy as np
import pandas as pd

class MUSPlanningResult:
    # Placeholder for the actual MUSPlanningResult class implementation
    def __init__(self, data, col_name_book_values, high_value_threshold, n):
        self.data = data  # pandas DataFrame
        self.col_name_book_values = col_name_book_values
        self.High_value_threshold = high_value_threshold
        self.n = n

class MUSExtractionResult(dict):
    # Dict-based for easy attribute access, can be replaced with a dataclass
    pass

def mus_extraction(plan, start_point=None, seed=None, obey_n_as_min=False, combined=False):
    # Validate plan
    if not isinstance(plan, MUSPlanningResult):
        raise TypeError("plan has to be an object of type MUSPlanningResult. Use MUSPlanning to create such an object.")
    # Validate seed
    if seed is not None:
        if not (isinstance(seed, (int, np.integer)) and seed >= 0):
            raise ValueError("seed has to be an integer value greater or equal to 0.")
        np.random.seed(seed)
    # Validate obey_n_as_min
    if not isinstance(obey_n_as_min, bool):
        raise ValueError("obey_n_as_min has to be True or False.")

    # Split data into high values and population for sampling
    high_values = plan.data[plan.data[plan.col_name_book_values] >= plan.High_value_threshold].copy()
    sample_population = plan.data[plan.data[plan.col_name_book_values] < plan.High_value_threshold].copy()

    # Set interval
    interval = plan.High_value_threshold

    if obey_n_as_min:
        old_interval = interval
        if plan.n - len(high_values) <= 0:
            raise ValueError("Not enough items to sample after excluding high values.")
        interval = sample_population[plan.col_name_book_values].sum() / (plan.n - len(high_values))
        while not np.isclose(old_interval, interval):
            high_values = plan.data[plan.data[plan.col_name_book_values] >= interval].copy()
            sample_population = plan.data[plan.data[plan.col_name_book_values] < interval].copy()
            old_interval = interval
            if plan.n - len(high_values) <= 0:
                raise ValueError("Not enough items to sample after excluding high values.")
            interval = sample_population[plan.col_name_book_values].sum() / (plan.n - len(high_values))

    # Validate start_point
    if start_point is not None:
        if not (isinstance(start_point, (int, float)) and 0 <= start_point <= interval):
            raise ValueError("start_point has to be a numeric value between 0 and the interval.")

    # If no start_point, set a random one
    if start_point is None:
        start_point = np.random.uniform(0, interval)

    # Calculate sampling units (one in each interval)
    num_samples = plan.n - len(high_values)
    sampling_units = np.round(start_point + np.arange(num_samples + 1) * round(interval, 2))

    # Add running sum (MUS.total)
    sample_population = sample_population.copy()
    sample_population['MUS_total'] = sample_population[plan.col_name_book_values].cumsum()

    # Cut sampling_units if above maximum book value
    max_book_value = sample_population[plan.col_name_book_values].sum()
    sampling_units = sampling_units[sampling_units <= max_book_value]

    # Find the index in sample_population for each sampling_unit
    mus_hit_indices = np.searchsorted(sample_population['MUS_total'].values, sampling_units, side='right')
    mus_hit_indices = np.clip(mus_hit_indices, 0, len(sample_population) - 1)
    sample_rows = sample_population.iloc[mus_hit_indices].copy()
    sample_rows['MUS_hit'] = sampling_units

    # Calculate revised sampling interval
    if len(sample_rows) == 0:
        revised_interval = 0
    else:
        revised_interval = sample_population[plan.col_name_book_values].sum() / len(sample_rows)

    # Prepare result
    result = MUSExtractionResult({
        **plan.__dict__,
        'start_point': start_point,
        'seed': seed,
        'obey_n_as_min': obey_n_as_min,
        'high_values': high_values,
        'sample_population': sample_population,
        'sampling_interval': revised_interval,
        'sample': sample_rows,
        'extensions': 0,
        'n_qty': [len(sample_rows) + len(high_values)],
        'combined': combined
    })
    result.__class__ = MUSExtractionResult  # for type
    return result
