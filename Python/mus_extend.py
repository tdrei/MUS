import numpy as np
import pandas as pd
from copy import deepcopy

class MUSPlanningResult:
    def __init__(self, data, col_name_book_values, confidence_level, tolerable_error,
                 expected_error, book_value, n, high_value_threshold, tolerable_taintings, combined):
        self.data = data
        self.col_name_book_values = col_name_book_values
        self.confidence_level = confidence_level
        self.tolerable_error = tolerable_error
        self.expected_error = expected_error
        self.book_value = book_value
        self.n = n
        self.high_value_threshold = high_value_threshold
        self.tolerable_taintings = tolerable_taintings
        self.combined = combined

class MUSExtractionResult:
    def __init__(self, sample, sample_population, high_values, data, col_name_book_values,
                 confidence_level, tolerable_error, expected_error, book_value, n, combined,
                 extensions=0, n_qty=None, seed=None, obey_n_as_min=None):
        self.sample = sample
        self.sample_population = sample_population
        self.high_values = high_values
        self.data = data
        self.col_name_book_values = col_name_book_values
        self.confidence_level = confidence_level
        self.tolerable_error = tolerable_error
        self.expected_error = expected_error
        self.book_value = book_value
        self.n = n
        self.combined = combined
        self.extensions = extensions
        self.n_qty = n_qty if n_qty is not None else [n]
        self.seed = seed
        self.obey_n_as_min = obey_n_as_min
        self.interval = None
        self.additional_sample = None

def MUS_extraction(plan, start_point=None, seed=None, obey_n_as_min=None, combined=None):
    # Dummy placeholder for the actual extraction logic.
    # Replace this with your implementation!
    # This should return a MUSExtractionResult object.
    raise NotImplementedError("MUS_extraction must be implemented.")

def MUS_extend(extract, new_plan=None, additional_n=None):
    if not isinstance(extract, MUSExtractionResult):
        raise TypeError("extract has to be an object from type MUSExtractionResult. Use function MUS_extraction to create such an object.")
    if new_plan is not None and not isinstance(new_plan, MUSPlanningResult):
        raise TypeError("new_plan has to be an object from type MUSPlanningResult. Use function MUS_planning to create such an object or None.")
    if additional_n is not None and not isinstance(additional_n, (int, float)):
        raise TypeError("additional_n must be numeric or None.")

    if additional_n is None:
        additional_n = 0

    # Rebuild plan from extract object if new_plan is None
    if new_plan is None:
        n_final = extract.n + additional_n
        interval = extract.book_value / n_final if n_final != 0 else 0
        tol_taint = extract.expected_error / extract.book_value * n_final if extract.book_value != 0 else 0

        new_plan = MUSPlanningResult(
            data=extract.data,
            col_name_book_values=extract.col_name_book_values,
            confidence_level=extract.confidence_level,
            tolerable_error=extract.tolerable_error,
            expected_error=extract.expected_error,
            book_value=extract.book_value,
            n=n_final,
            high_value_threshold=interval,
            tolerable_taintings=tol_taint,
            combined=extract.combined
        )
    else:
        additional_n = new_plan.n - extract.n

    if additional_n < 1:
        extract.additional_sample = extract.sample.iloc[0:0].copy()
        return extract

    total_n = additional_n + extract.n
    colunas = extract.sample_population.columns
    sample_cols = extract.sample.columns

    # Split data into high values and population from which will be sampled
    if isinstance(extract.high_values, pd.DataFrame):
        old_high_values = extract.high_values.copy()
    else:
        old_high_values = extract.sample_population.iloc[0:0].copy()
    old_high_values['MUS.total'] = 0
    old_high_values['MUS.hit'] = 0

    old_sample = extract.sample.copy()
    old_audited = pd.concat([
        old_sample.loc[:, sample_cols],
        old_high_values.loc[:, sample_cols]
    ], axis=0)

    # Create a brand new sample with the new n
    new_extract = MUS_extraction(new_plan, start_point=None, seed=extract.seed,
                                 obey_n_as_min=extract.obey_n_as_min, combined=extract.combined)

    if isinstance(new_extract.high_values, pd.DataFrame):
        new_high_values = new_extract.high_values.copy()
    else:
        new_high_values = new_extract.sample_population.iloc[0:0].copy()
    new_high_values['MUS.total'] = 0
    new_high_values['MUS.hit'] = 0

    new_sample_index = new_extract.sample.index
    new_n = len(new_sample_index)
    selected = old_audited.index.difference(new_high_values.index)
    new_basedraw = new_sample_index.difference(selected)
    new_size = max(0, new_n - len(selected))

    if new_size > 0:
        adding = np.random.choice(new_basedraw, new_size, replace=False)
        final_sample = selected.tolist() + list(adding)
    else:
        final_sample = selected.tolist()
        adding = []

    if "MUS.total" not in new_extract.sample_population.columns:
        new_extract.sample_population["MUS.total"] = 0
    if "MUS.hit" not in new_extract.sample_population.columns:
        new_extract.sample_population["MUS.hit"] = 0

    new_extract.sample = new_extract.sample_population.loc[
        new_extract.sample_population.index.isin(final_sample), sample_cols
    ]
    new_extract.sample_population = new_extract.sample_population.loc[:, colunas]

    # Calculate revised sampling interval (used for evaluation of the sample population)
    total_book_value = new_extract.sample_population[new_extract.col_name_book_values].sum()
    nrows = new_extract.sample.shape[0]
    new_extract.interval = total_book_value / nrows if nrows != 0 else 0

    new_extract.extensions = extract.extensions + 1
    new_extract.n_qty = extract.n_qty + [additional_n]

    return new_extract
