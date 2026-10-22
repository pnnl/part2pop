import numpy as np
import pandas as pd
from part2pop.population.factory.helpers.edx import _extract_element_mass_fractions


def test_element_in_first_dataframe_column_is_found():
    data = pd.DataFrame({'C':[0.7, 0.6], 'O':[0.3, 0.4]})
    values, names = _extract_element_mass_fractions(data, ['C', 'O'], 'synthetic.csv')
    assert list(names) == ['C', 'O']
    np.testing.assert_allclose(values, [[0.7, 0.3], [0.6, 0.4]])
