# pip install pyreadr
import pyreadr

# pip install statsmodels
import statsmodels.formula.api as smf

import numpy as np


def main():
    data = pyreadr.read_r('./data.RData')
    # nervlove_data has the following columns:
    # totcost, output, plabor, pfuel, pkap
    nervlove_data = data['NervloveData']

    # Each variable in the formula is a key in the input data
    # Each variable can be transformed by a Python function
    formula = 'log(totcost) ~ 1 + log(output) + log(plabor) + log(pfuel) + log(pkap)'
    ols_model = smf.ols(formula=formula, data=nervlove_data).fit()
    print(ols_model.summary())


def log(x):
    return np.log(x)


if __name__ == '__main__':
    main()
