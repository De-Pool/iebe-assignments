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
    opdracht1(nervlove_data)
    opdracht2(nervlove_data)


def log(x):
    return np.log(x)


def opdracht1(data):
    # Each variable in the formula is a key in the input data
    # Each variable can be transformed by a Python function
    formula = 'log(totcost) ~ 1 + log(output) + log(plabor) + log(pfuel) + log(pkap)'
    ols_model = smf.ols(formula=formula, data=data).fit()
    print(ols_model.summary())


def opdracht2(data):
    # output, plabor
    output_plabor_corr = np.correlate(data['output'].values, data['plabor'])
    print(output_plabor_corr)

    # output, pfuel
    output_pfuel_corr = np.correlate(data['output'].values, data['pfuel'])
    print(output_pfuel_corr)

    # output, pkap
    output_pkap_corr = np.correlate(data['output'].values, data['pkap'])
    print(output_pkap_corr)

    # plabor, pfuel
    plabor_pfuel_corr = np.correlate(data['plabor'].values, data['pfuel'])
    print(plabor_pfuel_corr)

    # plabor, pkap
    plabor_pkap_corr = np.correlate(data['plabor'].values, data['pkap'])
    print(plabor_pkap_corr)

    # pfuel, pkap
    pfuel_pkap_corr = np.correlate(data['pfuel'].values, data['pkap'])
    print(pfuel_pkap_corr)


if __name__ == '__main__':
    main()
