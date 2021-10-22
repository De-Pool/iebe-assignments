import pyreadr
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as stats
import pylab
import seaborn as sns
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def main():
    sns.set_theme(color_codes=True)
    sns.set_theme(style="ticks")

    data = pyreadr.read_r('./data.RData')
    # nervlove_data has the following columns:
    # totcost, output, plabor, pfuel, pkap
    nervlove_data = data['NervloveData']

    model = exercise1(nervlove_data, True)
    exercise2(nervlove_data, model, True)
    exercise3(model)
    exercise4(nervlove_data, model)
    exercise5(nervlove_data, True)
    exercise6(nervlove_data, True)


def exercise1(data, print_all):
    # Each variable in the formula is a key in the input data
    # Each variable can be transformed by a Python function
    formula = 'log(totcost) ~ 1 + log(output) + log(plabor) + log(pfuel) + log(pkap)'
    ols_model = smf.ols(formula=formula, data=data)
    ols_model_fit = ols_model.fit()

    u_output_cov = np.cov(ols_model_fit.resid, log(data['output']))[0][1]
    u_plabor_cov = np.cov(ols_model_fit.resid, log(data['plabor']))[0][1]
    u_pfuel_cov = np.cov(ols_model_fit.resid, log(data['pfuel']))[0][1]
    u_pkap_cov = np.cov(ols_model_fit.resid, log(data['pkap']))[0][1]

    if print_all:
        print('Exercise 1')
        print(ols_model_fit.summary(), '\n')
        print('Covariance between u and output:', u_output_cov, np.nanmean(log(data['output'])))
        print('Covariance between u and plabor:', u_plabor_cov, np.nanmean(log(data['plabor'])))
        print('Covariance between u and pfuel:', u_pfuel_cov, np.nanmean(log(data['pfuel'])))
        print('Covariance between u and pkap:', u_pkap_cov, np.nanmean(log(data['pkap'])))
        print('Average of residuals', np.nanmean(ols_model_fit.resid.values), '\n')

    return ols_model


def exercise2(data, model, print_all):
    model_fit = model.fit()

    output_plabor_corr = np.corrcoef(data['output'].values, data['plabor'])
    output_pfuel_corr = np.corrcoef(data['output'].values, data['pfuel'])
    output_pkap_corr = np.corrcoef(data['output'].values, data['pkap'])
    plabor_pfuel_corr = np.corrcoef(data['plabor'].values, data['pfuel'])
    plabor_pkap_corr = np.corrcoef(data['plabor'].values, data['pkap'])
    pfuel_pkap_corr = np.corrcoef(data['pfuel'].values, data['pkap'])

    # Test the coefficients by using heteroskedasticity-robust standard errors
    hc_robust_se = model_fit.HC0_se
    b_hat = model_fit.params
    t_hc_robust = b_hat / hc_robust_se

    # F = (R*B - r)' * (R*Var(B)*R')^-1 * (RB -r)
    # B = vector of estimated B parameters k+1 --> the intercept B0 is also included
    # R = matrix of full row rank, qxk+1
    # r = vector of qx1
    # X = matrix of nxp, with n samples and p regressors
    # Var(B|X) = sigma_u^2 * (X'X)^-1
    # where sigma_u^2 is the variance of the error terms
    # an unbiased estimator for sigma_u^2 = u'u / (n-p), where u is a nx1 vector with all the residuals
    # our final F can then be written as:  F = (R*B - r)' * (R*(u'u/(n-p)) * (X'X)^-1*R')^-1 * (RB - r)
    X = np.array([np.ones(len(data['output'].values)), log(data['output'].values), log(data['plabor'].values),
                  log(data['pfuel'].values), log(data['pkap'].values)]).T
    R = np.array([0, 0, 1, 1, 1])
    B_hat = model_fit.params.values
    r = np.array(1)
    u = model_fit.resid.values
    sigma = u.T @ u / (len(X) - len(X[0]))
    var_b = sigma * linalg.inv(X.T @ X)

    F = ((R @ B_hat - r) * (R @ B_hat - r)) / (R @ var_b @ R.T)

    if print_all:
        print('Exercise 2')
        print('output, plabor', output_plabor_corr[0][1])
        print('output, pfuel', output_pfuel_corr[0][1])
        print('output, pkap', output_pkap_corr[0][1])
        print('plabor, pfuel', plabor_pfuel_corr[0][1])
        print('plabor, pkap', plabor_pkap_corr[0][1])
        print('pfuel, pkap', pfuel_pkap_corr[0][1], '\n')
        print('heteroskedasticity-robust standard error t-statistic\n', t_hc_robust, '\n')
        print('F-statistic', F, '\n')


def exercise3(model):
    model_fit = model.fit()
    data = pd.DataFrame()
    data['residuals'] = model_fit.resid.values
    data['fitted'] = model_fit.fittedvalues.values

    # Plot the residuals against the fitted values
    sns.lmplot(x="fitted", y="residuals", data=data, height=5, aspect=1.5, truncate=False, order=2)
    plt.scatter(data['fitted'], data['residuals'])
    plt.title("Residuals plotted against the fitted values (Question 1 model)")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

    # QQ plot
    sm.qqplot(data['residuals'], line='45')
    pylab.show()

    # E[u'u|X] = I*sigma^squared --> show the presence of homoskedasticity
    # Null hypothesis of the Breusch Pagan test is the presence of homoskedasticity
    # if the pvalue is less than 0.05, we reject the null hypothesis,
    # so we can conclude heteroskedasticity is present.
    _, _, _, fpvalue = stats.het_breuschpagan(model_fit.resid, model_fit.model.exog)
    print('p-value breusch pagan', fpvalue, '\n')


def exercise4(data, model):
    model_fit = model.fit()

    data['residuals'] = model_fit.resid.values
    data['log_q'] = log(data['output'])

    # Plot the residuals against log(Q)
    sns.lmplot(x="log_q", y="residuals", data=data, height=5, aspect=1.5, truncate=False, order=2)
    plt.scatter(data['log_q'], data['residuals'])
    plt.title("Residuals plotted against log(Q) (Question 1 model)")
    plt.xlabel("log(Q)")
    plt.ylabel("Residuals")
    plt.show()


def exercise5(data, print_all):
    # Each variable in the formula is a key in the input data
    # Each variable can be transformed by a Python function
    formula = 'log(totcost) ~ 1 + log(output) + log_squared(output) + log(plabor) + log(pfuel) + log(pkap)'
    ols_model = smf.ols(formula=formula, data=data)
    ols_model_fit = ols_model.fit()

    data['residuals'] = ols_model_fit.resid.values
    data['fitted'] = ols_model_fit.fittedvalues.values

    # Plot the residuals against the fitted values
    sns.lmplot(x="fitted", y="residuals", data=data, height=5, aspect=1.5, truncate=False)
    plt.scatter(data['fitted'], data['residuals'])
    plt.title("Residuals plotted against the fitted values (Question 5 model)")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

    # QQ plot
    sm.qqplot(data['residuals'], line='45')
    pylab.show()

    if print_all:
        print('Exercise 5')
        print(ols_model_fit.summary())


def exercise6(data, print_all):
    subsample1 = data[:29]
    subsample2 = data[29:29 * 2]
    subsample3 = data[29 * 2:29 * 3]
    subsample4 = data[29 * 3:29 * 4]
    subsample5 = data[29 * 4:]

    if print_all:
        print('Exercise 6')
        print('Subsample 1\n', subsample1.describe(), '\n')
        print('Subsample 2\n', subsample2.describe(), '\n')
        print('Subsample 3\n', subsample3.describe(), '\n')
        print('Subsample 4\n', subsample4.describe(), '\n')
        print('Subsample 5\n', subsample5.describe(), '\n')


def log(x):
    return np.log(x)


def log_squared(x):
    return np.square(np.log(x))


if __name__ == '__main__':
    main()
