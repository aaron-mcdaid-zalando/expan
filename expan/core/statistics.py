import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _delta_mean(x, y):
    """Implemented as function to allow calling from bootstrap. """
    return np.nanmean(x) - np.nanmean(y)


def make_delta(assume_normal=True, percentiles=[2.5, 97.5],
               min_observations=20, nruns=10000, relative=False, multi_test_correction=False, num_tests=1):
    """ a closure to the below delta function """

    def f(x, y, x_denominators=1, y_denominators=1):
        return delta(x, y, assume_normal, percentiles, min_observations,
                     nruns, relative, x_denominators, y_denominators, multi_test_correction, num_tests)

    return f


def delta(x, y, assume_normal=True, percentiles=[2.5, 97.5],
          min_observations=20, nruns=10000, relative=False, x_denominators=1, y_denominators=1,
          multi_test_correction=False, num_tests=1):
    """
    Calculates the difference of means between the samples (x-y) in a
    statistical sense, i.e. with confidence intervals.

    NaNs are ignored: treated as if they weren't included at all. This is done
    because at this level we cannot determine what a NaN means. In some cases,
    a NaN represents missing data that should be completely ignored, and in some
    cases it represents inapplicable (like PCII for non-ordering customers) - in
    which case the NaNs should be replaced by zeros at a higher level. Replacing
    with zeros, however, would be completely incorrect for return rates.

    Computation is done in form of treatment minus control, i.e. x-y

    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        assume_normal (boolean): specifies whether normal distribution
            assumptions can be made
        percentiles (list): list of percentile values for confidence bounds
        min_observations (integer): minimum number of observations needed
        nruns (integer): only used if assume normal is false
        relative (boolean): if relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
        x_denominator (list): ...
        y_denominator (list): ...
        multi_test_correction (boolean): flag of whether the correction for multiple testing is needed.
        num_tests (integer): number of tests or reported kpis used for multiple correction.

    Returns:
        DeltaStatistics object
    """
    # Checking if data was provided
    if x is None or y is None:
        raise ValueError('Please provide two non-None samples.')

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)
    # I would like to assert there are no 'inf's in the input,
    # but I'm scared to do so

    _x_ratio = _x / x_denominators
    _y_ratio = _y / y_denominators

    # 0/0 will give 'nan' above, which is what we want.

    x_nan = np.isnan(_x_ratio).sum()
    y_nan = np.isnan(_y_ratio).sum()
    if x_nan > 0:
        warnings.warn('Discarding ' + str(x_nan) + ' NaN(s) in the x array!')
    if y_nan > 0:
        warnings.warn('Discarding ' + str(y_nan) + ' NaN(s) in the y array!')

    ss_x = sample_size(_x_ratio)
    ss_y = sample_size(_y_ratio)

    # Checking if enough observations are left after dropping NaNs
    if min(ss_x, ss_y) < min_observations:
        # Set mean to nan
        mu = np.nan
        # Create nan dictionary
        c_i = dict(list(zip(percentiles, np.empty(len(percentiles)) * np.nan)))
        # Return the result structure
        # return mu, c_i, ss_x, ss_y, np.nanmean(_x), np.nanmean(_y)
        c_i = [{'percentile': p, 'value': v} for (p, v) in c_i.items()]

        return {'delta'                 : float(mu),
                'confidence_interval'   : c_i,
                'treatment_sample_size' : int(ss_x),
                'control_sample_size'   : int(ss_y),
                'treatment_mean'        : float(np.nanmean(_x)),
                'control_mean'          : float(np.nanmean(_y)),
                'treatment_variance'    : float(np.nanvar(_x)),
                'control_variance'      : float(np.nanvar(_y))}

    else:
        # Computing the mean
        # Computing the confidence intervals
        if assume_normal:
            nswd = normal_sample_weighted_difference(
                            x_numerators=_x, y_numerators=_y,
                            x_denominators = x_denominators, y_denominators = y_denominators,
                            percentiles=percentiles, relative=relative,
                                           multi_test_correction=multi_test_correction, num_tests=num_tests)
            c_i =   nswd['c_i']

            assert ss_x == nswd['x_n']
            assert ss_y == nswd['y_n']

            # Return the result structure
            # return mu, c_i, ss_x, ss_y, np.nanmean(_x), np.nanmean(_y)
            c_i = [{'percentile': p, 'value': v} for (p, v) in c_i.items()]


            return {'delta'                 : float(nswd['mean(x)-mean(y)']),
                    'confidence_interval'   : c_i,
                    'treatment_sample_size' : int(ss_x),
                    'control_sample_size'   : int(ss_y),
                    'treatment_mean'        : float(nswd['x_mean']),
                    'control_mean'          : float(nswd['y_mean']),
                    'treatment_variance'    : float(nswd['x_var']),
                    'control_variance'      : float(nswd['y_var']),
                    }
        else:
            mu = _delta_mean(_x, _y)
            # We need to consider 'bootstrap' more with weighting. Should it be adjusted for
            # weighting? Probably yes, just not sure how to do it yet. TODO
            c_i, _ = bootstrap(x=_x_ratio, y=_y_ratio, percentiles=percentiles, nruns=nruns, relative=relative,
                               multi_test_correction=multi_test_correction, num_tests=num_tests)

            # Return the result structure
            # return mu, c_i, ss_x, ss_y, np.nanmean(_x), np.nanmean(_y)
            c_i = [{'percentile': p, 'value': v} for (p, v) in c_i.items()]

            return {'delta'                 : float(mu),
                    'confidence_interval'   : c_i,
                    'treatment_sample_size' : int(ss_x),
                    'control_sample_size'   : int(ss_y),
                    'treatment_mean'        : float(np.nanmean(_x)),
                    'control_mean'          : float(np.nanmean(_y)),
                    'treatment_variance'    : float(np.nanvar(_x)),
                    'control_variance'      : float(np.nanvar(_y))}


def sample_size(x):
    """
    Calculates sample size of a sample x
    Args:
        x (array_like): sample to calculate sample size

    Returns:
        int: sample size of the sample excluding nans
    """
    # cast into a dummy numpy array to infer the dtype
    x_as_array = np.array(x)

    if np.issubdtype(x_as_array.dtype, np.number):
        _x = np.array(x, dtype=float)
        x_nan = np.isnan(_x).sum()
    # assuming categorical sample
    elif isinstance(x, pd.core.series.Series):
        x_nan = x.str.contains('NA').sum()
    else:
        x_nan = list(x).count('NA')

    return len(x) - x_nan


def estimate_sample_size(x, mde, r, alpha=0.05, beta=0.2):
    """
    Estimates sample size based on sample mean and variance given MDE (Minimum Detectable effect), number of variants and variant split ratio

    Args:
        x (pd.Series or pd.DataFrame): sample to base estimation on
        mde (float): minimum detectable effect
        r (float): variant split ratio
        alpha (float): significance level
        beta (float): type II error

    Returns:
        float or pd.Series: estimated sample size

    """
    if not isinstance(x, pd.Series) and not isinstance(x, pd.DataFrame):
        raise TypeError("Sample x needs to be either Series or DataFrame.")

    if r <= 0:
        raise ValueError("Variant split ratio needs to be higher than 0.")

    ppf = stats.norm.ppf
    c1 = (ppf(1.0 - alpha/2.0) - ppf(beta))**2
    c2 = (1.0 + r) * c1 * (1.0 + 1.0 / r)
    return c2 * x.var() / (mde * x.mean())**2


def chi_square(x, y, min_counts=5):
    """
    Performs the chi-square homogeneity test on categorical arrays x and y

    Args:
        x (array_like): sample of the treatment variable to check
        y (array_like): sample of the control variable to check
        min_counts (int): drop categories where minimum number of observations
                        or expected observations is below min_counts for x or y

    Returns:
        tuple:
            * float: p-value
            * float: chi-square value
            * int: number of attributes used (after dropping)
    """
    # Checking if data was provided
    if x is None or y is None:
        raise ValueError('Please provide two samples.')

    # Check if data is not empty
    if not len(x) or not len(y):
        return np.nan

    # Transform input to categorical variable
    _x = pd.Categorical(x)
    _y = pd.Categorical(y)

    #
    treat_counts = _x.value_counts()
    control_counts = _y.value_counts()
    # Get observed counts for both _x and _y for each category
    # (=contingency table) and set the counts for non occuring categories to 0
    # This is a workaround to fix the bug #56, to cast the output of value_counts
    # into a Series with the normal Index instead of the CategoricalIndex
    tcs = pd.Series(treat_counts.values, index=treat_counts.index.astype(list))
    ccs = pd.Series(control_counts.values, index=control_counts.index.astype(list))
    observed_ct = pd.DataFrame([tcs, ccs]).fillna(0)
    # Ensure at least a frequency of 5 at every location in observed_ct,
    # otherwise drop categorie see
    # http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.chisquare.html
    observed_freqs = observed_ct[observed_ct >= min_counts].dropna(axis=1)

    # Calculate expected counts for chi-square homogeneity test
    # expected_freqs = group_totals*category_totals/all_totals
    # see e.g. Fahrmeir, L., Kuenstler, R., Pigeot, I., & Tutz, G. (2007).
    #          Statistik: Der Weg zur Datenanalyse. Springer-Verlag.
    all_totals = observed_freqs.sum().sum()
    category_totals = observed_freqs.sum(axis=0)
    expected_freqs = np.outer(category_totals,
                              observed_freqs.sum(axis=1) / all_totals).T

    # The actual degrees of freedom for the test are dof=(num_categories-1)
    # however the chisquare() function assumes dof=k-1, with
    # k = num_variants*num_categories
    # see http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.chisquare.html
    # Therefore, we have to correct using
    # ddof=(2*num_categories-1) - num_categories-1
    # Calculate the chi-square statistic and p-value
    es = expected_freqs.shape
    delta_dof = (es[0] * es[1] - 1) - (es[0] - 1) * (es[1] - 1)
    chisqr, p_val = stats.chisquare(f_obs=observed_freqs,
                                    f_exp=expected_freqs,
                                    ddof=delta_dof,
                                    axis=None)
    # Return the p-value
    return p_val, chisqr, es[1]


def alpha_to_percentiles(alpha):
    """
    Transforms alpha value to corresponding percentile.

    Args:
        alpha (float): alpha values to transform

    Returns:
        list of percentiles corresponding to given alpha
    """
    # Compute the percentiles
    return [100. * alpha / 2, 100. * (1 - (alpha / 2))]


def bootstrap(x, y, func=_delta_mean, nruns=10000, percentiles=[2.5, 97.5],
              min_observations=20, return_bootstraps=False, relative=False,
              multi_test_correction=False, num_tests=1):
    """
    Bootstraps the Confidence Intervals for a particular function comparing
    two samples. NaNs are ignored (discarded before calculation).

    Args:
        x (array like): sample of treatment group
        y (array like): sample of control group
        func (function): function of which the distribution is to be computed.
            The default comparison metric is the difference of means. For
            bootstraping correlation: func=lambda x,y: np.stats.pearsonr(x,y)[0]
        nruns (integer): number of bootstrap runs to perform
        percentiles (list):	The values corresponding to the given percentiles
            are returned. The default percentiles (2.5% and 97.5%) correspond to
            an alpha of 0.05.
        min_observations (integer): minimum number of observations necessary
        return_bootstraps (boolean): If this variable is set the bootstrap sets
            are returned otherwise the first return value is empty.
        relative (boolean): if relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In	this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
        multi_test_correction (boolean): flag of whether the correction for multiple testing is needed.
        num_tests (integer): number of tests or reported kpis used for multiple correction.

    Returns:
        tuple:
            * dict: percentile levels (index) and values
            * np.array (nruns): array containing the bootstraping results per run
    """
    # Checking if data was provided
    if x is None or y is None:
        raise ValueError('Please provide two non-None samples.')

    # Transform data to appropriate format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)
    ss_x = _x.size - np.isnan(_x).sum()
    ss_y = _y.size - np.isnan(_y).sum()

    # Adjusting percentiles, Bonferroni correction
    if multi_test_correction:
        percentiles = [float(p) / num_tests if p < 50.0
                       else 100 - (100 - float(p)) / num_tests if p > 50.0 else p for p in percentiles]

    # Checking if enough observations are left after dropping NaNs
    if min(ss_x, ss_y) < min_observations:
        # Create nan percentile dictionary
        c_val = dict(list(zip(percentiles, np.empty(len(percentiles)) * np.nan)))
        return (c_val, None)
    else:
        # Initializing bootstraps array and random sampling for each run
        bootstraps = np.ones(nruns) * np.nan
        for run in range(nruns):
            # Randomly choose values from _x and _y with replacement
            xp = _x[np.random.randint(0, len(_x), size=(len(_x),))]
            yp = _y[np.random.randint(0, len(_y), size=(len(_y),))]
            # Application of the given function to the bootstraps
            bootstraps[run] = func(xp, yp)
        # If relative is set subtract mean from bootstraps
        if relative:
            bootstraps -= np.nanmean(bootstraps)
        # Confidence values per given percentile as dictionary
        c_val = dict(list(zip(percentiles, np.percentile(bootstraps, q=percentiles))))
        return (c_val, None) if not return_bootstraps else (c_val, bootstraps)


def pooled_std(std1, n1, std2, n2):
    """
    Returns the pooled estimate of standard deviation. Assumes that population
    variances are equal (std(v1)**2==std(v2)**2) - this assumption is checked
    for reasonableness and an exception is raised if this is strongly violated.

    Args:
        std1 (float): standard deviation of first sample
        n1 (integer): size of first sample
        std2 (float): standard deviation of second sample
        n2 (integer): size of second sample

    Returns:
        float: Pooled standard deviation

    For further information visit:
        http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html

    Todo:
        Also implement a version for unequal variances.
    """
    if std1 != std2 and not (0.5 < (std1 ** 2) / (std2 ** 2) < 2.):
        warnings.warn('Sample variances differ too much to assume that '
                      'population variances are equal.')

    return np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))


def normal_percentiles(mean, std, n, percentiles=[2.5, 97.5], relative=False):
    """
    Calculate the percentile values for a normal distribution with parameters
    estimated from samples.

    Args:
        mean (float): mean value of the distribution
        std (float): standard deviation of the distribution
        n (integer): number of samples
        percentiles (list): list of percentile values to compute
        relative (boolean): if relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.

    Returns:
        dict: percentiles and corresponding values

    For more information visit:
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
        http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/dist/stat_tut/weg/st_eg/tut_mean_intervals.html
        http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
    """
    # Computing standard error
    st_error = std / np.sqrt(n)

    # Mapping percentiles via standard error
    if relative:
        return dict([(p, stats.t.ppf(p / 100.0, df=n - 1) * st_error)
                     for p in percentiles])
    else:
        return dict([(p, mean + stats.t.ppf(p / 100.0, df=n - 1) * st_error)
                     for p in percentiles])


def normal_sample_percentiles(values, percentiles=[2.5, 97.5], relative=False):
    """
    Calculate the percentile values for a sample assumed to be normally
    distributed. If normality can not be assumed, use bootstrap_ci instead.
    NaNs are ignored (discarded before calculation).

    Args:
        values (array-like): sample for which the normal distribution
            percentiles are computed.
        percentiles (list): list of percentile values to compute
        relative (boolean): if relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In	this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.

    Returns:
        dict: percentiles and corresponding values

    For further information visit:
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
        http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/dist/stat_tut/weg/st_eg/tut_mean_intervals.html
        http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
    """
    # Coerce data to right format
    _x = np.array(values, dtype=float)
    _x = _x[~np.isnan(_x)]

    # Determine distribution parameters
    mean = np.mean(_x)
    std = np.std(_x)
    n = len(_x)

    # Actual computation is done in normal_percentiles
    return normal_percentiles(mean=mean, std=std, n=n, percentiles=percentiles,
                              relative=relative)


def normal_sample_difference(x, y, percentiles=[2.5, 97.5], relative=False, multi_test_correction=False, num_tests=1):
    """
    Calculates the difference distribution of two normal distributions given
    by their samples.

    Computation is done in form of treatment minus control. It is assumed that
    the standard deviations of both distributions do not differ too much.

    Args:
        x (array-like): sample of a treatment group
        y (array-like): sample of a control group
        percentiles (list): list of percentile values to compute
        relative (boolean): If relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
        multi_test_correction (boolean): flag of whether the correction for multiple testing is needed.
        num_tests (integer): number of tests or reported kpis used for multiple correction.

    Returns:
        dict: percentiles and corresponding values
    """
    # Coerce data to right format
    _x = np.array(x, dtype=float)
    _x = _x[~np.isnan(_x)]
    _y = np.array(y, dtype=float)
    _y = _y[~np.isnan(_y)]

    # Calculate statistics
    mean1 = np.mean(_x)
    mean2 = np.mean(_y)
    std1 = np.std(_x)
    std2 = np.std(_y)
    n1 = len(_x)
    n2 = len(_y)

    # Push calculation to normal difference function
    return normal_difference(mean1=mean1, std1=std1, n1=n1, mean2=mean2,
                             std2=std2, n2=n2, percentiles=percentiles, relative=relative,
                             multi_test_correction=multi_test_correction, num_tests=num_tests)

def normal_sample_weighted_difference(x_numerators, y_numerators, x_denominators, y_denominators, percentiles=[2.5, 97.5], relative=False, multi_test_correction=False, num_tests=1):
    """
    Calculates the difference distribution of two normal distributions given
    by their samples.

    Computation is done in form of treatment minus control. It is assumed that
    the standard deviations of both distributions do not differ too much.

    Args:
        x_numerators, x_denominators (array-like): sample of a treatment group
        y_numerators, y_denominators (array-like): sample of a control group
        percentiles (list): list of percentile values to compute
        relative (boolean): If relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
        multi_test_correction (boolean): flag of whether the correction for multiple testing is needed.
        num_tests (integer): number of tests or reported kpis used for multiple correction.

    Returns:
        a dict with a few items:
            - ['c_i']               dict: percentiles and corresponding values
            - ['mean(x)-mean(y)']   difference in means
            - ['x_mean']
            - ['y_mean']
            - ['x_n']               sample size: i.e. numerator and denominator has reasonable values
            - ['y_n']               "
            - ['x_var']             variance, corrected for the weight
            - ['y_var']             variance, corrected for the weight
    """
    assert hasattr(x_numerators, '__len__')
    assert hasattr(y_numerators, '__len__')

    # if the denominators are just int/floats, convert them into
    # 'array-likes' of the appropriate length
    if not hasattr(x_denominators, '__len__'): x_denominators = (x_numerators*0.0) + x_denominators
    if not hasattr(y_denominators, '__len__'): y_denominators = (y_numerators*0.0) + y_denominators
    assert hasattr(x_denominators, '__len__')
    assert hasattr(y_denominators, '__len__')

    # check they have the appropriate length
    assert len(x_numerators) == len(x_denominators)
    assert len(y_numerators) == len(y_denominators)

    # Coerce data to right format
    _x_ratio = np.array(x_numerators/x_denominators, dtype=float)
    _y_ratio = np.array(y_numerators/y_denominators, dtype=float)
    # 0/0 will result in 'nan' in those two lines; and that's the
    # desired behaviour. Such data points will be removed

    # identify the positions that need to be removed, as they are 'NaN'
    _x_nans = ~np.isnan(_x_ratio)
    _y_nans = ~np.isnan(_y_ratio)

    # ... remove them
    _x_ratio = _x_ratio[_x_nans]
    _y_ratio = _y_ratio[_y_nans]
    x_numerators = x_numerators[_x_nans]
    x_denominators = x_denominators[_x_nans]
    y_numerators = y_numerators[_y_nans]
    y_denominators = y_denominators[_y_nans]
    assert len(_x_ratio) == len(x_numerators)
    assert len(_x_ratio) == len(x_denominators)
    assert len(_y_ratio) == len(y_numerators)
    assert len(_y_ratio) == len(y_denominators)

    # these next four lines are what need to change for derived KPIs
    # Calculate statistics
    mean1 = np.sum(x_numerators) / np.sum(x_denominators)
    mean2 = np.sum(y_numerators) / np.sum(y_denominators)

    x_mean_denominator = np.mean(x_denominators)
    y_mean_denominator = np.mean(y_denominators)

    # To understand the next two lines, think of linear regression.
    # We're computing the different between the real numerator and
    # and our 'prediction' of the numerator
    x_predicted_numerators = x_denominators * mean1
    y_predicted_numerators = y_denominators * mean2
    x_error_from_prediction = (x_numerators - x_predicted_numerators)
    y_error_from_prediction = (y_numerators - y_predicted_numerators)
    # These next two lines finally compute the correct variance, as
    # defined in the pdf:
    # https://github.bus.zalan.do/axolotl/experimentation-library/blob/master/variance_of_derived_kpis/derivedKpisAndSignificance.pdf
    std1 = np.std(x_error_from_prediction / x_mean_denominator)
    std2 = np.std(y_error_from_prediction / y_mean_denominator)

    n1 = len(_x_ratio)
    n2 = len(_y_ratio)

    # Push calculation to normal difference function
    c_i = normal_difference(mean1=mean1, std1=std1, n1=n1, mean2=mean2,
                             std2=std2, n2=n2, percentiles=percentiles, relative=relative,
                             multi_test_correction=multi_test_correction, num_tests=num_tests)
    assert isinstance(c_i, dict)
    return  {   'c_i':c_i,
                'mean(x)-mean(y)': mean1 - mean2,
                'x_mean' : mean1,
                'y_mean' : mean2,
                'x_n'    : n1,
                'y_n'    : n2,
                'x_var'    : np.var(x_error_from_prediction / x_mean_denominator),
                'y_var'    : np.var(y_error_from_prediction / y_mean_denominator),
            }


def normal_difference(mean1, std1, n1, mean2, std2, n2, percentiles=[2.5, 97.5],
                      relative=False, multi_test_correction=False, num_tests=1):
    """
    Calculates the difference distribution of two normal distributions.

    Computation is done in form of treatment minus control. It is assumed that
    the standard deviations of both distributions do not differ too much.

    Args:
        mean1 (float): mean value of the treatment distribution
        std1 (float): standard deviation of the treatment distribution
        n1 (integer): number of samples of the treatment distribution
        mean2 (float): mean value of the control distribution
        std2 (float): standard deviation of the control distribution
        n2 (integer): number of samples of the control distribution
        percentiles (list): list of percentile values to compute
        relative (boolean): If relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In	this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
        multi_test_correction (boolean): flag of whether the correction for multiple testing is needed.
        num_tests (integer): number of tests or reported kpis used for multiple correction.

    Returns:
        dict: percentiles and corresponding values

    For further information vistit:
            http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html
    """
    # Compute combined parameters from individual parameters
    mean = mean1 - mean2
    std = pooled_std(std1, n1, std2, n2)
    # Computing standard error
    st_error = std * np.sqrt(1. / n1 + 1. / n2)
    # Computing degrees of freedom
    d_free = n1 + n2 - 2

    # Adjusting percentiles, Bonferroni correction.
    if multi_test_correction:
        percentiles = [float(p) / num_tests if p < 50.0
                       else 100 - (100 - float(p)) / num_tests if p > 50.0 else p for p in percentiles]

    # Mapping percentiles via standard error
    if relative:
        return dict([(round(p, 5), stats.t.ppf(p / 100.0, df=d_free) * st_error)
                     for p in percentiles])
    else:
        return dict([(round(p, 5), mean + stats.t.ppf(p / 100.0, df=d_free) * st_error)
                     for p in percentiles])


def estimate_std(x, mu, pctile):
    """
    Estimate the standard deviation from a given percentile, according to
    the z-score:
        z = (x - mu) / sigma

    Args:
        x (float): cumulated density at the given percentile
        mu (float): mean of the distribution
        pctile (float): percentile value (between 0 and 100)

    Returns:
        float: estimated standard deviation of the distribution
    """
    return (x - mu) / stats.norm.ppf(pctile / 100.0)


def compute_statistical_power(x, y, alpha=0.05):
    """
    Compute statistical power
    Args:
        x (array-like): sample of a treatment group
        y (array-like): sample of a control group
        alpha: Type I error (false positive rate)

    Returns:
        float: statistical power --- the probability of a test to detect an effect,
            if the effect actually exists.
    """
    z_1_minus_alpha = stats.norm.ppf(1 - alpha/2.)

    _x = np.array(x, dtype=float)
    _x = _x[~np.isnan(_x)]
    _y = np.array(y, dtype=float)
    _y = _y[~np.isnan(_y)]

    mean1 = np.mean(_x)
    mean2 = np.mean(_y)
    std1 = np.std(_x)
    std2 = np.std(_y)
    n1 = len(_x)
    n2 = len(_y)

    return _get_power(mean1, std1, n1, mean2, std2, n2, z_1_minus_alpha)


def _get_power(mean1, std1, n1, mean2, std2, n2, z_1_minus_alpha):
    """
    Compute statistical power.
    This is a helper function for compute_statistical_power(x, y, alpha=0.05)
    Args:
        mean1 (float): mean value of the treatment distribution
        std1 (float): standard deviation of the treatment distribution
        n1 (integer): number of samples of the treatment distribution
        mean2 (float): mean value of the control distribution
        std2 (float): standard deviation of the control distribution
        n2 (integer): number of samples of the control distribution
        z_1_minus_alpha (float): critical value for significance level alpha. That is, z-value for 1-alpha.

    Returns:
        float: statistical power --- that is, the probability of a test to detect an effect,
            if the effect actually exists.
    """
    effect_size = mean1 - mean2
    std = pooled_std(std1, n1, std2, n2)
    tmp = (n1 * n2 * effect_size**2) / ((n1 + n2) * std**2)
    z_beta = z_1_minus_alpha - np.sqrt(tmp)
    beta = stats.norm.cdf(z_beta)
    power = 1 - beta

    return power
