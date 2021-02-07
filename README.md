# syndat

Toolbox to generate synthetic data from a real data set

## Description
The joint (multivariate estimates) or individual (univariate estimates) distribution of the variables are estimated from the input data set using kernel density techniques. Synthetic data are generated by random sampling of the estimated distributions.

SynDat provides the option of using multivariate and univariate density estimates and can handle a mixture of continuous, discrete, ordinal, categorical, and date/time variables.

The multivariate approach is preferred over univariate when possible. The multivariate estimate, however, is more computationally demanding, and can become intractable especially with skewed distributions, in which case the user can switch to the univariate model.


## Usage
### Running from terminal
SynDat can be directly run from the unix shell:


```
python syndat.py --data <csv> --cols <json> --out <csv> [--dt_format <date/time format>] [--univariate] [--sampsize <desired sample size, default = 1000>]
```

Optional arguments are enclosed in brackets [ ]

Usage information can be found by running `python syndat.py --help`

### Using as a python class
Syndat can be imported as a class in python. The following notebook illustrates the basic usage:

https://github.com/kayvanrad/syndat/blob/master/syndat_usage_example.ipynb

## Supported data types
| Indicator | Description              |
|-----------|--------------------------|
| float     | Continuous quantitative  |
| int       | Discrete quantitative    |
| dt        | Date/time                |
| ord       | Ordered categorical      |
| unord     | Unordered categorical    |

## `cols` dictionary / json file example:

```
cols = {'x_continuous':'float', 'x_discrete':'int', 'x_categ':'unord', 'x_date':'dt'}
```

## Data with NA's
| Variable type | Recommendation |
|-|-|
| unord | NA’s are considered as their own level. Synthetic data will have NA’s with the same probabilities. If different codes are used for NA’s (e.g., Null, None, etc.) each will be considered as a separate level. The user can impute the NA’s or unify different NA codes  during preprocessing if desired. |
| int, ord | NA’s must be imputed or removed NA’s prior to passing the data to SynDat.|
| float | For the multivariate approach, NA’s must be imputed or removed NA’s prior to passing the data to SynDat. The univariate approach can handle NA’s - the synthetic data will have NA's with the same probability as the original data|
| dt | NA’s must be imputed or removed before prior to passing the data to SynDat. If NA values are meaningful (e.g., a future date) it is recommended to replace them with a designated date prior to passing the data to SynDat using the dt_null_encoder utility. Please refer to the usage notebook for details. |

