# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
logging.getLogger('fbprophet').disabled = True
import itertools


def wape(actual, predicted):
    return np.nansum(abs(actual - predicted)) / np.nansum(abs(actual))


def nfm(actual, predicted):
    return np.nansum(predicted - actual) / np.nansum(predicted + actual) * 2


def bias(actual, predicted):
    return np.nansum(predicted - actual)


def rmse(actual, predicted):
    return np.sqrt(np.mean((predicted - actual) ** 2))


def print_estimation(actual, predicted):
    print('wape:', wape(actual, predicted), '\n nfm:', nfm(actual, predicted), '\n bias:', bias(actual, predicted),
          '\n rmses:', rmse(actual, predicted))


def get_estimation(actual, predicted):
    return pd.DataFrame(
        {'wape': [wape(actual, predicted)], 'nfm': [nfm(actual, predicted)], 'bias': [bias(actual, predicted)],
         'rmses': [rmse(actual, predicted)]})

def get_arts_bid_gap(sales_df, max_gap_size=3, max_nan_count=15):
    '''
    Get list of articles id with big gaps to delete from calculation

    :param sales_df: Data Frame with articles(articleId), dates (months) and sales quantity (SalesQty)
    :param max_gap_size: how many consecutive months the maximum absences/gap can be
    :param max_nan_count: how many missed months there may be in total
    :return: numbers of articles with holes in the sequence of months more than  > max_gap_size
            or articles with total missed months > max_nan_count
    '''
    arts_bid_gap = []
    df_all = sales_df.set_index('articleId')
    df_all.index.freq='MS'

    for i in sales_df.articleId.unique():
        df = pd.DataFrame({'ds':df_all.loc[i].months.values, 'y':df_all.loc[i].SalesQty.values})
        df = df.set_index('ds').resample('MS').mean()
        mask = df['y'].isna()
        counts = mask.ne(mask.shift()).cumsum().where(mask).value_counts()

        if (counts.values > max_gap_size).sum() + (counts.sum() > max_nan_count).sum():
            arts_bid_gap.append(i)
        else:
            pass
    return arts_bid_gap

def finetunning_for_one_TS_estimate_all(df_train, cutoffs, time_t, forecast_horizon='183 days', parallel="processes"):
    """
    Get the best fine tunning params for a single article Prophet forecast (minimising RMSEs) and all interested ADD metrics
    - crossvalidation forecast ADD-like (6 month forehead) - '183 days'

    :param df_train: Data frame for one article forecast  'ds' - date(start of month), 'y' - sales quantity.
    :param cutoffs: list of dates for cutoffs
    :param time_t: Data Frame with ADD dates for cross-validation where 'forecastDate' - date for cutoff, 'ds'- forecast date
    :param forecast_horizon: only days, because of limitation TimeDelta method in Pandas (used in Prophet)
    :param parallel: how to optimize prophet
    :return: DataFrame with columns: articleid, wape, bias, rmse - metrics of the best result for this article,
    and also params - changepoint_prior_scale and seasonality_prior_scale for this best result
    """
    param_grid = {
        'yearly_seasonality': [True],
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    estimation_df = pd.DataFrame()  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(df_train)  # Fit model with given params
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon=forecast_horizon, parallel=parallel)
        df_cv['forecastDate'] = df_cv['cutoff'] + timedelta(days=1)
        df_cv = pd.merge(df_cv, time_t, on=['forecastDate', 'ds'], how='inner') # here saved forecasted values, column - yhat
        # df_p = performance_metrics(df_cv, rolling_window=1)                   # method from the Prophet library, but does not calculate WAPE
        df_p = get_estimation(df_cv.y.values, df_cv.yhat.values)                # so was wrote custom function  'get_estimation'
        estimation_df = pd.concat([estimation_df, df_p])                        # collect metrics for all applied param

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results = pd.concat([tuning_results, estimation_df.reset_index(drop=True)], axis=1)

    return tuning_results[tuning_results.rmses.eq(tuning_results.rmses.min())]   # return the best (min rmses)

def cv_for_one_TS(df_train, cutoffs, time_t, forecast_horizon='183 days', parallel="processes"):
    """
    Get a Prophet forecast for a single article over few cutoffs (segments) for later evaluation
    - crossvalidation forecast ADD like (6 month forehead) -'183 days'

    :param df_train: Data frame for one article forecast  'ds' - date(start of month), 'y' - sales quantity.
    :param cutoffs: list of dates for cutoffs
    :param time_t: Data Frame with ADD dates for cross-validation where 'forecastDate' - date for cutoff, 'ds'- forecast date
    :param forecast_horizon: only days, because of limitation TimeDelta method in Pandas (used in Prophet)
    :param parallel: how to optimize prophet
    :return:
    """
    # Use cross validation to evaluate all parameters
    m = Prophet(yearly_seasonality = True)
    m = m.fit(df_train)  # Fit model with given params
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon=forecast_horizon, parallel=parallel)
    df_cv['forecastDate'] = df_cv['cutoff'] + timedelta(days=1)
    df_cv = pd.merge(df_cv, time_t, on=['forecastDate', 'ds'], how='inner')

    return df_cv[['ds', 'forecastDate', 'y', 'yhat']].reset_index(drop=True)


def start_estimation(sales_df):

    # Prepare  time_t dataset with ADD dates, used for the evaluation forecast quality
    time_t = pd.read_csv('time_t.csv')
    time_t.rename(columns={'MonthStart': 'ds'}, inplace=True)
    time_t['ds'] = time_t.ds.astype('datetime64[ns]')
    time_t['forecastDate'] = time_t.forecastDate.astype('datetime64[ns]')
    time_t['forecastDate'][~time_t['forecastDate'].dt.is_month_start] = time_t['forecastDate'][~time_t[
        'forecastDate'].dt.is_month_start] - pd.offsets.MonthBegin(1)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12 + 12 * 2,
                          freq='MS')  # periods - отрезаем лишние месяцы - берем год на валидацию и два года на трейн


    # Prepare list of the dates for cutoffs (dates to start of forecasting - 6 times cutoff on 6 month ahead forecast each)
    cutoffs = dates[-12:-6]
    cutoffs = cutoffs - timedelta(days=1) # 1 day early then date of forecast

    sales_df.months = pd.to_datetime(sales_df.months )
    sales_df = sales_df[sales_df.months >= dates[0]]
    sales_df.sort_values('months', inplace=True)
    sales_df.months.freq = 'MS'

    # Prepare articles for forecasting (leave only articles without big gaps)
    articleId_from2020 = set(sales_df[sales_df.months.eq(dates[0])].articleId.values) # нет продаж на начало периода
    articleId_to2023 = set(sales_df[sales_df.months.eq(dates[-2])].articleId.values)  # нет продаж на конец периода
    articles2020_2023 = list(articleId_from2020 & articleId_to2023)                   # их обьединение - упрощает чистку от артикулов с большими пробелами продаж

    arts_bid_gap = get_arts_bid_gap(sales_df=sales_df[sales_df.months >= dates[0]], max_gap_size=3, max_nan_count=15)
    sales_df = sales_df[(sales_df.months >= dates[0]) & (~sales_df.articleId.isin(arts_bid_gap)) & (sales_df.articleId.isin(articles2020_2023))]
    articles = sales_df.articleId.unique()

    # предобработка
    # оставляем для прогноза только те артикула, которые еще не считали
    results = pd.read_csv('results_80arts.csv')
    results2 = pd.read_csv('results_27arts.csv')
    results3 = pd.read_csv('results_2mb.csv')
    #alredy_got_arts = set(results.articleId.values) and set(results2.articleId.values) and set(results3.articleId.values)
    #articles = np.array(list(set(articles) - alredy_got_arts))

    # Start of forecasting
    df_all = sales_df.set_index('articleId')
    cv_results = pd.DataFrame()
    for art in articles:
        print(art)
        df = pd.DataFrame({'ds': df_all.loc[art].months.values,
                           'y': df_all.loc[art].SalesQty.values})

        ## добавим два лишних месяца, чтобы встроенная функция кроссвалидации prophet не ругалась на выходящий за рамки датасета горизон (183 дня)
        # но учитывать для оценки будем только те месяцы, которы учитываются в ADD - time_t dataset
        df = pd.concat([df, pd.DataFrame(
            {'ds': [pd.Timestamp.today().normalize(), pd.Timestamp.today().normalize() + timedelta(days=1)],
             'y': [0, 0]})])

        #cv_result = cv_for_one_TS_estimate_all(df, cutoffs, time_t, forecast_horizon='183 days',
        #                                                     parallel="processes")
        cv_result = finetunning_for_one_TS_estimate_all(df, cutoffs, time_t, forecast_horizon='183 days',
                                               parallel="processes")
        cv_result['articleId'] = art
        cv_results = pd.concat([cv_results, cv_result])
        cv_results.to_csv('results2_ft_.csv')
    cv_results.to_csv('results2_tf.csv')

    # Get estimation of all articles forecasts
    #estimation_df = get_estimation(cv_result.y.values, cv_result.yhat.values)
    #estimation_df.to_csv('estimation_df_4.csv')
    #return estimation_df
    return cv_results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sales_df = pd.read_csv('sales_months_smooth.csv', index_col = 0)
    bests_results = start_estimation(sales_df)
    bests_results.to_csv('metrics_results.csv')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
