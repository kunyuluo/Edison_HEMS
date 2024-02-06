import math
import numpy as np
import pandas as pd
import pytz
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from Time_Features import TimeCovariates


class DefaultValueFiller:
    """
    Default_value_mode:
    0: Calculate default value based on the entire dataset.
    1: Calculate default value based on monthly data.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            feature_names=None,
            default_value_mode: int = 0,
            date_column: str = 'timestamp',
            freq: str = '5min'):

        feature_names = [] if feature_names is None else feature_names

        self.df = df
        self.feature_names = feature_names
        self.default_value_mode = default_value_mode
        self.datetime_column = date_column
        self.freq = freq
        self.feature_data = self.get_feature_data()
        # self.new_dataset = self.fill_missing_value()

    def format_date(self):
        self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
        df_local = self.df[self.datetime_column].dt.tz_localize(None).dt.floor('min')

        return df_local

    def transfer_time_zone(self):
        self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
        df_utc = self.df[self.datetime_column].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')

        return df_local

    def get_feature_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        # date_local = self.transfer_time_zone()
        date_local = self.format_date()

        # Get data from specific column
        # *******************************************************************************
        feature_data = pd.concat([date_local, self.df[self.feature_names]], axis=1)
        feature_data['weekday'] = feature_data[self.datetime_column].dt.weekday

        return feature_data

    @staticmethod
    def get_date_range(data: pd.DataFrame, date_column: str = 'data_time'):
        dt_min = data[date_column].min()
        dt_max = data[date_column].max()
        year_range = range(dt_min.year, dt_max.year + 1)

        months_range = {}
        days_range = {}

        for year in year_range:
            month_min = data[data[date_column].dt.year == year][date_column].dt.month.min()
            month_max = data[data[date_column].dt.year == year][date_column].dt.month.max()
            day_min = data[(data[date_column].dt.year == year) &
                           (data[date_column].dt.month == month_min)][date_column].dt.day.min()
            day_max = data[(data[date_column].dt.year == year) &
                           (data[date_column].dt.month == month_max)][date_column].dt.day.max()
            months_range[year] = range(month_min, month_max + 1)
            days_range[year] = (day_min, day_max)

        return year_range, months_range, days_range

    def fill_missing_value(self):

        data = self.get_feature_data()
        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        hr_interval = None
        min_interval = None
        if 'min' in self.freq:
            min_interval = int(self.freq.split('min')[0])
        elif 'h' in self.freq:
            hr_interval = int(self.freq.split('h')[0])
        else:
            pass

        datetimes = []
        year_range, months_range, days_range = DefaultValueFiller.get_date_range(data, date_column=self.datetime_column)

        # start_day, end_day = dt_min.day, dt_max.day
        num_days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for year in year_range:
            month_range = months_range[year]
            if len(months_range[year]) == 1:
                for day in range(1, days_range[year][1] + 1):
                    if min_interval is not None:
                        for hour in range(24):
                            for minute in range(0, 60, min_interval):
                                datetimes.append(
                                    dt.datetime(year=year, month=month_range[0], day=day, hour=hour, minute=minute))
                    else:
                        if hr_interval is not None:
                            for hour in range(0, 24, hr_interval):
                                datetimes.append(
                                    dt.datetime(year=year, month=month_range[0], day=day, hour=hour, minute=0))
            else:
                for i, month in enumerate(month_range):
                    if i == 0:
                        for day in range(days_range[year][0], num_days_month[month - 1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))
                    elif i == len(month_range) - 1:
                        for day in range(1, days_range[year][1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))
                    else:
                        for day in range(1, num_days_month[month - 1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))

        new_df = pd.DataFrame(pd.to_datetime(datetimes), columns=[self.datetime_column])
        new_df['weekday'] = new_df[self.datetime_column].dt.weekday

        for feature in self.feature_names:
            if self.default_value_mode == 0:
                default = self.calc_default_value(feature)
                filled_data = []
                for date in datetimes:
                    value = self.feature_data[(self.feature_data[self.datetime_column] == date)][feature].values

                    if len(value) == 0:
                        weekday = date.weekday()
                        if weekday in [0, 1, 2, 3, 4]:
                            value = default[0][date.hour][date.minute]
                        elif weekday in [5, 6]:
                            value = default[1][date.hour][date.minute]

                        filled_data.append(value)
                    else:
                        filled_data.append(value[0])
            else:
                default = self.calc_default_value_monthly(feature)
                filled_data = []
                for date in datetimes:
                    value = self.feature_data[(self.feature_data[self.datetime_column] == date)][feature].values

                    if len(value) == 0:
                        current_year = date.year
                        current_month = date.month
                        weekday = date.weekday()
                        if weekday in [0, 1, 2, 3, 4]:
                            value = default[current_year][current_month][0][date.hour][date.minute]
                        elif weekday in [5, 6]:
                            value = default[current_year][current_month][1][date.hour][date.minute]
                        filled_data.append(value)
                    else:
                        filled_data.append(value[0])

            # Fill the strange zero value:
            DefaultValueFiller.fill_zero_value(filled_data)
            # DefaultValueFiller.fill_strange_value(filled_data)
            DefaultValueFiller.fill_nan_value(filled_data)
            DefaultValueFiller.spike_smoother(filled_data)

            new_df[feature] = filled_data
            # new_df.drop(['weekday'], axis=1, inplace=True)

        return new_df

    @staticmethod
    def fill_nan_value(data, inplace=True):
        """
        Replace nan value in the input list with linear interpolation.
        """

        def interpolation(low_bound, high_bound, num_of_values):
            if low_bound == 0 and high_bound == 0:
                return [0] * num_of_values
            else:
                step = (high_bound - low_bound) / (num_of_values + 1)
                values = []
                for i in range(num_of_values):
                    value = low_bound + (i + 1) * step
                    values.append(value)
                return values

        # Find position (index) of all nan value in the list
        nan_position = [i for i, x in enumerate(data) if math.isnan(x)]

        # Find start and end positions of nan values sequence
        split_start_idx = []
        split_end_idx = []
        split_start_idx.append(nan_position[0])

        for i in range(1, len(nan_position)):
            if nan_position[i] - nan_position[i - 1] == 1:
                pass
            else:
                split_start_idx.append(nan_position[i])
                split_end_idx.append(nan_position[i - 1])

        split_end_idx.append(nan_position[-1])

        # Apply linear interpolation for missed values
        for i in range(len(split_start_idx)):
            missing_len = split_end_idx[i] - split_start_idx[i] + 1

            low = data[split_start_idx[i] - 1] if split_start_idx[i] != 0 else 0
            high = data[split_end_idx[i] + 1] if split_end_idx[i] != len(data) - 1 else data[split_end_idx[i] - 1]

            data[split_start_idx[i]:split_end_idx[i] + 1] = (interpolation(low, high, missing_len))

        if not inplace:
            return data

    @staticmethod
    def fill_zero_value(data):
        """
        Replace zero value in the input list with its previous value.
        """
        for i in range(len(data)):
            if data[i] == 0:
                if i == 0:
                    data[i] = data[i + 1]
                else:
                    data[i] = data[i - 1]

    @staticmethod
    def fill_strange_value(data):
        """
        Replace strange high or low value in the input list with modified ratio of its previous value.
        """
        delta_threshold = 0.8
        modified_ratio = 0.5
        for i in range(len(data)):
            if data[i] != 0 and data[i - 1] != 0:
                if data[i] > data[i - 1] and 1 - (data[i] / data[i - 1]) > delta_threshold:
                    data[i] = data[i - 1] * modified_ratio
                elif data[i] < data[i - 1] and 1 - (data[i - 1] / data[i]) > delta_threshold:
                    data[i] = data[i - 1] * modified_ratio
                else:
                    pass
            else:
                pass

    @staticmethod
    def spike_smoother(data):

        window_size = 3
        length = len(data) - window_size + 1
        bound_var_ratio_threshold = 0.4
        var_ratio_threshold = 0.5

        for i in range(length):
            bound_delta = abs(data[i] - data[i + 2]) / data[i]
            median = (data[i] + data[i + 2]) / 2
            delta = abs(median - data[i + 1]) / median

            if bound_delta <= bound_var_ratio_threshold:
                if delta > var_ratio_threshold:
                    data[i + 1] = median
            else:
                if data[i+1] > data[i] and data[i+1] > data[i+2]:
                    data[i + 1] = median

    def calc_default_value(self, column_name):
        """
            Calculate average value of every minute in a day by weekday (from Monday to Sunday).
            Use the calculated value to fill empty/missing value in the dataset.
        """
        hours = range(24)
        minutes = range(60)
        default_values = {}

        weekdays = {}
        weekends = {}

        for hour in hours:
            hours_wday = []
            hours_wend = []
            for minute in minutes:
                value_wday = self.feature_data[
                    ((self.feature_data['weekday'] == 0) |
                     (self.feature_data['weekday'] == 1) |
                     (self.feature_data['weekday'] == 2) |
                     (self.feature_data['weekday'] == 3) |
                     (self.feature_data['weekday'] == 4)) &
                    (self.feature_data[self.datetime_column].dt.hour == hour) &
                    (self.feature_data[self.datetime_column].dt.minute == minute)][column_name].mean()

                value_wend = self.feature_data[
                    ((self.feature_data['weekday'] == 5) |
                     (self.feature_data['weekday'] == 6)) &
                    (self.feature_data[self.datetime_column].dt.hour == hour) &
                    (self.feature_data[self.datetime_column].dt.minute == minute)][
                    column_name].mean()

                hours_wday.append(value_wday)
                hours_wend.append(value_wend)

            weekdays[hour] = hours_wday
            weekends[hour] = hours_wend

        default_values[0] = weekdays
        default_values[1] = weekends

        return default_values

    def calc_default_value_monthly(self, column_name):
        """
            Calculate average value of every minute in a day by monthly average.
            Use the calculated value to fill empty/missing value in the dataset.
        """
        days_threshold = 3

        hours = range(24)
        minutes = range(60)
        default_values = {}

        data = self.get_feature_data()
        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        year_range, months_range, days_range = DefaultValueFiller.get_date_range(data, date_column=self.datetime_column)

        for year in year_range:
            year_values = {}
            month_range = months_range[year]
            day_range = days_range[year]

            for month in month_range:
                # Calculate the number of days of the last month of the current year:
                days_of_last_month = day_range[1]

                weekdays = {}
                weekends = {}
                month_values = {}
                for hour in hours:
                    hours_wday = []
                    hours_wend = []
                    for minute in minutes:

                        # If number of days in this month is not enough (less than threshold) to
                        # calculate monthly average, then use the previous month to do calculation.
                        current_year = year
                        current_month = month
                        if days_of_last_month < days_threshold:
                            if month == 1:
                                current_year = year - 1
                                current_month = 12
                            else:
                                current_month = month - 1
                        else:
                            pass

                        value_wday = data[(data[self.datetime_column].dt.year == current_year) &
                                          (data[self.datetime_column].dt.month == current_month) &
                                          ((data[self.datetime_column].dt.weekday == 0) |
                                           (data[self.datetime_column].dt.weekday == 1) |
                                           (data[self.datetime_column].dt.weekday == 2) |
                                           (data[self.datetime_column].dt.weekday == 3) |
                                           (data[self.datetime_column].dt.weekday == 4)) &
                                          (data[self.datetime_column].dt.hour == hour) &
                                          (data[self.datetime_column].dt.minute == minute)][column_name].mean()

                        value_wend = data[(data[self.datetime_column].dt.year == current_year) &
                                          (data[self.datetime_column].dt.month == current_month) &
                                          ((data[self.datetime_column].dt.weekday == 5) |
                                           (data[self.datetime_column].dt.weekday == 6)) &
                                          (data[self.datetime_column].dt.hour == hour) &
                                          (data[self.datetime_column].dt.minute == minute)][column_name].mean()

                        hours_wday.append(value_wday)
                        hours_wend.append(value_wend)

                    weekdays[hour] = hours_wday
                    weekends[hour] = hours_wend

                month_values[0] = weekdays
                month_values[1] = weekends

                year_values[month] = month_values
            default_values[year] = year_values

        return default_values


class DataProcessor:
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame = None,
                 feature_names=None,
                 start_date: tuple = None,
                 end_date: tuple = None,
                 hour_range: tuple = None,
                 group_freq: int = None,
                 test_size: float = 0.2,
                 val_size: float = 0.05,
                 n_input: int = 5,
                 n_output: int = 5,
                 time_zone_transfer: bool = False,
                 date_column: str = 'data_time',
                 add_time_features: bool = False,
                 scaler=None):

        feature_names = [] if feature_names is None else feature_names

        data = df if df is not None else pd.read_csv(file_path, low_memory=False)

        self.df = data
        self.feature_names = feature_names
        self.start_date = start_date
        self.end_date = end_date
        self.hour_range = hour_range
        self.group_freq = group_freq
        self.test_size = test_size
        self.n_input = n_input
        self.n_output = n_output
        self.time_zone_transfer = time_zone_transfer
        self.date_column = date_column
        self.add_time_features = add_time_features
        self.scaler = scaler

        period_data = self.get_period_data()
        self.data = period_data[0]

        self.train_idx = round(len(self.data) * (1 - test_size))
        if val_size == 0:
            self.val_idx = len(self.data) - 1
        else:
            if val_size < 1:
                self.val_idx = round(len(self.data) * val_size)
            else:
                self.val_idx = len(self.data) - val_size

        self.time_feature_names = period_data[1]

        num_features = len(feature_names) if len(feature_names) > 0 else data.shape[1] - 1
        self.num_features = num_features + len(self.time_feature_names)

        self.train, self.test, self.val = self.get_train_test_val()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def format_date(self):
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        df_local = self.df[self.date_column].dt.tz_localize(None).dt.floor('min')

        return df_local

    def transfer_time_zone(self):
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        df_utc = self.df[self.date_column].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')
        # self.df['data_time'] = df_local
        # df_utc = df_utc.dt.tz_localize(None)

        return df_local

    def select_by_date(
            self,
            df: pd.DataFrame,
            start_year: int = dt.date.today().year,
            end_year: int = dt.date.today().year,
            start_month: int = None,
            start_day: int = None,
            end_month: int = None,
            end_day: int = None):

        # year = dt.date.today().year
        # date_start = dt.date(year, start_month, start_day)
        # date_end = dt.date(year, end_month, end_day)
        # days = (date_end - date_start).days + 1
        if (start_month is not None and
                end_month is not None and
                start_day is not None and
                end_day is not None):
            # df_selected = df[
            #     (df[self.date_column].dt.year >= start_year) &
            #     (df[self.date_column].dt.year <= end_year) &
            #     (df[self.date_column].dt.month >= start_month) &
            #     (df[self.date_column].dt.month <= end_month) &
            #     (df[self.date_column].dt.day >= start_day) &
            #     (df[self.date_column].dt.day <= end_day)]

            start = pd.to_datetime(dt.datetime(start_year, start_month, start_day))
            end = pd.to_datetime(dt.datetime(end_year, end_month, end_day))

            df_selected = df[(df[self.date_column] >= start) & (df[self.date_column] < end)]

            # df_selected.set_index('data_time', inplace=True)

            return df_selected

    def select_by_time(self, df: pd.DataFrame, start_hour: int = 8, end_hour: int = 22):
        df_selected = df[
            (df[self.date_column].dt.hour >= start_hour) &
            (df[self.date_column].dt.hour < end_hour)]

        return df_selected

    def get_period_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        if self.time_zone_transfer:
            date_local = self.transfer_time_zone()
        else:
            date_local = self.format_date()

        # Get data from specific column
        # *******************************************************************************
        if len(self.feature_names) != 0:
            target_data = pd.concat([date_local, self.df[self.feature_names]], axis=1)
        else:
            feature_df = self.df.drop([self.date_column], axis=1)
            target_data = pd.concat([date_local, feature_df], axis=1)

        # Get data for specified period
        # *******************************************************************************
        if self.start_date is not None and len(self.start_date) == 3:
            target_period = self.select_by_date(
                target_data, start_year=self.start_date[0], end_year=self.end_date[0],
                start_month=self.start_date[1], start_day=self.start_date[2],
                end_month=self.end_date[1], end_day=self.end_date[2])
        else:
            target_period = target_data

        if self.hour_range is not None:
            target_period = self.select_by_time(target_period, self.hour_range[0], self.hour_range[1])
        else:
            target_period = target_period

        target_period.set_index(self.date_column, inplace=True)

        if self.group_freq is not None:
            target_period = target_period.groupby(pd.Grouper(freq=f'{self.group_freq}min')).mean()

        target_period = target_period.dropna()
        # target_period = target_period.reset_index()
        # print(target_period)

        if self.scaler is not None:
            index = target_period.index
            column_names = target_period.columns
            target_period = self.scaler.fit_transform(target_period)
            target_period = pd.DataFrame(target_period, index=index, columns=column_names)

        time_feas = []

        # Add time features as dynamic covariates if needed:
        if self.add_time_features:
            dti = target_period.index
            time_covs = TimeCovariates(dti)
            time_features = time_covs.get_covariates()
            target_period = pd.concat([target_period, time_features], axis=1)

            time_feas = time_covs.get_feature_names()

        return target_period, time_feas

    def get_train_test_val(self) -> tuple[np.array, np.array, np.array]:
        """
        Runs complete ETL
        """
        train, test, val = self.split_data()
        return self.transform(train, test, val)

    def split_data(self):
        """
        Split data into train and test sets.
        """

        if len(self.data) != 0:
            # train_idx = round(len(self.data) * (1 - self.test_size))
            train = self.data[:self.train_idx]
            if self.val_idx == len(self.data) - 1:
                test = self.data[self.train_idx:]
                val = None
            else:
                test = self.data[self.train_idx: self.val_idx]
                val = self.data[self.val_idx:]
            # test = self.data[self.train_idx:]
            # train = np.array(np.split(train, train.shape[0] / self.timestep))
            # test = np.array(np.split(test, test.shape[0] / self.timestep))
            return train.values, test.values, val.values
        else:
            raise Exception('Data set is empty, cannot split.')

    def transform(self, train: np.array, test: np.array, val: np.array):

        train_remainder = train.shape[0] % self.n_input
        test_remainder = test.shape[0] % self.n_input
        val_remainder = val.shape[0] % self.n_input
        # print(train_remainder, test_remainder)

        if train_remainder != 0 and test_remainder != 0 and val_remainder != 0:
            # train = train[0: train.shape[0] - train_remainder]
            # test = test[0: test.shape[0] - test_remainder]
            train = train[train_remainder:]
            test = test[test_remainder:]
            val = val[:-val_remainder]
        elif train_remainder != 0:
            train = train[train_remainder:]
        elif test_remainder != 0:
            test = test[test_remainder:]
        elif val_remainder != 0:
            val = val[val_remainder:]

        return self.window_and_reshape(train), self.window_and_reshape(test), self.window_and_reshape(val)
        # return train, test

    def window_and_reshape(self, data) -> np.array:
        """
        Reformats data into shape our model needs,
        namely, [# samples, timestep, # feautures]
        samples
        """
        samples = int(data.shape[0] / self.n_input)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.n_input, self.num_features))

    def to_supervised(self, data) -> tuple:
        """
        Converts our time series prediction problem to a
        supervised learning problem.
        Input has to be reshaped to 3D [samples, timesteps, features]
        """
        # flatted the data
        data_flattened = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        X, y = [], []
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + self.n_input
            out_end = in_end + self.n_output
            # ensure we have enough data for this instance
            if out_end <= len(data_flattened):
                x_input = data_flattened[in_start:in_end, :]
                # x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], 1))
                X.append(x_input)
                y.append(data_flattened[in_end:out_end, 0])
                # move along one time step
                in_start += 1
        return np.array(X), np.array(y)


class PredictAndForecast:
    """
    model: tf.keras.Model
    train: np.array
    test: np.array
    Takes a trained model, train, and test datasets and returns predictions
    of len(test) with same shape.
    """

    def __init__(self, model, train, test, n_input=5, n_output=5) -> None:
        train = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        test = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))

        self.model = model
        self.train = train
        self.test = test
        self.n_input = n_input
        self.n_output = n_output
        # self.updated_test = self.updated_test()
        # self.predictions = self.get_predictions()

    def forcast(self, x_input) -> np.array:
        """
        Given last weeks actual data, forecasts next weeks' prices.
        """
        # Flatten data
        # data = np.array(history)
        # data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

        # retrieve last observations for input data
        # x_input = data[-self.n_input:, :]
        # x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

        # forecast the next week
        yhat = self.model.predict(x_input, verbose=0)

        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def get_predictions(self) -> np.array:
        """
        compiles models predictions week by week over entire test set.
        """
        # history is a list of flattened test data + last observation from train data
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        step = round(len(history) / self.n_output)
        # history = []

        # walk-forward validation
        predictions = []
        window_start = 0
        for i in range(step):

            if window_start <= len(history) - self.n_input - self.n_output:
                # print('pred no {}, window_start {}'.format(i+1, window_start))
                x_input = np.array(history[window_start:window_start + self.n_input])
                x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
                yhat_sequence = self.forcast(x_input)
                # print('pred no {}'.format(i))
                # store the predictions
                predictions.append(yhat_sequence)

            window_start += self.n_output
            # get real observation and add to history for predicting the next week
            # history.append(self.test[i, :])

        return np.array(predictions)

    def updated_test(self):
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
            return test
        else:
            return self.test

    def get_sample_prediction(self, index=0):

        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        if index < len(test) - self.n_output:
            index = index
        else:
            index = len(test) - self.n_output

        x_input = np.array(history[index:index + self.n_input])
        x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
        # print(x_input)
        yhat_sequence = self.forcast(x_input)
        actual = test[index:index + self.n_output, 0]

        return np.array(yhat_sequence), actual

    def walk_forward_validation(self, pred_length, start_point=0):
        """
        walk-forward validation for univariate data.
        """
        test_remainder = self.test.shape[0] % self.n_output
        if test_remainder != 0:
            test = self.test[:-test_remainder]
        else:
            test = self.test

        history = [x for x in self.train[-self.n_input:, :]]
        history.extend(test)

        if start_point < len(test) - pred_length:
            start_point = start_point
        else:
            start_point = len(test) - pred_length

        inputs = np.array(history[start_point:start_point + self.n_input])
        predictions = []
        actuals = []

        max_length = len(test) - (start_point + 1) - self.n_output

        if pred_length > max_length:
            pred_length = max_length
        else:
            pred_length = pred_length

        step = round(pred_length / self.n_output)

        for i in range(step):
            # Prepare the input sequence
            x_input = inputs[-self.n_input:]
            # print(x_input)
            x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

            # Make prediction
            yhat_sequence = self.forcast(x_input)
            for value in yhat_sequence:
                # print(yhat_sequence)
                predictions.append(value)

            # Get actual value for the current timestep
            actual = test[start_point:start_point + self.n_output, 0]
            for value in actual:
                # print(actual)
                actuals.append(value)

            # Update the input sequence
            x_input_new = test[start_point:start_point + self.n_output, :]
            # print(x_input_new)

            for j in range(len(yhat_sequence)):
                # np.put(x_input_new[j], 0, yhat_sequence[j])
                x_input_new[j, 0] = yhat_sequence[j]

            inputs = np.append(inputs, x_input_new, axis=0)

            start_point += self.n_output

        return np.array(predictions).reshape(-1, 1), np.array(actuals).reshape(-1, 1)


class Evaluate:
    def __init__(self, actual, predictions) -> None:
        if actual.shape[1] > 1:
            actual_values = actual[:, 0]
        else:
            actual_values = actual

        self.actual = actual_values
        self.predictions = predictions
        self.var_ratio = self.compare_var()
        self.mape = self.evaluate_model_with_mape()

    def compare_var(self) -> float:
        """
        Calculates the variance ratio of the predictions
        """
        # print(np.var(self.predictions))
        # print(np.var(self.actual))
        return abs(1 - (np.var(self.predictions)) / np.var(self.actual))

    def evaluate_model_with_mape(self) -> float:
        """
        Calculates the mean absolute percentage error
        """
        return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())


def inverse_transform_prediction(values, feature_num: int, scaler):
    values = values.flatten()

    dim = values.shape[0]
    dummy_array = np.zeros([dim, feature_num])
    for i in range(dim):
        np.put(dummy_array[i], 0, values[i])

    unscaled = scaler.inverse_transform(dummy_array)
    unscaled = unscaled[:, 0]
    unscaled = unscaled.reshape((unscaled.shape[0], 1))

    return unscaled


def scale_data(value, value_range=(0, 1), scaled_range=(0, 1)):
    ratio = (value - value_range[0]) / (value_range[1] - value_range[0])
    return ratio * (scaled_range[1] - scaled_range[0]) + scaled_range[0]
