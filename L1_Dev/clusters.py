import pandas as pd

from L0_Library.config import *


class Clusters:

    def __init__(self):
        pass

    @staticmethod
    def get_dates_with_jumps(df_input: pd.DataFrame):
        list_result = []
        for i in df_input.columns:
            if (df_input[i] == 1).any():
                list_result.append(i.strftime('%m/%d/%Y'))
        return list_result

    @staticmethod
    def calculate_jump_inter_times(df_input: pd.DataFrame, list_input: list):
        df_result = df_input.copy()
        for i in list_input:
            if df_input[i].sum() > 1:
                mask = np.array(df_input[i] == 1)
                series = pd.Series(range(len(df_input[i])))[mask]
                shifted_shifted = series.shift(-1) - series + 1
                shifted_shifted = shifted_shifted.replace(np.nan, 0)
                shifted_shifted.iloc[0] = np.inf
                df_result[i].iloc[shifted_shifted.index] = shifted_shifted
            else:
                mask = np.array(df_input[i] == 1)
                df_result[i][mask] = np.inf

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    @staticmethod
    def min_bernoulli_trials_p(row):
        if row['jumps'] > 0.05:
            return 0.05
        else:
            return row['jumps']

    def bernoulli_trials_p(self, df_input: pd.DataFrame, rolling_window: int):
        df_input_unstack = df_input.unstack().reset_index().rename(columns={0: 'jumps'})
        df_input_unstack['jumps'] = np.nan

        s_bernoulli_trials_p = df_input.unstack().dropna().rolling(rolling_window).sum() / rolling_window
        s_bernoulli_trials_p = s_bernoulli_trials_p.reset_index().rename(columns={0: 'jumps'})

        de_result = pd.merge(df_input_unstack, s_bernoulli_trials_p,
                             on=['Date', 'Hour'], how='left')[['Date', 'Hour', 'jumps_y']]
        de_result = de_result.rename(columns={'jumps_y': 'jumps'})

        de_result['jumps'] = de_result.apply(lambda row: self.min_bernoulli_trials_p(row), axis=1)

        de_result = de_result.pivot(index='Hour', columns='Date', values=f'jumps')

        pd.options.display.float_format = '{:,.5f}'.format

        return de_result

    @staticmethod
    def bernoulli_hypothesis_threshold(df_input: pd.DataFrame):
        df_threshold = (np.log(1 - 0.05) / np.log(1 - df_input)) - 1

        pd.options.display.float_format = '{:,.5f}'.format

        return df_threshold

    @staticmethod
    def cluster_jumps(df_jumps: pd.DataFrame, df_inter_times: pd.DataFrame, df_threshold: pd.DataFrame,
                      list_dates_w_jumps: list):
        df_clustered_jumps = df_jumps.where(df_threshold.notna(), np.nan)

        for i in list_dates_w_jumps:
            s_inter_times = df_inter_times[i]
            s_threshold = df_threshold[i]
            s_clustered_jumps = df_clustered_jumps[i]
            for j in range(len(s_inter_times)):
                if (s_inter_times[j] != 0) and pd.notna(s_clustered_jumps[j]):
                    if s_inter_times[j] < s_threshold[j]:
                        s_clustered_jumps[j] = 0
                    if s_inter_times[j] == np.inf:
                        s_clustered_jumps[j] = 1
            df_clustered_jumps[i] = s_clustered_jumps

        pd.options.display.float_format = '{:,.0f}'.format

        return df_clustered_jumps
