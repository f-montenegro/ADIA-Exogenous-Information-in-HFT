from L0_Library.config import *


class Clusters:

    def __init__(self):
        pass

    @staticmethod
    def jumps_timestamp(df_input: pd.DataFrame) -> list:
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
                shifted_shifted = series - series.shift(1) - 1
                shifted_shifted = shifted_shifted.replace(np.nan, np.inf)
                df_result[i].iloc[shifted_shifted.index] = shifted_shifted
            else:
                df_result[i] = df_input[i].replace(1, np.inf)

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    @staticmethod
    def min_bernoulli_trials_p(row, epsilon):
        if row['jumps'] > epsilon:
            return epsilon
        else:
            return row['jumps']

    def bernoulli_trials_p(self, df_input: pd.DataFrame, rolling_window: int, epsilon: float):
        df_input_unstack = df_input.unstack().reset_index().rename(columns={0: 'jumps'})
        df_input_unstack['jumps'] = np.nan

        s_bernoulli_trials_p = df_input.unstack().dropna().rolling(rolling_window).sum() / rolling_window
        s_bernoulli_trials_p = s_bernoulli_trials_p.reset_index().rename(columns={0: 'jumps'})

        df_result = pd.merge(df_input_unstack, s_bernoulli_trials_p,
                             on=['Date', 'Hour'], how='left')[['Date', 'Hour', 'jumps_y']]
        df_result = df_result.rename(columns={'jumps_y': 'jumps'})

        df_result['jumps'] = df_result.apply(lambda row: self.min_bernoulli_trials_p(row, epsilon), axis=1)

        df_result = df_result.pivot(index='Hour', columns='Date', values=f'jumps')

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def bernoulli_hypothesis_threshold(df_input: pd.DataFrame, epsilon: float):
        df_threshold = (np.log(1 - epsilon) / np.log(1 - df_input)) - 1

        pd.options.display.float_format = '{:,.5f}'.format

        return df_threshold

    def cluster_jumps(self, df_jumps: pd.DataFrame, df_inter_times: pd.DataFrame, df_threshold: pd.DataFrame):
        df_clustered_jumps = df_inter_times.where(df_threshold.notna(), np.nan)

        df_jumps = df_jumps.where(df_threshold.notna(), np.nan)
        l_jumps_timestamp = self.jumps_timestamp(df_input=df_jumps)

        for i in l_jumps_timestamp:
            s_inter_times = df_inter_times[i]
            s_threshold = df_threshold[i]
            s_clustered_jumps = df_clustered_jumps[i]
            for j in range(len(s_inter_times)):
                if s_inter_times[j] == np.inf:
                    s_clustered_jumps[j] = 1
                elif s_inter_times[j] < s_threshold[j]:
                    s_clustered_jumps[j] = 0
                elif s_inter_times[j] >= s_threshold[j]:
                    s_clustered_jumps[j] = 1
            df_clustered_jumps[i] = s_clustered_jumps

        pd.options.display.float_format = '{:,.0f}'.format

        return df_clustered_jumps
