from L0_Library.config import *


class Clusters:

    def __init__(self):
        pass

    @staticmethod
    def get_timestamps(df_input: pd.DataFrame) -> list:
        list_timestamps = []
        for i in df_input.columns:
            if (df_input[i] == 1).any():
                list_timestamps.append(i)
        return list_timestamps

    @staticmethod
    def event_inter_times(df_input: pd.DataFrame, list_input: list):
        df_inter_times = df_input.copy()
        for i in list_input:
            if df_input[i].sum() > 1:
                mask = np.array(df_input[i] == 1)
                series = pd.Series(range(len(df_input[i])))[mask]
                shifted_shifted = series - series.shift(1) - 1
                shifted_shifted = shifted_shifted.replace(np.nan, np.inf)
                df_inter_times[i].iloc[shifted_shifted.index] = shifted_shifted
            else:
                df_inter_times[i] = df_input[i].replace(1, np.inf)

        pd.options.display.float_format = '{:,.3f}'.format

        return df_inter_times

    @staticmethod
    def min_bernoulli_p(row, epsilon):
        if row['jumps'] > epsilon:
            return epsilon
        else:
            return row['jumps']

    def bernoulli_p(self, df_input: pd.DataFrame, rolling_window: int, epsilon: float):
        df_input_unstack = df_input.unstack().reset_index().rename(columns={0: 'jumps'})
        df_input_unstack['jumps'] = np.nan

        s_bernoulli_trials_p = df_input.unstack().dropna().rolling(rolling_window).sum() / rolling_window
        s_bernoulli_trials_p = s_bernoulli_trials_p.reset_index().rename(columns={0: 'jumps'})

        df_bernoulli_p = pd.merge(df_input_unstack, s_bernoulli_trials_p,
                                  on=['Date', 'Hour'], how='left')[['Date', 'Hour', 'jumps_y']]
        df_bernoulli_p = df_bernoulli_p.rename(columns={'jumps_y': 'jumps'})

        df_bernoulli_p['jumps'] = df_bernoulli_p.apply(lambda row: self.min_bernoulli_p(row, epsilon), axis=1)

        df_bernoulli_p = df_bernoulli_p.pivot(index='Hour', columns='Date', values=f'jumps')

        pd.options.display.float_format = '{:,.5f}'.format

        return df_bernoulli_p

    @staticmethod
    def bernoulli_threshold(df_bernoulli_p: pd.DataFrame, epsilon: float):
        df_threshold = (np.log(1 - epsilon) / np.log(1 - df_bernoulli_p)) - 1
        # Dividing by zero results in -np.inf which needs to be changed to np.inf
        df_threshold = df_threshold.replace(-np.inf, np.inf)

        pd.options.display.float_format = '{:,.5f}'.format

        return df_threshold

    def cluster_events(self, df_events: pd.DataFrame, df_inter_times: pd.DataFrame, df_threshold: pd.DataFrame):
        df_clustered_events = pd.DataFrame(data=0, index=df_inter_times.index, columns=df_inter_times.columns)
        # if the value in df_threshold is np.nan then change it to np.nan in df_jumps
        df_events = df_events.where(df_threshold.notna(), np.nan)
        list_events_timestamp = self.get_timestamps(df_input=df_events)

        for i in list_events_timestamp:
            s_inter_times = df_inter_times[i]
            s_threshold = df_threshold[i]
            s_clustered_events = df_clustered_events[i]
            for j in range(len(s_inter_times)):
                if s_inter_times[j] == np.inf:
                    s_clustered_events[j] = 1
                elif s_inter_times[j] < s_threshold[j]:
                    s_clustered_events[j] = 0
                elif s_inter_times[j] >= s_threshold[j]:
                    s_clustered_events[j] = 1
            df_clustered_events[i] = s_clustered_events

        df_clustered_events = df_clustered_events.where(df_threshold.notna(), np.nan)

        pd.options.display.float_format = '{:,.0f}'.format

        return df_clustered_events

    @staticmethod
    def clustered_data_list(df_input: pd.DataFrame, cluster: str):
        df_result = df_input
        df_result = pd.DataFrame(df_result.unstack()[df_result.unstack() == 1]).rename(columns={0: f'{cluster}'})

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    ##############################################################################
    ############################ DataFrame Compilation ###########################
    ##############################################################################

    def compilation(self, df_input: pd.DataFrame, cluster: str):

        l_data_timestamp = self.get_timestamps(df_input=df_input)

        df_inter_times = self.event_inter_times(df_input, l_data_timestamp)
        df_bernoulli_prob = self.bernoulli_p(df_input, rolling_window=7780, epsilon=0.05)
        df_bernoulli_threshold = self.bernoulli_threshold(df_bernoulli_prob, epsilon=0.05)
        df_clustered_data = self.cluster_events(df_input, df_inter_times, df_bernoulli_threshold)
        df_clustered_data_list = self.clustered_data_list(df_clustered_data, cluster)

        df_dict = {
            f"inter_times_{cluster}": df_inter_times,             # Inter-Times
            "bernoulli_prob": df_bernoulli_prob,                  # Bernoulli Hypothesis-Test Implied Probability
            "bernoulli_threshold": df_bernoulli_threshold,        # Bernoulli Hypothesis-Test Implied Threshold
            f"clustered_{cluster}": df_clustered_data,            # Clustered Data
            f"clustered_{cluster}_list": df_clustered_data_list,  # Clustered Data List
        }

        return df_dict
