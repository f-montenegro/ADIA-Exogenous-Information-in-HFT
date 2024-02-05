from L0_Library.config import *


def df_index_columns_correction(df_input: pd.DataFrame):

    df_input.columns.name = 'Date'
    df_input.index.name = 'Hour'

    try:
        df_input.index = [dt.strftime('%H:%M:%S') for dt in df_input.index.to_list()]
        df_input.columns = [dt.strftime('%Y-%m-%d') for dt in df_input.columns.to_list()]

    except:
        pass

    return df_input


def cluster_date_correction(start_date: datetime, df_input: pd.DataFrame):
    """
    Filter a DataFrame based on the type of input (columns or MultiIndex) and the given start_date.

    Parameters:
    - start_date: datetime object
    - df_input: pandas DataFrame

    Returns:
    - filtered DataFrame
    """

    try:
        # Try converting columns to datetime
        df_columns_dates = pd.to_datetime(df_input.columns)

        # Case: DataFrame with columns as dates
        df_columns_dates = df_columns_dates[df_columns_dates >= start_date]
        df_columns_dates = df_columns_dates.strftime('%Y-%m-%d').tolist()

        df_result = df_input[df_columns_dates]

        return df_result

    except:

        try:
            # Case: DataFrame with MultiIndex (Date, Hour)
            df_input.index.names = ['Date', 'Hour']
            df_input['Date'] = pd.to_datetime(df_input.index.get_level_values('Date'))

            df_result = df_input[df_input['Date'] >= start_date]
            df_result = df_result.drop('Date', axis=1)
            df_result = df_result.sort_values(by=['Date', 'Hour'])

            return df_result

        except:

            try:
                # Case: DataFrame with SingleIndex (Date)
                df_input.index = pd.MultiIndex.from_arrays([df_input.index.date, df_input.index.time],
                                                           names=['Date', 'Hour'])

                df_input['Date'] = pd.to_datetime(df_input.index.get_level_values('Date'))

                df_result = df_input[df_input['Date'] >= start_date]
                df_result = df_result.drop('Date', axis=1)
                df_result = df_result.sort_values(by=['Date', 'Hour'])

                return df_result

            except:

                try:
                    # Case: DataFrame has just a Date column (no hours)
                    df_result = df_input[pd.to_datetime(df_input['Date']) >= start_date]
                    df_result = df_result.sort_values(by='Date')
                    return df_result

                except:
                    # Case: DataFrame is probably date independent
                    return df_input
