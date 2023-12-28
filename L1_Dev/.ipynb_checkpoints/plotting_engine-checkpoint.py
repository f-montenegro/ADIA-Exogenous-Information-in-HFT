from L0_Library.config import *


class PlottingEngine:
    def __init__(self):
        pass

    @staticmethod
    def plot_dataframe_columns(df: pd.DataFrame, columns: list, title: str="DataFrame Columns Plot", kind: str='line', show_legend: bool=True):
        """
        Plots specified columns from a Pandas DataFrame.
        
        :param df: The Pandas DataFrame containing the data.
        :param columns: A list of column names to plot.
        :param title: The title of the plot (default is "DataFrame Columns Plot").
        :param kind: The kind of plot (default is 'line').
        :param show_legend: Whether to show a legend (default is True).
        """

        if not set(columns).issubset(df.columns):
            raise ValueError("All columns to plot must exist in the DataFrame")
        
        plt.figure(figsize=(10,6))
        df[columns].plot(kind=kind, ax=plt.gca(), marker="o", alpha=0.5)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.grid(False)
        if show_legend:
            plt.legend(columns)
        plt.show()


    @staticmethod
    def plot_histogram(df_input, column_name, bins=10, **kwargs):
        """
        Plot a histogram for a given column from a DataFrame.
        
        Parameters:
        - df_input: The input pandas DataFrame.
        - column_name: The name of the column to plot.
        - bins: Number of histogram bins. Default is 10.
        - **kwargs: Additional keyword arguments to be passed to the plt.hist function.
        """
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in DataFrame.")
            return
        
        plt.hist(df[column_name].dropna(), bins=bins, **kwargs)
        plt.title(f"Histogram of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.show()
