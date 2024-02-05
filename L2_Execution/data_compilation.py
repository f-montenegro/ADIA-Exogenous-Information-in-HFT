from L0_Library.config import *

from L1_Dev.data_prep import DataPrep
from L1_Dev.news import News
from L1_Dev.clusters import Clusters
from L1_Dev.classifier import Classifier
from L1_Dev.utils import cluster_date_correction, df_index_columns_correction


def get_data(start_date, end_date, ticker):

    ##############
    ## Get Data ##
    ##############

    # Databento: Prices, Metric & Jumps
    print("Databento: Prices, Metric & Jumps")
    databento = DataPrep(API_key='db-SEbmRhQ3ekjnrdRQLfK4iDYJhVkrL',
                         dataset='XNAS.ITCH',
                         start_date=start_date,
                         start_hour=9,
                         start_minute=30,
                         end_date=end_date,
                         end_hour=16,
                         end_minute=0,
                         time_zone='US/Eastern')
    jumps_dict = databento.compilation(generate_data=True, df_orders_storage=None, symbol=ticker)

    # StockNewsAPI: News
    print("StockNewsAPI: News")
    news = News(base_url='https://stocknewsapi.com/api/v1',
                API_key='7vjl2kzbnxdltdz2hxixyzhbc07yltk4keyh5az9',
                start_date=start_date,
                end_date=end_date,
                time_zone='US/Eastern')
    news_dict = news.compilation(generate_data=True, df_news_storage=None, symbol=ticker)

    # Clustering: Jumps & News
    cluster = Clusters()

    ### Jumps
    print("Clustering: Jumps")
    df_jumps = df_index_columns_correction(jumps_dict["jumps_bpv"])
    cluster_jumps_dict = cluster.compilation(df_jumps, "Jump")

    ### News (cluster in news is missing)
    print("Clustering: News")

    mask_for_news = df_index_columns_correction(jumps_dict["returns"])
    df_news = news_dict["news"]
    df_news = df_news.where(mask_for_news.notna(), np.nan)
    cluster_news_dict = cluster.compilation(df_news, "News")

    # Jumps Classification
    print("Jumps Classification")
    classifier = Classifier()
    df_jumps_clustered_list = cluster_jumps_dict["clustered_Jump_list"]
    df_news_clustered_list = cluster_news_dict["clustered_News_list"]
    classifier_dict = classifier.compilation(df_jumps_clustered_list, df_news_clustered_list)

    #####################
    # Final Compilation #
    #####################
    print("Final Compilation")

    jumps_dict = {key: cluster_date_correction(start_date, df) for key, df in jumps_dict.items()}
    cluster_jumps_dict = {key: cluster_date_correction(start_date, df) for key, df in cluster_jumps_dict.items()}
    news_dict = {key: cluster_date_correction(start_date, df) for key, df in news_dict.items()}
    cluster_news_dict = {key: cluster_date_correction(start_date, df) for key, df in cluster_news_dict.items()}
    classifier_dict = {key: cluster_date_correction(start_date, df) for key, df in classifier_dict.items()}

    compiled_data = {
        "Jumps_Data": {**jumps_dict, **cluster_jumps_dict},
        "News_Data": {**news_dict, **cluster_news_dict},
        "Classifier_Data": {**classifier_dict}
    }

    return compiled_data
