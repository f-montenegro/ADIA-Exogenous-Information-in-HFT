from L0_Library.config import *


class News:

    def __init__(self, base_url: str, API_key: str, start_date: datetime, end_date: datetime):

        self.base_url = base_url
        self.API_key = API_key
        self.start_date = start_date
        self.end_date = end_date

    def get_news(self, symbol: list, items: int, page: int, news_type: str):

        start_date = self.start_date.strftime('%m%d%Y')
        end_date = self.end_date.strftime('%m%d%Y')
        date = f'{start_date}-{end_date}'

        params = {'tickers': symbol,
                  'items': items,  # Number of news items to fetch
                  'page': page,  # Pagination for the results
                  'date': date,
                  'type': news_type,
                  'token': self.API_key}

        response = requests.get(self.base_url, params=params)
        response = response.json()

        if 'error' in response:
            print(f"API Error: {response['error']}")

            return False

        response = response['data']

        return response
