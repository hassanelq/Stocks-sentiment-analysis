import requests
from bs4 import BeautifulSoup
import pandas as pd


def fetch_stock_news(ticker):
    # finviz url
    finviz_url = 'https://finviz.com/quote.ashx?t='

    # En-têtes pour la requête
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # URL complète pour le ticker
    url = finviz_url + ticker

    # Récupérer les données avec requests
    response = requests.get(url, headers=headers)

    # Vérifier si la requête a réussi
    if response.status_code == 200:
        html = BeautifulSoup(response.text, features='html.parser')
        news_table = html.find(id='news-table')

        parsed_data = []
        previous_date = None  # Stocker la date précédente si manquante

        # Analyser les lignes de la table de nouvelles
        for row in news_table.findAll('tr'):
            title = row.a.text.strip()  # Extraire le titre de l'article
            date_data = row.td.text.strip()  # Extraire la date ou l'heure

            # Gérer les cas où seule l'heure est présente
            if len(date_data.split()) == 1:
                time = date_data
                date = previous_date  # Utiliser la date précédente
            else:
                # Extraire la date et l'heure
                date, time = date_data.split(' ', 1)
                previous_date = date  # Mettre à jour la date précédente

            parsed_data.append([ticker, date, time, title])

        # Créer le DataFrame à partir des données analysées
        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

        return df
    else:
        print(f"Échec de la récupération des données pour {ticker}. Code d'état HTTP : {response.status_code}")
        return None


