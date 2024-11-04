import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import emoji

nltk.download("stopwords")


def remove_emojis(text):
    """
    Supprime les emojis et les émoticônes d'un texte.

    Parameters:
    text (str): Texte à nettoyer

    Returns:
    str: Texte sans emojis
    """
    # Supprimer les emojis Unicode
    text = emoji.replace_emoji(text, "")

    # Supprimer les émoticônes textuelles communes
    emoticons_pattern = re.compile(r"[:;=]-?[)(/\\|dpDP]|[)(/<>]{}")
    text = emoticons_pattern.sub("", text)

    return text


def clean_reddit_data(df, similarity_threshold=0.9):
    """
    Nettoie les données Reddit en gérant les problèmes de formatage, les sauts de ligne et les emojis.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les colonnes 'title', 'selftext'
    similarity_threshold (float): Seuil de similarité cosinus pour les doublons

    Returns:
    pd.DataFrame: DataFrame nettoyé
    """
    # Copie pour éviter de modifier les données originales
    df = df.copy()

    # Step 1: Nettoyage initial des colonnes textuelles
    text_columns = ["title", "selftext"]
    for col in text_columns:
        if col in df.columns:
            # Convertir en string et gérer les NaN
            df[col] = df[col].fillna("")
            df[col] = df[col].astype(str)

            # Supprimer les emojis
            df[col] = df[col].apply(remove_emojis)

            # Normaliser les sauts de ligne
            df[col] = df[col].apply(lambda x: re.sub(r"\n+", " ", x))

            # Supprimer les quotes Reddit
            df[col] = df[col].apply(lambda x: re.sub(r"^\s*>\s*", "", x))

            # Nettoyer les espaces multiples
            df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", x.strip()))

    # Combiner title et selftext avec un séparateur clair
    df["full_text"] = df["title"] + " || " + df["selftext"]

    # Step 2: Supprimer le contenu indésirable
    unwanted_content = [
        r"\b\[removed\]\b",
        r"\b\[deleted\]\b",
        r"^\s*$",  # Lignes vides
        r"^deleted$",
        r"^removed$",
    ]
    df = df[
        ~df["full_text"].str.contains(
            "|".join(unwanted_content), flags=re.IGNORECASE, regex=True
        )
    ]

    # Step 3: Nettoyer le contenu Markdown et les URLs
    def clean_markdown_and_links(text):
        # Supprimer les liens Markdown
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Supprimer les URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Supprimer la syntaxe Markdown
        text = re.sub(r"[*_~#>`]", "", text)

        # Nettoyer les caractères spéciaux HTML
        text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#x200B;", " ", text)

        return text

    df["full_text"] = df["full_text"].apply(clean_markdown_and_links)

    # Step 4: Filtrage basé sur la longueur et la qualité
    df = df[df["full_text"].str.len() > 30]  # Texte minimum

    # Filtrer les posts avec trop de majuscules
    df = df[
        df["full_text"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) < 0.5)
    ]

    # Step 5: Nettoyage approfondi du texte
    def deep_clean_text(text):
        # Convertir en minuscules
        text = text.lower()

        # Supprimer les caractères non-alphanumériques
        text = re.sub(r"[^\w\s]", " ", text)

        # Supprimer les chiffres isolés
        text = re.sub(r"\b\d+\b", "", text)

        # Nettoyer les espaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    df["cleaned_text"] = df["full_text"].apply(deep_clean_text)

    # Step 6: Suppression des mots vides
    stop_words = set(nltk.corpus.stopwords.words("english"))
    df["cleaned_text"] = df["cleaned_text"].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )

    # Step 7: Suppression des doublons
    df = df.drop_duplicates(subset=["cleaned_text"])

    # Step 8: Suppression des posts quasi-identiques
    if len(df) > 1:
        tfidf = TfidfVectorizer().fit_transform(df["cleaned_text"])
        pairwise_sim = cosine_similarity(tfidf)

        to_drop = set()
        for idx in range(pairwise_sim.shape[0]):
            if idx in to_drop:
                continue
            duplicates = np.where(pairwise_sim[idx] > similarity_threshold)[0]
            duplicates = [i for i in duplicates if i != idx and i > idx]
            to_drop.update(duplicates)

        df = df.drop(df.index[list(to_drop)])

    # Nettoyage final
    df = df.drop(columns=["full_text", "cleaned_text"])
    df = df.reset_index(drop=True)

    return df


# Exemple d'utilisation
# data = pd.read_csv("reddit_02.csv")
# cleaned_df = clean_reddit_data(data)
# cleaned_df.to_csv("cleaned_reddit_data.csv", index=False)
