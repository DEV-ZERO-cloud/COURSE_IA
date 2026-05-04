import re
import unicodedata
import pandas as pd
from nltk import TweetTokenizer
from spacy.lang.es import Spanish
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessing:

    def __init__(self, lang: str = 'es'):
        self.lang = lang
        self._nlp_lang = None
        self.vectorizer = None

    def _get_lang_model(self):
        """Singleton: carga Spanish() o English() solo una vez"""
        if self._nlp_lang is None:
            self._nlp_lang = Spanish() if self.lang == 'es' else English()
        return self._nlp_lang

    # -----------------------------------------------
    # LIMPIEZA
    # -----------------------------------------------
    def _proper_encoding(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        return text.decode('utf-8')

    def _remove_stopwords(self, text: str) -> str:
        nlp = self._get_lang_model()
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_stop and len(t.text) > 1]
        return ' '.join(tokens) if tokens else None

    def clean(self, text: str, stopwords: bool = False,
              keep_hashtag_text: bool = True) -> str:
        try:
            if not isinstance(text, str):
                return None

            text = text.lower()

            # Reemplazos con tokens semánticos
            text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
            text = re.sub(r'rt\s', ' RETWEET ', text)
            text = re.sub(r'@[A-Za-z0-9_]{1,40}', ' MENTION ', text)
            text = re.sub(r'[\U0001f000-\U000e007f]', ' EMOJI ', text)

            if keep_hashtag_text:
                text = re.sub(r'#([A-Za-z0-9_]+)', r' HASHTAG_\1 ', text)
            else:
                text = re.sub(r'#([A-Za-z0-9_]+)', ' HASHTAG ', text)

            # Normalización
            text = self._proper_encoding(text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            if stopwords:
                text = self._remove_stopwords(text)

            return text if text and text.strip() else None

        except Exception as e:
            print(f'Error clean: {e}')
            return None

    def clean_dataset(self, df: pd.DataFrame, col: str,
                      stopwords: bool = False,
                      keep_hashtag_text: bool = True) -> pd.DataFrame:
        try:
            df = df.copy()
            df['clean'] = df[col].apply(
                lambda x: self.clean(x,
                                     stopwords=stopwords,
                                     keep_hashtag_text=keep_hashtag_text)
            )
            before = len(df)
            df = df.dropna(subset=['clean'])
            df = df[df['clean'].str.strip() != '']
            print(f'Tweets válidos: {len(df)} / {before}')
            return df
        except Exception as e:
            print(f'Error clean_dataset: {e}')

    # -----------------------------------------------
    # TOKENIZACIÓN
    # -----------------------------------------------
    def tokenize(self, text: str) -> list:
        try:
            tokenizer = TweetTokenizer()
            return tokenizer.tokenize(text)
        except Exception as e:
            print(f'Error tokenize: {e}')

    # -----------------------------------------------
    # LEXICAL VECTORIZER
    # -----------------------------------------------
    def fit_vectorizer(self, texts: pd.Series,
                       max_features: int = 10000,
                       ngram_range: tuple = (1, 2),
                       min_df: int = 2) -> None:
        """Entrena el vectorizador con el corpus"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True
            )
            self.vectorizer.fit(texts)
            print(f'Vocabulario: {len(self.vectorizer.vocabulary_)} términos')
        except Exception as e:
            print(f'Error fit_vectorizer: {e}')

    def transform(self, texts: pd.Series):
        """Transforma textos a matriz TF-IDF"""
        try:
            if self.vectorizer is None:
                raise ValueError('Primero llama a fit_vectorizer()')
            return self.vectorizer.transform(texts)
        except Exception as e:
            print(f'Error transform: {e}')

    def fit_transform(self, texts: pd.Series,
                      max_features: int = 10000,
                      ngram_range: tuple = (1, 2),
                      min_df: int = 2):
        """fit + transform en un solo paso"""
        try:
            self.fit_vectorizer(texts, max_features, ngram_range, min_df)
            return self.transform(texts)
        except Exception as e:
            print(f'Error fit_transform: {e}')

    def get_feature_names(self) -> list:
        """Retorna los términos del vocabulario"""
        try:
            if self.vectorizer is None:
                raise ValueError('Primero llama a fit_vectorizer()')
            return self.vectorizer.get_feature_names_out().tolist()
        except Exception as e:
            print(f'Error get_feature_names: {e}')