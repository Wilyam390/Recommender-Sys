import warnings
import os
import re
import zipfile
import subprocess
import gc
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.cli import download
from sentence_transformers import SentenceTransformer

from surprise import Dataset, Reader, NormalPredictor, KNNBasic, KNNWithMeans, SVD, accuracy, BaselineOnly
from surprise.model_selection import train_test_split as surprise_train_test_split

from recsys_eval_utils import (
    precision_at_k,
    recall_at_k,
    map_at_k,
    ndcg_at_k,
    ranking_metrics_from_topn,
)


def configure_notebook():
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 200)
    sns.set_theme(style="whitegrid")


__all__ = [
    "warnings", "os", "re", "zipfile", "subprocess", "gc", "time", "math", "Path",
    "np", "pd", "plt", "sns",
    "sparse", "csr_matrix",
    "train_test_split", "MinMaxScaler", "MultiLabelBinarizer", "StandardScaler",
    "mean_squared_error", "mean_absolute_error",
    "cosine_similarity", "pairwise_distances",
    "CountVectorizer", "TfidfVectorizer",
    "tqdm", "nltk", "stopwords", "spacy", "download", "SentenceTransformer",
    "Dataset", "Reader", "NormalPredictor", "KNNBasic", "KNNWithMeans", "SVD", "accuracy", "BaselineOnly", "surprise_train_test_split",
    "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k", "ranking_metrics_from_topn",
    "configure_notebook",
]
