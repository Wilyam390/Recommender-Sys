import numpy as np


def _eligible_user_groups(eval_df, threshold):
    """Yield per-user frames restricted to users with at least one relevant item."""
    grouped = eval_df.sort_values("pred", ascending=False).groupby("UserId")
    for _, user_df in grouped:
        if (user_df["Score"] >= threshold).sum() > 0:
            yield user_df


def precision_at_k(eval_df, k=10, threshold=4):
    """Average Precision@k across users using predicted scores for ranking."""
    precision_scores = []

    for user_df in _eligible_user_groups(eval_df, threshold):
        top_k = user_df.head(k)
        if len(top_k) == 0:
            continue
        precision_scores.append((top_k["Score"] >= threshold).mean())

    return float(np.mean(precision_scores)) if precision_scores else 0.0


def recall_at_k(eval_df, k=10, threshold=4):
    """Average Recall@k across users using predicted scores for ranking."""
    recall_scores = []

    for user_df in _eligible_user_groups(eval_df, threshold):
        n_relevant = (user_df["Score"] >= threshold).sum()

        top_k = user_df.head(k)
        relevant_in_top_k = (top_k["Score"] >= threshold).sum()
        recall_scores.append(relevant_in_top_k / n_relevant)

    return float(np.mean(recall_scores)) if recall_scores else 0.0


def map_at_k(eval_df, k=10, threshold=4):
    """Mean Average Precision@k across users."""
    ap_scores = []

    for user_df in _eligible_user_groups(eval_df, threshold):
        n_relevant = (user_df["Score"] >= threshold).sum()

        top_k = user_df.head(k)
        relevance = (top_k["Score"] >= threshold).astype(int).values

        hits = 0
        precision_sum = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                hits += 1
                precision_sum += hits / rank

        denom = min(n_relevant, k)
        ap_scores.append(precision_sum / denom if denom > 0 else 0.0)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def ndcg_at_k(eval_df, k=10, threshold=4):
    """Average NDCG@k across users with binary relevance from thresholded ratings."""
    ndcg_scores = []

    for user_df in _eligible_user_groups(eval_df, threshold):
        n_relevant = (user_df["Score"] >= threshold).sum()

        top_k = user_df.head(k)
        relevance = (top_k["Score"] >= threshold).astype(int).values

        dcg = 0.0
        for rank, rel in enumerate(relevance, start=1):
            dcg += rel / np.log2(rank + 1)

        ideal_len = min(n_relevant, k)
        idcg = 0.0
        for rank in range(1, ideal_len + 1):
            idcg += 1 / np.log2(rank + 1)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
