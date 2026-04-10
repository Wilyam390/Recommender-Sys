import numpy as np


def build_relevant_items_lookup(eval_df, threshold=4, user_col="UserId", item_col="ProductId", score_col="Score"):
    """Return mapping user -> set(relevant item ids) from an interaction dataframe."""
    rel = eval_df.loc[eval_df[score_col] >= threshold, [user_col, item_col]].dropna()
    if rel.empty:
        return {}
    return rel.groupby(user_col)[item_col].apply(set).to_dict()


def ranking_metrics_from_topn(
    topn,
    eval_df,
    k=10,
    threshold=4,
    user_col="UserId",
    item_col="ProductId",
    score_col="Score",
):
    """Compute Precision/Recall/MAP/NDCG@k from Top-N recommendations.

    Parameters
    - topn: dict-like mapping user_id -> list of item_ids OR list of (item_id, score) tuples.
    - eval_df: interactions dataframe with true ratings, used to derive relevant items.

    Notes
    - This is intended for realistic Top-N evaluation: recommend from unseen candidates,
      then compare the recommended items with relevant test items for each user.
    """

    relevant_by_user = build_relevant_items_lookup(
        eval_df, threshold=threshold, user_col=user_col, item_col=item_col, score_col=score_col
    )
    if not relevant_by_user:
        return {"precision": np.nan, "recall": np.nan, "map": np.nan, "ndcg": np.nan}

    user_ids = [u for u in topn.keys() if u in relevant_by_user]
    if not user_ids:
        return {"precision": np.nan, "recall": np.nan, "map": np.nan, "ndcg": np.nan}

    precisions, recalls, aps, ndcgs = [], [], [], []
    for user_id in user_ids:
        rel_items = relevant_by_user.get(user_id, set())
        if not rel_items:
            continue

        recs = topn.get(user_id, [])
        rec_items = []
        for entry in recs:
            if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                rec_items.append(entry[0])
            else:
                rec_items.append(entry)
        rec_items = rec_items[:k]
        if not rec_items:
            continue

        hits = [1 if item_id in rel_items else 0 for item_id in rec_items]
        hit_count = int(sum(hits))

        precisions.append(hit_count / float(k))
        recalls.append(hit_count / float(len(rel_items)))

        cum_hits = 0
        precisions_at_hits = []
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                cum_hits += 1
                precisions_at_hits.append(cum_hits / float(rank))
        aps.append(float(np.mean(precisions_at_hits)) if precisions_at_hits else 0.0)

        # binary relevance (0/1)
        dcg = float(sum(h / np.log2(rank + 1) for rank, h in enumerate(hits, start=1)))
        ideal_hits = [1] * min(len(rel_items), k) + [0] * max(0, k - min(len(rel_items), k))
        idcg = float(sum(h / np.log2(rank + 1) for rank, h in enumerate(ideal_hits, start=1)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision": float(np.mean(precisions)) if precisions else np.nan,
        "recall": float(np.mean(recalls)) if recalls else np.nan,
        "map": float(np.mean(aps)) if aps else np.nan,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else np.nan,
    }


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
