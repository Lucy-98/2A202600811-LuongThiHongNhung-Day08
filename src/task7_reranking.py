"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API hoặc local fallback).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    jina_api_key = os.getenv("JINA_API_KEY", "")
    if jina_api_key and jina_api_key != "jina_xxx":
        print("Using Jina AI Reranker API...")
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
        except Exception as e:
            print(f"Jina AI Reranker API error: {e}. Falling back to local similarity...")

    # Fallback: tính toán Cosine Similarity bằng sentence-transformers cục bộ
    print("Falling back to local similarity for rerank...")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = model.encode(query)
        doc_embs = model.encode([c["content"] for c in candidates])

        results = []
        for idx, (cand, d_emb) in enumerate(zip(candidates, doc_embs)):
            dot_product = np.dot(q_emb, d_emb)
            norm_q = np.linalg.norm(q_emb)
            norm_d = np.linalg.norm(d_emb)
            score = float(dot_product / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.0
            results.append({**cand, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Local similarity rerank failed: {e}. Returning original candidates...")
        candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return candidates_sorted[:top_k]


def cosine_sim(vec1, vec2):
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    # Đảm bảo các candidates có embedding vector
    from sentence_transformers import SentenceTransformer
    model = None
    for cand in candidates:
        if "embedding" not in cand:
            if model is None:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            cand["embedding"] = model.encode(cand["content"]).tolist()

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Tương đồng với truy vấn
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Độ trùng lặp lớn nhất với các tài liệu đã chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # Công thức MMR
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()
            else:
                # Giữ điểm tương đồng cao nhất của nguồn
                content_map[key]["score"] = max(content_map[key].get("score", 0), item.get("score", 0))

    # Sắp xếp theo điểm RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)  # Điểm mới là điểm RRF
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode(query).tolist()
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # Nếu truyền danh sách dẹt vào rrf, mặc định trả về top_k đầu tiên
        return candidates[:top_k]
    else:
        raise ValueError(f"Unknown rerank method: {method}")



if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
