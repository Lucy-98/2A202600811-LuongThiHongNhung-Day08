"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import pickle
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.pkl"
    if not vector_store_path.exists():
        print(f"[WARN] Vector store chua ton tai: {vector_store_path}")
        return []

    # Đọc chunks từ pickle
    with open(vector_store_path, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        return []

    # Load embedding model (Task 4)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query)

    # Tính cosine similarity cho từng chunk
    results = []
    for chunk in chunks:
        chunk_emb = chunk.get("embedding")
        if chunk_emb is None:
            continue
        
        # Chuyển thành numpy array
        q_vec = np.array(query_embedding)
        c_vec = np.array(chunk_emb)
        
        # Tính cosine similarity
        dot_product = np.dot(q_vec, c_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_c = np.linalg.norm(c_vec)
        
        if norm_q > 0 and norm_c > 0:
            score = float(dot_product / (norm_q * norm_c))
        else:
            score = 0.0

        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk.get("metadata", {})
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

