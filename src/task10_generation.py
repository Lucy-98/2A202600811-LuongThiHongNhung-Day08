"""
Task 10 — Generation Có Citation.

Yêu cầu:
    1. Chọn top_k, top_p phù hợp và giải thích trong comment.
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle".
    3. Inject context vào prompt.
    4. Gọi LLM trả lời có citation.
    5. Nếu không đủ evidence -> "I cannot verify this information".

Cài đặt:
    pip install openai python-dotenv

.env:
    OPENAI_API_KEY=your_api_key_here
"""

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    retrieve = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# TOP_K = 5 vì:
# - Đủ nhiều context để có evidence đa chiều.
# - Không quá dài, giảm rủi ro lost in the middle.
# - Phù hợp bài lab cá nhân, dễ debug citation.
TOP_K = 5

# TOP_P = 0.8 vì:
# - RAG cần factual, không cần quá sáng tạo.
# - 0.8 giúp model bớt lan man hơn 0.9.
# - Vẫn đủ linh hoạt để viết câu trả lời tự nhiên.
TOP_P = 0.8

# TEMPERATURE = 0.2 vì:
# - Citation generation cần ổn định.
# - Giảm hallucination.
# - Ưu tiên bám context hơn sáng tạo.
TEMPERATURE = 0.2

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CANNOT_VERIFY = "I cannot verify this information"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""
Answer the question in Vietnamese using ONLY the provided context.

Citation rules:
- Every factual claim MUST include a citation immediately after the claim.
- Citation format MUST be exactly: [Nguồn, Năm]
- Use the source and year shown in each context chunk.
- Do not cite sources that are not present in the context.
- If the context does not explicitly support the answer, reply exactly:
  {CANNOT_VERIFY}

Do not guess.
Do not use outside knowledge.
Do not mention information that is not in the context.
""".strip()


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để giảm lỗi "lost in the middle".

    Giả định input đã được rerank theo độ quan trọng:
        [1, 2, 3, 4, 5]

    Output mong muốn:
        [1, 3, 5, 4, 2]

    Lý do:
        - Chunk quan trọng nhất đặt ở đầu.
        - Chunk quan trọng thứ 2 đặt ở cuối.
        - Chunk ít quan trọng hơn đặt ở giữa.
    """
    if len(chunks) <= 2:
        return chunks

    front = []
    back = []

    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            front.append(chunk)
        else:
            back.append(chunk)

    back.reverse()

    return front + back


# =============================================================================
# METADATA HELPERS
# =============================================================================

def _safe_get_metadata(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _extract_source(chunk: dict, fallback: str) -> str:
    """
    Lấy tên nguồn từ metadata.
    Ưu tiên các field thường gặp trong pipeline.
    """
    metadata = _safe_get_metadata(chunk)

    source = (
        metadata.get("source")
        or metadata.get("title")
        or metadata.get("filename")
        or metadata.get("file_name")
        or metadata.get("url")
        or chunk.get("source")
        or fallback
    )

    source = str(source).strip()
    return source if source else fallback


def _extract_year(chunk: dict) -> str:
    """
    Lấy năm từ metadata hoặc source/content.
    Nếu không tìm được thì dùng 'unknown'.
    """
    metadata = _safe_get_metadata(chunk)

    candidates = [
        metadata.get("year"),
        metadata.get("date"),
        metadata.get("published_at"),
        metadata.get("created_at"),
        metadata.get("date_crawled"),
        metadata.get("source"),
        metadata.get("filename"),
        chunk.get("source"),
        chunk.get("content"),
    ]

    for value in candidates:
        if not value:
            continue

        match = re.search(r"(20\d{2}|19\d{2})", str(value))
        if match:
            return match.group(1)

    return "unknown"


def _get_content(chunk: dict) -> str:
    content = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("content_markdown")
        or chunk.get("markdown")
        or ""
    )

    return str(content).strip()


def _normalize_chunk(chunk: dict, index: int) -> dict:
    """
    Chuẩn hóa chunk để prompt luôn có source/year/content rõ ràng.
    """
    source = _extract_source(chunk, fallback=f"Source {index}")
    year = _extract_year(chunk)
    content = _get_content(chunk)

    return {
        **chunk,
        "_source_label": source,
        "_year_label": year,
        "_citation": f"[{source}, {year}]",
        "_content": content,
    }


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format context để LLM biết phải cite theo nguồn nào.
    """
    context_parts = []

    for i, raw_chunk in enumerate(chunks, start=1):
        chunk = _normalize_chunk(raw_chunk, i)

        if not chunk["_content"]:
            continue

        context_parts.append(
            f"### Chunk {i}\n"
            f"Source: {chunk['_source_label']}\n"
            f"Year: {chunk['_year_label']}\n"
            f"Citation to use: {chunk['_citation']}\n"
            f"Content:\n{chunk['_content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(query: str, context: str) -> str:
    return f"""
Context:
{context}

Question:
{query}

Write the answer in Vietnamese.
Every factual claim must include citation in this exact format: [Nguồn, Năm].
If the context is insufficient, reply exactly: {CANNOT_VERIFY}
""".strip()


# =============================================================================
# CITATION VALIDATION
# =============================================================================

def _extract_allowed_citations(chunks: list[dict]) -> set[str]:
    allowed = set()

    for i, raw_chunk in enumerate(chunks, start=1):
        chunk = _normalize_chunk(raw_chunk, i)
        allowed.add(chunk["_citation"])

    return allowed


def _answer_has_citation(answer: str) -> bool:
    """
    Kiểm tra answer có citation dạng [Nguồn, Năm] hay không.
    """
    pattern = r"\[[^\[\],]+,\s*(?:19\d{2}|20\d{2}|unknown)\]"
    return bool(re.search(pattern, answer))


def _uses_only_allowed_citations(answer: str, allowed_citations: set[str]) -> bool:
    """
    Không bắt buộc 100% nếu tên nguồn dài có ký tự lạ,
    nhưng vẫn check các citation bắt được có nằm trong context không.
    """
    found = set(re.findall(r"\[[^\[\]]+,\s*(?:19\d{2}|20\d{2}|unknown)\]", answer))

    if not found:
        return False

    return found.issubset(allowed_citations)


def _is_cannot_verify(answer: str) -> bool:
    return CANNOT_VERIFY.lower() in answer.lower()


# =============================================================================
# OFFLINE EXTRACTIVE FALLBACK
# =============================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Tách câu đơn giản cho tiếng Việt.
    """
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) >= 30]


def _query_terms(query: str) -> set[str]:
    """
    Lấy keyword đơn giản từ query để chọn câu liên quan.
    """
    words = re.findall(r"\w+", query.lower(), flags=re.UNICODE)

    stopwords = {
        "là", "và", "của", "cho", "theo", "về", "có", "những",
        "nào", "gì", "ai", "ở", "trong", "một", "các", "đã",
        "bị", "liên", "quan", "tới", "đến"
    }

    return {w for w in words if len(w) >= 3 and w not in stopwords}


def _offline_extractive_answer(query: str, chunks: list[dict], max_sentences: int = 4) -> str:
    """
    Fallback không gọi LLM.
    Không bịa thêm thông tin, chỉ trích câu từ context và gắn citation.
    """
    terms = _query_terms(query)
    candidates = []

    for i, raw_chunk in enumerate(chunks, start=1):
        chunk = _normalize_chunk(raw_chunk, i)
        sentences = _split_sentences(chunk["_content"])

        for sentence in sentences:
            sentence_lower = sentence.lower()
            overlap = sum(1 for term in terms if term in sentence_lower)

            if overlap > 0:
                candidates.append(
                    {
                        "sentence": sentence,
                        "overlap": overlap,
                        "citation": chunk["_citation"],
                    }
                )

    candidates.sort(key=lambda x: x["overlap"], reverse=True)

    if not candidates:
        return CANNOT_VERIFY

    selected = candidates[:max_sentences]

    answer_parts = [
        f"{item['sentence']} {item['citation']}"
        for item in selected
    ]

    return "\n\n".join(answer_parts)


# =============================================================================
# LLM CALL
# =============================================================================

def _call_openai_llm(query: str, context: str) -> str:
    """
    Gọi OpenAI Chat Completions.

    Lưu ý:
        - temperature thấp để giảm hallucination.
        - top_p thấp vừa phải để output ổn định.
        - Không set cả hai quá cao vì bài này cần citation chính xác.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or api_key == "sk-xxx":
        raise RuntimeError("OPENAI_API_KEY chưa được cấu hình.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(query, context),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content or ""
    return answer.strip()


def _build_generation_result(
    answer: str,
    sources: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Return a stable generation payload for tests, evaluation, and UI callers."""
    return {
        "answer": answer,
        "sources": sources or [],
        "top_k": top_k,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "model": MODEL_NAME,
    }


# =============================================================================
# GENERATION MAIN FUNCTION
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] = None,
    top_k: int = TOP_K,
    use_reranking: bool = True,
    use_hyde: bool = False,
) -> dict[str, Any]:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve/use provided context chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return dict with answer and source metadata
    """
    # Step 1: Retrieve if not provided
    if context_chunks is None:
        if retrieve is None:
            return _build_generation_result(CANNOT_VERIFY, [], top_k)
        context_chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking, use_hyde=use_hyde)

    if not query or not query.strip():
        return _build_generation_result(CANNOT_VERIFY, context_chunks or [], top_k)

    if not context_chunks:
        return _build_generation_result(CANNOT_VERIFY, [], top_k)

    # Chỉ lấy TOP_K chunk đầu vì input giả định đã rerank theo score.
    selected_chunks = context_chunks[:top_k]

    # Bỏ chunk rỗng.
    selected_chunks = [
        chunk for chunk in selected_chunks
        if _get_content(chunk)
    ]

    if not selected_chunks:
        return _build_generation_result(CANNOT_VERIFY, context_chunks, top_k)

    reordered_chunks = reorder_for_llm(selected_chunks)
    context = format_context(reordered_chunks)

    if not context.strip():
        return _build_generation_result(CANNOT_VERIFY, reordered_chunks, top_k)

    allowed_citations = _extract_allowed_citations(reordered_chunks)

    try:
        answer = _call_openai_llm(query, context)
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}")
        print("[INFO] Using offline extractive fallback from retrieved context.")
        answer = _offline_extractive_answer(query, reordered_chunks)

    # Nếu model tự nói không xác minh được thì giữ nguyên.
    if _is_cannot_verify(answer):
        return _build_generation_result(CANNOT_VERIFY, reordered_chunks, top_k)

    # Nếu answer không có citation thì không đạt yêu cầu.
    if not _answer_has_citation(answer):
        return _build_generation_result(CANNOT_VERIFY, reordered_chunks, top_k)

    # Nếu citation không nằm trong context thì không tin.
    if not _uses_only_allowed_citations(answer, allowed_citations):
        print("[WARN] Answer contains citation not present in context.")
        return _build_generation_result(CANNOT_VERIFY, reordered_chunks, top_k)

    return _build_generation_result(answer, reordered_chunks, top_k)


# =============================================================================
# OPTIONAL WRAPPER: RETRIEVE + GENERATE
# =============================================================================

def rag_generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Wrapper end-to-end:
        1. Retrieve từ task9.
        2. Generate có citation.

    Hàm này để chạy demo.
    Còn hàm chính theo đề là generate_with_citation(query, context_chunks).
    """
    if retrieve is None:
        raise ImportError("Không import được retrieve từ task9_retrieval_pipeline.")

    chunks = retrieve(query, top_k=top_k)
    return generate_with_citation(query, chunks, top_k=top_k)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Q: {query}")
        print("=" * 80)

        result = rag_generate_with_citation(query, top_k=TOP_K)

        print("\nA:")
        print(result["answer"])

        print(f"\nConfig: top_k={result['top_k']}, top_p={result['top_p']}, temperature={result['temperature']}")
        print(f"Sources used: {len(result['sources'])}")
