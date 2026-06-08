"""
Task 8 — PageIndex Vectorless RAG.

Yêu cầu:
    1. Upload tài liệu lên PageIndex.
    2. Viết function pageindex_search(query, top_k) trả về list[dict].
    3. Dùng PageIndex như fallback khi hybrid search không có kết quả tốt.

Cài đặt:
    pip install -U pageindex python-dotenv requests

.env:
    PAGEINDEX_API_KEY=your_api_key_here

Lưu ý:
    - PageIndex cloud SDK chính thức upload PDF bằng submit_document().
    - Nếu thư mục data/standardized chỉ có .md, cần chuyển sang PDF
      hoặc dùng Markdown API riêng để tạo tree.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
LANDING_DIR = BASE_DIR / "data" / "landing"
MANIFEST_PATH = LANDING_DIR / "pageindex_docs.json"

PAGEINDEX_API_BASE = "https://api.pageindex.ai"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_api_key() -> str:
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được cấu hình. "
            "Hãy tạo file .env và thêm PAGEINDEX_API_KEY=your_api_key_here"
        )
    return PAGEINDEX_API_KEY


def _get_client():
    """
    Khởi tạo PageIndex SDK client.
    """
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise ImportError(
            "Chưa cài pageindex. Chạy: pip install -U pageindex"
        ) from exc

    return PageIndexClient(api_key=_require_api_key())


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"documents": []}

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"documents": []}


def _save_manifest(manifest: dict) -> None:
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _wait_until_completed(client, doc_id: str, timeout_seconds: int = 600) -> dict:
    """
    Đợi PageIndex xử lý document xong.
    """
    start = time.time()

    while True:
        doc = client.get_document(doc_id)
        status = doc.get("status")

        if status == "completed":
            return doc

        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex xử lý thất bại: {doc}")

        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timeout khi chờ PageIndex xử lý doc_id={doc_id}")

        print(f"  [WAIT] doc_id={doc_id}, status={status}")
        time.sleep(5)


def upload_documents() -> list[dict]:
    """
    Upload toàn bộ file PDF trong data/standardized/ lên PageIndex.

    Returns:
        List document metadata đã upload:
        [
            {
                "doc_id": "...",
                "filename": "...",
                "path": "...",
                "status": "completed",
                "uploaded_at": "..."
            }
        ]
    """
    client = _get_client()
    manifest = _load_manifest()

    uploaded_paths = {
        item.get("path")
        for item in manifest.get("documents", [])
        if item.get("doc_id")
    }

    pdf_files = sorted(STANDARDIZED_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] Không tìm thấy file PDF trong: {STANDARDIZED_DIR}")
        print("[HINT] PageIndex submit_document() dùng tốt nhất với PDF.")
        return manifest.get("documents", [])

    for file_path in pdf_files:
        file_key = str(file_path.relative_to(BASE_DIR))

        if file_key in uploaded_paths:
            print(f"[SKIP] Đã upload trước đó: {file_key}")
            continue

        print(f"[UPLOAD] {file_key}")

        try:
            result = client.submit_document(str(file_path))
            doc_id = result["doc_id"]

            doc_info = _wait_until_completed(client, doc_id)

            item = {
                "doc_id": doc_id,
                "filename": file_path.name,
                "path": file_key,
                "status": doc_info.get("status", "completed"),
                "uploaded_at": _now_iso(),
            }

            manifest["documents"].append(item)
            _save_manifest(manifest)

            print(f"  [OK] Uploaded: {file_path.name} -> {doc_id}")

        except Exception as exc:
            print(f"  [ERROR] Upload failed for {file_path.name}: {exc}")

    return manifest.get("documents", [])


def _get_queryable_doc_ids() -> list[str]:
    """
    Lấy doc_id từ manifest local.
    """
    manifest = _load_manifest()

    doc_ids = [
        item["doc_id"]
        for item in manifest.get("documents", [])
        if item.get("doc_id") and item.get("status") == "completed"
    ]

    return doc_ids


def _check_retrieval_ready(doc_id: str) -> bool:
    """
    Kiểm tra document đã sẵn sàng retrieval chưa.
    """
    api_key = _require_api_key()

    response = requests.get(
        f"{PAGEINDEX_API_BASE}/doc/{doc_id}/",
        headers={"api_key": api_key},
        params={"type": "tree"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return bool(data.get("retrieval_ready"))


def _submit_retrieval_task(doc_id: str, query: str) -> str:
    """
    Gửi retrieval task cho một document.
    """
    api_key = _require_api_key()

    response = requests.post(
        f"{PAGEINDEX_API_BASE}/retrieval/",
        headers={"api_key": api_key},
        json={
            "doc_id": doc_id,
            "query": query,
            "thinking": False,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return data["retrieval_id"]


def _wait_retrieval_result(retrieval_id: str, timeout_seconds: int = 180) -> dict:
    """
    Đợi retrieval task hoàn thành.
    """
    api_key = _require_api_key()
    start = time.time()

    while True:
        response = requests.get(
            f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/",
            headers={"api_key": api_key},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        status = data.get("status")

        if status == "completed":
            return data

        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval thất bại: {data}")

        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timeout khi chờ retrieval_id={retrieval_id}"
            )

        time.sleep(3)


def _flatten_retrieval_result(result: dict, doc_id: str) -> list[dict]:
    """
    Chuyển response của PageIndex thành format chuẩn list[dict].
    """
    flattened = []

    retrieved_nodes = result.get("retrieved_nodes", [])

    for node_rank, node in enumerate(retrieved_nodes, start=1):
        title = node.get("title")
        node_id = node.get("node_id")

        relevant_contents = node.get("relevant_contents", [])

        for content_rank, item in enumerate(relevant_contents, start=1):
            text = (
                item.get("relevant_content")
                or item.get("text")
                or item.get("content")
                or ""
            ).strip()

            if not text:
                continue

            flattened.append(
                {
                    "content": text,
                    "score": 1.0 / (node_rank + content_rank - 1),
                    "metadata": {
                        "doc_id": doc_id,
                        "node_id": node_id,
                        "title": title,
                        "page_index": item.get("page_index"),
                        "retrieval_id": result.get("retrieval_id"),
                        "query": result.get("query"),
                    },
                    "source": "pageindex",
                }
            )

    return flattened


def _chat_fallback(query: str, doc_ids: list[str]) -> list[dict]:
    """
    Fallback thật bằng PageIndex Chat API nếu Retrieval API lỗi.
    Không dùng semantic_search local để tránh giả danh PageIndex.
    """
    client = _get_client()

    response = client.chat_completions(
        messages=[
            {
                "role": "user",
                "content": (
                    "Trả lời ngắn gọn dựa trên tài liệu đã upload. "
                    f"Câu hỏi: {query}"
                ),
            }
        ],
        doc_id=doc_ids,
        enable_citations=True,
    )

    answer = response["choices"][0]["message"]["content"]

    return [
        {
            "content": answer,
            "score": 1.0,
            "metadata": {
                "doc_ids": doc_ids,
                "mode": "pageindex_chat_fallback",
            },
            "source": "pageindex",
        }
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Dùng khi hybrid search không trả về kết quả phù hợp.

    Args:
        query: Câu truy vấn người dùng.
        top_k: Số lượng kết quả tối đa.

    Returns:
        List[dict]:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict,
                "source": "pageindex"
            }
        ]
    """
    if not query or not query.strip():
        return []

    doc_ids = _get_queryable_doc_ids()

    if not doc_ids:
        print("[WARN] Chưa có doc_id nào. Hãy chạy upload_documents() trước.")
        return []

    all_results = []

    for doc_id in doc_ids:
        try:
            if not _check_retrieval_ready(doc_id):
                print(f"[SKIP] Document chưa sẵn sàng retrieval: {doc_id}")
                continue

            retrieval_id = _submit_retrieval_task(doc_id, query)
            result = _wait_retrieval_result(retrieval_id)

            all_results.extend(_flatten_retrieval_result(result, doc_id))

        except Exception as exc:
            print(f"[WARN] Retrieval lỗi với doc_id={doc_id}: {exc}")

    all_results = sorted(
        all_results,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    if all_results:
        return all_results[:top_k]

    print("[WARN] Retrieval API không trả kết quả. Thử PageIndex Chat API fallback.")
    return _chat_fallback(query, doc_ids)[:top_k]


if __name__ == "__main__":
    print("Uploading documents to PageIndex...")
    upload_documents()

    print("\nTest PageIndex search:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)

    for index, item in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Score: {item['score']}")
        print(f"Source: {item['source']}")
        print(f"Metadata: {item['metadata']}")
        print(item["content"][:500])