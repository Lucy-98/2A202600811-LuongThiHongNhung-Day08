"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.

Yêu cầu:
    1. Crawl tối thiểu 5 bài báo.
    2. Sử dụng Crawl4AI.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON.
    5. Metadata gồm: URL gốc, ngày crawl, tiêu đề bài báo.

Cài đặt:
    pip install crawl4ai

Chạy:
    python crawl_news.py
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://cuoi.tuoitre.vn/miu-le-truoc-khi-bi-bat-qua-tang-dung-ma-tuy-toi-khong-biet-minh-thuc-su-muon-gi-20260511174121645.htm",
    "https://cafebiz.vn/long-nhat-hao-quang-30-nam-ca-hat-thi-phi-bua-vay-va-cu-soc-bi-bat-vi-ma-tuy-176260520130943245.chn",
    "https://giaoducthoidai.vn/ca-si-chi-dan-hien-ra-sao-sau-khi-bi-bat-vi-ma-tuy-post719183.html",
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://dantri.com.vn/giai-tri/binh-gold-tu-tho-xam-thanh-nhan-vat-tai-tieng-trong-lang-rap-viet-20250724090729539.htm",
]


def setup_directory() -> None:
    """Tạo thư mục data/landing/news/ nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Lấy thời gian crawl theo ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_markdown_text(result: Any) -> str:
    """
    Lấy markdown từ Crawl4AI.
    Một số version trả result.markdown là string,
    một số version trả object có raw_markdown / fit_markdown.
    """
    markdown = getattr(result, "markdown", "")

    if isinstance(markdown, str):
        return markdown.strip()

    for attr in ["fit_markdown", "raw_markdown", "markdown"]:
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return str(markdown).strip() if markdown else ""


def extract_title(result: Any, content_markdown: str) -> str:
    """
    Ưu tiên lấy title từ metadata.
    Nếu không có thì lấy dòng heading đầu tiên trong markdown.
    """
    metadata = getattr(result, "metadata", None)

    if isinstance(metadata, dict):
        title = metadata.get("title") or metadata.get("og:title")
        if title:
            return title.strip()

    for line in content_markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.replace("#", "").strip()

    return "Không xác định"


def safe_filename(title: str, index: int) -> str:
    """Tạo tên file an toàn."""
    text = title.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text).strip("-")

    if not text:
        text = f"article-{index:02d}"

    return f"{index:02d}_{text[:80]}.json"


async def crawl_article(
    crawler: AsyncWebCrawler,
    url: str,
    index: int,
    run_config: CrawlerRunConfig,
) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.
    Không dùng mock data để tránh sai yêu cầu bài.
    """
    print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")

    try:
        result = await crawler.arun(url=url, config=run_config)

        success = bool(getattr(result, "success", False))
        final_url = getattr(result, "url", url)
        error_message = getattr(result, "error_message", None)

        content_markdown = get_markdown_text(result)
        title = extract_title(result, content_markdown)

        article = {
            "source_url": url,
            "final_url": final_url,
            "title": title,
            "date_crawled": now_iso(),
            "success": success,
            "content_markdown": content_markdown,
            "error": error_message,
        }

        if not success:
            print(f"  [WARN] Crawl failed: {error_message}")
        elif len(content_markdown) < 200:
            print("  [WARN] Nội dung crawl hơi ngắn, nên kiểm tra lại URL.")
        else:
            print(f"  [OK] Crawled: {title}")

        return article

    except Exception as e:
        print(f"  [ERROR] {e}")

        return {
            "source_url": url,
            "final_url": url,
            "title": "Không xác định",
            "date_crawled": now_iso(),
            "success": False,
            "content_markdown": "",
            "error": str(e),
        }


async def crawl_all() -> None:
    """Crawl toàn bộ bài báo trong ARTICLE_URLS và lưu thành JSON."""
    if len(ARTICLE_URLS) < 5:
        raise ValueError("Cần tối thiểu 5 URL bài báo theo yêu cầu đề bài.")

    setup_directory()

    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000,
        word_count_threshold=50,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, url in enumerate(ARTICLE_URLS, start=1):
            article = await crawl_article(
                crawler=crawler,
                url=url,
                index=index,
                run_config=run_config,
            )

            filename = safe_filename(article["title"], index)
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"  [SAVED] {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())