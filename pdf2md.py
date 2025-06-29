#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdf2md.py  —  OCR‑конвертор PDF → Markdown на базе Mistral AI.

❯ python pdf2md.py <path_to_pdf> [-o OUTPUT] [--method {auto,base64,upload}]
    path_to_pdf     — входной PDF.
    -o, --output    — путь для Markdown (по умолчанию <basename>.md).
    --method        — стратегия передачи файла:
                      auto    — base64 ≤ 5 МБ, иначе upload  (default)
                      base64  — всегда вставляет файл как data‑URL
                      upload  — загружает через client.files.upload()
Переменная окружения MISTRAL_API_KEY должна быть установлена.


pip install mistralai tqdm
python pdf2md.py /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/LegalAlly/Lexi_brief/case_doc/Приложение7.pdf            # создаст example.md
python pdf2md.py ./docs/report.pdf -o out.md # явный путь к .md

Lexi_brief/case_doc/Приложение7.pdf

python pdf2md.py "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/LegalAlly/Lexi_brief/case_doc/Приложение7.pdf"


"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Literal

from mistralai import Mistral
from tqdm import tqdm


# ────────────────────────────  helpers  ────────────────────────────


def encode_pdf_to_base64(pdf_path: Path) -> str:
    """Читает PDF и возвращает base64‑строку (без переносов)."""
    with pdf_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_markdown(ocr_response) -> str:
    """
    Сборка Markdown из ответа OCR.
    Каждая страница приходит в ocr_response.pages[i].markdown.
    """
    pages = getattr(ocr_response, "pages", [])
    if not pages:
        raise ValueError("OCR‑ответ не содержит страниц.")
    md_parts: list[str] = []
    for page in pages:
        # page может быть dataclass или dict
        md_parts.append(page.markdown if hasattr(page, "markdown") else page["markdown"])
        md_parts.append("\n\n")  # разделитель между страницами
    return "".join(md_parts).strip()


def save_markdown(markdown: str, output_path: Path) -> None:
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ Markdown сохранён в: {output_path}")


# ─────────────────────────────── main  ─────────────────────────────


def process_pdf(
    pdf_path: Path,
    output_md: Path,
    method: Literal["auto", "base64", "upload"] = "auto",
    size_threshold: int = 5 * 1024 * 1024,
) -> None:
    """Основной рабочий цикл: подготовка, вызов OCR, запись Markdown."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("❌ Переменная MISTRAL_API_KEY не найдена.")

    client = Mistral(api_key=api_key)

    # ── выбор способа передачи документа ──────────────────────────
    use_upload = method == "upload" or (
        method == "auto" and pdf_path.stat().st_size > size_threshold
    )

    if use_upload:
        print("↑ Загрузка PDF в Mistral Storage…")
        with pdf_path.open("rb") as f:
            uploaded = client.files.upload(
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )
        signed_url = client.files.get_signed_url(file_id=uploaded.id).url
        doc_descriptor = {"type": "document_url", "document_url": signed_url}
    else:
        print("↻ Кодирование PDF в base64…")
        data_url = f"data:application/pdf;base64,{encode_pdf_to_base64(pdf_path)}"
        doc_descriptor = {"type": "document_url", "document_url": data_url}

    # ── запуск OCR ─────────────────────────────────────────────────
    print("🤖 Запуск OCR… (модель mistral-ocr-latest)")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document=doc_descriptor,
        include_image_base64=False,  # нам нужен только текст
    )

    # ── обработка результата ──────────────────────────────────────
    markdown_text = extract_markdown(ocr_response)
    save_markdown(markdown_text, output_md)


def cli() -> None:
    parser = argparse.ArgumentParser(description="PDF → Markdown через Mistral OCR")
    parser.add_argument("pdf", type=Path, help="Путь к PDF‑файлу")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Файл Markdown (по умолчанию <basename>.md)",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "base64", "upload"),
        default="auto",
        help="Стратегия передачи PDF в API",
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        sys.exit(f"❌ Файл не найден: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit("❌ Ожидается файл формата .pdf")

    output_md = args.output or pdf_path.with_suffix(".md")
    process_pdf(pdf_path, output_md, method=args.method)


if __name__ == "__main__":
    cli()