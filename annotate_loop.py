#!/usr/bin/env python
"""
CLI‑скрипт: «итеративный тест‑пайплайн аннотаций судебных решений».

— Принимает 3 Markdown‑файла с решениями.
— Генерирует аннотацию (o3) → самооценку (o4‑mini) → новый промпт (o3).
— Логирует все запросы/ответы в JSON.
— Складывает результаты в outputs/.
python annotate_loop.py docs/act1.md docs/act2.md docs/act3.md

python annotate_loop.py doc/doc_1.md doc/doc_2.md doc/doc_3.md
python annotate_loop.py doc/Приложение7.md doc/Приложение10.md doc/Приложение11.md

"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict

import typer
from openai import OpenAI
from tqdm import tqdm

# ──────────────────────────────── Константы ──────────────────────────────── #

LOG_DIR = Path("logs")
OUT_DIR = Path("outputs")
MODELS = {
    "annotation": "o3-mini",
    "review": "o4-mini",
    "optimization": "o3-mini",
}

# ──────────────────────────────── Utils ──────────────────────────────────── #


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def save_json(data: Any, stage: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fname = LOG_DIR / f"{utc_timestamp()}_{stage}.json"
    with fname.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    effort: str = "high",
    max_retries: int = 2,
) -> str:
    """Обёртка над OpenAI Response API (по умолчанию уровень рассуждений high) с простым повтором."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": effort},
                input=[{"role": "user", "content": prompt}],
            )
            return resp.output_text
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            typer.echo(f"LLM error: {exc}. Retry {attempt}/{max_retries}…", err=True)


# ──────────────────────────────── Основной поток ─────────────────────────── #


def main(
    doc_1: Path = typer.Argument(..., exists=True, readable=True),
    doc_2: Path = typer.Argument(..., exists=True, readable=True),
    doc_3: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    # Проверка ключа
    if not os.getenv("OPENAI_API_KEY"):
        typer.secho("✖ Переменная окружения OPENAI_API_KEY не установлена.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Импорт промптов (пользователь может их свободно менять)
    try:
        from prompt import (
            prompt_start,
            self_evaluation_prompt,
            process_optimization_prompt,
            legal_annotation_revision_prompt,
            legal_annotation_revision_prompt_end
            
        )
    except ImportError as exc:
        typer.secho(f"Не найден prompt.py: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    client = OpenAI()

    docs = [doc_1, doc_2, doc_3]
    doc_texts: List[str] = [p.read_text(encoding="utf-8") for p in docs]

    summaries: List[str] = []
    reviews: List[str] = []

    # 1) Аннотация
    annot_log: List[Dict[str, str]] = []
    for idx, (path, text) in enumerate(zip(docs, doc_texts), start=1):
        prompt = f"{prompt_start}\n---\n{text}"
        typer.echo(f"🔹 Генерация аннотации для {path.name}…")
        summary = call_llm(client, MODELS["annotation"], prompt)
        summaries.append(summary)
        write_text(OUT_DIR / f"summary_{idx}.md", summary)
        annot_log.append({"doc": path.name, "prompt": prompt, "response": summary})
    save_json(annot_log, "annotation")

    # 2) Самооценка
    review_log: List[Dict[str, str]] = []
    for idx, (path, text, summary) in enumerate(zip(docs, doc_texts, summaries), start=1):
        prompt = (
            f"{self_evaluation_prompt}\n---\nДокумент:\n{text}\n---\nАннотация:\n{summary}"
        )
        typer.echo(f"🔹 Самооценка аннотации {path.name}…")
        review = call_llm(client, MODELS["review"], prompt)
        reviews.append(review)
        write_text(OUT_DIR / f"review_{idx}.md", review)
        review_log.append({"doc": path.name, "prompt": prompt, "response": review})
    save_json(review_log, "review")

    # 3) Оптимизация промпта
    typer.echo("🔹 Генерация улучшенного промпта…")
    # Формируем большой единый prompt
    parts: List[str] = [process_optimization_prompt]
    for idx, (orig, summ, revw) in enumerate(zip(doc_texts, summaries, reviews), start=1):
        parts.append(
            f"### Документ {idx}\n{orig}\n---\nАннотация\n{summ}\n---\nЗамечания\n{revw}"
        )

    # Добавляем исходный промпт и явную инструкцию улучшить его
    parts.extend(
        [
            "Изучите версию промпта, на основании которой модель сформировала эти аннотации.",
            "Старая версия промпта:",
            prompt_start,
            "#####",
            "Проанализируйте всю информацию, подумайте и разработайте более эффективную подсказку‑промпт.",
        ]
    )

    # Используем более заметный разделитель между крупными блоками
    optimization_prompt = "\n---\n".join(parts)

    refined_prompt_text = call_llm(client, MODELS["optimization"], optimization_prompt)
    write_text(OUT_DIR / "prompt_refined.txt", refined_prompt_text)
    save_json(
        {
            "prompt": optimization_prompt,
            "response": refined_prompt_text,
        },
        "optimization",
    )

    # 4) Исправленная аннотация
    revision_summaries: List[str] = []
    revision_log: List[Dict[str, str]] = []
    for idx, (path, text, summary, review) in enumerate(
        zip(docs, doc_texts, summaries, reviews), start=1
    ):
        revision_prompt = (
            f"{legal_annotation_revision_prompt}\\n\\n####\\n"
            f"ИСХОДНЫЙ ЮРИДИЧЕСКИЙ ДОКУМЕНТ:\\n{text}\\n#####\\n"
            "ПЕРВОНАЧАЛЬНАЯ АННОТАЦИЯ: Черновик с ошибками, который нужно "
            "переработать.:\\n\\n"
            f"{summary}\\n\\n####\\n"
            "КРИТИЧЕСКИЕ ЗАМЕЧАНИЯ РУКОВОДИТЕЛЯ: Список конкретных недостатков, "
            "которые необходимо устранить.:\\n\\n"
            f"{review}\\n\\n####\\n"
            f"{legal_annotation_revision_prompt_end}\\n\\n####\\n"
            
        )
        typer.echo(f"🔹 Исправленная аннотация для {path.name}…")
        revised = call_llm(client, MODELS["annotation"], revision_prompt)
        revision_summaries.append(revised)
        # сохраняем под двумя понятными именами
        write_text(OUT_DIR / f"revision_{idx}.md", revised)
        write_text(OUT_DIR / f"summary2_{idx}.md", revised)
        revision_log.append(
            {
                "doc": path.name,
                "prompt": revision_prompt,
                "response": revised,
            }
        )
    save_json(revision_log, "revision")

    # 5) Второй раунд самооценки
    second_review_log: List[Dict[str, str]] = []
    for idx, (path, text, revised) in enumerate(
        zip(docs, doc_texts, revision_summaries), start=1
    ):
        review2_prompt = (
            f"{self_evaluation_prompt}\\n---\\nДокумент:\\n{text}\\n---\\nАннотация:\\n{revised}"
        )
        typer.echo(f"🔹 Повторное review {path.name}…")
        review2 = call_llm(client, MODELS["review"], review2_prompt)
        write_text(OUT_DIR / f"review2_{idx}.md", review2)
        second_review_log.append(
            {
                "doc": path.name,
                "prompt": review2_prompt,
                "response": review2,
            }
        )
    save_json(second_review_log, "second_review")

    typer.secho("✅ Готово! Все результаты — в каталоге outputs/.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        sys.exit("Прервано пользователем.")