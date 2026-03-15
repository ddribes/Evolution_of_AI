from __future__ import annotations

import os
import re
import base64
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# Prompt templates
# -----------------------------
CRITIQUE_TEMPLATE = '''
You are a demanding but constructive icon design studio critic.

ICON INTENDED MEANING
{icon_meaning}

Use the design context below when relevant.

DESIGN CONTEXT
{design_context}

Analyze whether the icon clearly communicates the intended meaning.

Focus on:
- clarity of the intended meaning
- likelihood that users interpret the icon correctly
- legibility at small size
- silhouette distinctiveness
- visual complexity
- concreteness versus abstraction
- perceived aesthetics
- balance of positive and negative space
- consistency of stroke, geometry, and spacing
- communicative precision
- coherence of the icon design language

Return your answer in this exact structure:

Meaning clarity:
<Does the icon communicate the intended meaning?>

Possible misinterpretations:
- ...
- ...

Overall impression:
<2 to 4 sentences>

Strengths:
- ...
- ...
- ...

Weaknesses:
- ...
- ...
- ...

Design scorecard:
- Meaning clarity: X/10
- Legibility: X/10
- Aesthetic appeal: X/10
- Complexity control: X/10
- Concreteness / abstraction: X/10
- System coherence: X/10

Next design moves:
- ...
- ...
- ...

Be concrete and avoid generic praise.
'''.strip()

COMPARISON_TEMPLATE = '''
You are reviewing two iterations of an icon design.

ICON INTENDED MEANING
{icon_meaning}

Use the design context below when relevant.

DESIGN CONTEXT
{design_context}

The first image is the earlier version.
The second image is the newer version.

Compare them as a design evolution.

Return your answer in this exact structure:

What improved:
- ...
- ...
- ...

What became weaker or less clear:
- ...
- ...
- ...

Design evolution:
<3 to 6 sentences>

Priority for the next iteration:
- ...
- ...
- ...

Be specific about legibility, visual complexity, abstraction, aesthetics, and communicative clarity.
'''.strip()

CONFUSION_TEST_TEMPLATE = '''
You are evaluating an icon as if you were a first-time user who has not been told its intended meaning.

Use the design context below when relevant.

DESIGN CONTEXT
{design_context}

Infer what the icon most likely means.

Return your answer in this exact structure:

Top 3 possible meanings:
1. ...
2. ...
3. ...

Best guess:
<1 short sentence>

Confidence in best guess:
<X/10>

Visual evidence used for the guess:
- ...
- ...
- ...

Potential sources of confusion:
- ...
- ...
- ...

Clarity assessment:
<2 to 4 sentences>

Do not assume the designer's intention. Base your answer only on the icon itself.
'''.strip()

MEANING_MATCH_TEMPLATE = '''
You are comparing an icon's predicted meaning with its intended meaning.

INTENDED MEANING
{icon_meaning}

MODEL-PREDICTED MEANING
{predicted_meaning}

ALTERNATIVE POSSIBLE MEANINGS
{alternative_meanings}

CLARITY ASSESSMENT
{clarity_assessment}

Return your answer in this exact structure:

Meaning match:
<good match / partial match / weak match>

Where the icon succeeds:
- ...
- ...
- ...

Where the icon fails or risks misunderstanding:
- ...
- ...
- ...

Recommendation:
<2 to 4 sentences explaining what should change to make the meaning clearer>
'''.strip()

FINAL_REVIEW_TEMPLATE = '''
You are a senior design tutor reviewing the progression of a student's icon design project.

ICON INTENDED MEANING
{icon_meaning}

Use the design context below when relevant.

DESIGN CONTEXT
{design_context}

INDIVIDUAL CRITIQUES
{individual_critiques}

VERSION COMPARISONS
{comparisons}

Write a final studio review in this exact structure:

Main direction of the project:
<short paragraph>

Most successful qualities:
- ...
- ...
- ...

Recurring issues:
- ...
- ...
- ...

Design evolution timeline:
- Version 1: ...
- Version 2: ...
- Version 3: ...
- Continue as needed

How well does the project control aesthetics, complexity, and concreteness?
<short paragraph>

Is the project converging toward a coherent icon language?
<short paragraph>

Recommended next steps:
- ...
- ...
- ...

Overall studio scorecard:
- Aesthetic quality: X/10
- Complexity control: X/10
- Concreteness control: X/10
- Meaning clarity: X/10
- System coherence: X/10
- Iteration quality: X/10
'''.strip()

DEFAULT_QUERY = (
    "Icon design theory about aesthetics, visual complexity, concreteness, icon comprehension, "
    "legibility, and consistency in icon systems."
)

# -----------------------------
# Utilities
# -----------------------------
def normalize_uploaded_files(files: Any) -> list[str]:
    if not files:
        return []
    out: list[str] = []
    for f in files:
        if isinstance(f, str):
            out.append(f)
        elif hasattr(f, "name"):
            out.append(f.name)
        elif isinstance(f, dict) and "name" in f:
            out.append(f["name"])
    return out


def image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in ["jpg", "jpeg"] else suffix or "png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def extract_text_from_pdfs(pdf_paths: list[str]) -> str:
    pages: list[str] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(book_text: str):
    chunks = chunk_text(book_text, chunk_size=1200, overlap=200)
    if not chunks:
        raise gr.Error("No text could be extracted from the PDF(s).")
    embedder = get_embedder()
    chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(query: str, k: int = 4) -> list[dict[str, Any]]:
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(chunk_embeddings, query_embedding)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            {
                "rank": rank,
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "text": chunks[idx],
            }
            for rank, idx in enumerate(top_idx, start=1)
        ]

    return retrieve


def format_context(retrieved: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        [f"[Chunk {r['chunk_id']} | score={r['score']:.3f}]\n{r['text']}" for r in retrieved]
    )


def build_design_query_for_single_icon(filename: str, icon_meaning: str) -> str:
    return f"""
I am trying to make an icon with this intended meaning:
{icon_meaning}

{DEFAULT_QUERY}
Filename: {filename}
""".strip()


def build_design_query_for_comparison(prev_name: str, curr_name: str, icon_meaning: str) -> str:
    return f"""
Compare two icon iterations with this intended meaning:
{icon_meaning}

{DEFAULT_QUERY}
Earlier version: {prev_name}
Newer version: {curr_name}
""".strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def call_vlm(client: InferenceClient, messages: list[dict[str, Any]], model_id: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def parse_section(text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\s*:\s*(.*?)(?=\n[A-Z][^\n]*:|\Z)"
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else text.strip()


def make_gallery(image_paths: list[str]) -> list[tuple[str, str]]:
    gallery = []
    for p in image_paths:
        gallery.append((p, Path(p).name))
    return gallery


def create_report_file(
    image_paths: list[str],
    icon_meaning: str,
    confusion_result: str,
    meaning_match_result: str,
    individual_results: list[dict[str, Any]],
    comparison_results: list[dict[str, Any]],
    final_review: str,
) -> str:
    report_name = safe_filename(f"icon_design_critic_report_{Path(image_paths[-1]).stem}.md")
    report_path = str(Path(tempfile.gettempdir()) / report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Icon Design Critic Report\n\n")
        f.write("## Intended Meaning\n\n")
        f.write(icon_meaning.strip() + "\n\n")
        f.write("## Confusion Test\n\n")
        f.write(confusion_result + "\n\n")
        f.write("## Meaning Match Review\n\n")
        f.write(meaning_match_result + "\n\n")
        f.write("## Individual Critiques\n\n")
        for r in individual_results:
            f.write(f"### {r['image']}\n\n{r['critique']}\n\n")
        f.write("## Version Comparisons\n\n")
        for r in comparison_results:
            f.write(f"### {r['pair']}\n\n{r['comparison']}\n\n")
        f.write("## Final Studio Review\n\n")
        f.write(final_review + "\n")
    return report_path


def run_analysis(
    hf_token: str,
    model_id: str,
    pdf_files: Any,
    icon_files: Any,
    icon_meaning: str,
    top_k: int,
    max_tokens: int,
    temperature: float,
    run_comparisons: bool,
) -> tuple[Any, ...]:
    if not hf_token.strip():
        raise gr.Error("Please enter a Hugging Face token.")

    pdf_paths = normalize_uploaded_files(pdf_files)
    icon_paths = sorted(normalize_uploaded_files(icon_files), key=lambda p: Path(p).name)

    if not pdf_paths:
        # Try local default if present
        default_pdf = "/mnt/data/Collaud-2022-Design_Standards.pdf"
        if Path(default_pdf).exists():
            pdf_paths = [default_pdf]
        else:
            raise gr.Error("Please upload at least one PDF with design context.")

    if not icon_paths:
        raise gr.Error("Please upload at least one icon image.")

    if not icon_meaning.strip():
        raise gr.Error("Please provide the intended meaning of the icon.")

    client = InferenceClient(provider="auto", api_key=hf_token.strip())
    book_text = extract_text_from_pdfs(pdf_paths)
    retrieve_design_context = build_retriever(book_text)

    # Confusion test on the latest icon
    confusion_retrieved = retrieve_design_context(DEFAULT_QUERY, k=top_k)
    confusion_prompt = CONFUSION_TEST_TEMPLATE.format(design_context=format_context(confusion_retrieved))
    test_image = icon_paths[-1]
    confusion_messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": confusion_prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(test_image)}},
        ],
    }]
    confusion_result = call_vlm(client, confusion_messages, model_id, max_tokens=max_tokens, temperature=temperature)

    predicted_meaning = parse_section(confusion_result, "Best guess")
    alternative_meanings = parse_section(confusion_result, "Top 3 possible meanings")
    clarity_assessment = parse_section(confusion_result, "Clarity assessment")

    meaning_match_prompt = MEANING_MATCH_TEMPLATE.format(
        icon_meaning=icon_meaning,
        predicted_meaning=predicted_meaning,
        alternative_meanings=alternative_meanings,
        clarity_assessment=clarity_assessment,
    )
    meaning_match_result = call_vlm(
        client,
        [{"role": "user", "content": [{"type": "text", "text": meaning_match_prompt}]}],
        model_id,
        max_tokens=min(max_tokens, 700),
        temperature=0.2,
    )

    # Individual critiques
    individual_results: list[dict[str, Any]] = []
    individual_md_parts: list[str] = []
    summary_rows: list[dict[str, str]] = []
    for path in icon_paths:
        retrieved = retrieve_design_context(build_design_query_for_single_icon(Path(path).name, icon_meaning), k=top_k)
        prompt = CRITIQUE_TEMPLATE.format(
            icon_meaning=icon_meaning,
            design_context=format_context(retrieved),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}},
            ],
        }]
        critique = call_vlm(client, messages, model_id, max_tokens=max_tokens, temperature=temperature)
        result = {"image": Path(path).name, "retrieved": retrieved, "critique": critique}
        individual_results.append(result)
        summary_rows.append({"image": Path(path).name, "critique": critique})
        individual_md_parts.append(f"## Critique — {Path(path).name}\n\n{critique}")

    # Comparisons
    comparison_results: list[dict[str, Any]] = []
    comparison_md_parts: list[str] = []
    if run_comparisons and len(icon_paths) > 1:
        for i in range(1, len(icon_paths)):
            prev_path = icon_paths[i - 1]
            curr_path = icon_paths[i]
            retrieved = retrieve_design_context(
                build_design_query_for_comparison(Path(prev_path).name, Path(curr_path).name, icon_meaning),
                k=top_k,
            )
            prompt = COMPARISON_TEMPLATE.format(
                icon_meaning=icon_meaning,
                design_context=format_context(retrieved),
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(prev_path)}},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(curr_path)}},
                ],
            }]
            comparison = call_vlm(client, messages, model_id, max_tokens=max_tokens, temperature=temperature)
            pair_label = f"{Path(prev_path).name} → {Path(curr_path).name}"
            result = {"pair": pair_label, "retrieved": retrieved, "comparison": comparison}
            comparison_results.append(result)
            comparison_md_parts.append(f"## Comparison — {pair_label}\n\n{comparison}")

    individual_text = "\n\n".join([f"Icon iteration: {r['image']}\n{r['critique']}" for r in individual_results])
    comparison_text = "\n\n".join([f"Comparison: {r['pair']}\n{r['comparison']}" for r in comparison_results]) or "No comparisons were run."

    final_retrieved = retrieve_design_context(
        (
            "Overall review of a student icon design project across iterations, focusing on icon aesthetics, "
            "visual complexity, concreteness, communicative clarity, and design consistency. "
            f"Intended meaning: {icon_meaning}"
        ),
        k=top_k,
    )

    final_prompt = FINAL_REVIEW_TEMPLATE.format(
        icon_meaning=icon_meaning,
        design_context=format_context(final_retrieved),
        individual_critiques=individual_text,
        comparisons=comparison_text,
    )
    final_messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(icon_paths[-1])}},
        ],
    }]
    final_review = call_vlm(client, final_messages, model_id, max_tokens=min(max_tokens + 300, 1400), temperature=0.3)

    report_path = create_report_file(
        icon_paths,
        icon_meaning,
        confusion_result,
        meaning_match_result,
        individual_results,
        comparison_results,
        final_review,
    )

    gallery = make_gallery(icon_paths)
    summary_df = pd.DataFrame(summary_rows)

    return (
        gallery,
        confusion_result,
        meaning_match_result,
        "\n\n".join(individual_md_parts),
        "\n\n".join(comparison_md_parts) if comparison_md_parts else "No comparisons were run.",
        final_review,
        summary_df,
        report_path,
    )


with gr.Blocks(title="Icon Design Critic Agent") as demo:
    gr.Markdown(
        "# Icon Design Critic Agent\n\n"
        "Upload a design paper and one or more icon iterations. The app retrieves relevant design context from the PDF, "
        "runs a confusion test, critiques each icon, compares iterations, and generates a final studio review."
    )

    with gr.Row():
        with gr.Column(scale=1):
            hf_token = gr.Textbox(label="Hugging Face token", type="password", placeholder="hf_...")
            model_id = gr.Textbox(label="Model ID", value="Qwen/Qwen3-VL-8B-Instruct")
            pdf_files = gr.Files(label="Design paper PDF(s)", file_types=[".pdf"])
            icon_files = gr.Files(label="Icon image(s)", file_types=["image"])
            icon_meaning = gr.Textbox(
                label="Intended meaning",
                lines=4,
                value="The icon represents: Upload file to cloud storage.\n\nExpected user interpretation:\nA file is being sent from a local device to a cloud service.",
            )
            with gr.Row():
                top_k = gr.Slider(1, 6, value=4, step=1, label="Retrieved chunks")
                max_tokens = gr.Slider(300, 1400, value=900, step=50, label="Max output tokens")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Temperature")
                run_comparisons = gr.Checkbox(value=True, label="Compare consecutive iterations")
            run_btn = gr.Button("Run analysis", variant="primary")

        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Uploaded icons", columns=3, height=260)
            summary_df = gr.Dataframe(label="Summary table")
            report_file = gr.File(label="Download report")

    with gr.Tabs():
        with gr.Tab("Confusion test"):
            confusion_md = gr.Markdown()
        with gr.Tab("Meaning match"):
            meaning_match_md = gr.Markdown()
        with gr.Tab("Individual critiques"):
            individual_md = gr.Markdown()
        with gr.Tab("Comparisons"):
            comparisons_md = gr.Markdown()
        with gr.Tab("Final review"):
            final_review_md = gr.Markdown()

    run_btn.click(
        fn=run_analysis,
        inputs=[hf_token, model_id, pdf_files, icon_files, icon_meaning, top_k, max_tokens, temperature, run_comparisons],
        outputs=[gallery, confusion_md, meaning_match_md, individual_md, comparisons_md, final_review_md, summary_df, report_file],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
