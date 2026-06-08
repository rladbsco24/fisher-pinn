from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def _set_run_font(run, *, size: float | None = None, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = "Malgun Gothic"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def _set_paragraph_spacing(paragraph, *, before: float = 0, after: float = 6, line: float = 1.28) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def _add_bottom_border(paragraph, color: str = "D9D9D9") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def _add_paragraph(document: Document, text: str = "", *, style: str | None = None):
    paragraph = document.add_paragraph(style=style)
    if text:
        run = paragraph.add_run(text)
        _set_run_font(run, size=10.5)
    _set_paragraph_spacing(paragraph)
    return paragraph


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.95)
    section.right_margin = Inches(0.95)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Malgun Gothic"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    for style_name, size, bold, before, after in [
        ("Heading 1", 14, True, 12, 8),
        ("Heading 2", 12, True, 10, 4),
        ("Heading 3", 10.5, True, 8, 2),
    ]:
        style = styles[style_name]
        style.font.name = "Malgun Gothic"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.18


def _write_inline_markdown(paragraph, text: str, *, size: float = 10.5, bold: bool = False) -> None:
    # Minimal inline handling for backtick code spans; keep the parser deterministic.
    parts = text.split("`")
    for idx, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        _set_run_font(run, size=size, bold=bold)
        if idx % 2 == 1:
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            run.font.size = Pt(size - 0.5)
            run.font.color.rgb = RGBColor(70, 70, 70)


def build_docx(markdown_path: Path, output_path: Path) -> None:
    document = Document()
    _configure_document(document)
    lines = markdown_path.read_text(encoding="utf-8").splitlines()

    in_code = False
    code_buffer: list[str] = []
    title_written = False
    metadata_lines: list[str] = []

    def flush_code() -> None:
        nonlocal code_buffer
        if not code_buffer:
            return
        paragraph = document.add_paragraph()
        _set_paragraph_spacing(paragraph, before=3, after=7, line=1.0)
        run = paragraph.add_run("\n".join(code_buffer))
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        run.font.size = Pt(9.0)
        run.font.color.rgb = RGBColor(55, 55, 55)
        code_buffer = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_buffer = []
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if not line.strip():
            continue

        if line.startswith("# "):
            title = line[2:].strip()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_paragraph_spacing(paragraph, before=72, after=30, line=1.0)
            run = paragraph.add_run(title)
            _set_run_font(run, size=17, bold=True)
            title_written = True
            continue

        if title_written and (line.startswith("작성:") or line.startswith("일자:")):
            metadata_lines.append(line)
            if len(metadata_lines) == 2:
                for meta in metadata_lines:
                    paragraph = document.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    _set_paragraph_spacing(paragraph, after=2, line=1.0)
                    run = paragraph.add_run(meta)
                    _set_run_font(run, size=10.5)
                document.add_paragraph()
            continue

        if line.startswith("#### "):
            paragraph = document.add_paragraph(style="Heading 3")
            _write_inline_markdown(paragraph, line[5:].strip(), size=10.5, bold=True)
            continue

        if line.startswith("### "):
            paragraph = document.add_paragraph(style="Heading 2")
            _write_inline_markdown(paragraph, line[4:].strip(), size=11.2, bold=True)
            continue

        if line.startswith("## "):
            paragraph = document.add_paragraph(style="Heading 1")
            _write_inline_markdown(paragraph, line[3:].strip(), size=13.5, bold=True)
            if line[3:].strip() == "개론":
                _add_bottom_border(paragraph)
            continue

        if line.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.first_line_indent = Inches(-0.12)
            _set_paragraph_spacing(paragraph, after=3)
            _write_inline_markdown(paragraph, line[2:].strip(), size=10.2)
            continue

        paragraph = document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Inches(0.18)
        _set_paragraph_spacing(paragraph, after=7, line=1.35)
        _write_inline_markdown(paragraph, line, size=10.5)

    flush_code()

    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.text = ""
    run = footer.add_run("Fisher-KPP PINN 산림청 적용 검토의견 회신서")
    _set_run_font(run, size=8.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Fisher-KPP PINN review-response DOCX.")
    parser.add_argument("--input", type=Path, default=Path("docs") / "fisher_kpp_pinn_review_response.md")
    parser.add_argument("--output", type=Path, default=Path("docs") / "fisher_kpp_pinn_review_response.docx")
    args = parser.parse_args()
    build_docx(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
