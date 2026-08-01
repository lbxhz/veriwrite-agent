"""Assemble, audit, confirm, and export the final MVP paper."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from io import BytesIO
from math import ceil

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.final_delivery import (
    FinalMatterProposal,
    FinalPaperAudit,
    FinalPaperAuditIssue,
    FinalPaperPackage,
    FinalReferenceEntry,
)
from veriwrite_agent.models.writing import BodyDraftPackage
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.requirement_policy import (
    RequirementPolicyCompiler,
    ai_generation_prohibitions,
    source_restriction_reasons,
)


class FinalDeliveryError(ValueError):
    """Raised when final delivery would violate a confirmed contract."""


class LLMFinalMatterWriter:
    """Generate title, abstract, keywords, and conclusion from the confirmed body."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def draft(
        self,
        handoff: V04WritingHandoff,
        body: BodyDraftPackage,
    ) -> FinalMatterProposal:
        policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
            handoff.requirement
        )
        prohibitions = ai_generation_prohibitions(policy)
        if prohibitions:
            raise FinalDeliveryError(
                "AI final-matter generation is prohibited: " + "; ".join(prohibitions)
            )
        schema = json.dumps(
            FinalMatterProposal.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON only. Using only the confirmed body, write a final "
                        "paper title, abstract, 3-8 keywords, and conclusion. Do not add "
                        "citations, DOI values, papers, methods, results, numbers, or claims "
                        "that are absent from the body. The conclusion must synthesize the "
                        "body rather than introduce new evidence. Use the requested output "
                        f"language: {policy.output_language}. Satisfy this schema: {schema}"
                    ),
                },
                {"role": "user", "content": body.markdown},
            ],
            response_format={"type": "json_object"},
        )
        try:
            return FinalMatterProposal.model_validate_json(raw)
        except ValidationError as exc:
            raise FinalDeliveryError(
                f"LLM final matter violates the data contract: {exc.errors(include_url=False)[:8]}"
            ) from exc


class FinalPaperAssembler:
    """Build citations and bibliography in code, then run the release audit."""

    def assemble(
        self,
        *,
        handoff: V04WritingHandoff,
        body: BodyDraftPackage,
        final_matter: FinalMatterProposal,
        ai_declaration: str | None = None,
    ) -> FinalPaperPackage:
        policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
            handoff.requirement
        )
        records = {record.doi: record for record in handoff.evidence_library.records}
        missing_dois = [doi for doi in body.source_dois if doi not in records]
        if missing_dois:
            raise FinalDeliveryError(
                "body citations are missing from the evidence library: " + ", ".join(missing_dois)
            )
        citation_key_to_doi = {binding.citation_key: binding.doi for binding in body.citations}
        ordered_dois = list(dict.fromkeys(body.source_dois))
        references = [
            _reference_entry(
                records[doi],
                index=index,
                citation_key=next(
                    key for key, value in citation_key_to_doi.items() if value == doi
                ),
                bibliography_style=policy.references.bibliography_style,
                numeric=_numeric_citations(policy),
            )
            for index, doi in enumerate(ordered_dois, 1)
        ]
        reference_by_key = {entry.citation_key: entry for entry in references}
        transformed_body, unknown_keys = _render_final_citations(
            body.markdown,
            reference_by_key,
            numeric=_numeric_citations(policy),
        )
        transformed_body = _remove_leading_title(transformed_body)
        markdown = _assemble_markdown(
            final_matter,
            transformed_body,
            references,
            ai_declaration=ai_declaration,
            output_language=policy.output_language,
        )
        audit = _audit_final_paper(
            policy=policy,
            handoff=handoff,
            body=body,
            final_matter=final_matter,
            references=references,
            ai_declaration=ai_declaration,
            unknown_citation_keys=unknown_keys,
            markdown=markdown,
        )
        return FinalPaperPackage(
            status=("needs_revision" if audit.blocking_count else "ready_for_confirmation"),
            requirement_policy=policy,
            title=final_matter.title,
            abstract=final_matter.abstract,
            keywords=final_matter.keywords,
            body_markdown=transformed_body,
            conclusion=final_matter.conclusion,
            ai_declaration=ai_declaration,
            references=references,
            markdown=markdown,
            audit=audit,
        )

    def confirm(
        self,
        package: FinalPaperPackage,
        *,
        confirmed_by: str,
    ) -> FinalPaperPackage:
        if package.status != "ready_for_confirmation":
            raise FinalDeliveryError("final paper still has blocking audit issues")
        name = confirmed_by.strip()
        if not name:
            raise FinalDeliveryError("confirmed_by cannot be blank")
        return package.model_copy(
            update={
                "status": "confirmed",
                "confirmed_by": name,
                "confirmed_at": datetime.now(timezone.utc),
            }
        )


class FinalPaperDocxExporter:
    """Export a confirmed package using the narrative_proposal base preset."""

    def export(self, package: FinalPaperPackage) -> bytes:
        if package.status != "confirmed":
            raise FinalDeliveryError("DOCX export requires a confirmed final paper")
        try:
            from docx import Document
            from docx.enum.section import WD_SECTION_START
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            from docx.shared import Inches, Mm, Pt, RGBColor
        except ImportError as exc:
            raise FinalDeliveryError("python-docx is required for DOCX export") from exc

        policy = package.requirement_policy
        document = Document()
        section = document.sections[0]
        _ = WD_SECTION_START
        if policy.formatting.paper_size and "letter" in policy.formatting.paper_size.casefold():
            section.page_width = Inches(8.5)
            section.page_height = Inches(11)
        else:
            section.page_width = Mm(210)
            section.page_height = Mm(297)
        section.top_margin = Inches(1)
        section.right_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.header_distance = Inches(0.492)
        section.footer_distance = Inches(0.492)

        body_font = policy.formatting.body_font or (
            "SimSun" if policy.output_language in {"Chinese", "bilingual"} else "Times New Roman"
        )
        body_size = _font_size(policy.formatting.body_font_size)
        line_spacing = policy.formatting.line_spacing or 1.333
        normal = document.styles["Normal"]
        normal.font.name = body_font
        normal.font.size = Pt(body_size)
        normal._element.rPr.rFonts.set(qn("w:ascii"), body_font)
        normal._element.rPr.rFonts.set(qn("w:hAnsi"), body_font)
        normal._element.rPr.rFonts.set(qn("w:eastAsia"), body_font)
        normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        normal.paragraph_format.space_before = Pt(0)
        normal.paragraph_format.space_after = Pt(8)
        normal.paragraph_format.line_spacing = line_spacing

        heading_tokens = {
            "Heading 1": (16, "2E74B5", 18, 10),
            "Heading 2": (13, "2E74B5", 12, 6),
            "Heading 3": (12, "1F4D78", 8, 4),
        }
        for name, (size, color, before, after) in heading_tokens.items():
            style = document.styles[name]
            style.font.name = body_font
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor.from_string(color)
            style._element.rPr.rFonts.set(qn("w:eastAsia"), body_font)
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)

        document.core_properties.title = package.title
        document.core_properties.subject = (
            "VeriWrite final paper; base preset narrative_proposal; "
            "named override academic_course_paper"
        )
        title = document.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_after = Pt(12)
        title_run = title.add_run(package.title)
        _set_run_font(title_run, body_font, 18, bold=True)

        english = policy.output_language == "English"
        _add_section(document, "Abstract" if english else "摘要", package.abstract, level=1)
        keyword_paragraph = document.add_paragraph()
        keyword_run = keyword_paragraph.add_run("Keywords: " if english else "关键词：")
        _set_run_font(keyword_run, body_font, body_size, bold=True)
        keyword_values = keyword_paragraph.add_run(
            (", " if english else "；").join(package.keywords)
        )
        _set_run_font(keyword_values, body_font, body_size)
        _append_markdown(document, package.body_markdown, body_font, body_size)
        _add_section(
            document,
            "Conclusion" if english else "结论",
            package.conclusion,
            level=1,
        )
        document.add_heading("References" if english else "参考文献", level=1)
        for entry in package.references:
            paragraph = document.add_paragraph(entry.formatted_text)
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        if package.ai_declaration:
            _add_section(
                document,
                "AI Usage Declaration" if english else "AI 使用声明",
                package.ai_declaration,
                level=1,
            )

        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = footer.add_run()
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        run._r.append(field)

        output = BytesIO()
        document.save(output)
        return output.getvalue()


def _audit_final_paper(
    *,
    policy,
    handoff: V04WritingHandoff,
    body: BodyDraftPackage,
    final_matter: FinalMatterProposal,
    references: list[FinalReferenceEntry],
    ai_declaration: str | None,
    unknown_citation_keys: list[str],
    markdown: str,
) -> FinalPaperAudit:
    issues: list[FinalPaperAuditIssue] = []
    counted = body.counted_words
    if policy.length.minimum_units is not None and counted < policy.length.minimum_units:
        issues.append(
            _issue(
                "length_below_minimum",
                "blocking",
                "length",
                f"required >= {policy.length.minimum_units}; actual={counted}",
            )
        )
    if policy.length.maximum_units is not None and counted > policy.length.maximum_units:
        issues.append(
            _issue(
                "length_above_maximum",
                "blocking",
                "length",
                f"required <= {policy.length.maximum_units}; actual={counted}",
            )
        )
    if len(references) < policy.references.minimum_total:
        issues.append(
            _issue(
                "reference_count_below_minimum",
                "blocking",
                "references.minimum_total",
                f"required={policy.references.minimum_total}; actual={len(references)}",
            )
        )
    elif len(references) < policy.references.target_total:
        issues.append(
            _issue(
                "reference_count_below_target",
                (
                    "warning"
                    if policy.references.target_is_approximate
                    else "blocking"
                ),
                "references.target_total",
                (
                    f"target={policy.references.target_total}; "
                    f"actual={len(references)}; "
                    f"origin={policy.references.target_origin}"
                ),
            )
        )
    foreign_count = sum(entry.is_foreign for entry in references)
    required_foreign = (
        ceil(len(references) * policy.references.minimum_foreign_ratio)
        if policy.references.minimum_foreign_ratio is not None
        else None
    )
    if required_foreign is not None and foreign_count < required_foreign:
        issues.append(
            _issue(
                "foreign_ratio_below_minimum",
                "blocking",
                "references.minimum_foreign_ratio",
                f"required_count={required_foreign}; actual={foreign_count}",
            )
        )
    if policy.references.year_from is not None:
        outside = [
            entry.doi
            for entry in references
            if entry.year < policy.references.year_from
            or (policy.references.year_to is not None and entry.year > policy.references.year_to)
        ]
        if outside:
            issues.append(
                _issue(
                    "hard_year_window_violation",
                    "blocking",
                    "references.recent_year_window",
                    ", ".join(outside),
                )
            )
    for entry in references:
        reasons = source_restriction_reasons(
            policy, publisher=entry.publisher, journal=entry.journal, source_type=entry.source_type
        )
        if reasons:
            issues.append(
                _issue(
                    "source_restriction_violation",
                    "blocking",
                    "references.restriction_rules",
                    f"{entry.doi}: {', '.join(reasons)}",
                )
            )
    if unknown_citation_keys:
        issues.append(
            _issue(
                "unknown_citation_key", "blocking", "references", ", ".join(unknown_citation_keys)
            )
        )
    cluster_limit = policy.references.max_references_per_citation_cluster
    if cluster_limit is not None:
        clusters: dict[tuple[str, int], set[str]] = {}
        for citation in body.citations:
            clusters.setdefault((citation.section_id, citation.paragraph_number), set()).add(
                citation.doi
            )
        excessive = [
            f"{section}:{paragraph}={len(dois)}"
            for (section, paragraph), dois in clusters.items()
            if len(dois) > cluster_limit
        ]
        if excessive:
            issues.append(
                _issue(
                    "citation_cluster_too_large",
                    "blocking",
                    "references.max_references_per_citation_cluster",
                    ", ".join(excessive),
                )
            )
    required_sections = policy.structure.required_sections
    for section in required_sections:
        if not _section_present(section, markdown):
            issues.append(
                _issue(
                    "required_section_missing", "blocking", "structure.required_sections", section
                )
            )
    for deliverable in policy.deliverables:
        normalized_deliverable = deliverable.casefold()
        deliverable_missing = (
            ("正文" in normalized_deliverable or "body" in normalized_deliverable)
            and not body.markdown.strip()
        ) or (
            (
                "参考文献" in normalized_deliverable
                or "reference" in normalized_deliverable
            )
            and not references
        )
        if deliverable_missing:
            issues.append(
                _issue(
                    "required_deliverable_missing",
                    "blocking",
                    "deliverables",
                    deliverable,
                )
            )
    compact_markdown = re.sub(r"\s+", "", markdown.casefold())
    for theme_element in policy.required_theme_elements:
        normalized_element = re.sub(r"\s+", "", theme_element.casefold())
        if normalized_element and normalized_element not in compact_markdown:
            issues.append(
                _issue(
                    "theme_element_requires_user_review",
                    "warning",
                    "required_theme_elements",
                    theme_element,
                )
            )
    if policy.structure.must_include_original_analysis:
        issues.append(
            _issue(
                "original_analysis_requires_user_review",
                "warning",
                "structure.must_include_original_analysis",
                "User must confirm that synthesis contains original analysis.",
            )
        )
    if policy.references.required_management_tools:
        issues.append(
            _issue(
                "reference_tool_usage_requires_attestation",
                "warning",
                "references.required_management_tools",
                ", ".join(policy.references.required_management_tools),
            )
        )
    if policy.ai_usage.declaration_required and not ai_declaration:
        issues.append(
            _issue(
                "ai_declaration_missing",
                "blocking",
                "ai_policy.declaration_required",
                "A confirmed AI usage declaration is required.",
            )
        )
    for unresolved in policy.unresolved_requirements:
        severity = (
            "blocking"
            if (
                "output_language" in unresolved
                or "bibliography_style" in unresolved
                or unresolved.startswith("hard_rule_not_fully_executable:")
            )
            else "warning"
        )
        issues.append(
            _issue("unresolved_requirement", severity, "unresolved_requirements", unresolved)
        )
    if policy.output_language == "pending_confirmation":
        issues.append(
            _issue(
                "output_language_unconfirmed",
                "blocking",
                "output_language",
                "Final document language is not confirmed.",
            )
        )
    return FinalPaperAudit(
        policy_fingerprint=policy.requirement_fingerprint,
        counted_units=counted,
        reference_count=len(references),
        foreign_reference_count=foreign_count,
        issues=issues,
    )


def _issue(code: str, severity: str, path: str, detail: str) -> FinalPaperAuditIssue:
    return FinalPaperAuditIssue(code=code, severity=severity, requirement_path=path, detail=detail)


def _reference_entry(
    record, *, index: int, citation_key: str, bibliography_style: str, numeric: bool
) -> FinalReferenceEntry:
    authors = record.authors or ["Anonymous"]
    author_text = ", ".join(authors)
    journal = record.journal or record.publisher or "Unknown source"
    doi_url = f"https://doi.org/{record.doi}"
    if "gb/t" in bibliography_style.casefold() or "7714" in bibliography_style:
        formatted = f"[{index}] {author_text}. {record.title}[J]. {journal}, {record.year}. DOI:{record.doi}."
    elif numeric:
        formatted = f"[{index}] {author_text}. {record.title}. {journal}. {record.year}. {doi_url}."
    else:
        formatted = f"{author_text} ({record.year}). {record.title}. {journal}. {doi_url}."
    return FinalReferenceEntry(
        citation_key=citation_key,
        index=index,
        doi=record.doi,
        authors=record.authors,
        year=record.year,
        title=record.title,
        journal=record.journal,
        publisher=record.publisher,
        source_type=record.source_type,
        is_foreign=record.is_foreign,
        formatted_text=formatted,
    )


def _numeric_citations(policy) -> bool:
    return (
        policy.references.in_text_style == "numeric_superscript"
        or "gb/t" in policy.references.bibliography_style.casefold()
        or "7714" in policy.references.bibliography_style
    )


def _render_final_citations(
    markdown: str, references: dict[str, FinalReferenceEntry], *, numeric: bool
) -> tuple[str, list[str]]:
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        rendered: list[str] = []
        for raw in match.group(1).split(";"):
            item = raw.strip()
            parsed = re.fullmatch(r"@([a-z0-9_]+)(?:,\s*(.+))?", item)
            if parsed is None:
                return match.group(0)
            key, locator = parsed.groups()
            reference = references.get(key)
            if reference is None:
                unknown.append(key)
                return match.group(0)
            if numeric:
                rendered.append(f"{reference.index}{', ' + locator if locator else ''}")
            else:
                surname = _surname(reference.authors[0] if reference.authors else "Anonymous")
                rendered.append(f"{surname}, {reference.year}{', ' + locator if locator else ''}")
        return f"[{'; '.join(rendered)}]" if numeric else f"({'; '.join(rendered)})"

    return re.sub(r"\[(@[^\]]+)\]", replace, markdown), list(dict.fromkeys(unknown))


def _surname(author: str) -> str:
    return author.split(",", 1)[0].strip() if "," in author else author.split()[-1]


def _remove_leading_title(markdown: str) -> str:
    return re.sub(r"^#\s+[^\n]+\n+", "", markdown.strip())


def _assemble_markdown(
    final_matter: FinalMatterProposal,
    body: str,
    references: list[FinalReferenceEntry],
    *,
    ai_declaration: str | None,
    output_language: str,
) -> str:
    english = output_language == "English"
    abstract_heading = "Abstract" if english else "摘要"
    keyword_label = "Keywords:" if english else "关键词："
    keyword_separator = ", " if english else "；"
    conclusion_heading = "Conclusion" if english else "结论"
    references_heading = "References" if english else "参考文献"
    parts = [
        f"# {final_matter.title}",
        f"## {abstract_heading}",
        final_matter.abstract,
        f"**{keyword_label}** {keyword_separator.join(final_matter.keywords)}",
        body,
        f"## {conclusion_heading}",
        final_matter.conclusion,
        f"## {references_heading}",
        "\n".join(entry.formatted_text for entry in references),
    ]
    if ai_declaration:
        parts.extend(
            [
                "## AI Usage Declaration" if english else "## AI 使用声明",
                ai_declaration,
            ]
        )
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) + "\n"


def _section_present(required: str, markdown: str) -> bool:
    normalized = re.sub(r"\s+", "", required.casefold())
    aliases = {
        "标题": "#",
        "title": "#",
        "摘要": "##摘要",
        "abstract": "##abstract",
        "关键词": "关键词",
        "keywords": "keywords:",
        "正文": "##",
        "body": "##",
        "结论": "##结论",
        "conclusion": "##conclusion",
        "参考文献": "##参考文献",
        "references": "##references",
    }
    compact_markdown = re.sub(r"\s+", "", markdown.casefold())
    if normalized in aliases:
        marker = aliases[normalized]
        if marker in compact_markdown:
            return True
        bilingual_markers = {
            "##摘要": "##abstract",
            "关键词": "keywords:",
            "##结论": "##conclusion",
            "##参考文献": "##references",
        }
        counterpart = bilingual_markers.get(marker)
        if counterpart is None:
            counterpart = next(
                (left for left, right in bilingual_markers.items() if right == marker),
                None,
            )
        return counterpart in compact_markdown if counterpart else False
    headings = [
        re.sub(r"\s+", "", value.casefold())
        for value in re.findall(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)
    ]
    return any(normalized in heading or heading in normalized for heading in headings)


def _font_size(value: str | None) -> float:
    if not value:
        return 11.0
    mapping = {"小四": 12.0, "四号": 14.0, "五号": 10.5, "小五": 9.0}
    if value.strip() in mapping:
        return mapping[value.strip()]
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else 11.0


def _set_run_font(run, font_name: str, size: float, *, bold: bool = False) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:ascii"), font_name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), font_name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)


def _add_section(document, title: str, content: str, *, level: int) -> None:
    document.add_heading(title, level=level)
    document.add_paragraph(content)


def _append_markdown(document, markdown: str, font_name: str, font_size: float) -> None:
    for block in re.split(r"\n\s*\n", markdown.strip()):
        clean = block.strip()
        if not clean:
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", clean)
        if heading:
            document.add_heading(heading.group(2).strip(), level=len(heading.group(1)))
            continue
        paragraph = document.add_paragraph()
        run = paragraph.add_run(re.sub(r"\*\*([^*]+)\*\*", r"\1", clean))
        _set_run_font(run, font_name, font_size)
