import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import requests


PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_MAX_RESULTS = 8

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _keyword_terms(text: str) -> List[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", cleaned.lower())
    return [token for token in tokens if token not in _STOPWORDS]


def build_pubmed_query(task_name: str, clinical_goal: str, modality: str) -> str:
    phrases = [_clean_text(task_name), _clean_text(clinical_goal), _clean_text(modality)]
    query_parts: List[str] = []

    for phrase in phrases:
        if phrase:
            query_parts.append(f"({phrase}[Title/Abstract])")

    modality_tokens = _keyword_terms(modality)
    if modality_tokens:
        query_parts.append("(" + " OR ".join(f"{token}[Title/Abstract]" for token in modality_tokens) + ")")

    goal_tokens = _keyword_terms(clinical_goal)
    if goal_tokens:
        query_parts.append("(" + " OR ".join(f"{token}[Title/Abstract]" for token in goal_tokens[:6]) + ")")

    task_tokens = _keyword_terms(task_name)
    if task_tokens:
        query_parts.append("(" + " OR ".join(f"{token}[Title/Abstract]" for token in task_tokens[:6]) + ")")

    if not query_parts:
        return "biomedical imaging[Title/Abstract]"
    return " AND ".join(query_parts)


def _extract_year(pubmed_article: ET.Element) -> int | None:
    for xpath in (
        ".//PubDate/Year",
        ".//ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ):
        node = pubmed_article.find(xpath)
        if node is not None and node.text and node.text.isdigit():
            return int(node.text)
    medline_date = pubmed_article.findtext(".//PubDate/MedlineDate", default="")
    match = re.search(r"(19|20)\d{2}", medline_date)
    return int(match.group(0)) if match else None


def _abstract_snippet(pubmed_article: ET.Element, max_chars: int = 400) -> str:
    sections: List[str] = []
    for node in pubmed_article.findall(".//Abstract/AbstractText"):
        label = _clean_text(node.attrib.get("Label", ""))
        content = _clean_text("".join(node.itertext()))
        if not content:
            continue
        sections.append(f"{label}: {content}" if label else content)
    snippet = _clean_text(" ".join(sections))
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def _parse_pubmed_articles(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean_text(article.findtext(".//PMID", default=""))
        title = _clean_text("".join(article.find(".//ArticleTitle").itertext())) if article.find(".//ArticleTitle") is not None else ""
        if not pmid or not title:
            continue
        records.append(
            {
                "title": title,
                "year": _extract_year(article),
                "pmid": pmid,
                "abstract_snippet": _abstract_snippet(article),
            }
        )
    return records


def retrieve_pubmed_evidence(
    task_name: str,
    clinical_goal: str,
    modality: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = 20,
) -> Dict[str, Any]:
    clamped_max_results = min(max(5, max_results), 10)
    query = build_pubmed_query(task_name=task_name, clinical_goal=clinical_goal, modality=modality)

    try:
        esearch_response = requests.get(
            PUBMED_ESEARCH_URL,
            params={
                "db": "pubmed",
                "retmode": "json",
                "sort": "relevance",
                "retmax": clamped_max_results,
                "term": query,
            },
            timeout=timeout,
        )
        esearch_response.raise_for_status()
        id_list = esearch_response.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {
                "query": query,
                "source": "pubmed",
                "count": 0,
                "results": [],
            }

        efetch_response = requests.get(
            PUBMED_EFETCH_URL,
            params={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(id_list),
            },
            timeout=timeout,
        )
        efetch_response.raise_for_status()
        results = _parse_pubmed_articles(efetch_response.text)[:clamped_max_results]
        return {
            "query": query,
            "source": "pubmed",
            "count": len(results),
            "results": results,
        }
    except (requests.RequestException, ValueError, ET.ParseError) as exc:
        return {
            "query": query,
            "source": "pubmed",
            "count": 0,
            "results": [],
            "error": str(exc),
        }


def build_evidence_packet(task_name: str, clinical_goal: str, modality: str, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
    return retrieve_pubmed_evidence(
        task_name=task_name,
        clinical_goal=clinical_goal,
        modality=modality,
        max_results=max_results,
    )


def render_evidence_packet_markdown(packet: Dict[str, Any]) -> str:
    lines = [
        "# Evidence Packet",
        "",
        f"- source: {packet.get('source', 'pubmed')}",
        f"- query: {packet.get('query', '')}",
        f"- count: {packet.get('count', 0)}",
    ]
    if packet.get("error"):
        lines.extend(["", f"- error: {packet['error']}"])

    for idx, item in enumerate(packet.get("results", []), start=1):
        lines.extend(
            [
                "",
                f"## Paper {idx}",
                f"- title: {item.get('title', '')}",
                f"- year: {item.get('year', '')}",
                f"- pmid: {item.get('pmid', '')}",
                f"- abstract_snippet: {item.get('abstract_snippet', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_evidence_packet(output_dir: str | Path, packet: Dict[str, Any]) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_packet.json"
    md_path = out_dir / "evidence_packet.md"
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_evidence_packet_markdown(packet), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path)}
