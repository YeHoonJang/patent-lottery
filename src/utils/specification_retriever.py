"""
Specification Retriever — structure-aware retrieval from patent specifications.

Patents are structured documents. Instead of dumping raw text,
we parse the structure and retrieve relevant paragraphs per claim element.

Patent Specification Structure:
  ├── TECHNICAL FIELD
  ├── BACKGROUND
  ├── SUMMARY
  ├── BRIEF DESCRIPTION OF DRAWINGS
  └── DETAILED DESCRIPTION
      ├── ¶001 ... ¶N (paragraphs)

Each claim element maps to specific paragraphs.
This module:
  1. Parses specification into structured sections/paragraphs
  2. Embeds each paragraph
  3. Given claim elements, retrieves the most relevant paragraphs
  4. Returns structured context per element (not raw text dump)
"""

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("vpatent.spec_retriever")


@dataclass
class SpecParagraph:
    """A single paragraph from the specification."""
    index: int
    text: str
    section: str = ""       # BACKGROUND, SUMMARY, DETAILED_DESCRIPTION, etc.
    embedding: np.ndarray = field(default=None, repr=False)


@dataclass
class ElementSupport:
    """Support evidence for one claim element from the specification."""
    element_id: str
    element_text: str
    supporting_paragraphs: list[dict] = field(default_factory=list)
    # Each dict: {"paragraph_index", "section", "text", "similarity"}
    has_support: bool = False
    best_similarity: float = 0.0


# Section detection patterns
SECTION_PATTERNS = [
    (r"(?i)technical\s+field", "TECHNICAL_FIELD"),
    (r"(?i)background", "BACKGROUND"),
    (r"(?i)summary\s+of\s+(the\s+)?invention|summary", "SUMMARY"),
    (r"(?i)brief\s+description\s+of\s+(the\s+)?drawings", "DRAWINGS"),
    (r"(?i)detailed\s+description", "DETAILED_DESCRIPTION"),
    (r"(?i)examples?|embodiments?", "EXAMPLES"),
    (r"(?i)claims", "CLAIMS"),
]


def parse_specification(text: str) -> list[SpecParagraph]:
    """
    Parse patent specification text into structured paragraphs.
    Handles multiple formats:
    - [0001] numbered paragraphs (USPTO XML)
    - Newline-separated with section headers (BigQuery)
    """
    if not text:
        return []

    paragraphs = []
    current_section = "PREAMBLE"

    # Try [0001] format first
    numbered_para = re.split(r'\[(\d{4})\]', text)
    if len(numbered_para) > 2:
        i = 1
        while i < len(numbered_para) - 1:
            try:
                para_num = int(numbered_para[i])
                para_text = numbered_para[i + 1].strip()
            except (ValueError, IndexError):
                i += 2
                continue
            if len(para_text) < 10:
                i += 2
                continue
            for pattern, section_name in SECTION_PATTERNS:
                if re.search(pattern, para_text[:100]):
                    current_section = section_name
                    break
            paragraphs.append(SpecParagraph(
                index=para_num, text=para_text[:1000], section=current_section,
            ))
            i += 2
    else:
        # BigQuery format: newline-separated text with section headers
        # Split by single newlines, merge short lines, detect sections
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Merge: section headers are short lines in CAPS, content follows
        merged_paras = []
        current_para = ""
        for line in lines:
            # Check if this line is a section header
            is_header = (len(line) < 80 and
                        (line.isupper() or
                         any(re.match(p, line, re.I) for p, _ in SECTION_PATTERNS)))

            if is_header:
                # Save current paragraph
                if current_para.strip() and len(current_para.strip()) > 30:
                    merged_paras.append((current_section, current_para.strip()))
                current_para = ""
                # Update section
                for pattern, section_name in SECTION_PATTERNS:
                    if re.search(pattern, line):
                        current_section = section_name
                        break
            else:
                # Content line — if it starts a new logical paragraph (long enough)
                if len(line) > 100 and current_para and len(current_para) > 200:
                    merged_paras.append((current_section, current_para.strip()))
                    current_para = line
                else:
                    current_para += " " + line if current_para else line

        # Don't forget the last paragraph
        if current_para.strip() and len(current_para.strip()) > 30:
            merged_paras.append((current_section, current_para.strip()))

        for i, (section, text_content) in enumerate(merged_paras):
            paragraphs.append(SpecParagraph(
                index=i, text=text_content[:1000], section=section,
            ))

    logger.info("Parsed specification: %d paragraphs, sections: %s",
                len(paragraphs),
                dict(sorted(((p.section, 1) for p in paragraphs), key=lambda x: x[0])))
    return paragraphs


class SpecificationRetriever:
    """
    Retrieve relevant specification paragraphs for claim elements.
    Uses the same bge-large embedding model as Prior Art Retriever.
    """

    def __init__(self, device: str = "cuda:0"):
        import os
        self._model = None
        self.device = os.environ.get("BGE_DEVICE", device)
        self.paragraphs: list[SpecParagraph] = []
        self.embeddings: np.ndarray = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=self.device)
        return self._model

    def index_specification(self, description_text: str):
        """Parse and embed a patent specification."""
        self.paragraphs = parse_specification(description_text)
        if not self.paragraphs:
            return

        model = self._load_model()
        texts = [f"{p.section}: {p.text}" for p in self.paragraphs]
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        logger.info("Indexed %d paragraphs", len(self.paragraphs))

    def find_support(self, claim_element: str, top_k: int = 3) -> list[dict]:
        """Find specification paragraphs supporting a claim element."""
        if self.embeddings is None or len(self.paragraphs) == 0:
            return []

        model = self._load_model()
        query = f"Represent this sentence for searching relevant documents: {claim_element}"
        query_emb = model.encode([query], normalize_embeddings=True)

        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            p = self.paragraphs[idx]
            results.append({
                "paragraph_index": p.index,
                "section": p.section,
                "text": p.text,
                "similarity": float(scores[idx]),
            })

        return results

    def find_support_for_elements(self, elements: list[dict], top_k: int = 3) -> list[ElementSupport]:
        """
        Find specification support for multiple claim elements.

        Args:
            elements: List of {"element_id": "1a", "text": "..."}
            top_k: Paragraphs per element

        Returns:
            List of ElementSupport with supporting paragraphs
        """
        results = []
        for elem in elements:
            paras = self.find_support(elem["text"], top_k=top_k)
            best_sim = max((p["similarity"] for p in paras), default=0.0)

            results.append(ElementSupport(
                element_id=elem.get("element_id", ""),
                element_text=elem["text"][:200],
                supporting_paragraphs=paras,
                has_support=best_sim > 0.5,  # threshold for "supported"
                best_similarity=best_sim,
            ))

        return results

    def build_context_for_role(self, role: str, claim_elements: list[dict] = None) -> str:
        """
        Build role-specific context from specification.

        Instead of dumping raw text, returns only relevant paragraphs
        formatted for the specific role's needs.
        """
        if not self.paragraphs:
            return ""

        if role == "claim_interpretation":
            # Need: Summary + term definitions from Detailed Description
            summary_paras = [p for p in self.paragraphs if p.section == "SUMMARY"]
            context = "## Specification Summary\n"
            for p in summary_paras[:5]:
                context += f"[¶{p.index:04d}] {p.text}\n\n"
            return context[:3000]

        elif role == "novelty_assessment" or role == "obviousness_analysis":
            # Need: Background (what's known) — minimal spec context
            bg_paras = [p for p in self.paragraphs if p.section == "BACKGROUND"]
            context = "## Background Art (from specification)\n"
            for p in bg_paras[:3]:
                context += f"[¶{p.index:04d}] {p.text}\n\n"
            return context[:2000]

        elif role == "written_desc_check":
            # Need: Per-element supporting paragraphs (the critical one)
            if not claim_elements:
                return ""
            supports = self.find_support_for_elements(claim_elements, top_k=2)
            context = "## Specification Support for Claim Elements\n"
            for s in supports:
                status = "SUPPORTED" if s.has_support else "NO CLEAR SUPPORT"
                context += f"\n### Element: {s.element_text[:100]}\n"
                context += f"Status: {status} (best similarity: {s.best_similarity:.2f})\n"
                for p in s.supporting_paragraphs:
                    context += f"  [¶{p['paragraph_index']:04d}] ({p['section']}, sim={p['similarity']:.2f}) {p['text'][:300]}\n"
            return context

        elif role == "prior_art_mapping":
            # Need: Only claim elements — spec context is minimal
            return ""

        else:
            # Generic: return summary + first few detailed paragraphs
            summary = [p for p in self.paragraphs if p.section == "SUMMARY"][:3]
            detail = [p for p in self.paragraphs if p.section == "DETAILED_DESCRIPTION"][:3]
            context = ""
            for p in summary + detail:
                context += f"[¶{p.index:04d}] ({p.section}) {p.text}\n\n"
            return context[:3000]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json

    # Test with a BQ-enriched patent
    with open("data/pilot_bench/enriched_bq/pilot_bench_bq_v2.json") as f:
        data = json.load(f)

    case = data["cases"][0]
    desc = case["bq_data"].get("desc_summary", "")
    claims_text = case["bq_data"].get("claims_text", "")

    print(f"Patent: {case['bq_data']['publication_number']}")
    print(f"Description: {len(desc)} chars")

    retriever = SpecificationRetriever()
    retriever.index_specification(desc)

    print(f"\nParagraphs: {len(retriever.paragraphs)}")
    sections = {}
    for p in retriever.paragraphs:
        sections[p.section] = sections.get(p.section, 0) + 1
    for s, c in sorted(sections.items()):
        print(f"  {s}: {c} paragraphs")

    # Test element search
    elements = [
        {"element_id": "1a", "text": "first comparator receiving a first reference address"},
        {"element_id": "1b", "text": "memory access address comparator"},
    ]

    print("\n=== Element Support ===")
    supports = retriever.find_support_for_elements(elements)
    for s in supports:
        print(f"\n{s.element_id}: {s.element_text[:60]}...")
        print(f"  Supported: {s.has_support} (best sim: {s.best_similarity:.3f})")
        for p in s.supporting_paragraphs[:2]:
            print(f"  [¶{p['paragraph_index']:04d}] ({p['section']}, {p['similarity']:.2f}) {p['text'][:80]}...")

    # Test role-specific context
    print("\n=== Role Contexts (char counts) ===")
    for role in ["claim_interpretation", "novelty_assessment", "written_desc_check"]:
        ctx = retriever.build_context_for_role(role, claim_elements=elements)
        print(f"  {role}: {len(ctx)} chars")
