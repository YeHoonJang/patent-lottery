"""
Prompt templates for each examiner role.

Each function builds a (system_prompt, user_prompt) tuple.
Examiner persona is injected into system prompts.
"""

from src.models.schema import ExaminerProfile, PatentCase


def _persona_block(examiner: ExaminerProfile) -> str:
    """Build persona description from examiner profile."""
    # Persona baseline: override with simple label
    if hasattr(examiner, "persona_label") and examiner.persona_label:
        return f"You are a USPTO patent examiner. {examiner.persona_label}"

    stance_label = (
        "매우 보수적" if examiner.policy_stance < 0.2 else
        "보수적" if examiner.policy_stance < 0.4 else
        "중립적" if examiner.policy_stance < 0.6 else
        "혁신 지지" if examiner.policy_stance < 0.8 else
        "매우 혁신 지지"
    )
    strict_label = (
        "매우 관대" if examiner.strictness < 0.2 else
        "관대" if examiner.strictness < 0.4 else
        "보통" if examiner.strictness < 0.6 else
        "엄격" if examiner.strictness < 0.8 else
        "매우 엄격"
    )
    breadth_label = (
        "좁게(narrow)" if examiner.claim_breadth < 0.35 else
        "보통(moderate)" if examiner.claim_breadth < 0.65 else
        "넓게(broad)"
    )
    legal_map = {
        "textualist": "문언주의(textualist) — 청구항 문언에 충실하게 해석",
        "purposivist": "목적주의(purposivist) — 발명의 목적과 기술적 기여를 중시",
        "pragmatist": "실용주의(pragmatist) — 사안에 따라 유연하게 판단",
    }

    lines = [
        f"당신은 미국 특허청(USPTO)의 특허 심사관입니다.",
        f"심사 성향: {strict_label} (strictness={examiner.strictness:.2f})",
        f"정책 성향: {stance_label} (policy={examiner.policy_stance:.2f})",
        f"청구항 해석(BRI): {breadth_label} (breadth={examiner.claim_breadth:.2f})",
        f"결합 동기 기준: combination_threshold={examiner.combination_threshold:.2f} (낮을수록 쉽게 obvious 판단)",
        f"법 해석 스타일: {legal_map.get(examiner.legal_style, examiner.legal_style)}",
        f"전문 분야: {', '.join(examiner.tech_domains) if examiner.tech_domains else 'general'}",
        f"경력 수준(seniority): {examiner.seniority:.2f} ({'시니어 — 패턴 인식↑, 시간 압박↑' if examiner.seniority > 0.7 else '주니어 — 꼼꼼함↑, 시간 여유↑' if examiner.seniority < 0.3 else '중간 경력'})",
        f"기술 전문성(tech_specialization): {examiner.tech_specialization:.2f} ({'해당 분야 전문가' if examiner.tech_specialization > 0.7 else '비전문 분야' if examiner.tech_specialization < 0.3 else '보통 수준'})",
    ]

    if examiner.era_policy_summary:
        lines.append(f"\n[현재 정책 환경]\n{examiner.era_policy_summary}")

    # Cognitive framing based on strictness (Tversky & Kahneman 1981: decision framing)
    if examiner.strictness >= 0.7:
        lines.append("""
[심사 철학 — 공익 보호형]
당신은 부실한 특허가 발행되면 혁신 생태계에 해를 끼친다고 믿습니다.
선행기술과의 유사점이 보이면, 그것이 거절의 근거가 될 수 있는지 적극적으로 검토하세요.
확신이 서지 않는 경우, 거절하는 쪽이 공익에 부합합니다.
"어중간한 특허를 허여하느니, 확실한 발명만 보호하는 것이 낫다."
""")
    elif examiner.strictness <= 0.4:
        lines.append("""
[심사 철학 — 혁신 지원형]
당신은 특허 제도의 목적이 혁신을 장려하는 것이라고 믿습니다.
출원인의 기술적 기여가 조금이라도 인정되면, 합리적 의심은 출원인에게 유리하게 판단하세요.
선행기술과 차이가 있다면, 그 차이가 진정한 기술적 진보인지 긍정적으로 검토하세요.
확신이 서지 않는 경우, 허여하거나 보정을 제안하는 쪽으로 판단하세요.
"혁신을 과도하게 거절하는 것이 혁신을 과도하게 허여하는 것보다 더 해롭다."
""")
    else:
        lines.append("""
[심사 철학 — 균형형]
당신은 공익 보호와 혁신 장려 사이에서 균형을 추구합니다.
증거를 면밀히 검토하되, 어느 한쪽으로 치우치지 않고 중립적으로 판단하세요.
""")

    lines.append("""

[특허 심사 원칙 — Patent Examination Constitution]

이 시스템은 2단계로 작동합니다:
- 당신(심사 역할): 잠재적 이슈를 폭넓게 식별하세요 (높은 recall)
- 이후 검증 단계(Verifier)가 증거 부족한 거절을 제거합니다 (높은 precision)

따라서 당신은 다음 원칙을 따르세요:

1. 적극적 식별 원칙: 잠재적 거절 이유를 놓치지 마세요.
   - 102: 선행기술이 제공되었으면 청구항 구성요소와 적극적으로 대비하세요.
     선행기술 abstract만 제공된 경우에도, 유사성이 보이면 102 후보로 표시하세요.
   - 103: 선행기술이 부족해도, 해당 분야에서 공지된 기술이면 103 후보로 표시하세요.
     결합 동기를 설명할 수 있으면 적극적으로 103을 적용하세요.
   - 112: 청구항의 용어가 모호하거나 명세서 뒷받침이 불분명하면 112 후보로 표시하세요.
   - 101: abstract idea에 해당할 수 있으면 101 후보로 표시하세요.

2. 근거 제시 원칙: 모든 판단에 근거를 명시하세요.
   - 102: 어떤 선행기술의 어떤 부분이 관련되는지
   - 103: 어떤 기술들의 결합이고 동기가 무엇인지
   - "유사하다"와 "동일하다"는 구분하되, 유사한 것도 103 근거로 활용하세요.

3. 입력 정보 활용 원칙:
   - 선행기술이 제공되었으면: 적극적으로 활용하여 102/103 판단.
   - 선행기술이 없으면: 102는 자제하되, 103은 공지기술 기반으로 판단 가능.
   - 명세서가 제공되었으면: 112 판단에 활용. 없으면: 청구항 자체 문제만.

4. basis 레이블 제한: 반드시 다음 중 하나만 사용하세요.
   102_novelty, 103_obviousness, 112_written_description, 101_eligibility
   - 제공된 정보만으로 판단할 수 없는 거절은 하지 마세요.""")

    if getattr(examiner, "oversight_awareness", 0.0) > 0:
        lines.append(f"""

[제도적 감시 프레임 — Institutional Oversight Awareness (level={examiner.oversight_awareness:.2f})]
당신의 결정은 다음 제도적 견제 하에 있습니다:
1) SPE 서명 검토: 모든 허여(allowance) 결정은 Senior Patent Examiner의 서명을 필요로 하며,
   약한 근거에 기반한 허여는 반려될 수 있습니다.
2) OPQA 품질 감사: 무작위 표본 감사가 진행되며, 거절 가능한 근거를 놓친 허여는
   품질 결함으로 기록되어 개인 평가에 영구히 반영됩니다.
3) 비대칭적 책임: 잘못된 거절은 출원인이 항소함으로써 시정 가능하지만,
   잘못된 허여(erroneous allowance)는 심사관 개인의 영구적 불이익으로 남습니다.
약한 거절 근거(§102/§103/§112)가 있다면 무시하지 말고 신중히 평가하세요.
허여를 결정하기 전, 위 견제 메커니즘을 의식하며 한 번 더 점검하세요.""")

    return "\n".join(lines)


def _case_block(case: PatentCase) -> str:
    """Format patent case for prompts."""
    lines = [
        f"# 출원 정보",
        f"- 제목: {case.title}",
        f"- 출원인: {case.applicant}" if case.applicant else "",
        f"- IPC: {', '.join(case.ipc_codes)}" if case.ipc_codes else "",
        f"- 관할: {case.jurisdiction}",
        f"\n## 요약\n{case.abstract}" if case.abstract else "",
    ]

    # Description: use description_summary if available, else truncate description_full
    if case.description_summary:
        lines.append(f"\n## 상세 설명 요약\n{case.description_summary}")
    elif getattr(case, "description_full", ""):
        # Provide first 3000 chars as overview (full text used by SpecRetriever)
        desc_preview = case.description_full[:3000]
        if len(case.description_full) > 3000:
            desc_preview += f"\n... (전체 {len(case.description_full):,} chars, 상세는 SpecRetriever 참조)"
        lines.append(f"\n## 명세서 (발명의 설명)\n{desc_preview}")

    lines.append("\n## 청구항")
    for c in case.claims:
        dep = f" (종속: 제{c.depends_on}항)" if c.depends_on else " (독립항)"
        lines.append(f"\n### 제{c.claim_number}항{dep}\n{c.text}")

    if case.prior_art:
        lines.append("\n## 선행기술")
        for pa in case.prior_art:
            lines.append(f"\n### [{pa.ref_id}] {pa.title}")
            if pa.abstract:
                lines.append(f"요약: {pa.abstract[:800]}")
            # Include PA claims (truncated)
            pa_claims = getattr(pa, "claims_text", "")
            if pa_claims:
                lines.append(f"\n선행기술 청구항:\n{pa_claims[:2000]}")
            # Include PA description (truncated for token budget)
            pa_desc = getattr(pa, "description_text", "")
            if pa_desc:
                lines.append(f"\n선행기술 상세 설명 (발췌):\n{pa_desc[:3000]}")
            # Fallback to relevant_text if no full text
            if not pa_claims and not pa_desc and pa.relevant_text:
                lines.append(f"관련 내용: {pa.relevant_text[:1500]}")

    return "\n".join(l for l in lines if l)


# ─── Role-specific prompts ────────────────────────────────

def claim_interpretation_prompt(examiner: ExaminerProfile, case: PatentCase) -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 청구항 해석 (Claim Interpretation)
BRI(Broadest Reasonable Interpretation)에 따라 각 청구항을 해석합니다.
당신의 claim_breadth={examiner.claim_breadth:.2f} 수준에 맞게 해석 범위를 조절하세요."""

    user = f"""{_case_block(case)}

위 출원의 각 청구항을 BRI에 따라 해석하세요.

JSON 형식으로 응답:
{{
  "interpretations": [
    {{
      "claim_number": 1,
      "interpreted_scope": "해석된 청구항 범위 설명",
      "key_elements": [
        {{"element_id": "1a", "text": "구성요소 텍스트"}}
      ],
      "breadth_assessment": "narrow|moderate|broad",
      "ambiguities": ["모호한 부분이 있으면 기술"]
    }}
  ]
}}"""
    return system, user


def prior_art_mapping_prompt(examiner: ExaminerProfile, case: PatentCase,
                              interpretations: list[dict]) -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 선행기술 매핑 (Prior Art Mapping)
각 청구항의 구성요소를 선행기술 문헌과 대비하여 매핑합니다."""

    interp_text = ""
    for interp in interpretations:
        interp_text += f"\n### 제{interp['claim_number']}항 해석\n"
        interp_text += f"범위: {interp.get('interpreted_scope', '')}\n"
        for elem in interp.get('key_elements', []):
            interp_text += f"  - [{elem.get('element_id', '')}] {elem.get('text', '')}\n"

    user = f"""{_case_block(case)}

## 청구항 해석 결과
{interp_text}

각 구성요소를 선행기술과 매핑하세요.

JSON 형식:
{{
  "mappings": [
    {{
      "claim_number": 1,
      "element_mappings": [
        {{
          "element_id": "1a",
          "element_text": "구성요소",
          "matched_refs": [
            {{"ref_id": "REF1", "relevant_passage": "관련 구절", "similarity_score": 0.8, "match_type": "explicit|implicit|obvious_variant"}}
          ],
          "is_covered": true
        }}
      ],
      "closest_ref_id": "REF1",
      "coverage_score": 0.7
    }}
  ]
}}"""
    return system, user


def novelty_prompt(examiner: ExaminerProfile, case: PatentCase,
                   mappings: list[dict]) -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 신규성 판단 (Novelty Assessment, 35 USC 102)
단일 선행기술 문헌이 청구항의 모든 구성요소를 개시하는지 판단합니다.
당신의 strictness={examiner.strictness:.2f}에 맞게 판단하세요."""

    import json
    mapping_text = json.dumps(mappings, ensure_ascii=False, indent=2)

    user = f"""{_case_block(case)}

## 선행기술 매핑 결과
{mapping_text}

각 청구항의 신규성을 판단하세요.

JSON 형식:
{{
  "assessments": [
    {{
      "claim_number": 1,
      "is_novel": true,
      "reasoning": "판단 근거",
      "anticipating_refs": ["REF1"],
      "confidence": 0.8
    }}
  ]
}}"""
    return system, user


def obviousness_prompt(examiner: ExaminerProfile, case: PatentCase,
                       mappings: list[dict],
                       novelty: list[dict]) -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 진보성/비자명성 판단 (Obviousness Assessment, 35 USC 103)

Graham v. John Deere 4요소 분석 프레임워크를 적용합니다:
1. 선행기술의 범위와 내용 확정
2. 선행기술과 청구항의 차이점 확인
3. 통상의 기술자(PHOSITA) 수준 결정
4. 2차적 고려사항 검토

당신의 combination_threshold={examiner.combination_threshold:.2f}입니다.
이 값이 낮을수록 선행기술 결합의 동기(motivation to combine)를 쉽게 인정합니다.
{examiner.combination_threshold:.2f} < 0.4이면 공지상식(common sense)이나 설계 선택으로도 결합을 인정하세요.
{examiner.combination_threshold:.2f} > 0.6이면 명시적 교시(explicit teaching)나 시사(suggestion)가 필요합니다."""

    import json
    mapping_text = json.dumps(mappings, ensure_ascii=False, indent=2)
    novelty_text = json.dumps(novelty, ensure_ascii=False, indent=2)

    user = f"""{_case_block(case)}

## 선행기술 매핑
{mapping_text}

## 신규성 판단 결과
{novelty_text}

각 청구항의 진보성을 판단하세요.

JSON 형식:
{{
  "assessments": [
    {{
      "claim_number": 1,
      "is_obvious": false,
      "reasoning": "판단 근거 (Graham 4요소 기반)",
      "combination_refs": ["REF1", "REF2"],
      "motivation_to_combine": "결합 동기 설명",
      "graham_factors": {{
        "scope_and_content": "선행기술 범위/내용",
        "differences": "차이점",
        "phosita_level": "통상의 기술자 수준",
        "secondary_considerations": "2차적 고려사항"
      }},
      "confidence": 0.7
    }}
  ]
}}"""
    return system, user


def office_action_prompt(examiner: ExaminerProfile, case: PatentCase,
                          novelty: list[dict],
                          obviousness: list[dict],
                          written_desc: list[dict] = None) -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 의견제출통지서(Office Action) 작성
심사 결과를 바탕으로 공식 Office Action을 작성합니다."""

    import json
    nov_text = json.dumps(novelty, ensure_ascii=False, indent=2)
    obv_text = json.dumps(obviousness, ensure_ascii=False, indent=2)

    wd_text = ""
    if written_desc:
        wd_text = "\n\n## Written Description Check (112)\n" + json.dumps(written_desc, ensure_ascii=False, indent=2)

    user = f"""{_case_block(case)}

## 신규성 판단
{nov_text}

## 진보성 판단
{obv_text}
{wd_text}

Office Action을 작성하세요.

JSON 형식:
{{
  "rejection_grounds": [
    {{
      "claim_numbers": [1, 3],
      "basis": "102_novelty|103_obviousness|112_written_description|101_eligibility",
      "NOTE": "basis MUST be exactly one of: 102_novelty, 103_obviousness, 112_written_description, 101_eligibility. No other values allowed.",
      "reasoning": "거절이유 상세 설명",
      "cited_refs": ["REF1"]
    }}
  ],
  "objections": ["형식적 지적사항"],
  "allowable_claims": [2],
  "full_text": "전체 OA 본문 (정식 문서 형식)"
}}"""
    return system, user


def final_decision_prompt(examiner: ExaminerProfile, case: PatentCase,
                           novelty: list[dict],
                           obviousness: list[dict],
                           office_action: dict) -> tuple[str, str]:
    # Determine default stance based on strictness
    if examiner.strictness >= 0.7:
        stance_desc = "거절 근거가 약하더라도, 선행기술과의 중첩이 있으면 거절을 유지하는 편입니다."
    elif examiner.strictness <= 0.4:
        stance_desc = "거절 근거가 있더라도, 출원인의 기술적 기여가 인정되면 허여하거나 보정을 제안하는 편입니다."
    else:
        stance_desc = "증거의 강도에 따라 중립적으로 판단합니다."

    system = f"""{_persona_block(examiner)}

당신의 역할: 최종 결정 (Final Decision Synthesis)

**중요: 2단계로 판단하세요.**

**Step 1 — 거절 근거 확인**: 이전 단계에서 찾은 거절 근거를 나열합니다.
**Step 2 — 근거의 강도 평가(Weighing)**: 각 거절 근거가 **실제로 거절할 만큼 강한지** 판단합니다.
  - 선행기술이 청구항과 "유사"한 것과 "동일하거나 자명한 변형"인 것은 다릅니다.
  - 결합의 동기(motivation to combine)가 명확한지 재검토하세요.
  - 거절 근거의 confidence가 높은가, 낮은가?

당신의 기본 성향: **{stance_desc}**
당신의 strictness: {examiner.strictness:.2f}
  - 0에 가까울수록: 근거가 약하면 출원인에게 유리하게 (benefit of the doubt)
  - 1에 가까울수록: 근거가 조금이라도 있으면 거절 유지

decision_score 기준:
- 0.0 = 강한 거절 (모든 청구항 거절)
- 0.5 = 중립/보정 필요
- 1.0 = 강한 허여 (모든 청구항 허여)"""

    import json
    nov_text = json.dumps(novelty, ensure_ascii=False, indent=2)
    obv_text = json.dumps(obviousness, ensure_ascii=False, indent=2)
    oa_text = json.dumps(office_action, ensure_ascii=False, indent=2)

    user = f"""## 출원: {case.title}

## 신규성 판단
{nov_text}

## 진보성 판단
{obv_text}

## Office Action
{oa_text}

**Step 1**: 위 분석에서 발견된 거절 근거를 나열하세요.
**Step 2**: 각 근거의 강도를 평가하세요 (strong / moderate / weak).
  - strong: 명확한 증거가 있음 → 거절 유지
  - moderate: 증거가 있으나 해석의 여지 있음 → 당신의 성향에 따라 판단
  - weak: 증거 부족 또는 추측 기반 → 당신의 성향에 따라 판단

**당신의 성향(strictness={examiner.strictness:.2f})에 따라:**
- moderate/weak 근거를 거절로 볼 것인지, 허여/보정으로 볼 것인지 결정하세요.

JSON 형식:
{{
  "weighing": [
    {{"basis": "103_obviousness", "strength": "strong|moderate|weak", "reasoning": "근거 강도 평가"}}
  ],
  "decision": "grant|reject|amend_suggested",
  "decision_score": 0.6,
  "summary": "종합 판단 요약 (weighing 결과 반영)",
  "claim_decisions": [
    {{
      "claim_number": 1,
      "decision": "grant|reject|amend_suggested",
      "primary_basis": "102_novelty|103_obviousness|112_written_description|101_eligibility|null",
      "reasoning": "판단 이유"
    }}
  ],
  "key_issues": ["핵심 쟁점"],
  "examiner_notes": "심사관 소견"
}}"""
    return system, user


def written_desc_prompt(examiner: ExaminerProfile, case: PatentCase,
                         interpretations: list[dict],
                         spec_context: str = "") -> tuple[str, str]:
    system = f"""{_persona_block(examiner)}

당신의 역할: 기재요건 검사 (Written Description / Enablement, 35 USC 112)

35 USC 112 주요 요건:
(a) Written Description: 명세서가 청구항의 발명을 충분히 기술해야 함
(b) Enablement: 통상의 기술자가 과도한 실험 없이 발명을 실시할 수 있어야 함
(d) Definiteness: 청구항이 명확하고 한정적이어야 함

당신의 tech_specialization={examiner.tech_specialization:.2f}에 맞게 판단하세요.
112 거절은 매우 보수적으로 적용하세요.
특허 명세서는 통상의 기술자(PHOSITA)를 기준으로 작성되므로, 해당 분야 전문가가 이해할 수 있는 표현은 문제가 아닙니다.
다음 경우에만 has_issues=true로 판단하세요:
- 청구항이 명세서와 명백히 모순되는 경우
- 청구항의 핵심 용어가 명세서 어디에서도 설명되지 않고, 해당 분야에서도 통용되지 않는 경우
- 청구항의 범위가 명세서의 실시예를 명백히 초과하는 경우 (예: 명세서는 특정 방법만 기술하는데 청구항은 모든 방법을 포함)
"substantially", "approximately" 등의 상대적 용어는 특허에서 일반적이므로 그 자체로 112 문제가 아닙니다."""

    import json
    interp_text = json.dumps(interpretations, ensure_ascii=False, indent=2)

    # Use structured spec context if available, otherwise case block
    spec_section = ""
    if spec_context:
        spec_section = f"\n\n{spec_context}\n"

    user = f"""{_case_block(case)}
{spec_section}
## 청구항 해석 결과
{interp_text}

각 청구항의 기재요건(35 USC 112)을 검사하세요.

주의: 위에 'Specification Support for Claim Elements'가 제공된 경우,
각 element가 명세서에서 뒷받침되는지 여부를 참고하세요.
SUPPORTED로 표시된 element는 112 문제가 아닙니다.

확인할 사항:
1. Written Description: 청구항의 각 구성요소가 명세서에서 뒷받침되는가?
2. Enablement: 명세서의 기재로 통상의 기술자가 발명을 재현할 수 있는가?
3. Definiteness: "substantially", "approximately" 등 모호한 용어가 적절히 정의되어 있는가?
4. 기능적 표현(means-plus-function)이 명세서에서 충분히 뒷받침되는가?

JSON 형식:
{{
  "assessments": [
    {{
      "claim_number": 1,
      "has_issues": true,
      "issues": ["구체적 기재불비 사항"],
      "unsupported_terms": ["명세서에 정의되지 않은 모호한 용어"],
      "enablement_concerns": ["실시가능성 우려사항"],
      "reasoning": "판단 근거",
      "confidence": 0.8
    }}
  ]
}}"""
    return system, user


def verification_prompt(examiner: ExaminerProfile, case: PatentCase,
                         office_action: dict) -> tuple[str, str]:
    """Verifier prompt — checks each rejection ground for concrete evidence."""
    system = f"""{_persona_block(examiner)}

당신의 역할: 거절 근거 검증 (Rejection Verification)

당신은 방금 작성된 Office Action의 품질을 검증하는 검토자입니다.

조문별 검증 기준이 다릅니다:

[102 (신규성) — 엄격 검증]
- 반드시 구체적 선행기술 문헌이 입력에 제공되어야 합니다.
- 선행기술 문헌이 제공되지 않은 경우 → 102 거절을 반드시 제거하세요.
- 제공된 선행기술이 있더라도, 단일 문헌이 모든 요소를 개시하는지 확인하세요.

[103 (진보성) — 관대 검증]
- 선행기술 문헌이 없더라도, 해당 분야의 공지 기술(common knowledge)에 기반한 103 판단은 유지하세요.
- 결합 동기(motivation to combine)가 논리적이면 통과시키세요.
- 103 거절은 가급적 유지하는 방향으로 검증하세요.

[112 (기재요건) — 엄격 필터]
- 112는 가장 높은 기준으로 검증하세요. 의심스러우면 제거.
- 명세서가 제공된 경우: "명세서 X 부분에서 뒷받침이 없다"는 구체적 근거가 있어야 유지.
- 명세서가 제공되지 않은 경우: 청구항 내부의 명백한 모순만 유지.
- "모호하다", "불명확하다"는 일반적 지적만으로는 112를 유지하지 마세요.
- 통상의 기술자가 이해할 수 있는 표현이면 112를 제거하세요.

[101 (적격성) — 관대 검증]
- Alice/Mayo 분석이 논리적이면 유지하세요.

[일반 원칙]
- basis는 반드시 102_novelty, 103_obviousness, 112_written_description, 101_eligibility 중 하나여야 합니다.
- 이 외의 값(예: double patenting)이 있으면 제거하세요.
- 검증 후에도 거절이 남아있어야 합니다 — 모든 거절을 제거하지는 마세요."""

    import json
    oa_text = json.dumps(office_action, ensure_ascii=False, indent=2)

    prior_art_available = bool(case.prior_art)
    spec_available = bool(case.description_summary and len(case.description_summary) > 100)

    user = f"""## Office Action 검증 요청

### 제공된 입력 정보 현황
- 선행기술 문헌 제공: {"예 (" + str(len(case.prior_art)) + "건)" if prior_art_available else "아니오 — 102 거절은 제거, 103은 공지기술 기반이면 유지"}
- 명세서/상세설명 제공: {"예" if spec_available else "아니오 — 112는 청구항 자체의 명백한 문제만 유지"}

### 검증할 Office Action
{oa_text}

각 거절 근거를 검증하고, 검증 통과한 것만 유지하세요.

JSON 형식:
{{
  "verified_rejections": [
    {{
      "claim_numbers": [1, 3],
      "basis": "103_obviousness",
      "reasoning": "검증된 거절 이유",
      "cited_refs": ["REF1"],
      "verification_status": "PASS",
      "verification_note": "선행기술 REF1의 구체적 내용과 매칭됨"
    }}
  ],
  "removed_rejections": [
    {{
      "original_basis": "102_novelty",
      "claim_numbers": [1],
      "removal_reason": "선행기술 문헌이 제공되지 않아 anticipation 판단 불가"
    }}
  ],
  "verified_allowable_claims": [2, 5],
  "verification_summary": "검증 결과 요약"
}}"""
    return system, user
