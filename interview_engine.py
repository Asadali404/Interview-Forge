import json
import os
import re
from typing import Any, Dict, List

from groq import Groq


class InterviewForgeEngine:
    """
    Orchestrates the Interview Forge pipeline around the local RAG retriever
    and Groq-hosted LLM.

    Default model is openai/gpt-oss-20b because it is currently available on
    Groq's free tier. Change GROQ_MODEL in Streamlit secrets if desired.
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to Streamlit Cloud → App settings → Secrets."
            )

        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self.client = Groq(api_key=api_key)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError("The model returned invalid JSON.")

    def chat(self, system: str, user: str, max_tokens: int = 6000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, max_tokens: int = 6000) -> Dict[str, Any]:
        """Call Groq in JSON mode and parse the result safely.

        The previous implementation relied only on prompt instructions such as
        'return valid JSON'. LLMs can occasionally emit a trailing comma,
        markdown, or an incomplete object, which caused Streamlit errors such as
        `Expecting ',' delimiter`. Groq's JSON Object Mode guarantees valid JSON
        for supported models, so use it directly.
        """
        prompt = (
            system
            + "\nReturn ONLY a single JSON object. Do not use markdown fences. "
              "Do not add commentary before or after the JSON."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                reasoning_format="hidden",
            )
            text = response.choices[0].message.content or "{}"
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Groq returned malformed JSON. Please retry; the response may have been truncated. "
                f"JSON error: {exc}"
            ) from exc
        except Exception:
            # Some older Groq SDK/API combinations may reject one of the JSON-mode
            # parameters. Fall back once to the plain chat endpoint rather than
            # breaking the entire Streamlit app.
            text = self.chat(prompt, user, max_tokens=max_tokens)
            return self._extract_json(text)

    def forge(
        self,
        rag,
        company: str,
        role: str,
        categories: List[str],
        questions_per_category: int,
        difficulty: str,
        top_k: int,
    ) -> Dict[str, Any]:

        candidate_context = rag.context(
            "Extract the candidate's identity, education, skills, tools, projects, work experience, achievements, certifications, and measurable evidence.",
            k=top_k,
            max_chars=9000,
        )
        job_context = rag.context(
            f"Extract the target job requirements, responsibilities, required skills, preferred skills, seniority, technologies, and company information. Target role: {role or 'not specified'}. Company: {company or 'not specified'}.",
            k=top_k,
            max_chars=9000,
        )

        profile = self.chat_json(
            """You are Interview Forge's profile extraction engine.
Use ONLY evidence from the supplied RAG context. Do not invent candidate facts.
If something is unknown, use null or an empty list.""",
            f"""CANDIDATE RAG CONTEXT:
{candidate_context}

JOB/COMPANY RAG CONTEXT:
{job_context}

Return:
{{
  "candidate_profile": {{
    "name": null,
    "summary": "",
    "education": [],
    "skills": [],
    "tools": [],
    "projects": [],
    "experience": [],
    "certifications": [],
    "achievements": [],
    "evidence": []
  }},
  "job_company_profile": {{
    "role": "",
    "company": "",
    "must_have": [],
    "nice_to_have": [],
    "responsibilities": [],
    "technologies": [],
    "seniority": "",
    "company_signals": []
  }},
  "information_check": {{
    "status": "sufficient",
    "missing_information": [],
    "follow_up_questions": []
  }}
}}""",
            max_tokens=3500,
        )

        match_context = rag.context(
            "Compare candidate evidence with job requirements. Identify strongest matches, partial matches, gaps, and evidence-backed concerns.",
            k=top_k,
            max_chars=10000,
        )

        match = self.chat_json(
            """You are a strict job-match analyst.
Score only from the evidence in the RAG context. Never claim a skill is present without evidence.""",
            f"""Candidate profile:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job profile:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Evidence:
{match_context}

Return:
{{
  "job_match_analysis": "A concise but detailed evidence-based analysis with strengths, gaps and recommendations.",
  "match_table": [
    {{
      "requirement": "",
      "candidate_evidence": "",
      "match": "Strong|Partial|Gap",
      "interview_action": ""
    }}
  ]
}}""",
            max_tokens=3200,
        )

        risk = self.chat_json(
            """You are an interview risk engine. Find realistic interview risks from the candidate/job evidence.
Avoid discrimination or protected-attribute inference. Focus only on job-relevant evidence.""",
            f"""Candidate:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Match analysis:
{json.dumps(match, ensure_ascii=False)}

Return:
{{
  "swot": {{
    "strengths": [],
    "weaknesses": [],
    "opportunities": [],
    "threats": []
  }},
  "risk_areas": {{
    "30_seconds": [],
    "60_seconds": []
  }}
}}""",
            max_tokens=2600,
        )

        question_prompt = []
        for category in categories:
            question_prompt.append(
                f"{category}: exactly {questions_per_category} questions"
            )

        questions = self.chat_json(
            """You are the Interview Forge question engine.
Create realistic interview questions using the candidate evidence and job requirements.
Questions must be specific enough to expose weak evidence, not generic filler.""",
            f"""Candidate:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Risk engine:
{json.dumps(risk, ensure_ascii=False)}

Difficulty: {difficulty}
Requested categories:
{chr(10).join(question_prompt)}

Return a JSON object whose keys are exactly the requested category names.
Each value must be a list of objects:
{{"question": "", "why": "", "risk": ""}}""",
            max_tokens=5000,
        )

        all_questions = []
        for category, items in questions.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        all_questions.append(
                            {
                                "category": category,
                                "question": item.get("question", ""),
                                "why": item.get("why", ""),
                            }
                        )

        blueprint = self.chat_json(
            """You are an expert interview answer coach.
Build evidence-first answer blueprints. Do not write fake accomplishments.
Use STAR when appropriate and explicitly connect answers to CV evidence.""",
            f"""Candidate profile:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job profile:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Questions:
{json.dumps(all_questions, ensure_ascii=False)}

Return:
{{
  "question_priority": [
    {{"question": "", "priority": "HIGH|MEDIUM|LOW", "reason": ""}}
  ],
  "answers": [
    {{
      "question": "",
      "why_asked": "",
      "what_to_show": "",
      "cv_evidence": "",
      "answer_structure": ""
    }}
  ]
}}""",
            max_tokens=5000,
        )

        tough = self.chat_json(
            """You are a tough-question interview coach. Prepare the candidate for
follow-ups, contradictions, weak areas, missing experience, and pressure questions.
Never fabricate facts.""",
            f"""Candidate:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Weaknesses/risk:
{json.dumps(risk, ensure_ascii=False)}

Return:
{{
  "tough_question_coach": [
    {{
      "question": "",
      "trap": "",
      "coaching": "",
      "answer_angle": ""
    }}
  ],
  "interview_battle_plan": {{
    "before_interview": [],
    "opening": [],
    "technical": [],
    "behavioral": [],
    "closing": []
  }},
  "top_improvement_areas": []
}}""",
            max_tokens=4200,
        )

        score = self.chat_json(
            """You are the final Interview Forge scoring engine.
Return conservative scores from 0 to 100 based only on the evidence.
Job fit is alignment with requirements; technical readiness is evidence of technical
skills; behavioral readiness is evidence of communication/leadership/behavioral stories.
Do not score protected attributes.""",
            f"""Candidate:
{json.dumps(profile.get('candidate_profile', {}), ensure_ascii=False)}

Job:
{json.dumps(profile.get('job_company_profile', {}), ensure_ascii=False)}

Match:
{json.dumps(match, ensure_ascii=False)}

Risk:
{json.dumps(risk, ensure_ascii=False)}

Return:
{{
  "forge_score": {{
    "overall": 0,
    "job_fit": 0,
    "technical_readiness": 0,
    "behavioral_readiness": 0,
    "reasoning": ""
  }}
}}""",
            max_tokens=1400,
        )

        result = {}
        result.update(profile)
        result.update(match)
        result["interview_risk_engine"] = risk
        result["question_engine"] = questions
        result["answer_blueprint"] = blueprint
        result.update(tough)
        result.update(score)
        return result

    def evaluate_answer(self, question: str, answer: str, rag) -> str:
        context = rag.context(question, k=5, max_chars=7000)
        return self.chat(
            """You are a demanding but constructive interview coach.
Evaluate the answer against the candidate/job evidence. Do not invent facts.
Return sections: Score, What Worked, Weak Areas, Evidence to Add, Better Structure,
and a Short Improved Answer Framework.""",
            f"""QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

RETRIEVED EVIDENCE:
{context}""",
            max_tokens=1800,
        )
