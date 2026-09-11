import json
import streamlit as st

from rag import RAGPipeline
from interview_engine import InterviewForgeEngine

st.set_page_config(
    page_title="Interview Forge",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {color: #6b7280; margin-bottom: 1.5rem;}
    .score-card {padding: 1rem; border-radius: 0.8rem; border: 1px solid #e5e7eb; text-align:center;}
    .small {font-size: 0.85rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚒️ Interview Forge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">RAG-powered interview preparation from your CV, job description, and company context.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Inputs")
    cv_file = st.file_uploader("Upload CV / Resume", type=["pdf", "txt"], key="cv")
    jd_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"], key="jd")
    company = st.text_input("Company (optional)", placeholder="e.g. Google, Huawei, Siemens")
    role = st.text_input("Target role (optional)", placeholder="e.g. ML Engineer")

    st.divider()
    st.header("2. Interview settings")
    interview_type = st.multiselect(
        "Question categories",
        ["HR", "Behavioral", "Technical", "Resume-based", "Job-specific", "Company-specific"],
        default=["HR", "Behavioral", "Technical", "Resume-based", "Job-specific"],
    )
    questions_per_category = st.slider("Questions per category", 1, 5, 3)
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    top_k = st.slider("RAG chunks per analysis", 2, 8, 5)

    analyze = st.button("🔥 Forge Interview Plan", type="primary", use_container_width=True)

    st.divider()
    st.caption("Free-stack MVP: Streamlit + local embeddings + FAISS + Groq.")

if "result" not in st.session_state:
    st.session_state.result = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "error" not in st.session_state:
    st.session_state.error = None

def read_uploaded(file):
    if file is None:
        return ""
    data = file.getvalue()
    if file.name.lower().endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    # PDF extraction is kept in rag.py so the same parser is used by the RAG layer.
    return RAGPipeline.extract_pdf_bytes(data)

if analyze:
    st.session_state.error = None

    if cv_file is None:
        st.error("Please upload a CV / resume.")
        st.stop()

    with st.spinner("Reading documents and building your local FAISS knowledge base..."):
        cv_text = read_uploaded(cv_file)
        jd_text = read_uploaded(jd_file) if jd_file else ""

        if len(cv_text.strip()) < 80:
            st.error("The CV contains too little selectable text. Try a text-based PDF or TXT file.")
            st.stop()

        rag = RAGPipeline()
        documents = [
            {"source": "CV", "text": cv_text},
        ]
        if jd_text.strip():
            documents.append({"source": "JOB_DESCRIPTION", "text": jd_text})
        if company.strip():
            documents.append({"source": "COMPANY_CONTEXT", "text": f"Company: {company}\nTarget role: {role}"})

        stats = rag.build(documents)
        st.session_state.rag = rag

    with st.spinner("Running information check and extracting candidate/job profiles..."):
        engine = InterviewForgeEngine()
        try:
            result = engine.forge(
                rag=st.session_state.rag,
                company=company,
                role=role,
                categories=interview_type,
                questions_per_category=questions_per_category,
                difficulty=difficulty,
                top_k=top_k,
            )
            st.session_state.result = result
            st.session_state.result["_stats"] = stats
        except Exception as exc:
            st.session_state.error = str(exc)
            st.session_state.result = None

if st.session_state.error:
    st.error(st.session_state.error)

result = st.session_state.result

if result:
    # Top score row
    scores = result.get("forge_score", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forge Score", f"{scores.get('overall', 0)}/100")
    c2.metric("Job Fit", f"{scores.get('job_fit', 0)}/100")
    c3.metric("Technical Readiness", f"{scores.get('technical_readiness', 0)}/100")
    c4.metric("Behavioral Readiness", f"{scores.get('behavioral_readiness', 0)}/100")

    tabs = st.tabs([
        "📋 Profiles & Match",
        "⚠️ Risk Engine",
        "❓ Question Engine",
        "🧠 Answer Blueprint",
        "🎯 Battle Plan",
        "🔁 Practice Loop",
        "📚 RAG Evidence",
    ])

    with tabs[0]:
        st.subheader("Candidate Profile")
        st.json(result.get("candidate_profile", {}))
        st.subheader("Job + Company Profile")
        st.json(result.get("job_company_profile", {}))
        st.subheader("Information Check")
        info = result.get("information_check", {})
        st.write("**Status:**", info.get("status", "unknown"))
        missing = info.get("missing_information", [])
        if missing:
            st.warning("Missing / uncertain information")
            for item in missing:
                st.write(f"- {item}")
        followups = info.get("follow_up_questions", [])
        if followups:
            st.write("**Follow-up questions:**")
            for q in followups:
                st.write(f"- {q}")

        st.subheader("Job Match Analysis")
        st.markdown(result.get("job_match_analysis", "No analysis returned."))

    with tabs[1]:
        risk = result.get("interview_risk_engine", {})
        st.subheader("SWOT")
        cols = st.columns(4)
        for col, title, key in zip(
            cols,
            ["Strengths", "Weaknesses", "Opportunities", "Threats"],
            ["strengths", "weaknesses", "opportunities", "threats"],
        ):
            with col:
                st.markdown(f"### {title}")
                for item in risk.get("swot", {}).get(key, []):
                    st.write(f"- {item}")

        st.subheader("30-second / 60-second risk areas")
        short = risk.get("risk_areas", {}).get("30_seconds", [])
        long = risk.get("risk_areas", {}).get("60_seconds", [])
        a, b = st.columns(2)
        with a:
            st.markdown("**30-second answers**")
            for x in short:
                st.write(f"- {x}")
        with b:
            st.markdown("**60-second answers**")
            for x in long:
                st.write(f"- {x}")

    with tabs[2]:
        questions = result.get("question_engine", {})
        for category, items in questions.items():
            with st.expander(category, expanded=True):
                if isinstance(items, list):
                    for i, item in enumerate(items, 1):
                        if isinstance(item, dict):
                            st.markdown(f"**{i}. {item.get('question', '')}**")
                            if item.get("why"):
                                st.caption(f"Why: {item['why']}")
                            if item.get("risk"):
                                st.caption(f"Risk: {item['risk']}")
                        else:
                            st.markdown(f"**{i}. {item}**")

    with tabs[3]:
        blueprint = result.get("answer_blueprint", {})
        st.subheader("Question Priority")
        priorities = blueprint.get("question_priority", [])
        for p in priorities:
            st.write(f"**{p.get('priority', 'MEDIUM')}** — {p.get('question', '')}")

        st.subheader("Answer Blueprint")
        for item in blueprint.get("answers", []):
            with st.expander(item.get("question", "Answer")):
                st.markdown("**Why asked**")
                st.write(item.get("why_asked", ""))
                st.markdown("**What to show**")
                st.write(item.get("what_to_show", ""))
                st.markdown("**CV evidence**")
                st.write(item.get("cv_evidence", ""))
                st.markdown("**Answer structure**")
                st.write(item.get("answer_structure", ""))

        st.subheader("Tough Question Coach")
        for item in result.get("tough_question_coach", []):
            with st.expander(item.get("question", "Tough question")):
                st.write("**Trap:**", item.get("trap", ""))
                st.write("**Coaching:**", item.get("coaching", ""))
                st.write("**Recommended answer angle:**", item.get("answer_angle", ""))

    with tabs[4]:
        plan = result.get("interview_battle_plan", {})
        for section in ["before_interview", "opening", "technical", "behavioral", "closing"]:
            if plan.get(section):
                st.subheader(section.replace("_", " ").title())
                for item in plan[section]:
                    st.write(f"- {item}")
        st.subheader("Top Improvement Areas")
        for item in result.get("top_improvement_areas", []):
            st.write(f"- {item}")

    with tabs[5]:
        st.subheader("Refine & Practice Loop")
        st.info(
            "After each practice answer, paste the answer below. Interview Forge will "
            "evaluate it against the retrieved CV/job evidence and return weak areas."
        )
        answer = st.text_area("Practice answer", height=180)
        practice_question = st.text_input("Question you answered")
        if st.button("Evaluate practice answer"):
            if not answer.strip() or not practice_question.strip():
                st.warning("Enter both the question and your answer.")
            else:
                with st.spinner("Coaching..."):
                    engine = InterviewForgeEngine()
                    feedback = engine.evaluate_answer(
                        question=practice_question,
                        answer=answer,
                        rag=st.session_state.rag,
                    )
                st.markdown(feedback)

    with tabs[6]:
        st.subheader("Local RAG knowledge base")
        stats = result.get("_stats", {})
        st.write(
            f"Indexed **{stats.get('chunks', 0)} chunks** from "
            f"{stats.get('documents', 0)} document(s)."
        )
        query = st.text_input("Test a retrieval query", placeholder="What projects prove my Python skills?")
        if query:
            hits = st.session_state.rag.retrieve(query, k=top_k)
            for hit in hits:
                st.markdown(f"**{hit['source']} · score {hit['score']:.3f}**")
                st.write(hit["text"][:1200])

    st.download_button(
        "⬇️ Download interview plan JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="interview_forge_plan.json",
        mime="application/json",
    )
else:
    st.info(
        "Upload your CV and optionally a job description, then click **Forge Interview Plan**. "
        "The app follows the architecture in your diagram: profile extraction → information check → "
        "job match → risk engine → question engine → priority → answer blueprint → tough-question coach → "
        "battle plan → Forge Score → improvement loop."
    )
    st.markdown("### Free MVP architecture")
    st.code(
        "CV + Job Description\n"
        "        ↓\n"
        "Candidate / Job Profiles\n"
        "        ↓\n"
        "Information Check → Follow-up Questions\n"
        "        ↓\n"
        "Job Match Analysis\n"
        "        ↓\n"
        "Interview Risk Engine → SWOT + 30s/60s Risk Areas\n"
        "        ↓\n"
        "Question Engine → HR / Behavioral / Technical / Resume / Job / Company\n"
        "        ↓\n"
        "Question Priority → High / Medium / Low\n"
        "        ↓\n"
        "Answer Blueprint → Why Asked / What to Show / CV Evidence\n"
        "        ↓\n"
        "Tough Question Coach\n"
        "        ↓\n"
        "Interview Battle Plan\n"
        "        ↓\n"
        "Forge Score → Job Fit / Technical / Behavioral\n"
        "        ↓\n"
        "Top Improvement Areas → Refine & Practice Loop",
        language="text",
    )
