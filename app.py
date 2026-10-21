"""Streamlit frontend for the Agentic RAG-Based Scholarship Research Assistant.

Run with:  streamlit run app.py

All business logic lives in ``src/scholarship_assistant``; this file only renders the UI
and calls :class:`scholarship_assistant.services.ScholarshipAssistant`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from scholarship_assistant.config import get_settings  # noqa: E402
from scholarship_assistant.models import ScholarshipReport, VerificationStatus  # noqa: E402
from scholarship_assistant.services import ScholarshipAssistant, ServiceError  # noqa: E402
from scholarship_assistant.utils import format_timestamp, setup_logging, slugify  # noqa: E402

st.set_page_config(page_title="Scholarship Research Assistant", page_icon="🎓", layout="wide")

EXAMPLE_QUESTIONS = [
    "What is this scholarship about?",
    "Who is eligible?",
    "What is the maximum financial assistance?",
    "Which documents are required?",
    "How do I apply?",
    "What is the application deadline?",
]


@st.cache_resource(show_spinner=False)
def get_assistant() -> ScholarshipAssistant:
    settings = get_settings()
    setup_logging(settings.log_level)
    return ScholarshipAssistant(settings)


def init_state() -> None:
    st.session_state.setdefault("active", None)       # active scholarship name
    st.session_state.setdefault("chats", {})          # slug -> list of messages
    st.session_state.setdefault("last_result", None)  # ResearchResult from this session


# --------------------------------------------------------------------------- sidebar
def render_sidebar(assistant: ScholarshipAssistant) -> bool:
    status = assistant.status()
    with st.sidebar:
        st.header("🎓 Scholarship Assistant")

        st.subheader("Setup status")
        st.markdown(
            f"{'✅' if status.groq_configured else '⚠️'} **Groq LLM** — "
            f"{status.groq_model if status.groq_configured else 'not configured'}\n\n"
            f"{'✅' if status.search_configured else '⚠️'} **Web search (Tavily)** — "
            f"{'configured' if status.search_configured else 'not configured'}\n\n"
            f"ℹ️ **Embeddings** — {status.embedding_backend}"
            + (f" (`{status.embedding_model.split('/')[-1]}`)" if status.embedding_backend == "sentence-transformers" else "")
        )
        if status.problems:
            with st.expander("Configuration notes", expanded=not status.search_configured):
                for problem in status.problems:
                    st.caption(f"• {problem}")
                st.caption("Copy `.env.example` to `.env`, add your keys, then restart the app.")

        demo = st.toggle(
            "Demo mode (sample data)",
            value=not status.search_configured,
            help="Uses fictional sample scholarships so you can try the UI, RAG and PDF report without API keys.",
        )

        st.subheader("Knowledge bases")
        try:
            names = assistant.indexed_scholarships()
        except ServiceError as exc:
            st.error(str(exc))
            names = []
        if names:
            current = st.session_state.active if st.session_state.active in names else names[0]
            st.session_state.active = st.radio(
                "Active scholarship", names, index=names.index(current),
                help="Each scholarship has its own isolated FAISS index.",
            )
            if st.button("Delete this knowledge base", width="stretch"):
                assistant.delete_scholarship(st.session_state.active)
                st.session_state.chats.pop(slugify(st.session_state.active), None)
                st.session_state.active = None
                st.rerun()
        else:
            st.caption("No scholarships researched yet.")
    return demo


# --------------------------------------------------------------------------- research tab
def status_badge(report: ScholarshipReport, field_name: str) -> str:
    if report.is_sample:
        return "🧪 Sample"
    status = report.field_status.get(field_name, VerificationStatus.UNVERIFIED)
    icon = {"verified_official": "✅", "secondary_only": "🟡", "conflicting": "⛔", "unverified": "❔"}.get(status.value, "")
    return f"{icon} {status.label}"


def render_report_summary(assistant: ScholarshipAssistant, report: ScholarshipReport) -> None:
    if report.is_sample:
        st.info("🧪 **Sample data** — fictional content for demonstrating the app. Not real scholarship information.")

    cols = st.columns(4)
    cols[0].metric("Sources", len(report.sources))
    cols[1].metric("Conflicts", len(report.conflicts))
    cols[2].metric("Unconfirmed fields", 0 if report.is_sample else len(report.unverified_fields))
    cols[3].metric("Closing date", report.closing_date or "Not found")

    if report.conflicts:
        for conflict in report.conflicts:
            st.warning(f"⚠️ {conflict}")
    st.caption(f"🗓️ {report.deadline_note}")

    st.markdown(f"**Provider:** {report.provider or 'Not identified'}")
    st.markdown(report.overview or "_No overview found._")

    sections = [
        ("Eligibility criteria", "eligibility"),
        ("Amount and benefits", "benefits"),
        ("Required documents", "required_documents"),
        ("Application procedure", "application_steps"),
        ("Terms and conditions", "terms"),
    ]
    left, right = st.columns(2)
    for i, (title, field_name) in enumerate(sections):
        with (left if i % 2 == 0 else right).expander(f"{title} · {status_badge(report, field_name)}", expanded=i < 2):
            items = getattr(report, field_name)
            if items:
                st.markdown("\n".join(f"- {item}" for item in items))
            else:
                st.caption("Not found in the collected sources.")

    if report.application_url:
        st.markdown(f"**Application link** ({status_badge(report, 'application_url')}): {report.application_url}")

    pdf_path = assistant.report_pdf_path(report.name)
    if pdf_path.exists():
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            type="primary",
        )
    st.caption(f"Extraction: {report.extraction_method or 'n/a'} · Generated {format_timestamp(report.generated_at)}")


def render_research_tab(assistant: ScholarshipAssistant, demo: bool) -> None:
    st.subheader("Research a scholarship")
    if demo:
        samples = assistant.list_samples()
        if not samples:
            st.error("No sample files found in data/sample/.")
            return
        st.caption("Demo mode: pick a fictional sample scholarship. No web search or API key is used.")
        name = st.selectbox("Sample scholarship", samples)
    else:
        name = st.text_input("Scholarship name", placeholder="e.g. the full official name of the scholarship")

    if st.button("🔎 Research", type="primary", disabled=not name):
        progress = st.progress(0.0, text="Starting…")

        def on_progress(message: str, fraction: float) -> None:
            progress.progress(min(max(fraction, 0.0), 1.0), text=message)

        try:
            with st.spinner("Researching… (the first run also downloads the embedding model)"):
                result = assistant.research(name, progress=on_progress, demo=demo)
        except ServiceError as exc:
            progress.empty()
            st.error(str(exc))
            return
        except Exception as exc:  # unexpected; keep the app alive
            progress.empty()
            st.exception(exc)
            return
        st.session_state.last_result = result
        st.session_state.active = result.report.name
        st.success(f"Research complete: {result.chunks_indexed} chunks indexed for **{result.report.name}**.")
        for error in result.errors:
            st.warning(error)
        with st.expander("Research log"):
            for line in result.status_log:
                st.text(line)

    active = st.session_state.active
    if active:
        report = assistant.get_report(active)
        if report:
            st.divider()
            st.subheader(report.name)
            render_report_summary(assistant, report)


# --------------------------------------------------------------------------- chat tab
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(
                f"**[{src['number']}]** [{src['title']}]({src['url']}) · {src['source_type']} · "
                f"similarity {src['score']:.2f}"
            )


def render_chat_tab(assistant: ScholarshipAssistant) -> None:
    active = st.session_state.active
    if not active:
        st.info("Research a scholarship first (or load a sample in demo mode), then ask questions here.")
        return
    st.subheader(f"Ask about: {active}")
    st.caption("Try: " + " · ".join(EXAMPLE_QUESTIONS))

    history: list[dict] = st.session_state.chats.setdefault(slugify(active), [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("notice"):
                st.caption(f"ℹ️ {message['notice']}")
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about this scholarship")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            try:
                plain_history = [{"role": m["role"], "content": m["content"]} for m in history]
                response = assistant.ask(active, question, plain_history)
            except ServiceError as exc:
                st.error(str(exc))
                return
        answer = response.answer if response.sufficient_evidence else f"⚠️ {response.answer}"
        st.markdown(answer)
        if response.notice:
            st.caption(f"ℹ️ {response.notice}")
        sources = [vars(s) for s in response.sources]
        render_sources(sources)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer, "sources": sources, "notice": response.notice})


# --------------------------------------------------------------------------- sources tab
def render_sources_tab(assistant: ScholarshipAssistant) -> None:
    active = st.session_state.active
    report = assistant.get_report(active) if active else None
    if not report:
        st.info("No researched scholarship selected yet.")
        return
    st.subheader(f"Sources for {report.name}")
    rows = [
        {
            "Title": s.title,
            "URL": s.url,
            "Type": s.source_type.label,
            "Retrieved": format_timestamp(s.retrieved_at),
            "Status": (VerificationStatus.SAMPLE if s.is_sample else s.verification).label,
            "Note": s.error,
        }
        for s in report.sources
    ]
    st.dataframe(rows, width="stretch", hide_index=True,
                 column_config={"URL": st.column_config.LinkColumn("URL")})
    st.caption("Official government and provider sources are read first. Wikipedia is used for background only.")


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    try:
        assistant = get_assistant()
    except Exception as exc:
        st.error(f"The app could not start: {exc}")
        return
    demo = render_sidebar(assistant)

    st.title("Scholarship Research Assistant")
    st.caption("Agentic research · grounded answers with citations · downloadable PDF reports")
    research_tab, chat_tab, sources_tab = st.tabs(["🔎 Research", "💬 Ask Questions", "📚 Sources"])
    with research_tab:
        render_research_tab(assistant, demo)
    with chat_tab:
        render_chat_tab(assistant)
    with sources_tab:
        render_sources_tab(assistant)


main()
