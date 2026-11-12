import streamlit as st
import plotly.graph_objects as go
import re
from typing import Dict, Any, List
from src.llm_rag.macro_summarizer import MacroRAGSummarizer
from app.components.ui_styles import get_institutional_chart_layout

@st.cache_resource
def get_rag_summarizer() -> MacroRAGSummarizer:
    return MacroRAGSummarizer()

def is_single_ticker_or_micro_equity(query: str) -> bool:
    """
    Detects whether user inquiry is a pure single-stock ticker or micro-equity query without macro context.
    """
    q = query.strip()
    words = q.split()
    
    # Check if query is 1-2 words representing a ticker or company name
    common_tickers = {"AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NOK", "NOKIA", "NFLX", "AMD", "INTC", "BABA"}
    cleaned_ticker = re.sub(r'[^A-Za-z]', '', q).upper()
    
    if len(words) <= 2 and cleaned_ticker in common_tickers:
        return True
    
    # Phrases like "analyze nokia shares" or "buy apple stock"
    lower_q = q.lower()
    micro_patterns = [
        r"^(analyze|buy|sell|price target for)\s+[a-z0-9]+\s+(shares|stock)?$",
        r"^[a-z0-9]+\s+(stock|shares|earnings)$"
    ]
    for pat in micro_patterns:
        if re.match(pat, lower_q):
            return True

    return False

def render_macro_tab():
    st.markdown("### 🏛️ Macro Intelligence & Qualitative NLP RAG Engine")
    st.caption("Dense vector semantic retrieval and regime classification across Central Bank statements, speeches, and SEC filings.")

    rag = get_rag_summarizer()

    # 1. Macro Overview Metric Cards with Tooltips
    corpus_docs = rag.loader.load_all_text_documents()
    combined_corpus_text = " ".join([d.get("content", "") for d in corpus_docs])
    global_sentiment = rag.analyze_macro_sentiment(combined_corpus_text)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        regime_label = global_sentiment['macro_regime']
        badge_cls = "badge-hawkish" if "Hawkish" in regime_label else ("badge-dovish" if "Dovish" in regime_label else "badge-neutral")
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Dominant Regime</span>
                <span class="tooltip-icon" title="Current macroeconomic policy classification derived from central bank communication corpus">ℹ️</span>
            </div>
            <div style="margin: 4px 0;"><span class="{badge_cls}">{regime_label}</span></div>
            <div class="metric-subtext">Active Monetary Policy Bias</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        score = global_sentiment['hawkish_score']
        score_color = "#fb7185" if score > 0.1 else ("#34d399" if score < -0.1 else "#fbbf24")
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Policy Tone Score</span>
                <span class="tooltip-icon" title="Normalized net policy bias score: -1.0 (Dovish Easing) to +1.0 (Hawkish Tightening)">ℹ️</span>
            </div>
            <div class="metric-value" style="color: {score_color};">{score:+.2f}</div>
            <div class="metric-subtext">Scale: -1.0 (Dovish) to +1.0 (Hawkish)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Vector Knowledge Base</span>
                <span class="tooltip-icon" title="Total embedded text chunks indexed in the FAISS IndexFlatIP 384-dimensional dense vector space">ℹ️</span>
            </div>
            <div class="metric-value">{rag.vector_store.index.ntotal if rag.vector_store.index else len(corpus_docs)} Chunks</div>
            <div class="metric-subtext">FAISS IndexFlatIP (384-dim)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Hawk vs Dove Cues</span>
                <span class="tooltip-icon" title="Ratio of restrictive vs accommodative keyword occurrences across policy communication">ℹ️</span>
            </div>
            <div class="metric-value">{global_sentiment['hawk_mentions']} / {global_sentiment['dove_mentions']}</div>
            <div class="metric-subtext">{global_sentiment['risk_mentions']} Tail Risk Keywords Identified</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Sub-views: Search vs. Speech Archive Browser
    view_choice = st.radio(
        "Workspace View:",
        ["🔍 Semantic RAG Intelligence Search", "📚 Central Bank & 10-K Document Archive"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if view_choice == "🔍 Semantic RAG Intelligence Search":
        # Quick Query Chips
        st.markdown("##### ⚡ Quick Institutional Inquiries")
        chip_cols = st.columns(4)
        selected_chip = None
        if chip_cols[0].button("🏛️ FOMC Rate Trajectory", use_container_width=True):
            selected_chip = "How will sticky inflation impact Fed rate cut timing and balance sheet runoff?"
        if chip_cols[1].button("💼 Labor Market Cooling", use_container_width=True):
            selected_chip = "What did Chair Powell state regarding labor market normalization and wages?"
        if chip_cols[2].button("🏦 JPMorgan 10-K Risks", use_container_width=True):
            selected_chip = "What are the macroeconomic and liquidity risk factors in JPMorgan SEC 10-K?"
        if chip_cols[3].button("🛢️ Oil Price Pass-Through", use_container_width=True):
            selected_chip = "How does crude oil price volatility impact headline inflation expectations?"

        # Search Controls
        col_input, col_ctrl = st.columns([3, 1])
        with col_input:
            default_val = selected_chip if selected_chip else "How will sticky inflation impact Fed rate cut timing?"
            user_query = st.text_input(
                "Macro Query / Search Term:",
                value=default_val,
                placeholder="How will sticky inflation impact Fed rate cut timing?",
                help="Enter macroeconomic inquiries regarding monetary policy, inflation trends, yield curves, or systemic risks."
            )
        with col_ctrl:
            source_filter = st.selectbox(
                "Source Filter:",
                ["All Sources", "Central Bank", "SEC Filing", "Macro Intelligence"],
                help="Filter documents by publishing authority or filing category."
            )
            top_k = st.slider(
                "Top Chunks (k):",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of highest-relevance context passages injected into the qualitative synthesis prompt."
            )

        filter_val = None if source_filter == "All Sources" else source_filter

        # Domain Guardrail Check
        if is_single_ticker_or_micro_equity(user_query):
            st.info(
                "ℹ️ **QuantMacro AI is a top-down macro research engine.** "
                "For single-stock fundamental analysis, please formulate questions around macro drivers "
                "(e.g., *'How do rate hikes affect tech sector valuations?'* or *'What is the credit risk exposure of large-cap banks in 10-K filings?'*)."
            )

        if st.button("🚀 Execute Semantic Synthesis", type="primary", use_container_width=True):
            with st.spinner("Embedding query and retrieving dense vector context..."):
                rag_output = rag.query_and_summarize(user_query, top_k=top_k, source_filter=filter_val)
                sent = rag_output["sentiment_metrics"]

                # Two-column layout for Briefing and Sentiment Breakdown
                col_b1, col_b2 = st.columns([1.8, 1.2])

                with col_b1:
                    st.markdown(rag_output["executive_summary"])

                with col_b2:
                    st.markdown("#### 🎯 Tone & Topic Distribution")

                    # Donut chart for sentiment mentions
                    donut_labels = ["Hawkish / Restrictive", "Dovish / Accommodative", "Risk & Uncertainty"]
                    donut_values = [sent["hawk_mentions"], sent["dove_mentions"], sent["risk_mentions"]]
                    donut_colors = ["#f43f5e", "#10b981", "#f59e0b"]

                    fig_donut = go.Figure(data=[go.Pie(
                        labels=donut_labels,
                        values=donut_values,
                        hole=.6,
                        marker=dict(colors=donut_colors),
                        textinfo='percent',
                        hoverinfo='label+value+percent'
                    )])
                    chart_layout = get_institutional_chart_layout(title="Evidence Keyword Tone", height=230)
                    chart_layout["showlegend"] = True
                    chart_layout["legend"] = dict(orientation="v", y=0.5, x=1.0)
                    fig_donut.update_layout(**chart_layout)
                    st.plotly_chart(fig_donut, use_container_width=True)

                    # Gauge chart for net policy bias
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sent["hawkish_score"],
                        title={'text': "Policy Tone Score (-1 to +1)", 'font': {'size': 13, 'color': '#94a3b8'}},
                        gauge={
                            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#38bdf8"},
                            'bgcolor': "#1e293b",
                            'steps': [
                                {'range': [-1, -0.2], 'color': "rgba(16, 185, 129, 0.25)"},
                                {'range': [-0.2, 0.2], 'color': "rgba(245, 158, 11, 0.25)"},
                                {'range': [0.2, 1], 'color': "rgba(244, 63, 94, 0.25)"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        height=180,
                        margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#f3f4f6")
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Retrieved Context Cards with Cosine Similarity Tooltips
                st.markdown("#### 📑 Retrieved Evidence Sources")
                for doc in rag_output["retrieved_documents"]:
                    sim = doc.get("similarity_score", 0.0)
                    sim_pct = int(min(100, max(0, sim * 100)))
                    with st.expander(f"[{doc.get('source')}] {doc.get('title')} — Cosine Similarity: {sim:.4f} ({sim_pct}%)", expanded=False):
                        st.caption(f"**Category:** {doc.get('category')} | **Date:** {doc.get('date')} | **Metric:** Cosine Similarity in 384d FAISS space")
                        st.markdown(doc.get("content", "").strip())

    else:
        # Document Archive Browser
        st.markdown("##### 📚 Central Bank Speeches & SEC 10-K Archive")
        st.caption("Inspect verified institutional sources currently indexed in the FAISS vector database.")

        selected_doc_id = st.selectbox(
            "Select Document to Read:",
            [d["id"] for d in corpus_docs],
            format_func=lambda i: next(f"[{d['source']}] {d['title']} ({d['date']})" for d in corpus_docs if d["id"] == i)
        )

        chosen_doc = next(d for d in corpus_docs if d["id"] == selected_doc_id)
        doc_sent = rag.analyze_macro_sentiment(chosen_doc["content"])

        st.markdown(f"""
        <div class="section-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="badge-info">{chosen_doc['category']}</span>
                <span style="font-size: 0.82rem; color: #94a3b8; font-family: 'JetBrains Mono';">Date: {chosen_doc['date']}</span>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #f8fafc;">{chosen_doc['title']}</h4>
            <div style="font-size: 0.88rem; color: #38bdf8; font-weight: 600; margin-bottom: 14px;">Source: {chosen_doc['source']}</div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 8px; border: 1px solid #1e293b; font-size: 0.92rem; line-height: 1.7; color: #cbd5e1;">
                {chosen_doc['content']}
            </div>
            <div style="margin-top: 14px; display: flex; gap: 16px; font-size: 0.84rem; color: #94a3b8;">
                <span>Regime: <b style="color: #f8fafc;">{doc_sent['macro_regime']}</b></span>
                <span>Tone Score: <b style="color: #38bdf8;">{doc_sent['hawkish_score']:+.2f}</b></span>
                <span>Hawkish Mentions: <b style="color: #fb7185;">{doc_sent['hawk_mentions']}</b></span>
                <span>Dovish Mentions: <b style="color: #34d399;">{doc_sent['dove_mentions']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Document Uploader Expander
    st.markdown("---")
    with st.expander("➕ Upload Custom Speech / Report to Knowledge Base"):
        uploaded_file = st.file_uploader("Upload Text (.txt) or Central Bank Speech", type=["txt"])
        if uploaded_file is not None:
            text_str = uploaded_file.read().decode("utf-8", errors="ignore")
            if st.button("Index Uploaded Document"):
                save_path = rag.loader.raw_dir / uploaded_file.name
                save_path.write_text(text_str, encoding="utf-8")
                rag._initialize_knowledge_base()
                st.success(f"Indexed '{uploaded_file.name}' into FAISS vector space! Total vectors: {rag.vector_store.index.ntotal}")
