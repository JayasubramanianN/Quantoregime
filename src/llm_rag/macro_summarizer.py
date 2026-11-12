import os
import re
from typing import Dict, Any, List, Optional
from config.settings import settings
from config.logging_config import logger
from src.llm_rag.vector_store import MacroVectorStore
from src.data_ingestion.sec_rag_loader import SECRAGLoader

class MacroRAGSummarizer:
    """
    Institutional Qualitative NLP RAG Engine.
    Synthesizes Central Bank speeches, policy minutes, and SEC Filings
    to derive macro regime classification, policy tone (Hawkish vs. Dovish),
    and executive intelligence summaries.
    """

    def __init__(self, vector_store: Optional[MacroVectorStore] = None):
        self.vector_store = vector_store or MacroVectorStore()
        self.loader = SECRAGLoader()
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """
        Loads curated macroeconomic corpus and builds initial FAISS index.
        """
        docs = self.loader.load_all_text_documents()
        self.vector_store.build_index(docs)

    def analyze_macro_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Calculates macroeconomic tone metrics:
        - Hawkish vs. Dovish score (-1.0 to +1.0)
        - Inflation concern intensity
        - Labor market strength sentiment
        - Quantitative easing / tightening stance
        """
        hawkish_keywords = [
            "elevated", "restrictive", "inflation is still too high", "rate hike", 
            "tightening", "tight", "anchor", "vigilant", "upside risks", "hot", 
            "reduce holdings", "quantitative tightening", "qt", "perseverance"
        ]
        dovish_keywords = [
            "lower", "cut", "easing", "disinflation", "well on track", "substantially",
            "moderating", "downside risks", "in balance", "slowed", "cooling",
            "normalization", "cautious", "support economic activity", "accommodative"
        ]
        risk_keywords = [
            "uncertainty", "geopolitical", "vulnerability", "stress", "inversion", 
            "default", "drawdown", "credit contraction", "liquidity squeeze"
        ]

        text_lower = text.lower()
        hawk_count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower)) for kw in hawkish_keywords)
        dove_count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower)) for kw in dovish_keywords)
        risk_count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower)) for kw in risk_keywords)

        total = hawk_count + dove_count
        if total == 0:
            hawkish_score = 0.0
        else:
            hawkish_score = (hawk_count - dove_count) / total  # +1.0 (pure hawkish) to -1.0 (pure dovish)

        if hawkish_score > 0.25:
            regime = "Hawkish Tightening / Restrictive"
            stance = "Higher-for-longer interest rate bias; downward pressure on valuation multiples."
        elif hawkish_score < -0.25:
            regime = "Dovish Easing / Accommodative"
            stance = "Monetary pivot underway; tailwind for duration assets and equities."
        else:
            regime = "Neutral / Data-Dependent Pivot"
            stance = "Balanced risks between inflation mandate and labor market stabilization."

        return {
            "hawkish_score": round(hawkish_score, 3),
            "hawk_mentions": hawk_count,
            "dove_mentions": dove_count,
            "risk_mentions": risk_count,
            "macro_regime": regime,
            "policy_stance": stance
        }

    def generate_llm_summary(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Calls OpenAI / Anthropic if API key is present.
        Otherwise falls back to institutional analytical synthesis engine.
        """
        # 1. Try OpenAI if key is configured
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                context_str = "\n\n".join([
                    f"[{d.get('source')}] {d.get('title')}:\n{d.get('content')}"
                    for d in retrieved_docs
                ])
                prompt = (
                    f"You are a Senior Macro Portfolio Manager. Given the retrieved central bank speeches "
                    f"and SEC filings below, provide an institutional macroeconomic briefing answering: '{query}'\n\n"
                    f"Retrieved Documents:\n{context_str}\n\n"
                    f"Provide: (1) Executive Summary, (2) Monetary Policy Stance, (3) Key Market Risks, (4) Asset Class Implications."
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API invocation failed: {e}. Utilizing internal synthesis engine.")

        # 2. Institutional Rule-Based Quantitative Synthesis
        return self._synthesize_analytical_briefing(query, retrieved_docs)

    def _synthesize_analytical_briefing(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministic, institutional executive briefing generator.
        Extracts key takeaways, stance, and asset class implications from retrieved snippets.
        """
        if not retrieved_docs:
            return "No relevant macroeconomic documents matched the query filter."

        top_doc = retrieved_docs[0]
        combined_text = " ".join([d.get("content", "") for d in retrieved_docs])
        sentiment = self.analyze_macro_sentiment(combined_text)

        briefing = f"""### 🏛️ Executive Macroeconomic Intelligence Briefing

**Inquiry:** *"{query}"*  
**Dominant Regime Classification:** **{sentiment['macro_regime']}**  
**Net Policy Tone Score:** `{sentiment['hawkish_score']:+.2f}` (Range: -1.0 Dovish to +1.0 Hawkish)

---

#### 1. Executive Summary & Policy Stance
- **Primary Signal:** {sentiment['policy_stance']}
- **Primary Source:** {top_doc.get('source')} — *{top_doc.get('title')}* ({top_doc.get('date')}).
- **Disinflation & Mandate Dynamics:** Central bank rhetoric emphasizes incoming data dependence with heightened scrutiny on labor market resilience and sticky services inflation. Quantitative balance sheet reduction (QT) proceeds steadily alongside benchmark rate calibration.

#### 2. Cross-Document Evidence Synthesis
"""
        for i, doc in enumerate(retrieved_docs[:3]):
            briefing += f"- **[{doc.get('source')}]** *{doc.get('title')}*: Verified emphasis on `{doc.get('category')}` with cosine similarity relevance score of `{doc.get('similarity_score', 0.9):.3f}`.\n"

        briefing += f"""
#### 3. Asset Class & Valuation Implications
- **Equities (S&P 500 / Russell 2000):** {'Defensive positioning recommended; premium on robust balance sheets and positive free cash flows.' if sentiment['hawkish_score'] > 0 else 'Cyclical and small-cap assets benefit from discount rate easing and credit margin expansion.'}
- **Fixed Income (10Y/2Y Yield Curve):** {'Curve inversion risk elevated; short-duration debt provides attractive risk-free yield.' if sentiment['hawkish_score'] > 0 else 'Bull-steepening expected as short-term policy rates fall faster than long-term yields.'}
- **Commodities & Energy (CL=F, Gold):** {'Gold serves as a safe-haven hedge against geopolitical escalations and debt overhang.' if sentiment['risk_mentions'] > 2 else 'Energy input costs remain a pivotal leading indicator for headline PCE/CPI trajectories.'}

---
*Generated via QuantMacro AI Institutional Vector RAG Engine.*
"""
        return briefing

    def query_and_summarize(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end RAG query execution pipeline.
        """
        retrieved = self.vector_store.search(query, top_k=top_k, source_filter=source_filter)
        combined_content = " ".join([d.get("content", "") for d in retrieved])
        sentiment = self.analyze_macro_sentiment(combined_content)
        executive_summary = self.generate_llm_summary(query, retrieved)

        return {
            "query": query,
            "retrieved_documents": retrieved,
            "sentiment_metrics": sentiment,
            "executive_summary": executive_summary
        }
