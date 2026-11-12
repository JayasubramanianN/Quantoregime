import os
import glob
from pathlib import Path
from typing import List, Dict, Optional
import requests
from config.settings import settings
from config.logging_config import logger

class SECRAGLoader:
    """
    Ingestion engine for Central Bank speeches, policy minutes, and SEC Filings.
    Provides automated downloads via sec-edgar-downloader and curated institutional archives.
    """

    CURATED_MACRO_CORPUS = [
        {
            "id": "FOMC_2024_09",
            "source": "Federal Open Market Committee (FOMC)",
            "title": "FOMC Monetary Policy Statement & Rate Decision",
            "date": "2024-09-18",
            "category": "Central Bank Policy",
            "content": """
            Recent indicators suggest that economic activity has continued to expand at a solid pace. 
            Job gains have slowed, and the unemployment rate has edged up but remains low. 
            Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated.
            The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. 
            The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, 
            and judges that the risks to achieving its employment and inflation goals are roughly in balance. 
            The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.
            In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range 
            for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent. In considering additional adjustments 
            to the target range for the federal funds rate, the Committee will carefully assess incoming data, 
            the evolving outlook, and the balance of risks. The Committee will continue reducing its holdings of Treasury 
            securities and agency debt and agency mortgage-backed securities.
            """
        },
        {
            "id": "FOMC_2024_06",
            "source": "Federal Reserve Chair Jerome Powell",
            "title": "Monetary Policy Report & Labor Market Dynamics",
            "date": "2024-06-12",
            "category": "Central Bank Speech",
            "content": """
            The U.S. economy has made considerable progress toward our dual-mandate goals. Inflation has eased substantially 
            from its highs, while the labor market has remained resilient. However, inflation is still too high, and ongoing 
            progress in bringing it down is not assured. Restrictive monetary policy continues to exert downward pressure on demand, 
            particularly in interest-sensitive sectors like housing and corporate fixed capital investment.
            We have stated that we do not expect it will be appropriate to reduce the target range for the federal funds rate 
            until we have gained greater confidence that inflation is moving sustainably toward 2 percent. 
            Supply-demand balance in the labor market is normalizing: job openings have declined, and wage growth has moderated 
            toward levels consistent with long-run 2 percent inflation given productivity growth. 
            We remain committed to returning inflation to our 2 percent target and keeping longer-term inflation expectations well anchored.
            """
        },
        {
            "id": "ECB_2024_10",
            "source": "European Central Bank (ECB)",
            "title": "Governing Council Monetary Policy Decision",
            "date": "2024-10-17",
            "category": "Central Bank Policy",
            "content": """
            The Governing Council decided to lower the three key ECB interest rates by 25 basis points. 
            In particular, the decision to lower the deposit facility rate—the rate through which the Governing Council 
            steers the monetary policy stance—reflects an updated assessment of the inflation outlook, the dynamics of underlying 
            inflation and the strength of monetary policy transmission. The incoming information on inflation shows that 
            the disinflationary process is well on track. The inflation outlook is also affected by recent downside surprises in indicators 
            of economic activity. Meanwhile, financing conditions remain restrictive. 
            Inflation is expected to rise in the coming months, before declining to target in the course of next year. Domestic inflation 
            remains high, as wages are still rising at an elevated pace. At the same time, labor cost pressures are set to continue 
            moderating, with profits partially buffering their impact on inflation.
            """
        },
        {
            "id": "SEC_10K_MACRO_RISKS_JPM",
            "source": "SEC Form 10-K (JPMorgan Chase & Co.)",
            "title": "Item 1A: Macroeconomic & Capital Markets Risk Factors",
            "date": "2024-02-28",
            "category": "SEC Filing",
            "content": """
            Our financial results are impacted by global macroeconomic and geopolitical conditions. Unfavorable economic 
            conditions, including persistently elevated interest rates, inflationary pressure, and sovereign debt distress, 
            could adversely affect borrower creditworthiness and loan repayment performance. 
            Furthermore, sharp fluctuations in the yield curve, particularly prolonged yield curve inversion (negative 10Y-2Y spread), 
            compress Net Interest Margin (NIM) and can lead to unrealized losses in Held-To-Maturity (HTM) and Available-For-Sale (AFS) 
            securities portfolios. A severe liquidity contraction or unexpected rapid central bank rate adjustments could impair wholesale 
            funding markets, increase counterparty credit risk, and trigger sudden asset repricing across equity and fixed income benchmarks.
            """
        },
        {
            "id": "OIL_ENERGY_MACRO_2024",
            "source": "Energy Information Administration (EIA) & Macro Review",
            "title": "Global Crude Oil Supply Dynamics and Inflation Pass-Through",
            "date": "2024-08-15",
            "category": "Macro Intelligence",
            "content": """
            Crude oil price shocks have historically served as a critical exogenous variable driving headline consumer price indices (CPI) 
            and short-term inflation expectations. With OPEC+ production quotas remaining tightly disciplined, supply inelasticity 
            creates upside volatility in WTI and Brent crude futures contracts. Empirical econometric analyses demonstrate that 
            a sustained $10/barrel surge in West Texas Intermediate (CL=F) translates to an approximately 25-40 basis point increase in 
            headline annual inflation within 3 to 6 months. Conversely, slowing industrial demand from manufacturing hubs puts downward 
            pressure on commodities, offering central banks margin for monetary easing.
            """
        }
    ]

    def __init__(self):
        self.raw_dir = settings.DATA_RAW_DIR
        self.filings_dir = self.raw_dir / "sec_filings"
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = settings.SEC_USER_AGENT

    def download_filing(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 1
    ) -> List[Path]:
        """
        Downloads corporate SEC filings via sec-edgar-downloader.
        """
        downloaded_files: List[Path] = []
        try:
            from sec_edgar_downloader import Downloader
            # Email and company name required by SEC EDGAR policy
            email = "analyst@quantmacro.org"
            company = "QuantMacroAI"
            dl = Downloader(company, email, str(self.filings_dir))
            dl.get(form_type, ticker, limit=limit)
            
            # Find downloaded files
            pattern = str(self.filings_dir / f"sec-edgar-filings/{ticker}/{form_type}/**/*.txt")
            files = [Path(p) for p in glob.glob(pattern, recursive=True)]
            logger.info(f"Downloaded {len(files)} filings for {ticker} ({form_type}).")
            return files
        except Exception as e:
            logger.warning(f"Error downloading SEC filing for {ticker}: {e}")
            return []

    def load_all_text_documents(self) -> List[Dict[str, str]]:
        """
        Returns unified corpus of Central Bank speeches, policy reports, and filings.
        Combines curated macro corpus with any local parsed files.
        """
        docs = list(self.CURATED_MACRO_CORPUS)
        
        # Scan local text or markdown files in raw_dir
        for file_path in self.raw_dir.glob("*.txt"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                docs.append({
                    "id": file_path.stem,
                    "source": "Local Ingestion Archive",
                    "title": file_path.name,
                    "date": "2024-01-01",
                    "category": "Custom Document",
                    "content": content
                })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        return docs
