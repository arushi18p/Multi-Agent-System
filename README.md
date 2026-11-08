# Fraud Analysis App: Multi-Agent System on AWS EC2 and Streamlit Cloud

A fraud-analysis web app deployed on **AWS EC2** and **Streamlit Cloud**, with security controls in the app layer: input validation, prompt-injection defense, rate limiting, and secrets management.

**Live demo:** https://multiagentsystem3000.streamlit.app

*The EC2 instance is stopped when not in use to save free-tier hours, and its public IP changes on restart. The Streamlit link is the always-on demo.*

## Cloud Deployment

| | AWS EC2 | Streamlit Cloud |
|---|---|---|
| Purpose | Self-managed hosting to learn AWS deployment | Public demo |
| Instance / plan | t2.micro (free tier), Ubuntu | Community Cloud |
| Deploy method | git clone + pip install | Auto-deploy from this repo |
| Process management | nohup (background process, no auto-restart) | Managed |
| Secrets | `.env` file on the server, gitignored and never committed | Streamlit secrets manager |

**EC2 setup:**
1. Launched an instance and attached a key pair for SSH access
2. Configured the security group: SSH (22) and the Streamlit port (8501) both open to 0.0.0.0/0 (see limitations)
3. Installed dependencies and stored the API key in a `.env` file
4. Started the app with `python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0` and kept it running with `nohup`

## Security Controls

| Threat | Control |
|---|---|
| Malformed or malicious input | Regex allowlist (alphanumeric only) and length limits on all IDs |
| Prompt injection | Pattern blocklist (e.g. "ignore previous instructions") checked before any LLM call |
| API abuse / cost | Global rate limit of 10 requests/min (one shared bucket, not per user) |
| Credential leakage | API key kept in a gitignored `.env` or Streamlit secrets, never hardcoded |
| LLM making unsafe decisions | LLM limited to explanation only; final decision made by deterministic rules |
| Data exposure | Synthetic data only (BankSim); no real PII |

**Known limitations and next steps:**
- SSH is currently open to all IPs; restrict it to my IP
- The EC2 endpoint is plain HTTP; add HTTPS via a reverse proxy
- Rate limiting is one global limit; production would use per-user or IP-based limits
- The injection blocklist can be bypassed by rephrasing; the ID format check is the stronger control
- Move LLM access to AWS Bedrock, use an IAM role and Secrets Manager, and add CloudWatch logging

## What It Does

Three agents run in sequence (LangGraph):
1. **Intake:** validates and sanitizes input; stops the pipeline early on bad input
2. **Analysis:** looks up risk data and uses Gemini to write a short explanation
3. **Decision:** rule-based result: APPROVE, FLAG FOR REVIEW, or BLOCK

## Fraud Graph Analysis

Transactions from the BankSim dataset are modeled as a graph, with a Random Forest risk score on each edge. BFS finds the shortest path by edge count and ignores risk, while A* uses cost = 1 - risk so it is drawn toward suspicious paths. Sensitivity testing (edge noise, node removal, threshold sweeps, Monte Carlo) showed A* is much less stable than BFS when risk scores are noisy, which is why the LLM only explains the score and never makes the decision.

## Tech Stack

Python, Streamlit, LangGraph, Gemini API, NetworkX, scikit-learn, AWS EC2

## Live demo above. Local setup requires your own Gemini API key. Below are the instructions to run locally. 

```bash
git clone https://github.com/arushi18p/Multi-Agent-System
cd Multi-Agent-System
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
streamlit run app.py
```
