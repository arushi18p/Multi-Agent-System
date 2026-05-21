from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import re
import time
from collections import defaultdict

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# max 10 requests per minute per session
request_log = defaultdict(list)
MAX_REQUESTS = 10
WINDOW_SECONDS = 60

# Prompt injection patterns to block
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"you are now",
    r"act as",
    r"jailbreak",
    r"<script>",
    r"DROP TABLE",
    r"SELECT \*",
    r"system prompt",
]

def is_rate_limited(session_id: str) -> bool:
    now = time.time()
    request_log[session_id] = [
        t for t in request_log[session_id]
        if now - t < WINDOW_SECONDS
    ]
    if len(request_log[session_id]) >= MAX_REQUESTS:
        return True
    request_log[session_id].append(now)
    return False

def contains_injection(value: str) -> bool:
    lower = value.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False

def sanitize_input(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", value).strip()

def intake_agent(state: dict) -> dict:
    customer_raw = state.get("customer", "")
    merchant_raw = state.get("merchant", "")
    session_id = state.get("session_id", "default")

    if is_rate_limited(session_id):
        return {**state, "error": "Rate limit exceeded. Please wait before submitting again."}

    if contains_injection(customer_raw) or contains_injection(merchant_raw):
        return {**state, "error": "Invalid input detected. Potential injection attempt blocked."}

    if len(customer_raw) > 50 or len(merchant_raw) > 50:
        return {**state, "error": "Input too long. IDs must be under 50 characters."}
    customer = sanitize_input(customer_raw)
    merchant = sanitize_input(merchant_raw)

    if not customer or not merchant:
        return {**state, "error": "Missing customer or merchant ID."}

    if not customer.startswith("C"):
        return {**state, "error": "Invalid customer ID. Must start with C."}

    if not merchant.startswith("M"):
        return {**state, "error": "Invalid merchant ID. Must start with M."}

    return {
        **state,
        "customer": customer,
        "merchant": merchant,
        "error": None,
        "status": "intake_complete"
    }