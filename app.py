import json
from typing import List, Optional

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Loan Triage Agent",
    page_icon="💸",
    layout="wide",
)

st.markdown("""
<style>
:root {
    --bg-card: #0b1730;
    --bg-soft: rgba(255,255,255,0.04);
    --border-soft: rgba(120, 160, 255, 0.16);
    --text-muted: #9ca3af;

    --green-bg: rgba(34,197,94,0.16);
    --green-text: #86efac;

    --yellow-bg: rgba(250,204,21,0.18);
    --yellow-text: #fde68a;

    --red-bg: rgba(239,68,68,0.18);
    --red-text: #fca5a5;

    --orange-bg: rgba(249,115,22,0.18);
    --orange-text: #fdba74;
}

html, body, [class*="css"] {
    font-size: 16px;
}

.block-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.card-title {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 18px;
    color: white;
}

.field-label {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.35;
    margin-bottom: 6px;
}

.field-value {
    color: white;
    font-size: 1.15rem;
    font-weight: 600;
    line-height: 1.45;
    margin-bottom: 18px;
}

.summary-label {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 10px;
    line-height: 1.35;
}

.summary-value {
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 10px;
}

.pill {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.98rem;
    line-height: 1.2;
    margin-bottom: 12px;
}

.pill-high {
    background: var(--green-bg);
    color: var(--green-text);
}

.pill-medium {
    background: var(--yellow-bg);
    color: var(--yellow-text);
}

.pill-low {
    background: var(--red-bg);
    color: var(--red-text);
}

.status-chip {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 0.98rem;
    font-weight: 700;
    line-height: 1.35;
    margin-top: 2px;
    margin-bottom: 8px;
}

.status-green {
    background: var(--green-bg);
    color: var(--green-text);
}

.status-red {
    background: var(--red-bg);
    color: var(--red-text);
}

.status-orange {
    background: var(--orange-bg);
    color: var(--orange-text);
}

.card-list ul {
    padding-left: 22px;
    margin-top: 8px;
    margin-bottom: 4px;
}

.card-list li {
    margin-bottom: 14px;
    color: white;
    line-height: 1.55;
    font-size: 1rem;
}

.reply-box {
    background: var(--bg-soft);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    white-space: pre-wrap;
    color: white;
    line-height: 1.65;
    font-size: 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    min-height: 210px;
}

.small-note {
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.45;
    margin-top: 10px;
}

.subsection-title {
    color: white;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.summary-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1px 1fr 1px 1.2fr;
    gap: 0;
    align-items: center;
}

.summary-section {
    min-width: 0;
    padding: 0 32px;
}

.summary-section:first-child {
    padding-left: 0;
}

.summary-section:last-child {
    padding-right: 0;
}

.summary-divider-line {
    width: 1px;
    height: 60px;
    background: rgba(255, 255, 255, 0.10);
}

.summary-progress-track {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.10);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 16px;
}

.summary-progress-fill {
    height: 100%;
    background: #60a5fa;
    border-radius: 999px;
}

.summary-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 8px 0 0 0;
}

div[data-testid="stProgressBar"] > div > div > div > div {
    background-color: #60a5fa;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

@media (max-width: 900px) {
    .summary-grid {
        grid-template-columns: 1fr;
        gap: 20px;
    }
    .summary-divider-line {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("💸 AI Loan Triage Agent")
st.caption("Analyze borrower inquiries, flag risks, and recommend the next action.")


# -----------------------------
# OpenAI client
# -----------------------------
def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


# -----------------------------
# Output schema
# -----------------------------
class TriageResult(BaseModel):
    business_name: Optional[str] = None
    owner_name: Optional[str] = None
    location: Optional[str] = None
    years_in_business: Optional[float] = None
    monthly_revenue: Optional[float] = None
    requested_amount: Optional[float] = None
    use_of_funds: Optional[str] = None
    existing_debt: Optional[str] = None

    missing_information: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    positive_signals: List[str] = Field(default_factory=list)

    confidence_score: float = 0.0
    priority: str = "Medium"
    recommended_action: str = "Request more information"
    explanation: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    draft_reply_email: Optional[str] = None


# -----------------------------
# LLM prompt
# -----------------------------
SYSTEM_PROMPT = """
You are an AI loan application triage analyst for a fintech lender serving SMBs.

Your job is to read an inbound borrower email and produce a structured triage result.

Rules:
1. Do not invent facts.
2. If a field is not explicitly supported, leave it null and add it to missing_information where relevant.
3. Normalize money values as numbers without currency symbols or commas.
4. Focus on lending-workflow risks, not generic AI risks.
5. Recommended actions must be one of:
   - Advance to underwriting
   - Request more information
   - Reject / low priority
6. Priority must be one of:
   - High
   - Medium
   - Low
7. Confidence score must be between 0 and 1.
8. Explanation must be concise and business-relevant.
9. Follow-up questions should be practical for a loan officer to ask.
10. Draft reply email should be professional, short, and no more than 80 words.
11. Prefer "Advance to underwriting" when the borrower shows credible operating history, clear revenue, and a plausible use case, even if some standard underwriting documents are still missing.
12. Missing information should be practical and not excessive. Only include the most important items.
"""


def build_user_prompt(email_text: str) -> str:
    return f"""
Analyze the following borrower inquiry email.

Borrower email:
\"\"\"
{email_text}
\"\"\"

Return a JSON object with this exact structure:
{{
  "business_name": string|null,
  "owner_name": string|null,
  "location": string|null,
  "years_in_business": number|null,
  "monthly_revenue": number|null,
  "requested_amount": number|null,
  "use_of_funds": string|null,
  "existing_debt": string|null,
  "missing_information": [string],
  "risk_flags": [string],
  "positive_signals": [string],
  "confidence_score": number,
  "priority": "High" | "Medium" | "Low",
  "recommended_action": "Advance to underwriting" | "Request more information" | "Reject / low priority",
  "explanation": [string],
  "follow_up_questions": [string],
  "draft_reply_email": string|null
}}

Important workflow-specific risk examples:
- unclear revenue cadence (monthly vs annual)
- debt burden already exists
- expansion risk for second location
- vague use of funds
- insufficient business history
- missing business fundamentals needed for underwriting
- possible low-quality lead due to lack of detail

Do not include markdown. Return valid JSON only.
"""


# -----------------------------
# LLM call
# -----------------------------
def analyze_email(email_text: str) -> TriageResult:
    client = get_client()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(email_text)},
        ],
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return TriageResult(**data)


# -----------------------------
# Helper UI functions
# -----------------------------
def get_priority_class(priority: str) -> str:
    p = (priority or "").lower()
    if p == "high":
        return "pill pill-high"
    if p == "medium":
        return "pill pill-medium"
    return "pill pill-low"


def format_value(label, value):
    if value in [None, "", []]:
        return "Not provided"
    if isinstance(value, (int, float)) and ("Revenue" in label or "Amount" in label):
        return f"${value:,.0f}"
    if label == "Years in Business" and isinstance(value, (int, float)):
        return f"{value:.0f}"
    return str(value)


def render_html_list(items, empty_text="None identified.", list_type="default"):
    if not items:
        items = [empty_text]

    if list_type == "positive":
        bullet_class = "status-green"
    elif list_type == "risk":
        bullet_class = "status-red"
    elif list_type == "missing":
        bullet_class = "status-orange"
    else:
        bullet_class = "status-orange"

    lis = "".join(
        f"<li><span class='status-chip {bullet_class}'>{item}</span></li>"
        for item in items
    )
    return f"<div class='card-list'><ul>{lis}</ul></div>"


def card_start(title: str):
    st.markdown(f"<div class='block-card'><div class='card-title'>{title}</div>", unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Main input
# -----------------------------
default_email = st.session_state.get("sample_email", "")

email_text = st.text_area(
    "Paste borrower inquiry email",
    value=default_email,
    height=220,
    placeholder="Paste the inbound financing email here...",
)

analyze_clicked = st.button("Run AI Triage", type="primary")


# -----------------------------
# Run analysis
# -----------------------------
if analyze_clicked:
    if not email_text.strip():
        st.warning("Please paste an email first.")
        st.stop()

    with st.spinner("Analyzing borrower profile..."):
        try:
            result = analyze_email(email_text)
        except json.JSONDecodeError:
            st.error("The model returned invalid JSON. Try again.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.divider()

    progress_pct = int(max(0.0, min(1.0, result.confidence_score)) * 100)
    priority_class = get_priority_class(result.priority)

    summary_html = (
    f'<div style="padding: 3px 3px 3px 4px; background: #60a5fa; border-radius: 20px; margin-bottom: 24px;">'
    f'<div class="summary-card" style="margin-bottom: 0;">'
    f'<div class="summary-grid">'
    f'<div class="summary-section"><div class="summary-label">Priority</div>'
    f'<div class="{priority_class}">{result.priority}</div></div>'
    f'<div class="summary-divider-line"></div>'
    f'<div class="summary-section"><div class="summary-label">Confidence</div>'
    f'<div class="summary-value">{result.confidence_score:.2f}</div>'
    f'<div class="summary-progress-track"><div class="summary-progress-fill" style="width: {progress_pct}%;"></div></div></div>'
    f'<div class="summary-divider-line"></div>'
    f'<div class="summary-section"><div class="summary-label">Recommended Action</div>'
    f'<div class="summary-value">{format_value("Recommended Action", result.recommended_action)}</div></div>'
    f'</div></div></div>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    with st.expander("🔍 Why this decision?"):
        if result.explanation:
            for item in result.explanation:
                st.write(f"• {item}")
        else:
            st.write("No additional explanation available.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Row 1
    row1_col1, row1_col2 = st.columns(2, gap="large")

    with row1_col1:
        card_start("🏢 Borrower Snapshot")
        snap_left, snap_right = st.columns(2)

        with snap_left:
            st.markdown("<div class='field-label'>Business Name</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Business Name', result.business_name)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Owner Name</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Owner Name', result.owner_name)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Location</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Location', result.location)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Years in Business</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Years in Business', result.years_in_business)}</div>", unsafe_allow_html=True)

        with snap_right:
            st.markdown("<div class='field-label'>Monthly Revenue</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Monthly Revenue', result.monthly_revenue)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Requested Amount</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Requested Amount', result.requested_amount)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Use of Funds</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Use of Funds', result.use_of_funds)}</div>", unsafe_allow_html=True)

            st.markdown("<div class='field-label'>Existing Debt</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='field-value'>{format_value('Existing Debt', result.existing_debt)}</div>", unsafe_allow_html=True)

        card_end()

    with row1_col2:
        card_start("⚖️ Decision Summary")
        st.markdown("<div class='field-label'>Priority</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='{get_priority_class(result.priority)}'>{result.priority}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='field-label'>Recommended Action</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='field-value'>{format_value('Recommended Action', result.recommended_action)}</div>", unsafe_allow_html=True)

        st.markdown("<div class='field-label'>Confidence Score</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='field-value'>{result.confidence_score:.2f}</div>", unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, result.confidence_score)))

        if result.recommended_action == "Advance to underwriting":
            decision_note = "Borrower appears strong enough to move forward, with standard underwriting verification still required."
        elif result.recommended_action == "Request more information":
            decision_note = "Borrower shows some promise, but key underwriting details are still needed before advancing."
        else:
            decision_note = "Current information suggests low fit or elevated risk relative to the inquiry."

        st.markdown(f"<div class='small-note'>{decision_note}</div>", unsafe_allow_html=True)
        card_end()

    # Row 2
    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row2_col1:
        card_start("⚠️ Missing Information")
        st.markdown(
            render_html_list(result.missing_information, "No major missing information.", list_type="missing"),
            unsafe_allow_html=True
        )
        card_end()

    with row2_col2:
        card_start("📊 Positive Signals & Risk Flags")
        pr_col1, pr_col2 = st.columns(2)

        with pr_col1:
            st.markdown("<div class='subsection-title'>Positive Signals</div>", unsafe_allow_html=True)
            st.markdown(
                render_html_list(result.positive_signals, "No strong positive signals identified.", list_type="positive"),
                unsafe_allow_html=True
            )

        with pr_col2:
            st.markdown("<div class='subsection-title'>Risk Flags</div>", unsafe_allow_html=True)
            st.markdown(
                render_html_list(result.risk_flags, "No major risks identified.", list_type="risk"),
                unsafe_allow_html=True
            )

        card_end()

    # Row 3
    row3_col1, row3_col2 = st.columns(2, gap="large")

    with row3_col1:
        card_start("❓ Suggested Follow-Up Questions")
        st.markdown(
            render_html_list(result.follow_up_questions, "No follow-up questions needed.", list_type="default"),
            unsafe_allow_html=True
        )
        card_end()

    with row3_col2:
        card_start("✉️ Draft Reply Email")
        safe_reply = (result.draft_reply_email or "No draft reply generated.").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f"<div class='reply-box'>{safe_reply}</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Tip: this draft is intentionally short and copy-ready.</div>", unsafe_allow_html=True)
        card_end()