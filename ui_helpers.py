from html import escape


def admin_headers(admin_api_key: str | None) -> dict[str, str]:
    key = (admin_api_key or "").strip()
    return {"X-Admin-Key": key} if key else {}


def confidence_label(confidence: float | None) -> str:
    score = max(0.0, min(1.0, float(confidence or 0.0)))
    percent = int(round(score * 100))
    if score >= 0.75:
        label = "High confidence"
    elif score >= 0.45:
        label = "Medium confidence"
    else:
        label = "Low confidence"
    return f"{label} · {percent}%"


def render_citations_html(citations) -> str:
    if not citations:
        return ""

    items = ['<div class="citation-wrap">']
    for citation in citations:
        source = escape(str(citation.get("source", "Unknown")))
        page = citation.get("page")
        score = citation.get("score")
        snippet = escape(str(citation.get("snippet", "")))
        meta = source
        if page is not None:
            meta += f" · p.{escape(str(page))}"
        if score is not None:
            meta += f" · {int(round(float(score) * 100))}%"
        items.append(
            '<div class="citation-card">'
            f'<div class="citation-meta">{meta}</div>'
            f'<div class="citation-snippet">{snippet}</div>'
            "</div>"
        )
    items.append("</div>")
    return "".join(items)
