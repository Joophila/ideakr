# -*- coding: utf-8 -*-
"""
ideakr daily updater (KST 00:00 via GitHub Actions)

수집
- Google News (BUSINESS/SCIENCE/TECHNOLOGY)
- Reddit (주간 Top from subreddits) + 동적 키워드 Reddit 검색
- 정치/선거 관련 문구 제외, 최근 72시간만 사용

가공
- 동적 키워드 그룹 추출
- Naver DataLab 검색량 시계열(최근 14일)
- 간단 점수(Trend/Market/Competition 등) 계산
- (선택) OpenAI로 아이디어 카드 섹션 작성

출력 (프론트 스키마에 맞춘 "배열" JSON)
- data/rawitems.json
- data/trends.json
- data/ideas.json
- data/signals.json  (참고)
"""

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtp

# ---------------- Config ----------------
MAX_GROUPS_DATALAB = 5  # DataLab 요청은 5개 이내 권장
DEFAULT_SEEDS = ["인공지능", "사업", "친환경", "AI", "구독 서비스", "부업", "부수입"]
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

# 수집 범위/양
WINDOW_HOURS = 72
DATALAB_DAYS = 14
MAX_GNEWS = 50          # per topic
MAX_REDDIT = 12         # per subreddit
MAX_GROUPS = 8          # keyword groups 갯수
MIN_TERM_COUNT = 3

# 소스
TOPICS = ["BUSINESS", "SCIENCE", "TECHNOLOGY"]
REDDIT_SUBS = ["Entrepreneur", "startups", "technology", "Futurology", "smallbusiness"]

# 파일 경로
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "rawitems.json")
TRENDS_PATH = os.path.join(DATA_DIR, "trends.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")

# 필터(정치/선거 등 제외)
POLITICS_BLOCK = [
    "대통령", "총선", "국회", "정치", "여당", "야당", "민주당", "국민의힘", "선거", "의회", "외교", "북한"
]
STOPWORDS = set(["속보", "뉴스", "단독", "영상", "사진", "기자", "오늘", "이번", "관련"])

# 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


# ---------------- Utils ----------------
def is_korean(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def valid_for_datalab(term: str, allow_space=True) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    if not allow_space and " " in t:
        return False
    if is_korean(t):
        return len(t.replace(" ", "")) >= 2
    else:
        return len(t.replace(" ", "")) >= 3

def select_groups_for_datalab(groups: list[dict], allow_space=True, include_english=True, max_groups=MAX_GROUPS_DATALAB) -> list[dict]:
    picked = []
    for g in groups:
        name = g.get("groupName") or ""
        kws  = g.get("keywords") or []
        # 대표어 1개만 검증
        rep = name if name else (kws[0] if kws else "")
        if not rep:
            continue
        if not include_english and not is_korean(rep):
            continue
        if not valid_for_datalab(rep, allow_space=allow_space):
            continue
        picked.append({"groupName": rep, "keywords": [rep]})
        if len(picked) >= max_groups:
            break
    return picked

def strip_html(s: str) -> str:
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)


def as_iso(dt_any) -> str | None:
    if isinstance(dt_any, str):
        try:
            dt_any = dtp.parse(dt_any)
        except Exception:
            return None
    if not getattr(dt_any, "tzinfo", None):
        dt_any = dt_any.replace(tzinfo=timezone.utc)
    return dt_any.isoformat()


def is_recent_iso(iso_str: str | None, hours: int = WINDOW_HOURS) -> bool:
    if not iso_str:
        return False
    try:
        d = dtp.parse(iso_str)
        if not d.tzinfo:
            d = d.replace(tzinfo=timezone.utc)
        return d >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except Exception:
        return False


def is_blocked_text(text: str | None) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in POLITICS_BLOCK)


def dump_json(path: str, obj) -> bool:
    """내용이 바뀐 경우에만 저장하고 True 반환."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        pass
    if old == obj:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return True


# ---------------- Collectors ----------------
def fetch_gnews_topic(topic: str) -> list[dict]:
    url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"
    out: list[dict] = []
    try:
        feed = feedparser.parse(url)
    except Exception:
        return out
    for e in feed.entries[:MAX_GNEWS]:
        title = e.get("title") or ""
        link = e.get("link") or ""
        desc = strip_html(e.get("summary") or e.get("description") or "")
        pub = e.get("published") or e.get("updated")
        pub_iso = as_iso(pub) if pub else None
        if not pub_iso or not is_recent_iso(pub_iso):
            continue
        if is_blocked_text(title) or is_blocked_text(desc):
            continue
        out.append(
            {
                "title": title,
                "url": link,
                "published_at": pub_iso,
                "description": desc,
                "source_name": f"GoogleNews:{topic}",
                "source_url": "",
                "topic": topic,
            }
        )
    return out


def fetch_reddit_sub(sub: str) -> list[dict]:
    url = f"https://www.reddit.com/r/{sub}/top.json"
    qs = {"t": "week", "limit": MAX_REDDIT}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=qs, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return []
    out: list[dict] = []
    for ch in (js.get("data", {}) or {}).get("children", []):
        d = ch.get("data", {}) or {}
        title = d.get("title") or ""
        desc = d.get("selftext") or ""
        created = d.get("created_utc")
        if not created:
            continue
        pub_iso = as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent_iso(pub_iso):
            continue
        if is_blocked_text(title) or is_blocked_text(desc):
            continue
        permalink = d.get("permalink")
        urlp = f"https://www.reddit.com{permalink}" if permalink else (d.get("url") or "")
        out.append(
            {
                "title": title,
                "url": urlp,
                "published_at": pub_iso,
                "description": desc,
                "source_name": f"reddit:{sub}",
                "source_url": f"https://www.reddit.com/r/{sub}",
                "topic": "REDDIT",
                "ups": int(d.get("ups") or 0),
                "num_comments": int(d.get("num_comments") or 0),
            }
        )
    return out


def fetch_reddit_search(query: str, limit: int = 20) -> list[dict]:
    """동적 키워드 기반 reddit 검색."""
    headers = {"User-Agent": "Mozilla/5.0"}
    qs = {"q": query, "t": "week", "sort": "top", "limit": limit}
    try:
        r = requests.get("https://www.reddit.com/search.json", params=qs, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return []
    out: list[dict] = []
    for ch in (js.get("data", {}) or {}).get("children", []):
        d = ch.get("data", {}) or {}
        title = d.get("title") or ""
        desc = d.get("selftext") or ""
        created = d.get("created_utc")
        if not created:
            continue
        pub_iso = as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent_iso(pub_iso):
            continue
        if is_blocked_text(title) or is_blocked_text(desc):
            continue
        permalink = d.get("permalink")
        urlp = f"https://www.reddit.com{permalink}" if permalink else (d.get("url") or "")
        out.append(
            {
                "title": title,
                "url": urlp,
                "published_at": pub_iso,
                "description": desc,
                "source_name": "reddit:search",
                "source_url": "https://www.reddit.com/search",
                "topic": "REDDIT",
                "ups": int(d.get("ups") or 0),
                "num_comments": int(d.get("num_comments") or 0),
            }
        )
    return out


def collect_all() -> list[dict]:
    items: list[dict] = []
    # Google News
    for t in TOPICS:
        items.extend(fetch_gnews_topic(t))
        time.sleep(0.25)
    # Reddit subs
    for sub in REDDIT_SUBS:
        items.extend(fetch_reddit_sub(sub))
        time.sleep(0.25)
    # 중복 제거 (URL)
    seen = set()
    uniq = []
    for it in items:
        key = (it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq


# ---------------- Keywords ----------------
def tokenize_ko_en(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"&[a-z0-9#]+;", " ", s)
    s = re.sub(r"[^가-힣a-z0-9\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split(" ") if s else []


def build_keyword_groups(items: list[dict], max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT) -> dict:
    cnt: dict[str, int] = {}
    for it in items:
        text = f"{it.get('title','')} {it.get('description','')}"
        toks = [w for w in tokenize_ko_en(strip_html(text)) if len(w) >= 2 and w not in STOPWORDS]
        for w in toks:
            cnt[w] = cnt.get(w, 0) + 1
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            cnt[bg] = cnt.get(bg, 0) + 1

    terms = sorted([(k, v) for k, v in cnt.items() if v >= min_count], key=lambda x: x[1], reverse=True)[:80]
    groups: list[dict] = []
    used = set()
    for term, _ in terms:
        if len(groups) >= max_groups:
            break
        if term in used:
            continue
        groups.append({"groupName": term, "keywords": [term]})
        used.add(term)

    return {"keywordGroups": groups, "topTerms": terms, "itemCount": len(items)}


# ---------------- Naver DataLab ----------------
def naver_datalab(keyword_groups: list[dict], days=DATALAB_DAYS) -> dict:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"error": "missing_keys", "results": []}
    end_kst = NOW_KST.date()
    start_kst = (NOW_KST - timedelta(days=days)).date()
    body = {
    "startDate": str(start_kst),
    "endDate": str(end_kst),
    "timeUnit": "date",
    "keywordGroups": [
        {"groupName": g["groupName"], "keywords": g["keywords"][:3]}
        for g in keyword_groups[:MAX_GROUPS_DATALAB]
    ],
    "device": "",
    "ages": [],
    "gender": "",
}
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        r = requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            return {"error": f"http_{r.status_code}", "results": []}
        return r.json()
    except Exception:
        return {"error": "exception", "results": []}


def extract_trend_metrics(datalab: dict) -> list[dict]:
    """DataLab 결과 → 각 그룹별 마지막 volume, 첫/마지막 대비 growth%"""
    metrics: list[dict] = []
    for g in datalab.get("results", []):
        data = g.get("data", [])
        if not data:
            metrics.append({"groupName": g.get("title", ""), "volume": 0, "growth": "알 수 없습니다"})
            continue
        first = data[0]["ratio"]
        last = data[-1]["ratio"]
        try:
            growth = round(((last - first) / (first if first else 1)) * 100, 1)
        except Exception:
            growth = "알 수 없습니다"
        metrics.append({"groupName": g.get("title", ""), "volume": int(last), "growth": growth})
    return metrics


# ---------------- Scores & Community ----------------
def score_from_metrics(metrics: list[dict], items: list[dict]) -> dict:
    # Trend
    if metrics:
        vols = [m["volume"] for m in metrics if isinstance(m.get("volume"), (int, float))]
        grs = [m["growth"] for m in metrics if isinstance(m.get("growth"), (int, float))]
        v = sum(vols) / len(vols) if vols else 0
        g = sum(grs) / len(grs) if grs else 0
        trend = max(0, min(100, int(v * 0.5 + g * 1.0)))  # 휴리스틱 (추측한 내용입니다)
    else:
        trend = 0

    # Market: 기사/포스트 수 근사 (추측)
    market = max(0, min(100, len(items) * 2))

    # Competition invert: 도메인 다양성 근사 (추측)
    domains: dict[str, int] = {}
    for it in items:
        u = (it.get("url") or "").lower()
        host = u.split("/")[2] if u.startswith("http") and len(u.split("/")) > 2 else ""
        if host:
            domains[host] = domains.get(host, 0) + 1
    avg = sum(domains.values()) / len(domains) if domains else 0
    competition_invert = max(0, min(100, int(100 - min(90, avg * 10))))

    monetization = 50
    feasibility = 50
    regulatory_invert = 50

    overall = int(
        0.35 * trend + 0.25 * market + 0.15 * competition_invert + 0.1 * monetization + 0.15 * feasibility
    )  # (추측)

    return {
        "trend": trend,
        "market": market,
        "competition_invert": competition_invert,
        "feasibility": feasibility,
        "monetization": monetization,
        "regulatory_invert": regulatory_invert,
        "overall": overall,
    }


# ---------------- GPT Writer ----------------
def write_full_card_with_gpt(items: list[dict], trends: dict, metrics: list[dict], groups: dict) -> dict:
    """OpenAI 사용해 아이디어 카드 섹션 작성. 키 미설정 시 템플릿 반환."""
    if not OPENAI_API_KEY:
        return {
            "title": "GPT 미사용",
            "tagline": "알 수 없습니다",
            "sections": {
                "problem": "근거가 부족합니다",
                "solution": "근거가 부족합니다",
                "target_user": "근거가 부족합니다",
                "gtm": "근거가 부족합니다",
                "why_now": "근거가 부족합니다",
                "proof_signals": "근거가 부족합니다",
                "market_gap": "근거가 부족합니다",
                "execution_plan": {
                    "core": "근거가 부족합니다",
                    "growth": "근거가 부족합니다",
                    "lead_gen": "근거가 부족합니다",
                    "steps": [],
                },
            },
            "evidence": [],
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        payload = {
            "date_kr": NOW_KST.strftime("%Y-%m-%d"),
            "keywordGroups": groups.get("keywordGroups", []),
            "topTerms": groups.get("topTerms", [])[:20],
            "trends": trends.get("results", [])[:MAX_GROUPS],
            "trend_metrics": metrics,
            "articles": [{"title": it.get("title", ""), "url": it.get("url", "")} for it in items[:20]],
        }
        sys_prompt = (
            "너는 한국 시장 트렌드 리서처다. 아래 데이터(기사/키워드/검색트렌드)만 근거로 "
            "사업화 아이디어 카드를 한국어로 작성하라. 과장 금지, 정치/연예 배제. "
            "숫자/시장규모는 근거 없으면 '알 수 없습니다' 또는 '확실하지 않음'. "
            "JSON 하나로 출력: {title,tagline,sections{problem,solution,target_user,gtm,why_now,proof_signals,market_gap,execution_plan{core,growth,lead_gen,steps[]}},evidence[]}."
        )
        user_msg = "DATA:\n" + json.dumps(payload, ensure_ascii=False)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        card = json.loads(resp.choices[0].message.content)
        return card
    except Exception:
        # 실패 시 템플릿
        return {
            "title": "아이디어 생성 실패",
            "tagline": "확실하지 않음",
            "sections": {
                "problem": "알 수 없습니다",
                "solution": "알 수 없습니다",
                "target_user": "알 수 없습니다",
                "gtm": "알 수 없습니다",
                "why_now": "알 수 없습니다",
                "proof_signals": "알 수 없습니다",
                "market_gap": "알 수 없습니다",
                "execution_plan": {"core": "알 수 없습니다", "growth": "알 수 없습니다", "lead_gen": "알 수 없습니다", "steps": []},
            },
            "evidence": [],
        }


# ---------------- Frontend JSON (배열 스키마) ----------------
def make_id(prefix: str, n: int) -> str:
    return f"{prefix}_{n:04d}"


def to_frontend_json(items: list[dict], datalab: dict, metrics: list[dict], signals: dict, card: dict, scores: dict):
    # rawitems.json (Array)
    raw_list = []
    for idx, it in enumerate(items, 1):
        raw_list.append(
            {
                "raw_id": make_id("raw", idx),
                "source_platform": it.get("source_name") or "",
                "query_or_topic": it.get("topic") or "",
                "title": it.get("title") or "",
                "content_snippet": (it.get("description") or "")[:300],
                "url": it.get("url") or "",
                "metrics_upvotes": it.get("ups") or 0,
                "metrics_comments": it.get("num_comments") or 0,
                "search_volume": "",
                "language": "ko",
                "published_at": it.get("published_at") or "",
                "fetched_at": NOW_KST.isoformat(),
            }
        )

    # URL -> raw_id 매핑 (evidence 연결에 사용)
    url2rawid = {r["url"]: r["raw_id"] for r in raw_list if r.get("url")}

    # trends.json (Array)
    trend_arr = []
    for i, m in enumerate(metrics, 1):
        kw = m.get("groupName") or ""
        last_vol = m.get("volume") or 0
        gr = m.get("growth")
        growth_pct = (gr / 100.0) if isinstance(gr, (int, float)) else None
        tr_id = make_id("tr", i)
        trend_arr.append(
            {
                "trend_id": tr_id,
                "keyword": kw,
                "trend_score": scores.get("trend", 0),
                "volume": last_vol,
                "growth_percent": growth_pct,
                "region": "KR",
                "timespan": f"{DATALAB_DAYS}d",
                "evidence_rawitems": [],
                "updated_at": NOW_KST.isoformat(),
            }
        )

    # ideas.json (Array) - 프론트 기대 필드명
    sc = scores or {}
    score_breakdown = {
        "trend": sc.get("trend", 0),
        "market": sc.get("market", 0),
        "competition_invert": sc.get("competition_invert", 0),
        "feasibility": sc.get("feasibility", 50),
        "monetization": sc.get("monetization", 50),
        "regulatory_invert": sc.get("regulatory_invert", 50),
    }
    score_total = sc.get("overall", 0)

    evidence_urls = [e.get("url") for e in (card.get("evidence") or []) if isinstance(e, dict)]
    linked_raw_ids = [url2rawid[u] for u in evidence_urls if u in url2rawid]
    trend_link = [t["trend_id"] for t in trend_arr[:3]]

    idea_obj = {
        "idea_id": f"idea_{NOW_KST.strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or "",
        "one_liner": card.get("tagline") or "",
        "problem": card.get("sections", {}).get("problem", "근거가 부족합니다"),
        "solution": card.get("sections", {}).get("solution", "근거가 부족합니다"),
        "target_user": card.get("sections", {}).get("target_user", "—"),
        "why_now": card.get("sections", {}).get("why_now", "근거가 부족합니다"),
        "biz_model": "알 수 없습니다",
        "gtm_tactics": card.get("sections", {}).get("gtm", "—"),
        "validation_steps": "확실하지 않음",
        "tags": [g.get("groupName") for g in signals.get("keywordGroups", [])][:4],
        "score_breakdown": score_breakdown,
        "score_total": score_total,
        "trend_link": trend_link,
        "sources_linked": linked_raw_ids,
        "regulation_risk": "중간",
        "gov_support_fit": [],
        "created_at": NOW_KST.isoformat(),
        "is_today": True,
    }

    return raw_list, trend_arr, [idea_obj]


def save_frontend_files(items, datalab, metrics, signals, card, scores) -> bool:
    raw_list, trend_arr, idea_list = to_frontend_json(items, datalab, metrics, signals, card, scores)
    ch1 = dump_json(RAW_PATH, raw_list)
    ch2 = dump_json(TRENDS_PATH, trend_arr)
    ch3 = dump_json(IDEAS_PATH, idea_list)
    ch4 = dump_json(SIGNALS_PATH, signals)
    return ch1 or ch2 or ch3 or ch4


# ---------------- Main ----------------
def main():
    # 1) 수집
    items = collect_all()

    # 2) 1차 필터 (정치/선거 제외)
    items = [it for it in items if not is_blocked_text(it.get("title")) and not is_blocked_text(it.get("description"))]

    # 3) 키워드 그룹
    signals = build_keyword_groups(items, max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT)

    # 4) 동적 키워드 Reddit 검색 보강
    q_terms = [t for t, _ in signals.get("topTerms", [])[:5]]
    if q_terms:
        query = " OR ".join(q_terms)
        items.extend(fetch_reddit_search(query, limit=20))

    # 5) 최근 72h 재확인 + 중복 제거
    items = [it for it in items if is_recent_iso(it.get("published_at"))]
    seen = set()
    uniq = []
    for it in items:
        key = (it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    items = uniq

        # 6) DataLab (키워드 선별 → 실패 시 대체 시드 → 대체 전략 2차)
    groups_all = signals.get("keywordGroups", [])
    dl_groups = select_groups_for_datalab(groups_all, allow_space=True, include_english=True)

    if not dl_groups:
        print("DATALAB_INFO: no valid groups from signals, using DEFAULT_SEEDS")
        dl_groups = [{"groupName": s, "keywords": [s]} for s in DEFAULT_SEEDS]

    datalab = naver_datalab(dl_groups, days=DATALAB_DAYS)
    if not datalab.get("results"):
        print("DATALAB_EMPTY: trying fallback (no-space & Korean-only)")
        dl_groups2 = select_groups_for_datalab(groups_all, allow_space=False, include_english=False)
        if dl_groups2:
            datalab = naver_datalab(dl_groups2, days=DATALAB_DAYS)

    metrics = extract_trend_metrics(datalab)


    # 7) 점수
    scores = score_from_metrics(metrics, items)

    # 8) GPT 카드(전체 섹션)
    card = write_full_card_with_gpt(items, datalab, metrics, signals)

    # 9) 저장 (프론트 스키마)
    changed = save_frontend_files(items, datalab, metrics, signals, card, scores)

    # 로그
    print(
        f"[{NOW_KST.isoformat()}] items={len(items)} changed={changed} "
        f"(secrets: NAVER={'ok' if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET else 'missing'}, "
        f"OPENAI={'ok' if OPENAI_API_KEY else 'missing'})"
    )


if __name__ == "__main__":
    main()
