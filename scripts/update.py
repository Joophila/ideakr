# -*- coding: utf-8 -*-
"""
ideakr daily updater (KST 00:00 via GitHub Actions)

수집: Google News(BUSINESS/SCIENCE/TECH) + Reddit(주간 Top + 동적 검색)
필터: 최근 72h, 정치/선거 제외, 매체명/도메인 토큰 제거, BLOCK_TERMS 환경변수로 추가 차단
키워드: 동적 추출(레딧 가중치↑)
DataLab: 상위 5개 그룹만 요청, 시계열(series) 포함, 부적합 시 대체 시드
생성: OpenAI로 카드 작성(여러 모델 시도) + 실패 시 규칙기반 백업카드 → 항상 채워짐
출력: 프론트 스키마(배열)로 data/rawitems.json, data/trends.json(시계열 포함), data/ideas.json, data/signals.json
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

# ---------------- Time ----------------
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

# ---------------- Env / Config ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

ENABLE_GNEWS = (os.getenv("ENABLE_GNEWS", "1") != "0")
ENABLE_REDDIT = (os.getenv("ENABLE_REDDIT", "1") != "0")

WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))
DATALAB_DAYS = int(os.getenv("DATALAB_DAYS", "14"))
MAX_GNEWS = int(os.getenv("MAX_GNEWS", "50"))
MAX_REDDIT = int(os.getenv("MAX_REDDIT", "12"))
MAX_GROUPS = int(os.getenv("MAX_GROUPS", "8"))             # 추출 키워드 그룹 수
MAX_GROUPS_DATALAB = int(os.getenv("MAX_GROUPS_DATALAB", "5"))  # DataLab 요청 그룹 수(권장 5)
MIN_TERM_COUNT = int(os.getenv("MIN_TERM_COUNT", "3"))

# 소스
TOPICS = os.getenv("GNEWS_TOPICS", "BUSINESS,SCIENCE,TECHNOLOGY").split(",")
REDDIT_SUBS = os.getenv("REDDIT_SUBS", "Entrepreneur,startups,technology,Futurology,smallbusiness").split(",")

# 파일 경로
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "rawitems.json")
TRENDS_PATH = os.path.join(DATA_DIR, "trends.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")

# 필터(정치/선거 등 제외)
POLITICS_BLOCK = {
    "대통령","총선","국회","정치","여당","야당","민주당","국민의힘","선거","의회","외교","북한"
}
STOPWORDS = {"속보","뉴스","단독","영상","사진","기자","오늘","이번","관련"}

# 매체/포털 스톱리스트 + 환경변수 차단어
PUBLISHER_STOP = {
    "매일경제","한국경제","조선일보","중앙일보","연합뉴스","머니투데이","한경",
    "서울경제","전자신문","지디넷","zdnet","디지털데일리","네이트","다음","네이버",
    "mbn","sbs","kbs","jtbc","ytn","tv조선","채널a","아시아경제","파이낸셜뉴스"
}
TERM_BLOCK = set((os.getenv("BLOCK_TERMS") or "").split(",")) - {""}

# DataLab 대체 시드
DEFAULT_SEEDS = [s for s in (os.getenv("DEFAULT_SEEDS") or "인공지능,전기차,친환경,리필 스테이션,구독 서비스").split(",") if s]


# ---------------- Utils ----------------
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


def host_parts(url: str) -> set[str]:
    """URL의 호스트를 도메인 조각으로 분해해 토큰 리스트 생성."""
    if not url or not url.startswith("http"):
        return set()
    try:
        host = url.split("/")[2]
        parts = re.split(r"[.\-]", host.lower())
        return set(p for p in parts if p and p not in {"www","m","news","co","kr","com","net"})
    except Exception:
        return set()


def bad_token(tok: str, url: str) -> bool:
    """매체명/차단어/숫자-only/1글자/도메인 조각 등을 제거."""
    t = (tok or "").lower().strip()
    if not t:
        return True
    if t in PUBLISHER_STOP:
        return True
    if t in TERM_BLOCK:
        return True
    if t.isdigit():
        return True
    if len(t.replace(" ", "")) < 2:
        return True
    if t in host_parts(url):
        return True
    return False


def tokenize_ko_en(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"&[a-z0-9#]+;", " ", s)
    s = re.sub(r"[^가-힣a-z0-9\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split(" ") if s else []


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
    if ENABLE_GNEWS:
        for t in TOPICS:
            items.extend(fetch_gnews_topic(t))
            time.sleep(0.25)
    if ENABLE_REDDIT:
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


def select_groups_for_datalab(groups: list[dict], allow_space=True, include_english=True,
                              max_groups=MAX_GROUPS_DATALAB) -> list[dict]:
    picked = []
    for g in groups:
        name = g.get("groupName") or ""
        kws = g.get("keywords") or []
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


def build_keyword_groups(items: list[dict], max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT) -> dict:
    cnt: dict[str, int] = {}
    for it in items:
        text = f"{it.get('title','')} {it.get('description','')}"
        url = it.get("url", "")
        toks = [w for w in tokenize_ko_en(strip_html(text)) if not bad_token(w, url)]
        # Reddit 가중치 ↑ (커뮤니티 시그널 반영) - 추측한 내용입니다
        w = 2 if str(it.get("source_name", "")).startswith("reddit") else 1

        for tok in toks:
            cnt[tok] = cnt.get(tok, 0) + w
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            if not bad_token(bg, url):
                cnt[bg] = cnt.get(bg, 0) + w

    terms = sorted([(k, v) for k, v in cnt.items() if v >= min_count],
                   key=lambda x: x[1], reverse=True)[:80]

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
    """마지막 volume & growth% 계산."""
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


def summarize_community(items, metrics):
    red = [it for it in items if str(it.get("source_name","")).startswith("reddit")]
    return {
        "reddit": {
            "posts": len(red),
            "upvotes": sum(int(it.get("ups") or 0) for it in red),
            "comments": sum(int(it.get("num_comments") or 0) for it in red)
        },
        "naver": {
            "vol_last": (metrics[0].get("volume") if metrics else 0),
            "growth": (metrics[0].get("growth") if metrics else "알 수 없습니다")
        }
    }


# ---------------- Card Writers ----------------
def build_rule_based_card(items, metrics, groups) -> dict:
    """LLM 없이 규칙 기반 카드 → 항상 출력 보장."""
    top_kw = (metrics[0]["groupName"] if metrics else (groups.get("keywordGroups", [{}])[0].get("groupName", "신규 트렌드")))
    gr = metrics[0].get("growth") if metrics else None
    gr_txt = f"최근 {DATALAB_DAYS}일 {gr}%" if isinstance(gr, (int, float)) else "증감률 불명"

    def score(it): return (it.get("ups",0)*2) + it.get("num_comments",0)
    top_items = sorted(items, key=score, reverse=True)[:3]
    evidence = [{"title": it.get("title",""), "url": it.get("url","")} for it in top_items if it.get("url")]

    return {
        "title": f"{top_kw} 트렌드 리서치 카드",
        "tagline": f"{top_kw}: {gr_txt} · 근거 기반 요약",
        "sections": {
            "problem": f"‘{top_kw}’ 관련 관심은 늘지만 근거 중심 한국어 리서치가 부족합니다.",
            "solution": f"{gr_txt} 트렌드를 바탕으로 뉴스/커뮤니티 근거 링크와 요약을 제공.",
            "target_user": "초기 창업가, 리서치가 필요한 마케터/PM",
            "gtm": "매일 카드 1건 무료 → 이메일 구독 → 주간 리포트 유료",
            "why_now": f"관심 급증({gr_txt}) + 생성형 도구 확산으로 ‘근거 요약’ 수요 증가.",
            "proof_signals": "상위 기사·레딧 포스트 링크, 검색량 추세.",
            "market_gap": "한글/한국시장 맞춤의 근거 중심 요약 도구 부재.",
            "execution_plan": {
                "core": "상위 키워드 3개 자동 카드 → 검수",
                "growth": "SNS 미리보기 카드, 구독 리드 축적",
                "lead_gen": "키워드 알림·맞춤 리포트 폼",
                "steps": ["수집→키워드/검색량 갱신", "카드 자동 생성", "검수 후 Pages 배포"],
            },
        },
        "evidence": evidence,
    }


def write_full_card_with_gpt(items: list[dict], trends: dict, metrics: list[dict], groups: dict) -> dict:
    """OpenAI로 카드 작성. 실패 시 규칙 기반 카드로 대체."""
    # 키 없으면 바로 규칙기반
    if not OPENAI_API_KEY:
        print("OPENAI_INFO: missing API key → rule-based card")
        return build_rule_based_card(items, metrics, groups)

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

    for m in ["gpt-4o-mini", "gpt-4o"]:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            card = json.loads(resp.choices[0].message.content)
            print("CARD_SOURCE:", "gpt", m)
            return card
        except Exception as e:
            print(f"OPENAI_ERROR model={m} -> {repr(e)}")
            continue

    print("CARD_SOURCE:", "rule-based")
    return build_rule_based_card(items, metrics, groups)


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

    # URL -> raw_id 매핑 (evidence 연결)
    url2rawid = {r["url"]: r["raw_id"] for r in raw_list if r.get("url")}

    # datalab 시계열 → series 맵 (title -> [{date,value}...])
    series_map = {}
    for g in (datalab.get("results") or []):
        title = g.get("title") or ""
        data = g.get("data") or []
        series_map[title] = [
            {"date": p.get("period"), "value": p.get("ratio")}
            for p in data if "period" in p and "ratio" in p
        ]

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
                "series": series_map.get(kw, []),  # 차트용 시계열
            }
        )

    # ideas.json (Array)
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
    community = summarize_community(items, metrics)

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
        "community": community
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
    if ENABLE_REDDIT:
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

    # 6) DataLab (키워드 선별 → 실패 시 대체 시드)
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

    # 8) GPT 카드(여러 모델 시도, 실패 시 규칙기반)
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

