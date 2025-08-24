# -*- coding: utf-8 -*-
"""
ideakr daily updater (runs daily at KST 00:00 via GitHub Actions)

기능 요약
- 수집: Google News(BUSINESS/SCIENCE/TECH) + Reddit(주간 Top) + Reddit 검색(동적 키워드)
- 필터: 최근 72시간만, 정치 키워드 제외
- 키워드: 기사/포스트에서 동적 키워드 그룹 추출
- 트렌드: Naver DataLab 검색량 시계열 (최근 14일)
- 지표: Trend/Market/Competition 등 점수(데이터 기반 + 추정은 '추측한 내용입니다' 표시)
- 커뮤니티: Reddit 업보트/댓글 합산
- 생성: GPT로 '아이디어 카드'의 모든 섹션(JSON) 작성 (근거 부족 시 '알 수 없습니다/확실하지 않음')

출력 (리포 루트 기준 data/*)
- data/rawitems.json
- data/signals.json
- data/trends.json
- data/ideas.json
"""

import json, os, re, time
from datetime import datetime, timedelta, timezone
import requests, feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtp

# ---------------- Config ----------------
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

WINDOW_HOURS = 72
DATALAB_DAYS = 14

TOPICS = ["BUSINESS", "SCIENCE", "TECHNOLOGY"]                # Google News
REDDIT_SUBS = ["Entrepreneur","startups","technology","Futurology","smallbusiness"]  # Reddit
MAX_GNEWS = 50        # per topic (상한)
MAX_REDDIT = 12       # per subreddit
MAX_GROUPS = 8        # keyword groups 최대 개수
MIN_TERM_COUNT = 3    # 키워드 최소 빈도

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "rawitems.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")
TRENDS_PATH = os.path.join(DATA_DIR, "trends.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.json")

POLITICS_BLOCK = [
    "대통령","총선","국회","정치","여당","야당","민주당","국민의힘","선거","의회","외교","북한"
]
STOPWORDS = set(["속보","뉴스","단독","영상","사진","기자","오늘","이번","관련"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# --------------- Utils -------------------
def strip_html(s):
    return "" if not s else BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

def as_iso(dt):
    if isinstance(dt, str):
        try: dt = dtp.parse(dt)
        except Exception: return None
    if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def is_recent_iso(iso_str, hours=WINDOW_HOURS):
    try:
        d = dtp.parse(iso_str)
        if not d.tzinfo: d = d.replace(tzinfo=timezone.utc)
        return d >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except Exception:
        return False

def blocked(text):
    t = (text or "").lower()
    return any(k.lower() in t for k in POLITICS_BLOCK)

def dump_json(path, obj):
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

# --------------- Collectors ---------------
def fetch_gnews_topic(topic):
    url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:MAX_GNEWS]:
        title = e.get("title","") or ""
        link  = e.get("link","") or ""
        desc  = strip_html(e.get("summary") or e.get("description") or "")
        pub   = e.get("published") or e.get("updated")
        pub_iso = as_iso(pub) if pub else None
        if not pub_iso or not is_recent_iso(pub_iso): 
            continue
        if blocked(title) or blocked(desc):
            continue
        out.append({
            "title": title,
            "url": link,
            "published_at": pub_iso,
            "description": desc,
            "source_name": f"GoogleNews:{topic}",
            "source_url": "",
            "topic": topic
        })
    return out

def fetch_reddit_sub(sub):
    url = f"https://www.reddit.com/r/{sub}/top.json"
    qs = {"t":"week","limit":MAX_REDDIT}
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        r = requests.get(url, params=qs, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return []
    out = []
    for ch in (js.get("data",{}) or {}).get("children",[]):
        d = ch.get("data",{}) or {}
        title = d.get("title","") or ""
        desc  = d.get("selftext","") or ""
        created = d.get("created_utc")
        if not created:
            continue
        pub_iso = as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent_iso(pub_iso):
            continue
        if blocked(title) or blocked(desc):
            continue
        permalink = d.get("permalink")
        urlp = f"https://www.reddit.com{permalink}" if permalink else (d.get("url","") or "")
        out.append({
            "title": title,
            "url": urlp,
            "published_at": pub_iso,
            "description": desc,
            "source_name": f"reddit:{sub}",
            "source_url": f"https://www.reddit.com/r/{sub}",
            "topic": "REDDIT",
            "ups": int(d.get("ups") or 0),
            "num_comments": int(d.get("num_comments") or 0)
        })
    return out

def fetch_reddit_search(query, limit=20):
    """동적 키워드 기반 Reddit 검색"""
    qs = {"q": query, "t": "week", "sort": "top", "limit": limit}
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        r = requests.get("https://www.reddit.com/search.json", params=qs, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return []
    out=[]
    for ch in (js.get("data",{}) or {}).get("children",[]):
        d = ch.get("data",{}) or {}
        title = d.get("title","") or ""
        desc  = d.get("selftext","") or ""
        created = d.get("created_utc")
        if not created: 
            continue
        pub_iso = as_iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent_iso(pub_iso):
            continue
        if blocked(title) or blocked(desc):
            continue
        permalink = d.get("permalink")
        urlp = f"https://www.reddit.com{permalink}" if permalink else (d.get("url","") or "")
        out.append({
            "title": title,
            "url": urlp,
            "published_at": pub_iso,
            "description": desc,
            "source_name": "reddit:search",
            "source_url": "https://www.reddit.com/search",
            "topic": "REDDIT",
            "ups": int(d.get("ups") or 0),
            "num_comments": int(d.get("num_comments") or 0)
        })
    return out

def collect_all():
    items=[]
    # Google News(정치 제외는 후처리)
    for t in TOPICS:
        items.extend(fetch_gnews_topic(t)); time.sleep(0.3)
    # Reddit (서브레딧 Top)
    for sub in REDDIT_SUBS:
        items.extend(fetch_reddit_sub(sub)); time.sleep(0.3)
    # dedupe by URL
    seen=set(); uniq=[]
    for it in items:
        key=(it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen: 
            continue
        seen.add(key); uniq.append(it)
    return uniq

# --------------- Keywords ---------------
def tokenize_ko_en(s):
    s=(s or "").lower()
    s=re.sub(r"&[a-z0-9#]+;"," ", s)
    s=re.sub(r"[^가-힣a-z0-9\- ]"," ", s)
    s=re.sub(r"\s+"," ", s).strip()
    return s.split(" ") if s else []

def build_keyword_groups(items, max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT):
    cnt={}
    for it in items:
        text=f"{it.get('title','')} {it.get('description','')}"
        toks=[w for w in tokenize_ko_en(strip_html(text)) if len(w)>=2 and w not in STOPWORDS]
        for w in toks:
            cnt[w]=cnt.get(w,0)+1
        for i in range(len(toks)-1):
            bg=f"{toks[i]} {toks[i+1]}"
            cnt[bg]=cnt.get(bg,0)+1
    terms=sorted([(k,v) for k,v in cnt.items() if v>=min_count], key=lambda x:x[1], reverse=True)[:80]
    groups=[]; used=set()
    for term,_ in terms:
        if len(groups)>=max_groups: break
        if term in used: continue
        groups.append({"groupName":term, "keywords":[term]})
        used.add(term)
    return {"keywordGroups":groups, "topTerms":terms, "itemCount":len(items)}

# --------------- Naver DataLab ---------------
def naver_datalab(keyword_groups, days=DATALAB_DAYS):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"error":"missing_keys","results":[]}
    end_kst = NOW_KST.date()
    start_kst = (NOW_KST - timedelta(days=days)).date()
    body = {
        "startDate": str(start_kst),
        "endDate": str(end_kst),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": g["groupName"], "keywords": g["keywords"][:3]} for g in keyword_groups[:MAX_GROUPS]],
        "device": "",
        "ages": [],
        "gender": ""
    }
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        r = requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            return {"error": f"http_{r.status_code}", "results":[]}
        return r.json()
    except Exception:
        return {"error":"exception","results":[]}

def extract_trend_metrics(datalab):
    """DataLab 결과 → 각 그룹별 마지막 volume, 첫/마지막 대비 %growth"""
    metrics=[]
    for g in datalab.get("results",[]):
        data=g.get("data",[])
        if not data:
            metrics.append({"groupName":g.get("title",""),"volume":0,"growth":"알 수 없습니다"})
            continue
        first=data[0]["ratio"]; last=data[-1]["ratio"]
        try:
            growth = round(((last - first) / (first if first else 1)) * 100, 1)
        except Exception:
            growth = "알 수 없습니다"
        metrics.append({"groupName":g.get("title",""), "volume": int(last), "growth": growth})
    return metrics

# --------------- Scores & Community ---------------
def score_from_metrics(metrics, items):
    # Trend 점수(데이터 기반)
    if metrics:
        vols=[m["volume"] for m in metrics if isinstance(m.get("volume"), (int,float))]
        grs=[m["growth"] for m in metrics if isinstance(m.get("growth"), (int,float))]
        v = sum(vols)/len(vols) if vols else 0
        g = sum(grs)/len(grs) if grs else 0
        trend = max(0, min(100, int(v*0.5 + g*1.0)))  # 휴리스틱 (추측한 내용입니다)
    else:
        trend = 0

    # Market 근사: 최근 72h 수집량 (추측)
    market = max(0, min(100, len(items)*2))  # 50개≈100 (추측한 내용입니다)

    # Competition invert: 출처 도메인 다양성 (추측)
    domains={}
    for it in items:
        u=(it.get("url") or "").lower()
        host=u.split("/")[2] if u.startswith("http") and len(u.split("/"))>2 else ""
        if host: domains[host]=domains.get(host,0)+1
    avg = sum(domains.values())/len(domains) if domains else 0
    competition_invert = max(0, min(100, int(100 - min(90, avg*10))))  # (추측)

    monetization = 50   # 근거 부족
    feasibility = 50    # 근거 부족
    regulatory_invert = 50  # 근거 부족

    overall = int(0.35*trend + 0.25*market + 0.15*competition_invert + 0.1*monetization + 0.15*feasibility)  # (추측)

    notes = {
        "trend":"Naver DataLab volume/growth 기반",
        "market":"최근 72시간 기사/포스트 수로 근사(추측한 내용입니다)",
        "competition_invert":"도메인 다양성으로 근사(추측한 내용입니다)",
        "feasibility":"근거 부족: 50 고정",
        "monetization":"근거 부족: 50 고정",
        "regulatory_invert":"근거 부족: 50 고정"
    }
    return {
        "trend":trend, "market":market, "competition_invert":competition_invert,
        "feasibility":feasibility, "monetization":monetization, "regulatory_invert":regulatory_invert,
        "overall":overall, "notes":notes
    }

def community_from_items(items, datalab_metrics):
    red = [it for it in items if str(it.get("source_name","")).startswith("reddit")]
    reddit = {
        "posts": len(red),
        "upvotes": sum(int(it.get("ups",0)) for it in red),
        "comments": sum(int(it.get("num_comments",0)) for it in red)
    }
    naver = {}
    if datalab_metrics:
        last = datalab_metrics[0]
        naver = {
            "vol_last": last.get("volume", 0),
            "growth": last.get("growth","알 수 없습니다")
        }
    return {"reddit": reddit, "naver": naver}

# --------------- GPT Writer (모든 섹션) ---------------
def write_full_card_with_gpt(items, trends, metrics, groups):
    if not OPENAI_API_KEY:
        # GPT 미사용 시: 비어있는 템플릿
        return {
            "title":"GPT 미사용",
            "tagline":"알 수 없습니다",
            "sections":{
                "problem":"근거가 부족합니다",
                "solution":"근거가 부족합니다",
                "target_user":"근거가 부족합니다",
                "gtm":"근거가 부족합니다",
                "why_now":"근거가 부족합니다",
                "proof_signals":"근거가 부족합니다",
                "market_gap":"근거가 부족합니다",
                "execution_plan":{"core":"근거가 부족합니다","growth":"근거가 부족합니다","lead_gen":"근거가 부족합니다","steps":[]}
            },
            "evidence":[]
        }

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    payload = {
        "date_kr": NOW_KST.strftime("%Y-%m-%d"),
        "keywordGroups": groups.get("keywordGroups", []),
        "topTerms": groups.get("topTerms", [])[:20],
        "trends": trends.get("results", [])[:MAX_GROUPS],
        "trend_metrics": metrics,
        "articles": [{"title": it.get("title",""), "url": it.get("url","")} for it in items[:20]]
    }
    sys_prompt = (
        "너는 한국 시장 트렌드 리서처다. 아래 데이터(기사/키워드/검색트렌드)만 근거로 "
        "사업화 아이디어 카드를 한국어로 작성하라.\n"
        "원칙: 과장 금지, 정치/연예 배제. 숫자/시장규모는 근거가 없으면 '알 수 없습니다' 또는 '확실하지 않음'으로 표기.\n"
        "출력은 JSON 하나: {title,tagline,sections{problem,solution,target_user,gtm,why_now,proof_signals,market_gap,execution_plan{core,growth,lead_gen,steps[]}},evidence[]}.\n"
        "evidence에는 기사 URL 3~5개와 짧은 메모를 넣어라."
    )
    user_msg = "DATA:\n" + json.dumps(payload, ensure_ascii=False)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
            temperature=0.4,
            response_format={"type":"json_object"},
        )
        card = json.loads(resp.choices[0].message.content)
    except Exception:
        card = {
            "title":"아이디어 생성 실패",
            "tagline":"확실하지 않음",
            "sections":{"problem":"알 수 없습니다","solution":"알 수 없습니다","target_user":"알 수 없습니다",
                        "gtm":"알 수 없습니다","why_now":"알 수 없습니다","proof_signals":"알 수 없습니다",
                        "market_gap":"알 수 없습니다","execution_plan":{"core":"알 수 없습니다","growth":"알 수 없습니다","lead_gen":"알 수 없습니다","steps":[]}},
            "evidence":[]
        }
    return card

# --------------- Main ---------------------
def main():
    # 1) 수집
    items = collect_all()

    # 2) 정치 제외 (이중 방어)
    items = [it for it in items if not blocked(it.get("title")) and not blocked(it.get("description"))]

    # 3) 동적 키워드
    signals = build_keyword_groups(items, max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT)

    # 4) 동적 키워드로 Reddit 검색 보강
    q_terms = [t for t,_ in signals.get("topTerms", [])[:5]]
    if q_terms:
        query = " OR ".join(q_terms)
        items.extend(fetch_reddit_search(query, limit=20))

    # 5) 최근 72h 재확인 + 중복 제거
    items = [it for it in items if is_recent_iso(it.get("published_at",""))]
    seen=set(); uniq=[]
    for it in items:
        key=(it.get("url") or "").strip().lower().rstrip("/")
        if not key or key in seen: 
            continue
        seen.add(key); uniq.append(it)
    items = uniq

    # 6) Naver DataLab (키워드 그룹 기반)
    datalab = naver_datalab(signals.get("keywordGroups", []), days=DATALAB_DAYS)
    metrics = extract_trend_metrics(datalab)

    # 7) 점수/커뮤니티
    scores = score_from_metrics(metrics, items)
    community = community_from_items(items, metrics)

    # 8) GPT로 전체 섹션 생성
    card = write_full_card_with_gpt(items, datalab, metrics, signals)

    # 9) trends.json (원본 + metrics)
    trends_out = {
        "period": {"days": DATALAB_DAYS, "end_kr": NOW_KST.strftime("%Y-%m-%d")},
        "results": datalab.get("results", []),
        "metrics": metrics
    }

    # 10) ideas.json (프론트 렌더링 친화적)
    idea_payload = {
        "date_kr": NOW_KST.strftime("%Y-%m-%d"),
        "idea": {
            "title": card.get("title",""),
            "tagline": card.get("tagline",""),
            "sections": card.get("sections", {}),
            "scores": scores,
            "community": community,
            "evidence": card.get("evidence", []),
            "keywordGroups": signals.get("keywordGroups", []),
            "topTerms": signals.get("topTerms", [])[:20]
        }
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    ch1 = dump_json(RAW_PATH, items)
    ch2 = dump_json(SIGNALS_PATH, signals)
    ch3 = dump_json(TRENDS_PATH, trends_out)
    ch4 = dump_json(IDEAS_PATH, idea_payload)

    print(f"[{NOW_KST.isoformat()}] items={len(items)} "
          f"raw={ch1} signals={ch2} trends={ch3} ideas={ch4}")

if __name__ == "__main__":
    main()
