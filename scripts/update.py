#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ideakr — 사업화 아이템 카드 자동 생성기 (KR)
소스: Google News RSS + Google Custom Search + Reddit(OAuth) + YouTube Data API
지표: Naver DataLab(14d) / 커뮤니티 신호
산출: data/ideas.json, data/trends.json, data/rawitems.json, data/signals.json
- OPENAI_API_KEY 없으면 규칙 템플릿으로라도 카드 생성
"""

import os, re, json, time, math, hashlib, random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import requests, feedparser
from bs4 import BeautifulSoup
from dateutil import tz

# ---------- 환경변수 ----------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")

YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY", "")

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID       = os.getenv("GOOGLE_CSE_ID", "")

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "ideakr-bot/2.0 by joophila")

# ---------- 경로 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FN_IDEAS   = os.path.join(DATA_DIR, "ideas.json")
FN_TRENDS  = os.path.join(DATA_DIR, "trends.json")
FN_RAW     = os.path.join(DATA_DIR, "rawitems.json")
FN_SIGNALS = os.path.join(DATA_DIR, "signals.json")

KST = timezone(timedelta(hours=9))
NOW = datetime.now(KST)
UA  = {"User-Agent": "ideakr-bot/2.0 (+github actions)"}

# ---------- 유틸 ----------
DOMAIN_TOKEN_RE = re.compile(r"(https?://\S+)|([a-z0-9_-]+\.(co|com|kr|net|org)\b)", re.I)
PUBLISHER_BLACKLIST = {
    "매일경제","한국경제","한겨레","조선일보","중앙일보","동아일보","연합뉴스",
    "서울경제","머니투데이","이데일리","전자신문","블로터","네이버","다음","네이트",
    "kbs","mbc","sbs","ytn","jtbc","chosun","hankyung","zdnet","zumnet"
}
STOPWORDS = {
    "the","and","for","with","from","into","your","our","new","how","what","why","when","where","who",
    "this","that","are","is","was","to","of","in","on","by","at","as","it",
    "기사","속보","단독","사진","영상","인터뷰","전문","종합","오늘","어제","최근","관련",
    "https","http","com","kr","net","news","뉴스","기자","포토","업데이트",
}

def sha(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8","ignore")).hexdigest()[:12]

def strip_html(s: str) -> str:
    try:
        return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)
    except Exception:
        return s or ""

def clean_text(s: str) -> str:
    s = strip_html(s)
    s = DOMAIN_TOKEN_RE.sub(" ", s)
    s = re.sub(r"[\[\]()<>{}“”\"'`•·…~^_|=:;#@※]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_ko_en(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s\-]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    out = []
    for t in toks:
        if t.isdigit(): 
            continue
        if t in STOPWORDS or t in PUBLISHER_BLACKLIST:
            continue
        out.append(t)
    return out

def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def dedupe_by_url(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    seen, out = set(), []
    for it in items:
        key = sha((it.get("url") or it.get("title") or "")[:512])
        if key in seen: 
            continue
        seen.add(key); out.append(it)
    return out

# ---------- 수집: Google News RSS ----------
def fetch_gnews() -> List[Dict[str, Any]]:
    feeds = [
        # 기술/과학/스타트업 위주
        "https://news.google.com/rss/search?q=site:co.kr%20(%EA%B8%B0%EC%88%A0%20OR%20%EA%B3%BC%ED%95%99%20OR%20%EB%B9%84%EC%A6%88%EB%8B%88%EC%8A%A4%20OR%20%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85)&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=(SaaS%20OR%20%ED%94%84%EB%A1%9C%EB%8D%95%ED%8A%B8%20OR%20%EC%95%B1%20OR%20%ED%94%8C%EB%9E%AB%ED%8F%BC)%20site:co.kr&hl=ko&gl=KR&ceid=KR:ko",
    ]
    ban = re.compile(r"(총선|대선|국회|여야|정치|외교|북한|시위|탄핵|검찰|대법원|의회|청와대)")
    out = []
    since = NOW - timedelta(days=3)

    for u in feeds:
        try:
            f = feedparser.parse(u)
            for e in f.entries:
                title = e.get("title","")
                if ban.search(title): 
                    continue
                link  = e.get("link","")
                desc  = e.get("summary","")
                pub   = e.get("published_parsed") or e.get("updated_parsed")
                dt    = datetime.fromtimestamp(time.mktime(pub), tz=timezone.utc).astimezone(KST) if pub else NOW
                if dt < since: 
                    continue
                out.append({
                    "raw_id": f"gn_{sha(link or title)}",
                    "source_platform": "GoogleNews",
                    "source_name": e.get("source",{}).get("title",""),
                    "title": title,
                    "description": desc,
                    "url": link,
                    "published_at": dt.isoformat(),
                    "metrics_upvotes": 0,
                    "metrics_comments": 0
                })
        except Exception:
            continue
    return out

# ---------- 수집: Google Custom Search (Programmable Search) ----------
def fetch_google_cse() -> List[Dict[str,Any]]:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return []
    # 비즈니스/기술 키워드 중심 쿼리
    queries = [
        "site:co.kr (스타트업 OR 창업) (투자 OR 제품 OR 출시)",
        "site:co.kr (AI OR 인공지능 OR SaaS) (서비스 OR 도구 OR 툴)",
        "site:co.kr (핀테크 OR 헬스테크 OR 에듀테크) (서비스 OR 협업 OR 제휴)"
    ]
    ban = re.compile(r"(총선|대선|국회|정치|외교|북한)")
    out = []
    for q in queries:
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key":GOOGLE_API_KEY,"cx":GOOGLE_CSE_ID,"q":q,"num":10,"safe":"off","hl":"ko"},
                timeout=20
            )
            if r.status_code != 200: 
                continue
            for it in r.json().get("items",[]):
                title = it.get("title","")
                if ban.search(title): 
                    continue
                link = it.get("link","")
                snippet = it.get("snippet","")
                meta = it.get("pagemap",{}).get("metatags",[{}])[0]
                published = meta.get("article:published_time") or meta.get("datepublished") or meta.get("pubdate")
                if published:
                    try:
                        dt = datetime.fromisoformat(published.replace("Z","+00:00")).astimezone(KST)
                    except Exception:
                        dt = NOW
                else:
                    dt = NOW
                out.append({
                    "raw_id": f"gc_{sha(link)}",
                    "source_platform": "GoogleCSE",
                    "source_name": re.sub(r"^https?://(www\.)?","", it.get("displayLink","")),
                    "title": title,
                    "description": snippet,
                    "url": link,
                    "published_at": dt.isoformat(),
                    "metrics_upvotes": 0,
                    "metrics_comments": 0
                })
        except Exception:
            continue
    return out

# ---------- Reddit OAuth ----------
def reddit_token() -> str:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        return ""
    try:
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {"grant_type":"client_credentials"}
        headers = {"User-Agent": REDDIT_USER_AGENT}
        r = requests.post("https://www.reddit.com/api/v1/access_token",
                          auth=auth, data=data, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json().get("access_token","")
    except Exception:
        return ""
    return ""

def fetch_reddit_by_keywords(keywords: List[str]) -> List[Dict[str,Any]]:
    tok = reddit_token()
    if not tok:
        return []
    out = []
    since = NOW - timedelta(days=7)
    subs = ["Entrepreneur","startups","smallbusiness","technology","Futurology"]
    headers = {"Authorization": f"Bearer {tok}", "User-Agent": REDDIT_USER_AGENT}
    for kw in keywords[:6]:
        q = kw
        for sub in subs:
            url = f"https://oauth.reddit.com/r/{sub}/search"
            try:
                r = requests.get(url, headers=headers, params={
                    "q": q, "restrict_sr": "on", "sort": "top", "t": "week", "limit": 10
                }, timeout=20)
                if r.status_code != 200:
                    continue
                for c in r.json().get("data",{}).get("children",[]):
                    d = c.get("data",{})
                    dt = datetime.fromtimestamp(float(d.get("created_utc", time.time())),
                                                tz=timezone.utc).astimezone(KST)
                    if dt < since: 
                        continue
                    out.append({
                        "raw_id": f"rd_{sha(d.get('permalink','')+sub)}",
                        "source_platform": "Reddit",
                        "source_name": sub,
                        "title": d.get("title",""),
                        "description": (d.get("selftext") or "")[:400],
                        "url": "https://www.reddit.com"+d.get("permalink",""),
                        "published_at": dt.isoformat(),
                        "metrics_upvotes": int(d.get("ups",0)),
                        "metrics_comments": int(d.get("num_comments",0))
                    })
            except Exception:
                continue
    return out

# ---------- YouTube ----------
def fetch_youtube_for_keywords(keywords: List[str]) -> List[Dict[str,Any]]:
    if not YOUTUBE_API_KEY:
        return []
    out = []
    since = (NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for kw in keywords[:6]:
        try:
            sr = requests.get("https://www.googleapis.com/youtube/v3/search",
                              params={
                                  "part":"snippet","q":kw,"type":"video","maxResults":10,
                                  "order":"viewCount","publishedAfter":since,"regionCode":"KR",
                                  "relevanceLanguage":"ko","key":YOUTUBE_API_KEY
                              }, timeout=20)
            ids = [i["id"]["videoId"] for i in sr.json().get("items",[])]
            if not ids: 
                continue
            vr = requests.get("https://www.googleapis.com/youtube/v3/videos",
                              params={"part":"statistics,snippet","id":",".join(ids),"key":YOUTUBE_API_KEY},
                              timeout=20)
            for it in vr.json().get("items",[]):
                sn = it.get("snippet",{}); st = it.get("statistics",{})
                out.append({
                    "raw_id": f"yt_{it.get('id')}",
                    "source_platform": "YouTube",
                    "source_name": sn.get("channelTitle",""),
                    "title": sn.get("title",""),
                    "description": (sn.get("description") or "")[:400],
                    "url": "https://www.youtube.com/watch?v="+it.get("id",""),
                    "published_at": sn.get("publishedAt"),
                    "metrics_upvotes": int(st.get("likeCount","0") or 0),
                    "metrics_comments": int(st.get("commentCount","0") or 0)
                })
        except Exception:
            continue
    return out

# ---------- 키워드/트렌드 ----------
def build_keyword_groups(items: List[Dict[str,Any]], max_groups=8) -> List[Dict[str,Any]]:
    score = {}
    def bump(term: str, w: float):
        if not term: return
        if term in STOPWORDS or term in PUBLISHER_BLACKLIST: return
        if re.fullmatch(r"[a-z]{1,3}", term): return
        if re.search(r"\b(http|https|www|com|co|kr|net)\b", term): return
        score[term] = score.get(term, 0.0) + w

    for it in items:
        txt = clean_text((it.get("title","") or "") + " " + (it.get("description","") or ""))
        toks = tokenize_ko_en(txt)
        bigr = ngrams(toks, 2)
        trgr = ngrams(toks, 3)
        src = (it.get("source_platform","") or "").lower()
        w = 1.0
        if src.startswith("reddit"):  w = 1.5
        if src.startswith("youtube"): w = 1.4
        w *= (1.0 + 0.00002*it.get("metrics_upvotes",0) + 0.00005*it.get("metrics_comments",0))
        for t in toks: bump(t, w)
        for t in bigr: bump(t, w*1.2)
        for t in trgr: bump(t, w*1.4)

    cand = sorted(score.items(), key=lambda x:x[1], reverse=True)
    groups = []
    for k,_ in cand:
        if k in {"https","com"}: continue
        if any(k in " ".join(g["keywords"]) or g["groupName"] in k for g in groups):
            continue
        groups.append({"groupName": k, "keywords":[k]})
        if len(groups) >= max_groups: break
    if not groups:
        groups = [{"groupName":"시장 동향","keywords":["시장 동향"]}]
    return groups

def query_naver_datalab(groups: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {}
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json; charset=utf-8",
    }
    end = NOW.date()
    start = (NOW - timedelta(days=13)).date()
    body = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate":   end.strftime("%Y-%m-%d"),
        "timeUnit":  "date",
        "keywordGroups": [{"groupName": g["groupName"], "keywords": g["keywords"][:3]} for g in groups],
        "device": "", "gender": "", "ages": []
    }
    try:
        r = requests.post("https://openapi.naver.com/v1/datalab/search",
                          headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code != 200: return {}
        js = r.json()
        out = {}
        for series in js.get("results", []):
            name = series.get("title")
            data = series.get("data", [])
            out[name] = data
        return out
    except Exception:
        return {}

def build_trends(groups: List[Dict[str,Any]], datalab: Dict[str,Any]) -> List[Dict[str,Any]]:
    trends = []
    for i,g in enumerate(groups, start=1):
        name = g["groupName"]
        series = datalab.get(name, [])
        vol = sum([d.get("ratio",0) for d in series])
        growth = 0.0
        if len(series) >= 2:
            growth = (series[-1]["ratio"] - series[0]["ratio"]) / max(1.0, series[0]["ratio"])
        trend_score = min(100, round(0.7*vol + 30*growth))
        trends.append({
            "trend_id": f"tr_{i:04d}",
            "keyword": name,
            "trend_score": trend_score,
            "volume": round(vol),
            "growth_percent": float(growth),
            "region": "KR",
            "timespan": "14d",
            "series": [{"date": d["period"], "value": d["ratio"]} for d in series],
            "evidence_rawitems": [],
            "updated_at": NOW.isoformat()
        })
    return trends

def build_metrics(groups, datalab):
    metrics = []
    for g in groups:
        name = g["groupName"]
        data = datalab.get(name, [])
        vol = sum([d.get("ratio",0) for d in data])
        growth = 0.0
        if len(data) >= 2:
            growth = round( (data[-1]["ratio"]-data[0]["ratio"]) / max(1.0,data[0]["ratio"]) * 100, 1 )
        metrics.append({"groupName": name, "volume": round(vol), "growth": growth})
    metrics.sort(key=lambda m:(m["volume"], m["growth"]), reverse=True)
    return metrics

# ---------- 카드 생성 ----------
def make_business_card(top_kw: str,
                       metrics: List[Dict[str,Any]],
                       citations: List[Dict[str,str]]) -> Dict[str,Any]:
    if not OPENAI_API_KEY:
        # 규칙 기반 템플릿
        return {
            "title": f"{top_kw} 기반 솔루션",
            "tagline": f"{top_kw} 관련 수요가 관측됨 — 단계적 검증 권장",
            "sections": {
                "problem": f"‘{top_kw}’ 수요는 있으나 검증된 해결책과 사례가 부족.",
                "solution": f"{top_kw} JTBD 1~2개만 해결하는 경량 MVP로 시장성 검증.",
                "target_user": "초기 채택자/파워유저",
                "gtm": "콘텐츠/커뮤니티 중심의 저비용 채널 테스트"
            },
            "why_cards": ["검색량과 커뮤니티 언급이 증가 추세(정량 근거는 DataLab)"],
            "gap_notes": ["근거가 부족합니다"],
            "exec_steps": [
                "대기열 랜딩 · 인터뷰(2주)",
                "MVP(핵심 기능만, 4주) · 유료 베타",
                "채널: Reddit/YouTube/블로그 SEO",
            ],
            "offer_ladder": [
                {"name":"Lead Magnet","price":"Free","unit":"체크리스트/미니툴"},
                {"name":"Core MVP","price":"₩9,900~₩29,000/월","unit":"SaaS"},
                {"name":"Pro","price":"₩99,000+/월","unit":"확장"}
            ],
            "pricing": ["월 구독 + 연간 2개월 혜택"],
            "channels": ["YouTube, Reddit, 파트너 제휴"],
            "competitors": [],
            "personas": [],
        }

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    schema = {
      "type":"object",
      "properties":{
        "title":{"type":"string"},
        "tagline":{"type":"string"},
        "sections":{
          "type":"object",
          "properties":{
            "problem":{"type":"string"},
            "solution":{"type":"string"},
            "target_user":{"type":"string"},
            "gtm":{"type":"string"}
          }, "required":["problem","solution","target_user","gtm"]
        },
        "why_cards":{"type":"array","items":{"type":"string"}},
        "gap_notes":{"type":"array","items":{"type":"string"}},
        "exec_steps":{"type":"array","items":{"type":"string"}},
        "offer_ladder":{"type":"array","items":{"type":"object",
            "properties":{"name":{"type":"string"},"price":{"type":"string"},"unit":{"type":"string"}},
            "required":["name","price","unit"]}},
        "pricing":{"type":"array","items":{"type":"string"}},
        "channels":{"type":"array","items":{"type":"string"}},
        "competitors":{"type":"array","items":{"type":"string"}},
        "personas":{"type":"array","items":{"type":"object",
            "properties":{"name":{"type":"string"},"pain":{"type":"string"},"jtbd":{"type":"string"}},
            "required":["name","pain","jtbd"]}},
      },
      "required":["title","tagline","sections","why_cards","gap_notes","exec_steps"]
    }

    sys = (
      "너는 한국 스타트업 리서처다. 아래 정량지표와 인용 링크만 근거로 "
      "‘사업화 아이템 카드’를 한국어로 작성하라. 과장 금지, 근거가 약하면 "
      "‘근거가 부족합니다’를 사용. 응답은 JSON 하나만 반환. "
      "Offer Ladder, 가격, 채널, 경쟁사, 페르소나 포함."
    )
    payload = {"keyword": top_kw, "metrics": metrics[:5], "citations": citations[:8]}
    msg = "DATA:\n"+json.dumps(payload, ensure_ascii=False)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":msg}],
        temperature=0.2,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ---------- 메인 ----------
def main():
    # 1) RSS
    gnews = fetch_gnews()
    # 2) Google CSE
    cse   = fetch_google_cse()
    base  = dedupe_by_url(gnews + cse)

    # 3) 초벌 키워드
    groups1 = build_keyword_groups(base)

    # 4) Reddit / YouTube (키워드 기반)
    reddit  = fetch_reddit_by_keywords([g["groupName"] for g in groups1])
    youtube = fetch_youtube_for_keywords([g["groupName"] for g in groups1])

    # 5) 전체 raw
    raw = dedupe_by_url(base + reddit + youtube)

    # 6) 최종 키워드 & DataLab
    groups  = build_keyword_groups(raw)
    datalab = query_naver_datalab(groups)

    # 7) 지표/트렌드
    metrics = build_metrics(groups, datalab)
    trends  = build_trends(groups, datalab)

    # 8) 커뮤니티 시그널
    signals = {
        "reddit":  {"posts": len(reddit), "upvotes": sum(i["metrics_upvotes"] for i in reddit), "comments": sum(i["metrics_comments"] for i in reddit)},
        "youtube": {"videos": len(youtube)},
        "naver":   {"groups": len(datalab)}
    }

    # 9) 인용(상위 신호)
    citations = sorted(raw, key=lambda it:(it.get("metrics_upvotes",0)*2 + it.get("metrics_comments",0)), reverse=True)[:8]
    cite_min = [{"title": clean_text(c.get("title",""))[:120], "url": c.get("url","")} for c in citations]

    # 10) 사업화 카드
    top_kw = metrics[0]["groupName"] if metrics else (groups[0]["groupName"] if groups else "신규")
    card   = make_business_card(top_kw, metrics, cite_min)

    idea_obj = {
        "idea_id": f"idea_{NOW.strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or f"{top_kw} 아이디어",
        "one_liner": card.get("tagline") or "확실하지 않음",
        "problem": card.get("sections",{}).get("problem","알 수 없습니다"),
        "solution": card.get("sections",{}).get("solution","알 수 없습니다"),
        "target_user": card.get("sections",{}).get("target_user","알 수 없습니다"),
        "why_now": "",
        "biz_model": "알 수 없습니다",
        "gtm_tactics": card.get("sections",{}).get("gtm","알 수 없습니다"),
        "validation_steps": "확실하지 않음",
        "tags": [t.get("keyword") for t in trends[:4]],
        "score_breakdown": {},
        "score_total": 0,
        "trend_link": [t["trend_id"] for t in trends[:3]],
        "is_today": True,
        "why_cards": card.get("why_cards",[]),
        "gap_notes": card.get("gap_notes",[]),
        "exec_steps": card.get("exec_steps",[]),
        "evidence": cite_min,
        "offer_ladder": card.get("offer_ladder",[]),
        "pricing": card.get("pricing",[]),
        "channels": card.get("channels",[]),
        "competitors": card.get("competitors",[]),
        "personas": card.get("personas",[]),
    }

    # 11) 저장
    def dump(fp, obj):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    dump(FN_IDEAS, [idea_obj])
    dump(FN_TRENDS, trends)
    dump(FN_RAW, raw)
    dump(FN_SIGNALS, signals)

    print(f"[{NOW.isoformat()}] items={len(raw)} NAV={'ok' if NAVER_CLIENT_ID else 'no'} "
          f"OPENAI={'ok' if OPENAI_API_KEY else 'no'} YT={'ok' if YOUTUBE_API_KEY else 'no'} "
          f"CSE={'ok' if (GOOGLE_API_KEY and GOOGLE_CSE_ID) else 'no'} "
          f"RD={'ok' if (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET) else 'no'}")

if __name__ == "__main__":
    main()

