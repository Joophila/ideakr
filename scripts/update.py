#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ideakr daily updater (KR)
- Google News RSS 수집 → rawitems.json
- 키워드 추출(한글 명사, 블랙리스트 제거) → 후보키워드
- Naver DataLab OR rawitems 일자 집계로 trends.json (series/volume/growth)
- Reddit/YouTube 카운트 → signals.json
- GPT로 카드 작성(실패시 규칙템플릿) → ideas.json

필수 Secrets (GitHub Actions → Settings → Secrets and variables → Actions):
  OPENAI_API_KEY
  NAVER_CLIENT_ID
  NAVER_CLIENT_SECRET
  GOOGLE_API_KEY         # YouTube Data API v3
  GOOGLE_CSE_ID          # (선택) 구글 CSE 사용 시
  REDDIT_CLIENT_ID       # (선택)
  REDDIT_CLIENT_SECRET   # (선택)
  REDDIT_USER_AGENT      # (선택) e.g. ideakr/0.1 by joophila
"""

import os, re, sys, json, time, math, random, hashlib
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Any, Iterable, Tuple
import urllib.parse as urlparse

import requests
from bs4 import BeautifulSoup

# ---------- ENV ----------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY","")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID","")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET","")

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY","")     # YouTube용
GOOGLE_CSE_ID       = os.getenv("GOOGLE_CSE_ID","")      # (선택) 구글 CSE

REDDIT_CLIENT_ID    = os.getenv("REDDIT_CLIENT_ID","")
REDDIT_CLIENT_SECRET= os.getenv("REDDIT_CLIENT_SECRET","")
REDDIT_USER_AGENT   = os.getenv("REDDIT_USER_AGENT","ideakr/0.1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path, default):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def strip_html(s: str) -> str:
    if not s: return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

# ---------- 0. 소스(뉴스 RSS) ----------
# 사업/기술/경제 위주 한국 RSS (필요시 추가/수정)
RSS_SOURCES = [
    "https://news.google.com/rss/search?q=site:koreaherald.com+OR+site:koreatimes.co.kr+OR+site:zdnet.co.kr+OR+site:biz.chosun.com+OR+site:hankyung.com+OR+site:mk.co.kr+OR+site:etnews.com&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=%EA%B8%88%EC%9C%B5%20OR%20%EA%B2%BD%EC%A0%9C%20OR%20%EA%B8%B0%EC%88%A0%20OR%20%EC%B7%A8%EC%97%85%20OR%20%EC%82%AC%EC%9E%A5&hl=ko&gl=KR&ceid=KR:ko",
]

def fetch_rss_items(url: str) -> List[Dict[str,Any]]:
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        out=[]
        for it in soup.find_all("item"):
            title = strip_html(it.title.text if it.title else "")
            link  = it.link.text if it.link else ""
            pub   = it.pubDate.text if it.pubDate else ""
            # ISO 보정
            try:
                pub_dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc)
                pub_kst= pub_dt.astimezone(KST).isoformat()
            except:
                pub_kst = now_kst().isoformat()
            out.append({
                "raw_id": "gn_" + hashlib.md5((title+link).encode("utf-8")).hexdigest()[:12],
                "source_platform":"GoogleNews",
                "source_name":"news.google.com",
                "title":title,
                "description": strip_html(it.description.text if it.description else ""),
                "url": link,
                "published_at": pub_kst,
                "metrics_upvotes": 0,
                "metrics_comments": 0,
            })
        return out
    except Exception as e:
        print("RSS fail:", url, e)
        return []

def collect_rawitems() -> List[Dict[str,Any]]:
    all_items=[]
    for u in RSS_SOURCES:
        all_items += fetch_rss_items(u)
        time.sleep(0.7)
    # 중복 제거
    uniq={}
    for x in all_items:
        k = x["title"][:140]
        if k not in uniq: uniq[k]=x
    arr = list(uniq.values())
    print(f"[rawitems] {len(arr)} items")
    return arr[:300]

# ---------- 1. 키워드 추출 ----------
# 언론사/사이트/일반명사 블랙리스트 (뉴스 타이틀에서 제거)
BLACK = set("""
매일경제 한국경제 한겨레 조선일보 중앙일보 경향신문 연합뉴스 지디넷코리아
머니투데이 이데일리 디지털데일리 전자신문 비즈니스 조선 코리아헤럴드 코리아타임스
네이버 다음 네이트 구글 유튜브 트위터 페이스북 인스타그램 블로그 카페 커뮤니티
속보 단독 인터뷰 영상 사진 포토 사설 칼럼 오피니언 뉴스 기사
서비스 글로벌 기술 플랫폼 솔루션 시스템 프로그램 프로젝트 산업 기업 정부 한국 중국 일본 미국 유럽 세계
오늘 어제 내일 발표 정책 지원 증가 감축 강화 확대
""".split())

HANGUL_NOUN = re.compile(r"[가-힣]{2,}")

def extract_keywords(raw: List[Dict[str,Any]], topn=12) -> List[str]:
    c = Counter()
    for it in raw:
        t = (it.get("title") or "") + " " + (it.get("description") or "")
        for w in HANGUL_NOUN.findall(t):
            if w in BLACK: continue
            # 언론/사이트명 흔적 제거
            if w.endswith(("신문","일보","방송","뉴스")): continue
            c[w]+=1
    # 너무 일반적인 단어 제거
    cand = [(w,n) for w,n in c.most_common(50) if n>=2]
    out  = [w for w,_ in cand][:topn]
    if not out:
        out = ["스타트업", "AI 자동화", "물류 최적화", "에너지 관리", "리테일 분석"]
    print("[keywords]", out)
    return out

# ---------- 2. 트렌드 시계열 ----------
def group_counts_by_day(raw: List[Dict[str,Any]], kw: str) -> Dict[str,int]:
    d = defaultdict(int)
    for it in raw:
        text = (it.get("title","")+" "+it.get("description",""))
        if kw in text:
            try:
                dt = datetime.fromisoformat(it["published_at"]).astimezone(KST).date().isoformat()
            except:
                dt = now_kst().date().isoformat()
            d[dt]+=1
    return d

def build_trends(raw: List[Dict[str,Any]], kws: List[str]) -> List[Dict[str,Any]]:
    end = now_kst().date()
    days = [ (end - timedelta(days=i)).isoformat() for i in range(13,-1,-1) ]  # 14일
    trends=[]
    for i,kw in enumerate(kws, start=1):
        counts = group_counts_by_day(raw, kw)
        series = [{"date":d, "value": counts.get(d,0)} for d in days]
        vol = sum(x["value"] for x in series)
        prev = sum(x["value"] for x in series[:7])
        last = sum(x["value"] for x in series[7:])
        growth = 0.0
        if prev>0:
            growth = (last-prev)/prev
        trends.append({
            "trend_id": f"tr_{i:04d}",
            "keyword": kw,
            "trend_score": max(0, min(100, int(vol*5 + growth*100))),
            "volume": vol,
            "growth_percent": round(growth,3),
            "region":"KR",
            "timespan":"14d",
            "series": series,
            "evidence_rawitems": [],
            "updated_at": now_kst().isoformat()
        })
    return trends

# ---------- 3. 커뮤니티 신호 ----------
def reddit_counts(query: str) -> Dict[str,int]:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        return {"posts":0,"upvotes":0,"comments":0}
    try:
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": REDDIT_USER_AGENT}
        tok = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers, timeout=15).json()
        hdr = {"Authorization": f"bearer {tok['access_token']}", "User-Agent": REDDIT_USER_AGENT}
        r = requests.get("https://oauth.reddit.com/search",
                         params={"q":query, "sort":"hot", "t":"week", "limit":50},
                         headers=hdr, timeout=20)
        j = r.json()
        posts = j.get("data",{}).get("dist",0)
        ups = sum(max(0, c.get("data",{}).get("ups",0)) for c in j.get("data",{}).get("children",[]))
        com = sum(max(0, c.get("data",{}).get("num_comments",0)) for c in j.get("data",{}).get("children",[]))
        return {"posts":posts,"upvotes":ups,"comments":com}
    except Exception as e:
        print("reddit fail", e)
        return {"posts":0,"upvotes":0,"comments":0}

def youtube_counts(query: str) -> Dict[str,int]:
    if not GOOGLE_API_KEY:
        return {"videos": 0}
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"key":GOOGLE_API_KEY,"q":query,"part":"id","type":"video","maxResults":50,"publishedAfter":(now_kst()-timedelta(days=14)).astimezone(timezone.utc).isoformat()}
        r = requests.get(url, params=params, timeout=20)
        j = r.json()
        return {"videos": len(j.get("items",[]))}
    except Exception as e:
        print("youtube fail", e)
        return {"videos":0}

# ---------- 4. 증거(뉴스/레딧) ----------
def choose_evidence(raw: List[Dict[str,Any]], kw: str, limit=6) -> List[Dict[str,str]]:
    rows=[]
    for it in raw:
        t=(it.get("title") or "") + " " + (it.get("description") or "")
        if kw in t:
            rows.append({"title": it["title"][:120], "url": it["url"]})
    # 중복 제거
    seen=set(); ev=[]
    for r in rows:
        k=r["title"]
        if k not in seen:
            ev.append(r)
            seen.add(k)
        if len(ev)>=limit: break
    return ev

# ---------- 5. GPT 카드(실패해도 절대 빈칸 X) ----------
def make_business_card(top_kw: str,
                       metrics: Dict[str,Any],
                       evid: List[Dict[str,str]]) -> Dict[str,Any]:
    base = {
        "title": f"{top_kw} 기반 솔루션",
        "tagline": f"{top_kw} 관련 수요 증가를 겨냥한 경량 MVP",
        "sections": {
            "problem": f"‘{top_kw}’와 관련해 실사용자가 느끼는 구체적 문제/비효율이 많지만 검증된 제품이 부족합니다.",
            "solution": f"{top_kw} 핵심 과업 1~2개만 해결하는 간단한 SaaS/툴 형태로 빠른 검증을 목표로 합니다.",
            "target_user": "초기 채택자(해당 업무를 자주 수행하는 파워유저/팀)",
            "gtm": "커뮤니티(YouTube/Reddit)와 뉴스레터/SEO로 저비용 테스트 → 파트너십 확장"
        },
        "why_cards": [
            f"최근 2주 언급량 {metrics.get('volume',0)}건, 전주 대비 성장률 {round(metrics.get('growth',0)*100,1)}%",
            "유관 커뮤니티(YouTube/Reddit)에서 관련 주제 콘텐츠가 꾸준히 생성"
        ],
        "gap_notes": ["근거가 부족합니다"],
        "exec_steps": [
            "2주: 문제/해결 적합성 인터뷰·체크리스트",
            "4주: 핵심 기능 MVP 개발·유료 베타",
            "지표: 가입전환/재방문/유료전환/해지 사유 수집"
        ],
        "offer_ladder":[
            {"name":"Lead Magnet","price":"Free","unit":"체크리스트/미니툴"},
            {"name":"Core MVP","price":"₩9,900~₩29,000/월","unit":"SaaS"},
            {"name":"Pro","price":"₩99,000+/월","unit":"확장"}
        ],
        "pricing":["월 구독 + 연간 2개월 혜택"],
        "channels":["YouTube","SEO","파트너 제휴"],
        "competitors":[],
        "personas":[]
    }

    if not OPENAI_API_KEY:
        return base

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        sysmsg = ("너는 한국 스타트업 리서처다. 아래 정량근거만 사용하여 한국어 JSON으로 작성."
                  "과장 금지, 추정은 '근거가 부족합니다'로 표기. 응답은 JSON 하나.")
        payload = {"keyword": top_kw, "metrics": metrics, "evidence": evid}
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sysmsg},
                      {"role":"user","content":"DATA:\n"+json.dumps(payload,ensure_ascii=False)}],
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        out = json.loads(resp.choices[0].message.content)
        # 빈칸 보강
        for k,v in base.items():
            if k not in out or (isinstance(out[k],(str,list,dict)) and not out[k]):
                out[k]=v
        for sk,sv in base["sections"].items():
            if "sections" not in out: out["sections"]={}
            if not out["sections"].get(sk): out["sections"][sk]=sv
        return out
    except Exception as e:
        print("GPT fail -> fallback:", e)
        return base

# ---------- 6. 메인 ----------
def main():
    raw = collect_rawitems()
    save_json(os.path.join(DATA,"rawitems.json"), raw)

    # 키워드 & 트렌드
    kws = extract_keywords(raw, topn=8)
    trends = build_trends(raw, kws)

    # top 키워드 선택(최고 volume→growth)
    top = max(trends, key=lambda t: (t["volume"], t["growth_percent"])) if trends else {
        "keyword":"스타트업", "volume":0, "growth_percent":0.0
    }

    # 커뮤니티 신호 (top 키워드 기준)
    sig = {
        "reddit": reddit_counts(top["keyword"]),
        "youtube": youtube_counts(top["keyword"]),
        "naver": {"groups":0}
    }

    # 증거(뉴스/레딧 위주)
    evidence = choose_evidence(raw, top["keyword"], limit=6)

    # 카드 생성
    metrics = {"volume": top.get("volume",0), "growth": top.get("growth_percent",0.0)}
    card = make_business_card(top["keyword"], metrics, evidence)

    # ideas.json 구조
    idea = {
        "idea_id": f"idea_{now_kst().strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or f"{top['keyword']} 아이디어",
        "one_liner": card.get("tagline",""),
        "problem": card.get("sections",{}).get("problem",""),
        "solution": card.get("sections",{}).get("solution",""),
        "target_user": card.get("sections",{}).get("target_user",""),
        "why_now": " ".join(card.get("why_cards",[])[:1]),
        "biz_model": " · ".join(card.get("pricing",[])[:1]),
        "gtm_tactics": card.get("sections",{}).get("gtm",""),
        "validation_steps": "빠른 인터뷰/대기열/유료베타로 정량검증",
        "tags": [t["keyword"] for t in trends[:4]] if trends else [top["keyword"]],
        "score_breakdown": {  # 추정치
            "trend": max(0,min(100,int(metrics["volume"]*5 + metrics["growth"]*100))),
            "market": min(100, len(raw)*2),
            "competition_invert": max(0, 100 - min(90, (len(raw)/5)*10)),
            "feasibility": 50, "monetization": 50, "regulatory_invert": 50
        },
        "score_total": 0,
        "trend_link": [t["trend_id"] for t in trends[:3]],
        "is_today": True,
        "why_cards": card.get("why_cards",[]),
        "gap_notes": card.get("gap_notes",[]) or ["근거가 부족합니다"],
        "exec_steps": card.get("exec_steps",[]) or ["근거가 부족합니다"],
        "evidence": evidence,
        "offer_ladder": card.get("offer_ladder",[]),
        "pricing": card.get("pricing",[]),
        "channels": card.get("channels",[]),
        "competitors": card.get("competitors",[]),
        "personas": card.get("personas",[])
    }
    idea["score_total"] = int(0.35*idea["score_breakdown"]["trend"] + 0.25*idea["score_breakdown"]["market"]
                              + 0.15*idea["score_breakdown"]["competition_invert"] + 0.25*50)

    # 저장
    save_json(os.path.join(DATA,"trends.json"), trends)
    save_json(os.path.join(DATA,"signals.json"), sig)
    save_json(os.path.join(DATA,"ideas.json"), [idea])

    print(f"[{now_kst().isoformat()}] items={len(raw)} trends={len(trends)} top={top['keyword']}")

if __name__ == "__main__":
    main()
