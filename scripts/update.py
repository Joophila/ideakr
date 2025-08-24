# -*- coding: utf-8 -*-
# ideakr updater: GNews + Reddit + YouTube + Naver DataLab + (optional) GPT
# ⚠️ 프론트 스키마는 기존 그대로 유지합니다 (why_now / proof & signals 등 구조 변경 없음)

from __future__ import annotations
import json, os, re, time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import requests, feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtp

# ----- 환경 -----
KST = timezone(timedelta(hours=9))
NOW_UTC = datetime.now(timezone.utc)
NOW_KST = NOW_UTC.astimezone(KST)

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY")

ENABLE_GNEWS  = (os.getenv("ENABLE_GNEWS","1") != "0")
ENABLE_REDDIT = (os.getenv("ENABLE_REDDIT","1") != "0")
ENABLE_YT     = (os.getenv("ENABLE_YT","1") != "0")

WINDOW_HOURS  = int(os.getenv("WINDOW_HOURS","72"))
DATALAB_DAYS  = int(os.getenv("DATALAB_DAYS","14"))
MAX_GNEWS     = int(os.getenv("MAX_GNEWS","60"))
MAX_REDDIT    = int(os.getenv("MAX_REDDIT","20"))
MAX_YT_TREND  = int(os.getenv("MAX_YT_TREND","20"))
MAX_YT_SEARCH = int(os.getenv("MAX_YT_SEARCH","10"))
MAX_GROUPS          = int(os.getenv("MAX_GROUPS","8"))
MAX_GROUPS_DATALAB  = int(os.getenv("MAX_GROUPS_DATALAB","5"))
MIN_TERM_COUNT      = int(os.getenv("MIN_TERM_COUNT","3"))

TOPICS = [t.strip() for t in os.getenv("GNEWS_TOPICS","BUSINESS,SCIENCE,TECHNOLOGY").split(",") if t.strip()]
REDDIT_SUBS = [s.strip() for s in os.getenv("REDDIT_SUBS","Entrepreneur,startups,technology,Futurology,smallbusiness").split(",") if s.strip()]
YOUTUBE_REGION = os.getenv("YOUTUBE_REGION","KR")

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR,"rawitems.json")
TRN_PATH = os.path.join(DATA_DIR,"trends.json")
IDEA_PATH= os.path.join(DATA_DIR,"ideas.json")
SIG_PATH = os.path.join(DATA_DIR,"signals.json")

# ----- 유틸 -----
def strip_html(s:str)->str:
    try: return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)
    except Exception: return s or ""

def iso(dt_any)->str|None:
    if isinstance(dt_any,str):
        try: dt_any=dtp.parse(dt_any)
        except Exception: return None
    if not getattr(dt_any,"tzinfo",None): dt_any=dt_any.replace(tzinfo=timezone.utc)
    return dt_any.isoformat()

def is_recent(iso_str:str|None, hours:int=WINDOW_HOURS)->bool:
    if not iso_str: return False
    try:
        d=dtp.parse(iso_str)
        if not d.tzinfo: d=d.replace(tzinfo=timezone.utc)
        return d >= datetime.now(timezone.utc)-timedelta(hours=hours)
    except Exception:
        return False

def dump(path:str, obj:Any)->bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev=None
    try:
        with open(path,"r",encoding="utf-8") as f: prev=json.load(f)
    except Exception: pass
    if prev==obj: return False
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.write("\n")
    return True

# ----- 수집 -----
def fetch_gnews(topic:str)->list[dict]:
    url=f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"
    out=[]
    try: feed=feedparser.parse(url)
    except Exception: return out
    for e in feed.entries[:MAX_GNEWS]:
        ttl=e.get("title") or ""; link=e.get("link") or ""
        desc=strip_html(e.get("summary") or e.get("description") or "")
        pub=e.get("published") or e.get("updated"); pub_iso=iso(pub) if pub else None
        if not is_recent(pub_iso): continue
        out.append({
            "title":ttl,"url":link,"published_at":pub_iso,
            "description":desc,
            "source_name":f"GoogleNews:{topic}","source_url":url,"topic":topic,
            "ups":0,"num_comments":0
        })
    return out

def fetch_reddit(sub:str)->list[dict]:
    url=f"https://www.reddit.com/r/{sub}/top.json"
    qs={"t":"week","limit":MAX_REDDIT}; headers={"User-Agent":"Mozilla/5.0"}
    try:
        r=requests.get(url,params=qs,headers=headers,timeout=20); r.raise_for_status(); js=r.json()
    except Exception:
        return []
    out=[]
    for ch in (js.get("data",{}) or {}).get("children",[]):
        d=ch.get("data",{}) or {}
        ttl=d.get("title") or ""; body=d.get("selftext") or ""
        created=d.get("created_utc"); 
        if not created: continue
        pub_iso=iso(datetime.fromtimestamp(created, tz=timezone.utc))
        if not is_recent(pub_iso): continue
        per=d.get("permalink"); urlp=f"https://www.reddit.com{per}" if per else (d.get("url") or "")
        out.append({
            "title":ttl,"url":urlp,"published_at":pub_iso,"description":body,
            "source_name":f"reddit:{sub}","source_url":f"https://www.reddit.com/r/{sub}","topic":"REDDIT",
            "ups":int(d.get("ups") or 0),"num_comments":int(d.get("num_comments") or 0),
        })
    return out

# --- YouTube Data API v3
def yt_get(url:str, params:dict)->dict|None:
    if not YOUTUBE_API_KEY: return None
    try:
        params=dict(params or {}); params["key"]=YOUTUBE_API_KEY
        r=requests.get(url, params=params, timeout=20); 
        if r.status_code!=200: return None
        return r.json()
    except Exception:
        return None

def fetch_yt_trending()->list[dict]:
    if not (ENABLE_YT and YOUTUBE_API_KEY): return []
    js=yt_get("https://www.googleapis.com/youtube/v3/videos",{
        "part":"snippet,statistics","chart":"mostPopular","regionCode":YOUTUBE_REGION,"maxResults":min(50,MAX_YT_TREND)
    }) or {}
    out=[]
    for it in js.get("items",[]):
        sn=it.get("snippet",{}) or {}; st=it.get("statistics",{}) or {}
        vid=it.get("id"); url=f"https://www.youtube.com/watch?v={vid}" if vid else ""
        pub=sn.get("publishedAt"); pub_iso=iso(pub) if pub else None
        if not is_recent(pub_iso): continue
        out.append({
            "title":sn.get("title") or "", "url":url, "published_at":pub_iso,
            "description":sn.get("description") or "",
            "source_name":"YouTube:trending","source_url":"https://www.youtube.com/feed/trending","topic":"YOUTUBE",
            "ups":int(st.get("viewCount") or 0), "num_comments":int(st.get("commentCount") or 0),
        })
    return out

def fetch_yt_search(query:str, n:int=MAX_YT_SEARCH)->list[dict]:
    if not (ENABLE_YT and YOUTUBE_API_KEY): return []
    published_after=(datetime.now(timezone.utc)-timedelta(hours=WINDOW_HOURS)).isoformat()
    s=yt_get("https://www.googleapis.com/youtube/v3/search",{
        "part":"snippet","q":query,"type":"video","maxResults":min(50,n),
        "order":"viewCount","relevanceLanguage":"ko","regionCode":YOUTUBE_REGION,"publishedAfter":published_after
    }) or {}
    ids=[it["id"]["videoId"] for it in s.get("items",[]) if it.get("id",{}).get("videoId")]
    stats={}
    if ids:
        v=yt_get("https://www.googleapis.com/youtube/v3/videos",{"part":"statistics","id":",".join(ids)}) or {}
        for it in v.get("items",[]): stats[it.get("id")] = it.get("statistics",{}) or {}
    out=[]
    for it in s.get("items",[]):
        sn=it.get("snippet",{}) or {}; vid=it.get("id",{}).get("videoId")
        pub=sn.get("publishedAt"); pub_iso=iso(pub) if pub else None
        if not is_recent(pub_iso): continue
        st=stats.get(vid,{})
        out.append({
            "title":sn.get("title") or "", "url":f"https://www.youtube.com/watch?v={vid}" if vid else "",
            "published_at":pub_iso, "description":sn.get("description") or "",
            "source_name":"YouTube:search","source_url":"https://www.youtube.com","topic":query,
            "ups":int(st.get("viewCount") or 0), "num_comments":int(st.get("commentCount") or 0),
        })
    return out

# ----- 키워드/그룹(간단 TF)
def tokenize(s:str)->list[str]:
    s=(s or "").lower()
    s=re.sub(r"&[a-z0-9#]+;"," ",s)
    s=re.sub(r"[^가-힣a-z0-9\- ]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s.split(" ") if s else []

def build_groups(items:list[dict], max_groups=MAX_GROUPS, min_count=MIN_TERM_COUNT)->list[dict]:
    cnt={}
    for it in items:
        toks=tokenize(strip_html((it.get("title","")+" "+it.get("description",""))))
        w=2.0 if str(it.get("source_name","")).startswith(("reddit","YouTube")) else 1.0
        for t in toks:
            if len(t.replace(" ",""))<2: continue
            cnt[t]=cnt.get(t,0)+w
    terms=sorted([(k,v) for k,v in cnt.items() if v>=min_count], key=lambda x:x[1], reverse=True)[:120]
    groups=[]
    used=set()
    for t,_ in terms:
        if len(groups)>=max_groups: break
        if t in used: continue
        groups.append({"groupName":t,"keywords":[t]}); used.add(t)
    if not groups:
        groups=[{"groupName":"트렌드","keywords":["트렌드"]}]
    return groups

# ----- 네이버 데이터랩 -----
def datalab(groups:list[dict], days:int=DATALAB_DAYS)->dict:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"results":[]}
    end=NOW_KST.date(); start=(NOW_KST - timedelta(days=days)).date()
    body={"startDate":str(start),"endDate":str(end),"timeUnit":"date",
          "keywordGroups":[{"groupName":g["groupName"],"keywords":g["keywords"][:3]} for g in groups[:MAX_GROUPS_DATALAB]],
          "device":"","ages":[],"gender":""}
    headers={"X-Naver-Client-Id":NAVER_CLIENT_ID,"X-Naver-Client-Secret":NAVER_CLIENT_SECRET,"Content-Type":"application/json; charset=utf-8"}
    try:
        r=requests.post("https://openapi.naver.com/v1/datalab/search", headers=headers, data=json.dumps(body), timeout=20)
        if r.status_code!=200: return {"results":[]}
        return r.json()
    except Exception:
        return {"results":[]}

def to_metrics(dlab:dict)->list[dict]:
    out=[]
    for g in dlab.get("results",[]):
        data=g.get("data",[])
        if not data:
            out.append({"groupName":g.get("title",""),"volume":0,"growth":"알 수 없습니다"})
            continue
        first=data[0]["ratio"]; last=data[-1]["ratio"]
        try: growth=round(((last-first)/(first or 1))*100,1)
        except Exception: growth="알 수 없습니다"
        out.append({"groupName":g.get("title",""),"volume":int(last),"growth":growth})
    return out

# ----- 점수/신호/카드 -----
def score(metrics:list[dict], items:list[dict])->dict:
    vols=[m["volume"] for m in metrics if isinstance(m.get("volume"),(int,float))]
    grs=[m["growth"] for m in metrics if isinstance(m.get("growth"),(int,float))]
    v=sum(vols)/len(vols) if vols else 0
    g=sum(grs)/len(grs) if grs else 0
    trend=min(100, int(v*0.5 + g*1.0))
    market=min(100, len(items)*2)
    competition_invert=max(0, 100 - min(90, (len(items)/5)*10))
    return {
        "trend":trend,"market":market,
        "competition_invert":competition_invert,
        "feasibility":50,"monetization":50,"regulatory_invert":50,
        "overall": int(0.35*trend+0.25*market+0.15*competition_invert+0.25*50)
    }

def community(items:list[dict], metrics:list[dict])->dict:
    red=[it for it in items if str(it.get("source_name","")).startswith("reddit")]
    yts=[it for it in items if str(it.get("source_name","")).startswith("YouTube")]
    return {
        "reddit":{"posts":len(red),"upvotes":sum(int(it.get("ups") or 0) for it in red),"comments":sum(int(it.get("num_comments") or 0) for it in red)},
        "naver":{"groups":len(metrics),"vol":sum(m.get("volume",0) for m in metrics)},
        "youtube":{"videos":len(yts),"views":sum(int(it.get("ups") or 0) for it in yts)}
    }

def rule_card(top_kw:str, growth_txt:str, items:list[dict])->dict:
    top_items=sorted(items, key=lambda it:(it.get("ups",0)*2+it.get("num_comments",0)), reverse=True)[:3]
    return {
        "title": f"{top_kw} 인사이트 카드",
        "tagline": f"{top_kw}: 최근 {growth_txt}",
        "sections":{
            "problem": f"‘{top_kw}’에 대한 최신 근거 정리 니즈가 큼.",
            "solution": f"뉴스/커뮤니티/유튜브 근거를 한 장의 카드로 요약.",
            "target_user": "초기 창업가, PM/마케터",
            "gtm": "매일 1장 무료 발행 → 이메일 구독 → 주간 리포트",
            "why_now": f"최근 관심도 변화({growth_txt})와 창업 도구 확산."
        },
        "evidence":[{"title":it.get("title"),"url":it.get("url")} for it in top_items if it.get("url")]
    }

def write_card(items:list[dict], metrics:list[dict], groups:list[dict])->dict:
    # GPT는 옵션. 실패하면 규칙 기반.
    top_kw=metrics[0]["groupName"] if metrics else (groups[0]["groupName"] if groups else "신규")
    gr=metrics[0].get("growth"); growth_txt = (f"{gr}%" if isinstance(gr,(int,float)) else "증감률 불명")
    if not OPENAI_API_KEY:
        return rule_card(top_kw, growth_txt, items)
    try:
        from openai import OpenAI
        client=OpenAI(api_key=OPENAI_API_KEY)
        payload={"kw":top_kw,"growth":growth_txt,"articles":[{"t":it.get("title"),"u":it.get("url")} for it in items[:20]]}
        sys=("너는 한국어 리서처다. 아래 데이터만 근거로 한 장짜리 아이디어 카드를 JSON으로 반환하라. "
             "스키마: {title,tagline,sections:{problem,solution,target_user,gtm,why_now},evidence:[{title,url}]} "
             "근거가 없으면 '알 수 없습니다' 라고 적어라.")
        msg="DATA:\n"+json.dumps(payload,ensure_ascii=False)
        resp=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}],
            temperature=0.4, response_format={"type":"json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return rule_card(top_kw, growth_txt, items)

# ----- 프론트 스키마로 직렬화(이전 버전 유지) -----
def mkid(prefix:str, n:int)->str: return f"{prefix}_{n:04d}"

def to_frontend(items:list[dict], dlab:dict, metrics:list[dict], sigs:dict, card:dict):
    # rawitems.json
    raw=[]
    for i,it in enumerate(items,1):
        raw.append({
            "raw_id": mkid("raw",i),
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
            "fetched_at": NOW_KST.isoformat()
        })
    url2id={r["url"]:r["raw_id"] for r in raw if r.get("url")}

    # trends.json (series 옵션)
    series_map={}
    for g in dlab.get("results",[]):
        title=g.get("title") or ""; data=g.get("data") or []
        series_map[title] = [{"date":p.get("period"), "value":p.get("ratio")} for p in data if "period" in p]

    trends=[]
    for i,m in enumerate(metrics,1):
        kw=m.get("groupName") or ""
        vol=int(m.get("volume") or 0)
        gr=m.get("growth") if isinstance(m.get("growth"),(int,float)) else 0.0
        evid=[url2id.get(it.get("url")) for it in items if kw.lower() in (it.get("title","")+it.get("description","")).lower()]
        evid=[e for e in evid if e]
        trend_score=min(100, int(vol*1.0 + gr*0.8 + len(evid)*3))
        trends.append({
            "trend_id": mkid("tr",i),
            "keyword": kw,
            "trend_score": trend_score,
            "volume": vol,
            "growth_percent": round(gr/100.0,4),   # 내부 0~1 (예전 프론트가 *100해서 표시)
            "region": "KR",
            "timespan": f"{DATALAB_DAYS}d",
            "evidence_rawitems": evid[:3],
            "updated_at": NOW_KST.isoformat(),
            "series": series_map.get(kw, [])       # 없으면 프론트가 단일값 표시
        })

    # signals.json (예전 구조 유지 + youtube 추가)
    signals={
        "reddit": {"posts":sigs.get("reddit",{}).get("posts",0),"upvotes":sigs.get("reddit",{}).get("upvotes",0),"comments":sigs.get("reddit",{}).get("comments",0)},
        "naver":  {"groups":sigs.get("naver",{}).get("groups",0),"vol":sigs.get("naver",{}).get("vol",0)},
        "youtube":{"videos":sigs.get("youtube",{}).get("videos",0),"views":sigs.get("youtube",{}).get("views",0)},
        "updated_at": NOW_KST.isoformat()
    }

    # ideas.json (한 건)
    idea={
        "idea_id": f"idea_{NOW_KST.strftime('%Y_%m_%d')}_001",
        "title_ko": card.get("title") or "아이디어 생성 실패",
        "one_liner": card.get("tagline") or "확실하지 않음",
        "problem": card.get("sections",{}).get("problem","알 수 없습니다"),
        "solution": card.get("sections",{}).get("solution","알 수 없습니다"),
        "target_user": card.get("sections",{}).get("target_user","알 수 없습니다"),
        "why_now": card.get("sections",{}).get("why_now","알 수 없습니다"),
        "biz_model": "알 수 없습니다",
        "gtm_tactics": card.get("sections",{}).get("gtm","알 수 없습니다"),
        "validation_steps": "확실하지 않음",
        "tags": [t.get("keyword") for t in trends[:4]],
        "score_breakdown": {},  # 프론트가 overall만 쓰는 경우가 많아 비워둠(깨지지 않음)
        "score_total": 0,
        "trend_link": [t["trend_id"] for t in trends[:3]],
        "is_today": True
    }
    return raw, trends, [idea], signals

# ----- main -----
def main():
    # 1) 수집
    items=[]
    if ENABLE_GNEWS:
        for t in TOPICS:
            items.extend(fetch_gnews(t)); time.sleep(0.2)
    if ENABLE_REDDIT:
        for s in REDDIT_SUBS:
            items.extend(fetch_reddit(s)); time.sleep(0.2)
    if ENABLE_YT and YOUTUBE_API_KEY:
        items.extend(fetch_yt_trending())

    # 중복 제거
    seen=set(); uniq=[]
    for it in items:
        u=(it.get("url") or "").strip().lower().rstrip("/")
        if not u or u in seen: continue
        seen.add(u); uniq.append(it)
    items=uniq

    # 2) 키워드 → DataLab
    groups = build_groups(items)
    dlab   = datalab(groups)
    metrics= to_metrics(dlab)

    # YouTube 검색(대표 키워드 몇 개만)
    if ENABLE_YT and YOUTUBE_API_KEY and groups:
        for g in groups[:3]:
            items.extend(fetch_yt_search(g["groupName"], n=MAX_YT_SEARCH)); time.sleep(0.2)
        # 중복 재제거
        seen=set(); uniq=[]
        for it in items:
            u=(it.get("url") or "").strip().lower().rstrip("/")
            if not u or u in seen: continue
            seen.add(u); uniq.append(it)
        items=uniq

    # 3) 점수/커뮤니티/카드
    sc  = score(metrics, items)
    sig = community(items, metrics)
    card= write_card(items, metrics, groups)

    # 4) 프론트 JSON
    raw, trn, ideas, sigs = to_frontend(items, dlab, metrics, sig, card)

    c1=dump(RAW_PATH, raw)
    c2=dump(TRN_PATH, trn)
    c3=dump(IDEA_PATH, ideas)
    c4=dump(SIG_PATH, sigs)

    print(f"[{NOW_KST.isoformat()}] items={len(items)} changed={any([c1,c2,c3,c4])} (secrets: NAVER={'ok' if NAVER_CLIENT_ID else 'no'}, OPENAI={'ok' if OPENAI_API_KEY else 'no'}, YT={'ok' if YOUTUBE_API_KEY else 'no'})")

if __name__=="__main__":
    main()
