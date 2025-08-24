// Free build (모든 정보 공개). 탭: Today / Upgrade.
const state = { ideas: [], trends: [], insights: [], raw: [] };

document.addEventListener('DOMContentLoaded', () => {
  // nav
  document.querySelectorAll('.nav-link').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      const target = a.getAttribute('href').slice(1);
      document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
      document.getElementById(target).classList.remove('hidden');
      if(target === 'home') renderHome();
    });
  });
// === UI Labels (KR) ===
const LABELS = {
  appTitle: "오늘의 아이디어",
  tabs: { why: "타이밍", proof: "증거·신호", gap: "기회 공백", exec: "실행 전략" },
  timing: {
    title: "왜 지금인가?",
    boxes: {
      market: "시장 타이밍 포인트",
      tech: "기술 동인",
      risk: "리스크 완화 근거",
      data: "보강 데이터",
    },
  },
  proof: { title: "증거·신호", reddit: "커뮤니티 신호", naver: "검색 신호" },
  gap: { title: "기회 공백" },
  exec: { title: "실행 전략" },
};

  // tabs
  document.addEventListener('click', (e)=>{
    const btn = e.target.closest('.tab-btn'); if(!btn) return;
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
  });

  loadData();
});
const $ = (id) => document.getElementById(id);

$(`appTitle`) && ($(`appTitle`).textContent = LABELS.appTitle);
$(`btnWhyNow`) && ($(`btnWhyNow`).textContent = LABELS.tabs.why);
$(`btnProof`) && ($(`btnProof`).textContent = LABELS.tabs.proof);
$(`btnMarketGap`) && ($(`btnMarketGap`).textContent = LABELS.tabs.gap);
$(`btnExecPlan`) && ($(`btnExecPlan`).textContent = LABELS.tabs.exec);

$(`whyTitle`) && ($(`whyTitle`).textContent = LABELS.timing.title);
$(`proofTitle`) && ($(`proofTitle`).textContent = LABELS.proof.title);
$(`gapTitle`) && ($(`gapTitle`).textContent = LABELS.gap.title);
$(`execTitle`) && ($(`execTitle`).textContent = LABELS.exec.title);

$(`communityTitle`) && ($(`communityTitle`).textContent = LABELS.proof.reddit);
$(`searchTitle`) && ($(`searchTitle`).textContent = LABELS.proof.naver);
const $ = (id) => document.getElementById(id);

$(`appTitle`) && ($(`appTitle`).textContent = LABELS.appTitle);
$(`btnWhyNow`) && ($(`btnWhyNow`).textContent = LABELS.tabs.why);
$(`btnProof`) && ($(`btnProof`).textContent = LABELS.tabs.proof);
$(`btnMarketGap`) && ($(`btnMarketGap`).textContent = LABELS.tabs.gap);
$(`btnExecPlan`) && ($(`btnExecPlan`).textContent = LABELS.tabs.exec);

$(`whyTitle`) && ($(`whyTitle`).textContent = LABELS.timing.title);
$(`proofTitle`) && ($(`proofTitle`).textContent = LABELS.proof.title);
$(`gapTitle`) && ($(`gapTitle`).textContent = LABELS.gap.title);
$(`execTitle`) && ($(`execTitle`).textContent = LABELS.exec.title);

$(`communityTitle`) && ($(`communityTitle`).textContent = LABELS.proof.reddit);
$(`searchTitle`) && ($(`searchTitle`).textContent = LABELS.proof.naver);

async function loadData() {
  // 1) 원본 로드
  const [ideasRaw, trendsRaw, insightsRaw, rawRaw] = await Promise.all([
    fetch('data/ideas.json?cb=' + Date.now()).then(r => r.json()),
    fetch('data/trends.json?cb=' + Date.now()).then(r => r.json()),
    fetch('data/insights.json?cb=' + Date.now()).then(r => r.json()),
    fetch('data/rawitems.json?cb=' + Date.now()).then(r => r.json()),
  ]);

  // 2) 스키마 정규화 (서버/스크립트 버전에 따라 달라진 키를 프론트 기대치에 맞춤)
  const normalize = {
    // ideas.json
    ideas(arr = []) {
      return arr.map(i => ({
        idea_id: i.idea_id || i.id || '',
        title_ko: i.title_ko || i.title || '(제목 없음)',
        one_liner: i.one_liner || i.tagline || '',
        problem: i.problem ?? '근거가 부족합니다',
        solution: i.solution ?? '근거가 부족합니다',
        target_user: i.target_user ?? '—',
        gtm_tactics: i.gtm_tactics || i.sections?.execution_plan?.core || '—',
        why_now: i.why_now || i.sections?.why_now || '',
        // 점수 키 통일
        score_breakdown: i.score_breakdown || i.scores || {},
        score_total: (i.score_total != null) ? i.score_total : (i.scores?.overall ?? 0),
        // 링크/태그
        tags: Array.isArray(i.tags) ? i.tags : [],
        trend_link: i.trend_link || i.trends || [],
        sources_linked: i.sources_linked || i.evidence?.map(e => e.raw_id).filter(Boolean) || [],
        // 날짜 플래그
        created_at: i.created_at || i.date || '',
        is_today: !!i.is_today
      }));
    },
    // trends.json
    trends(arr = []) {
      return arr.map(t => {
        // growth_percent가 0~1(비율)로 온 경우를 0~100(%)로 보여주기 위해 보정은 렌더 함수에서 이미 함.
        // 여기서는 원본을 그대로 두고, series가 없으면 빈 배열을 채움.
        const series = Array.isArray(t.series) ? t.series
                      : Array.isArray(t.data) ? t.data.map(p => ({ date: p.date || p.period, volume: p.volume ?? p.ratio ?? 0 }))
                      : [];
        return {
          trend_id: t.trend_id || t.id || '',
          keyword: t.keyword || t.term || '',
          trend_score: t.trend_score ?? 0,
          volume: t.volume ?? 0,
          growth_percent: (typeof t.growth_percent === 'number' && t.growth_percent > 1)
                          ? (t.growth_percent / 100) : (t.growth_percent ?? 0), // 내부는 0~1로 정규화
          region: t.region || 'KR',
          timespan: t.timespan || '14d',
          evidence_rawitems: Array.isArray(t.evidence_rawitems) ? t.evidence_rawitems : [],
          updated_at: t.updated_at || t.updatedAt || '',
          series
        };
      });
    },
    // insights/ raw는 그대론데, 누락 필드를 안전하게 채움
    insights(arr = []) {
      return arr.map(x => ({
        title: x.title || '',
        notes: x.notes || '',
        pain_points_level: x.pain_points_level ?? '',
        solution_gap_level: x.solution_gap_level ?? '',
        revenue_potential: x.revenue_potential ?? '',
        related_idea: x.related_idea || ''
      }));
    },
    raw(arr = []) {
      return arr.map(r => ({
        raw_id: r.raw_id || r.id || '',
        source_platform: r.source_platform || r.source || '',
        query_or_topic: r.query_or_topic || r.topic || '',
        title: r.title || '',
        content_snippet: r.content_snippet || r.snippet || '',
        url: r.url || '',
        metrics_upvotes: parseInt(r.metrics_upvotes || r.upvotes || 0, 10) || 0,
        metrics_comments: parseInt(r.metrics_comments || r.comments || 0, 10) || 0,
        search_volume: r.search_volume || '',
        language: r.language || 'ko',
        published_at: r.published_at || r.date || '',
        fetched_at: r.fetched_at || ''
      }));
    }
  };

  // 3) 상태 반영
  state.ideas    = normalize.ideas(ideasRaw);
  state.trends   = normalize.trends(trendsRaw);
  state.insights = normalize.insights(insightsRaw);
  state.raw      = normalize.raw(rawRaw);

  // 4) 최초 렌더
  renderHome();
}


function pickTodayIdea() {
  const todays = state.ideas.filter(i => i && i.is_today);
  if (todays.length) return todays[0];
  return state.ideas.sort((a,b)=>(b?.score_total||0)-(a?.score_total||0))[0];
}

function renderHome() {
  const idea = pickTodayIdea();
  if (!idea) {
    document.getElementById('todayTitle').textContent = '데이터 없음';
    document.getElementById('todayOneLiner').textContent = '데이터 파일이 비어있거나 로딩 실패';
    // 점수/그래프/커뮤니티도 모두 ‘—’로 초기화
    document.getElementById('overallScore').textContent = '—';
    document.getElementById('trendChart').innerHTML = emptyChart('데이터 없음');
    document.getElementById('communityList').innerHTML = '';
    return;
  }
  // ... (기존 렌더링)
}


  document.getElementById('todayDate').textContent = (idea.created_at||'').split('T')[0] || '';
  document.getElementById('todayTitle').textContent = idea.title_ko || idea.title || '(제목 없음)';
  document.getElementById('todayOneLiner').textContent = idea.one_liner || '';
  const tagsEl = document.getElementById('todayTags');
  tagsEl.innerHTML = (idea.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('');

  document.getElementById('todayProblem').textContent = idea.problem || '근거가 부족합니다';
  document.getElementById('todaySolution').textContent = idea.solution || '근거가 부족합니다';
  document.getElementById('todayTarget').textContent = idea.target_user || '—';
  document.getElementById('todayGTM').textContent = idea.gtm_tactics || '—';

  const s = idea.score_breakdown || {};
  const kv = Object.entries(s);
  document.getElementById('scoreChips').innerHTML = kv.map(([k,v])=>scoreChip(k,v)).join('');
  document.getElementById('scoreTotal').textContent = (idea.score_total??'—');

  const src = (idea.sources_linked||[]).map(id => state.raw.find(r => r.raw_id===id)).filter(Boolean);
  const srcEl = document.getElementById('sourceLinks');
  srcEl.innerHTML = src.map(r=>`<a target="_blank" href="${r.url}">${r.source_platform||'src'}</a>`).join('');

  setupTrendBlock(idea);
  renderWhyProofGapExec(idea);
}

function scoreChip(label,val){
  const nice = label.replace(/([A-Z])/g,' $1').replace(/_/g,' ').trim();
  return `<div class="score-chip"><span>${nice}</span><b>${val ?? '—'}</b></div>`;
}

function setupTrendBlock(idea){
  const ids = idea.trend_link || [];
  const trends = state.trends.filter(t => ids.includes(t.trend_id));
  const sel = document.getElementById('trendKeywordSel');
  sel.innerHTML = trends.map(t=>`<option value="${t.trend_id}">${t.keyword}</option>`).join('');
  sel.onchange = () => renderTrendChart(sel.value);
  if(trends.length){ sel.value = trends[0].trend_id; renderTrendChart(trends[0].trend_id); }
  else { document.getElementById('trendChart').innerHTML = emptyChart('관련 트렌드 없음'); document.getElementById('volVal').textContent='—'; document.getElementById('growthVal').textContent='—'; }

  const ev = trends.flatMap(t => t.evidence_rawitems||[]);
  const items = state.raw.filter(r => ev.includes(r.raw_id));
  const group = groupBy(items, r => r.source_platform || 'src');
  const listEl = document.getElementById('communityList');
  listEl.innerHTML = Object.entries(group).map(([k,arr])=>{
    const metrics = arr.reduce((acc,r)=>{
      const c = parseInt(r.metrics_comments||0)||0, u = parseInt(r.metrics_upvotes||0)||0;
      return { comments: acc.comments + c, upvotes: acc.upvotes + u };
    }, {comments:0, upvotes:0});
    return `<li><span>${k}</span><span class="chip">posts ${arr.length} · 👍 ${metrics.upvotes} · 💬 ${metrics.comments}</span></li>`;
  }).join('') || '<li class="muted">근거가 부족합니다</li>';
}

function renderTrendChart(trendId){
  const t = state.trends.find(x=>x.trend_id===trendId);
  if(!t){ document.getElementById('trendChart').innerHTML = emptyChart('자료 없음'); return; }
  document.getElementById('volVal').textContent = (t.volume ?? '—').toLocaleString('en-US');
  document.getElementById('growthVal').textContent = t.growth_percent!=null ? (t.growth_percent*100).toFixed(1)+'%' : '—';

  const holder = document.getElementById('trendChart');
  if (Array.isArray(t.series) && t.series.length>=2){
    const xs = t.series.map(p => (p.volume ?? p.value ?? 0));
    holder.innerHTML = sparkline(xs);
  } else {
    holder.innerHTML = emptyChart('시계열 없음 — volume만 보유');
  }
}

function sparkline(values){
  const W=820, H=200, pad=12;
  const min = Math.min(...values), max = Math.max(...values);
  const norm = v => (H-pad*2) * (1 - (v-min)/(max-min || 1)) + pad;
  const step = (W-pad*2) / Math.max(1, (values.length-1));
  let d = ''; values.forEach((v,i)=>{ const x=pad+i*step, y=norm(v); d += (i?'L':'M')+x+','+y; });
  return `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}"><path d="${d}" fill="none" stroke="#7aa2ff" stroke-width="2"/></svg>`;
}
function emptyChart(text){ return `<div class="muted">${text}</div>`; }

function renderWhyProofGapExec(idea){
  // Why Now
  document.getElementById('whyBody').textContent = (idea?.why_now || '근거가 부족합니다');
  const whyCards = [
    {title:'Market Timing Factors', body: idea?.why_market || '근거가 부족합니다'},
    {title:'Technological Enablers', body: idea?.why_tech || '근거가 부족합니다'},
    {title:'Risk Reduction Factors', body: idea?.why_risk || '근거가 부족합니다'},
    {title:'Supporting Data Points', body: idea?.why_data || '근거가 부족합니다'},
  ];
  document.getElementById('whyCards').innerHTML = cardGrid(whyCards);

  // Proof & Signals
  const proofs = state.insights.filter(x=> !idea || x.related_idea===idea.idea_id);
  document.getElementById('proofCards').innerHTML = proofs.length ? proofs.map(p =>
    `<div class="info-card">
      <h4>${p.title}</h4>
      <div class="info-meta">Pain: ${p.pain_points_level} · Gap: ${p.solution_gap_level} · Revenue: ${p.revenue_potential}</div>
      <p class="para">${p.notes||'근거가 부족합니다'}</p>
    </div>`
  ).join('') : `<div class="info-card"><p class="info-meta">근거가 부족합니다</p></div>`;

  // Market Gap
  const gapData = [
    {title:'Underserved Segments', body: idea?.gap_segments || '근거가 부족합니다'},
    {title:'Feature Gaps', body: idea?.gap_features || '근거가 부족합니다'},
    {title:'Geographic Opportunities', body: idea?.gap_geo || '근거가 부족합니다'},
    {title:'Integration Opportunities', body: idea?.gap_integrations || '근거가 부족합니다'},
    {title:'Differentiation Levers', body: idea?.gap_diff || '근거가 부족합니다'}
  ];
  document.getElementById('gapCards').innerHTML = cardGrid(gapData);

  // Execution Plan
  const execData = [
    {title:'Core Strategy', body: idea?.exec_core || '근거가 부족합니다'},
    {title:'Lead Generation Strategy', body: idea?.exec_lead || '근거가 부족합니다'},
    {title:'Growth Strategy', body: idea?.exec_growth || '근거가 부족합니다'},
    {title:'Step-by-Step Execution', body: idea?.exec_steps || '근거가 부족합니다'}
  ];
  document.getElementById('execCards').innerHTML = cardGrid(execData);
}

function cardGrid(arr){
  return arr.map(x=>`<div class="info-card"><h4>${x.title}</h4><p class="para">${(x.body||'').replace(/\n/g,'<br>')}</p></div>`).join('');
}
function groupBy(arr, keyFn){ return arr.reduce((m,x)=>{ const k=keyFn(x); (m[k]=m[k]||[]).push(x); return m; }, {}); }
