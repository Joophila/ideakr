
// State & feature flags
const state = { ideas: [], trends: [], insights: [], raw: [] };
const features = { pro: false };  // future payment gate

// --- Boot ---
document.addEventListener('DOMContentLoaded', () => {
  // nav
  document.querySelectorAll('.nav-link').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      const target = a.getAttribute('href').slice(1);
      document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
      document.getElementById(target).classList.remove('hidden');
      if(target === 'home') renderHome();
      if(target === 'ideas') renderIdeas();
      if(target === 'trends') renderTrends();
      if(target === 'insights') renderInsights();
    });
  });
  // pro toggle (mock for future payment)
  document.getElementById('proToggle').addEventListener('click', () => {
    features.pro = !features.pro;
    document.getElementById('proState').textContent = features.pro ? 'ON' : 'OFF';
    renderWhyProofGapExec();
  });

  loadData();
});

async function loadData() {
  const [ideas, trends, insights, raw] = await Promise.all([
    fetch('data/ideas.json?cb='+Date.now()).then(r=>r.json()),
    fetch('data/trends.json?cb='+Date.now()).then(r=>r.json()),
    fetch('data/insights.json?cb='+Date.now()).then(r=>r.json()),
    fetch('data/rawitems.json?cb='+Date.now()).then(r=>r.json()),
  ]);
  state.ideas = ideas||[]; state.trends = trends||[]; state.insights = insights||[]; state.raw = raw||[];
  renderHome(); // initial
}

// ---------- Today detail ----------
function pickTodayIdea() {
  const todays = state.ideas.filter(i => i.is_today);
  if (todays.length) return todays[0];
  return [...state.ideas].sort((a,b)=> (b.score_total||0) - (a.score_total||0))[0];
}

function renderHome() {
  const idea = pickTodayIdea();
  if(!idea){ document.getElementById('todayTitle').textContent = '데이터 없음'; return; }

  // Head
  document.getElementById('todayDate').textContent = (idea.created_at||'').split('T')[0] || '';
  document.getElementById('todayTitle').textContent = idea.title_ko || idea.title || '(제목 없음)';
  document.getElementById('todayOneLiner').textContent = idea.one_liner || '';
  // tags
  const tagsEl = document.getElementById('todayTags');
  tagsEl.innerHTML = (idea.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('');

  // Summary
  document.getElementById('todayProblem').textContent = idea.problem || '근거가 부족합니다';
  document.getElementById('todaySolution').textContent = idea.solution || '근거가 부족합니다';
  document.getElementById('todayTarget').textContent = idea.target_user || '—';
  document.getElementById('todayGTM').textContent = idea.gtm_tactics || '—';

  // Scores
  const s = idea.score_breakdown || {};
  const kv = Object.entries(s);
  document.getElementById('scoreChips').innerHTML = kv.map(([k,v])=>scoreChip(k,v)).join('');
  document.getElementById('scoreTotal').textContent = (idea.score_total??'—');

  // Sources (linked raw ids -> urls)
  const src = (idea.sources_linked||[]).map(id => state.raw.find(r => r.raw_id===id)).filter(Boolean);
  const srcEl = document.getElementById('sourceLinks');
  srcEl.innerHTML = src.map(r=>`<a target="_blank" href="${r.url}">${r.source_platform||'src'}</a>`).join('');

  // Trends
  setupTrendBlock(idea);

  // Tabs (Why/Proof/Gap/Exec)
  setupTabs();
  renderWhyProofGapExec(idea);
}

function scoreChip(label,val){
  const nice = label.replace(/([A-Z])/g,' $1').replace(/_/g,' ').trim();
  return `<div class="score-chip"><span>${nice}</span><b>${val ?? '—'}</b></div>`;
}

function setupTrendBlock(idea){
  // gather trend ids that relate to this idea
  const ids = idea.trend_link || [];
  const trends = state.trends.filter(t => ids.includes(t.trend_id));
  const sel = document.getElementById('trendKeywordSel');
  sel.innerHTML = trends.map(t=>`<option value="${t.trend_id}">${t.keyword}</option>`).join('');
  sel.onchange = () => renderTrendChart(sel.value);
  if(trends.length){ sel.value = trends[0].trend_id; renderTrendChart(trends[0].trend_id); }
  else { document.getElementById('trendChart').innerHTML = emptyChart('관련 트렌드 없음'); document.getElementById('volVal').textContent='—'; document.getElementById('growthVal').textContent='—'; }

  // community signals list based on rawitems evidence
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
  // If a series exists, draw a tiny SVG line. Expected shape: t.series = [{date, volume}, ...]
  if (Array.isArray(t.series) && t.series.length>=2){
    const xs = t.series.map(p=>p.volume||0);
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
  return `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}"><path d="${d}" fill="none" stroke="#7aa2ff" stroke-width="2"/><rect x="0" y="${H-24}" width="${W}" height="24" fill="none"/></svg>`;
}
function emptyChart(text){ return `<div class="muted">${text}</div>`; }

function setupTabs(){
  document.querySelectorAll('.tab-btn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    });
  });
}

function renderWhyProofGapExec(idea){
  // --- Why Now ---
  document.getElementById('whyBody').textContent = (idea?.why_now || '근거가 부족합니다');
  // If pro-only extra cards (future monetization)
  const whyCards = document.getElementById('whyCards');
  whyCards.innerHTML = features.pro ? cardGrid([
    {title:'Market Timing Factors', body:'— 자료 준비 중'},
    {title:'Technological Enablers', body:'— 자료 준비 중'},
    {title:'Risk Reduction Factors', body:'— 자료 준비 중'},
    {title:'Supporting Data Points', body:'— 자료 준비 중'}
  ]) : `<div class="info-card"><h4>Pro 전용</h4><p class="info-meta">상세 Why Now 분석은 Pro에서 제공됩니다.</p></div>`;

  // --- Proof & Signals ---
  const proofs = state.insights.filter(x=> !idea || x.related_idea===idea.idea_id);
  const proofEl = document.getElementById('proofCards');
  proofEl.innerHTML = proofs.length ? proofs.map(p =>
    `<div class="info-card">
      <h4>${p.title}</h4>
      <div class="info-meta">Pain: ${p.pain_points_level} · Gap: ${p.solution_gap_level} · Revenue: ${p.revenue_potential}</div>
      <p class="para">${p.notes||''}</p>
    </div>`
  ).join('') : `<div class="info-card"><p class="info-meta">근거가 부족합니다</p></div>`;

  // --- Market Gap ---
  const gapEl = document.getElementById('gapCards');
  gapEl.innerHTML = features.pro
    ? cardGrid([
        {title:'Underserved Segments', body:'— 자료 준비 중'},
        {title:'Feature Gaps', body:'— 자료 준비 중'},
        {title:'Geographic Opportunities', body:'— 자료 준비 중'},
        {title:'Integration Opportunities', body:'— 자료 준비 중'},
        {title:'Differentiation Levers', body:'— 자료 준비 중'}
      ])
    : `<div class="info-card"><h4>Pro 전용</h4><p class="info-meta">마켓 갭 상세는 Pro에서 제공합니다.</p></div>`;

  // --- Execution Plan ---
  const execEl = document.getElementById('execCards');
  execEl.innerHTML = features.pro
    ? cardGrid([
        {title:'Core Strategy', body:`초기 오퍼: ${idea?.one_liner||''}\n가치제안: ${idea?.problem? '문제 해결 중심':'—'}`},
        {title:'Lead Generation', body:'레딧/유튜브 커뮤니티 협업 · 인플루언서 파일럿'},
        {title:'Growth Strategy', body:'파트너십 · 예측 유지보수 확장(예시)'},
        {title:'Step-by-Step', body:'베타 → 피드백 → 기능 확장 → 성과 광고'}
      ])
    : `<div class="info-card"><h4>Pro 전용</h4><p class="info-meta">실행 계획 카드는 Pro에서 제공합니다.</p></div>`;
}

function cardGrid(arr){
  return arr.map(x=>`<div class="info-card"><h4>${x.title}</h4><p class="para">${(x.body||'').replace(/\\n/g,'<br>')}</p></div>`).join('');
}

function groupBy(arr, keyFn){ return arr.reduce((m,x)=>{ const k=keyFn(x); (m[k]=m[k]||[]).push(x); return m; }, {}); }

// ---------- Legacy: lists (kept for browsing) ----------
function renderIdeas() {
  const listEl = document.getElementById('ideas-list');
  const search = document.getElementById('searchIdeas');
  const tagSel = document.getElementById('tagFilter');
  const allTags = [...new Set(state.ideas.flatMap(i=>i.tags||[]))];
  tagSel.innerHTML = '<option value=\"\">태그 전체</option>' + allTags.map(t=>`<option value=\"${t}\">${t}</option>`).join('');

  function apply() {
    const q = (search.value||'').trim().toLowerCase();
    const tag = tagSel.value||'';
    let arr = [...state.ideas].sort((a,b)=> (b.score_total||0)-(a.score_total||0));
    if(q) arr = arr.filter(i => (i.title_ko||'').toLowerCase().includes(q) || (i.one_liner||'').toLowerCase().includes(q));
    if(tag) arr = arr.filter(i => (i.tags||[]).includes(tag));
    listEl.innerHTML = arr.map(cardHtml).join('');
  }
  search.oninput = apply; tagSel.onchange = apply; apply();
}

function cardHtml(i){
  return `<div class="card">
    <h4>${i.title_ko||i.title||''}</h4>
    <div class="para">${i.one_liner||''}</div>
    <div class="badges">${(i.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('')}</div>
    <div class="para tiny">score ${i.score_total ?? '—'}</div>
  </div>`;
}

function renderTrends() {
  const el = document.getElementById('trends-list');
  el.innerHTML = state.trends.map(t => `<div class="card">
    <b>${t.keyword}</b>
    <div class="para tiny">vol ${t.volume?.toLocaleString()||'—'} · growth ${(t.growth_percent!=null)?(t.growth_percent*100).toFixed(1)+'%':'—'}</div>
  </div>`).join('');
}

function renderInsights() {
  const el = document.getElementById('insights-list');
  el.innerHTML = state.insights.map(p => `<div class="card">
    <b>${p.title}</b>
    <div class="para tiny">Pain ${p.pain_points_level} · Gap ${p.solution_gap_level} · Revenue ${p.revenue_potential}</div>
    <div class="para">${p.notes||''}</div>
  </div>`).join('');
}
