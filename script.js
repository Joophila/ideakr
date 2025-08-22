
const state = { ideas: [], trends: [], insights: [], raw: [] };

async function loadData() {
  const [ideas, trends, insights, raw] = await Promise.all([
    fetch('data/ideas.json').then(r=>r.json()),
    fetch('data/trends.json').then(r=>r.json()),
    fetch('data/insights.json').then(r=>r.json()),
    fetch('data/rawitems.json').then(r=>r.json()),
  ]);
  state.ideas = ideas;
  state.trends = trends;
  state.insights = insights;
  state.raw = raw;
  renderHome(); renderIdeas(); renderTrends(); renderInsights();
}

function renderHome() {
  const el = document.getElementById('home-cards');
  const todays = state.ideas.filter(i=>i.is_today);
  const list = (todays.length ? todays : [...state.ideas].sort((a,b)=>b.score_total-a.score_total)).slice(0,3);
  el.innerHTML = list.map(cardHtml).join(''); attachCardActions(el, list);
}

function renderIdeas() {
  const listEl = document.getElementById('ideas-list');
  const search = document.getElementById('searchIdeas');
  const tagSel = document.getElementById('tagFilter');
  const allTags = [...new Set(state.ideas.flatMap(i=>i.tags||[]))];
  tagSel.innerHTML = '<option value="">태그 전체</option>' + allTags.map(t=>`<option value="${t}">${t}</option>`).join('');

  function apply() {
    const q = (search.value||'').trim().toLowerCase();
    const tag = tagSel.value||'';
    let arr = [...state.ideas].sort((a,b)=>b.score_total-a.score_total);
    if (q) arr = arr.filter(i => (i.title_ko+i.one_liner).toLowerCase().includes(q));
    if (tag) arr = arr.filter(i => (i.tags||[]).includes(tag));
    listEl.innerHTML = arr.map(cardHtml).join(''); attachCardActions(listEl, arr);
  }
  search.oninput = apply; tagSel.onchange = apply; apply();
}

function renderTrends() {
  const el = document.getElementById('trends-list');
  const sorted = [...state.trends].sort((a,b)=> (b.growth_percent||0)-(a.growth_percent||0));
  el.innerHTML = sorted.map(t => `
    <div class="card">
      <h3>${t.keyword}</h3>
      <div class="subtitle">검색량: ${formatNum(t.volume)} / 증가율: ${(t.growth_percent*100).toFixed(1)}%</div>
      <div class="badges">
        <span class="badge">${t.region}</span>
        <span class="badge">${t.timespan}</span>
      </div>
      <button class="action" data-trend="${t.trend_id}">관련 아이디어 보기</button>
    </div>
  `).join('');
  el.querySelectorAll('button[data-trend]').forEach(btn=>{
    btn.onclick = ()=>{
      const id = btn.getAttribute('data-trend');
      const ideas = state.ideas.filter(i=> (i.trend_link||[]).includes(id));
      openModal(`<h3>관련 아이디어</h3>` + ideas.map(cardHtml).join(''));
      const body = document.getElementById('modalBody');
      attachCardActions(body, ideas);
    };
  });
}

function renderInsights() {
  const el = document.getElementById('insights-list');
  el.innerHTML = state.insights.map(mi => `
    <div class="card">
      <h3>${mi.title}</h3>
      <div class="badges">
        <span class="badge">Pain: ${mi.pain_points_level}</span>
        <span class="badge">Gap: ${mi.solution_gap_level}</span>
        <span class="badge">Revenue: ${mi.revenue_potential}</span>
      </div>
      <div class="subtitle small">커뮤니티 수: ${mi.communities_count}</div>
      ${mi.related_idea ? `<button class="action" data-idea="${mi.related_idea}">관련 아이디어</button>` : ''}
    </div>
  `).join('');
  el.querySelectorAll('button[data-idea]').forEach(btn=>{
    btn.onclick = ()=>{
      const id = btn.getAttribute('data-idea');
      const idea = state.ideas.find(i=>i.idea_id===id);
      if (idea) openIdea(idea);
    };
  });
}

function cardHtml(i) {
  return `
    <div class="card" data-idea="${i.idea_id}">
      <h3>${i.title_ko}</h3>
      <div class="subtitle">${i.one_liner||''}</div>
      <div class="badges">
        ${(i.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('')}
        <span class="badge score">Score ${i.score_total}</span>
      </div>
      <button class="action">자세히</button>
    </div>
  `;
}

function attachCardActions(scopeEl, arr) {
  scopeEl.querySelectorAll('.card .action').forEach(btn=>{
    btn.onclick = ()=>{
      const card = btn.closest('.card');
      const id = card.getAttribute('data-idea');
      const idea = arr.find(x=>x.idea_id===id) || state.ideas.find(x=>x.idea_id===id);
      if (idea) openIdea(idea);
    };
  });
}

function openIdea(i) {
  const sources = (i.sources_linked||[]).map(id=>{
    const s = state.raw.find(r=>r.raw_id===id);
    const link = s?.url || '#';
    const title = s?.title || id;
    return `<li><a href="${link}" target="_blank">${title}</a></li>`;
  }).join('');
  const scores = Object.entries(i.score_breakdown||{}).map(([k,v])=> `<span class="badge">${k} ${v}</span>`).join(' ');
  openModal(`
    <h3>${i.title_ko}</h3>
    <div class="subtitle">${i.one_liner||''}</div>
    <div class="badges">${scores}</div>
    <h4>Problem</h4><p>${i.problem||''}</p>
    <h4>Solution</h4><p>${i.solution||''}</p>
    <h4>Why Now</h4><p>${i.why_now||''}</p>
    <h4>Business Model</h4><p>${i.biz_model||''}</p>
    <h4>Go-To-Market</h4><p>${i.gtm_tactics||''}</p>
    <h4>Validation Steps</h4><p>${i.validation_steps||''}</p>
    <h4>Sources</h4><ul>${sources||'<li>없음</li>'}</ul>
  `);
}

function formatNum(n){ if(n===null||n===undefined||n==='') return '-'; return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

function openModal(html) {
  document.getElementById('modalBody').innerHTML = html;
  document.getElementById('modal').classList.remove('hidden');
}
document.getElementById('closeModal').onclick = ()=> document.getElementById('modal').classList.add('hidden');

function showView(hash) {
  document.querySelectorAll('.view').forEach(v=>v.classList.add('hidden'));
  const id = (hash || '#home').replace('#','');
  document.getElementById(id)?.classList.remove('hidden');
}
window.addEventListener('hashchange', ()=> showView(location.hash));
showView(location.hash);

loadData();
