// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = '';  // same-origin; change to 'http://localhost:5000' if needed

// ─── DATABASE (localStorage) ──────────────────────────────────────────────────
// Schema per entry:
//   { id, techniques, tagEmbeddings, problemText, addedAt }
//   tagEmbeddings: { "bitmask DP": [float, ...], "0-1 BFS": [...], ... }
const DB_KEY = 'cp_problems_v3';

function loadDB() {
  try { return JSON.parse(localStorage.getItem(DB_KEY)) || []; }
  catch { return []; }
}
function saveDB(data) {
  localStorage.setItem(DB_KEY, JSON.stringify(data));
}

let db = loadDB();

// ─── DOM READY ────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('threshold').addEventListener('input', (e) => {
    document.getElementById('thresholdVal').textContent =
      parseFloat(e.target.value).toFixed(2);
    redrawGraph();
  });
  renderList();
  redrawGraph();
});

// ─── STATUS HELPER ────────────────────────────────────────────────────────────
function setStatus(msg, isError = false) {
  const el = document.getElementById('status');
  el.className = isError ? 'error' : '';
  el.innerHTML = msg;
}

// ─── API CALLS ────────────────────────────────────────────────────────────────

/**
 * POST /api/analyze
 * Body:  { problem: string }
 * Reply: { techniques: string, summary: string }
 */
async function analyzeProblem(problemText) {
  const res  = await fetch(`${API_BASE}/api/analyze`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ problem: problemText }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return { techniques: data.techniques, summary: data.summary };
}

/**
 * POST /api/embed-tags
 * Body:  { tags: string[] }
 * Reply: { embeddings: { [tag]: float[] }, warnings?: string[] }
 */
async function embedTags(tags) {
  const res  = await fetch(`${API_BASE}/api/embed-tags`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ tags }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  if (data.warnings?.length) {
    console.warn('Embedding warnings:', data.warnings);
  }
  return data.embeddings;  // { tag: float[] }
}

/**
 * POST /api/modify
 * Body:  { problem: string, mode: "buff" | "nerf" }
 * Reply: { result: string }
 */

// ─── SIMILARITY: SOFT JACCARD WITH PER-TAG EMBEDDINGS ────────────────────────
//
// For each tag in A, find its best cosine match in B (and vice versa).
// Average all those best scores:
//
//   sim(A,B) = ( Σ_a max_b cos(a,b)  +  Σ_b max_a cos(b,a) ) / (|A| + |B|)
//
// This gives partial credit for semantically related but non-identical tags,
// e.g. "bitmask DP" vs "interval DP" will score ~0.8 instead of 0.

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

function softJaccard(embsA, embsB) {
  // embsA / embsB: { tag: float[] }
  const vecsA = Object.values(embsA);
  const vecsB = Object.values(embsB);

  if (!vecsA.length || !vecsB.length) return 0;

  // For each vec in A: best match in B
  let sumAtoB = 0;
  for (const a of vecsA) {
    let best = -Infinity;
    for (const b of vecsB) best = Math.max(best, cosine(a, b));
    sumAtoB += best;
  }

  // For each vec in B: best match in A
  let sumBtoA = 0;
  for (const b of vecsB) {
    let best = -Infinity;
    for (const a of vecsA) best = Math.max(best, cosine(b, a));
    sumBtoA += best;
  }

  return (sumAtoB + sumBtoA) / (vecsA.length + vecsB.length);
}

// ─── MAIN ANALYZE FLOW ────────────────────────────────────────────────────────
async function analyze(derivedFromId = null) {
  const text = document.getElementById('problemInput').value.trim();
  if (!text) { setStatus('⚠ Paste a problem statement first.', true); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;

  try {
    // 1. Extract techniques + summary in one request
    setStatus('<span class="spinner"></span>Analyzing…');
    const { techniques, summary } = await analyzeProblem(text);
    if (!techniques) throw new Error('No techniques extracted.');

    // 2. Embed tags
    const tags = techniques.split(',').map(t => t.trim()).filter(Boolean);
    setStatus(`<span class="spinner"></span>Embedding ${tags.length} tag${tags.length !== 1 ? 's' : ''}…`);
    let tagEmbeddings = {};
    try {
      tagEmbeddings = await embedTags(tags);
    } catch (e) {
      console.warn('Tag embedding failed, similarity will be unavailable:', e.message);
    }

    // 3. Persist
    const id    = Date.now();
    const entry = { id, techniques, tagEmbeddings, summary, problemText: summary || text, addedAt: new Date().toISOString(), derivedFromId: derivedFromId || null };
    db.push(entry);
    saveDB(db);

    document.getElementById('problemInput').value = '';
    const embeddedCount = Object.keys(tagEmbeddings).length;
    const preview = techniques.length > 60 ? techniques.slice(0, 60) + '…' : techniques;
    setStatus(`✓ Added — "${preview}" (${embeddedCount}/${tags.length} tags embedded)`);

    renderList();
    redrawGraph();
  } catch (e) {
    setStatus(`✗ ${e.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

// ─── PROBLEM LIST ─────────────────────────────────────────────────────────────
let highlightedId = null;

function renderList() {
  const list = document.getElementById('problemList');
  document.getElementById('problemCount').textContent = db.length;

  if (db.length === 0) {
    list.innerHTML =
      '<div style="padding:1.5rem;color:var(--muted);font-size:0.7rem;text-align:center">No problems yet</div>';
    return;
  }

  list.innerHTML = db.slice().reverse().map(p => {
    const tags = p.techniques
      .split(',').map(t => t.trim()).filter(Boolean)
      .map(t => `<span class="technique-tag">${t}</span>`).join('');

    const embCount   = Object.keys(p.tagEmbeddings || {}).length;
    const totalTags  = p.techniques.split(',').filter(t => t.trim()).length;
    const embLabel   = embCount === totalTags
      ? `<span style="color:var(--accent);font-size:0.58rem">✓ ${embCount} embedded</span>`
      : embCount > 0
        ? `<span style="color:var(--accent3);font-size:0.58rem">~ ${embCount}/${totalTags} embedded</span>`
        : `<span style="color:var(--muted);font-size:0.58rem">no embeddings</span>`;

    const highlighted = p.id === highlightedId ? 'highlighted' : '';
    const date        = new Date(p.addedAt).toLocaleString();

    return `
      <div class="problem-item ${highlighted}" onclick="highlightNode(${p.id})" data-id="${p.id}">
        <div class="problem-id">#${p.id} · ${date} · ${embLabel}</div>
        <div class="problem-techniques">${tags}</div>
        <button class="delete-btn" onclick="deleteProblem(event, ${p.id})">✕</button>
      </div>`;
  }).join('');
}

function deleteProblem(e, id) {
  e.stopPropagation();
  db = db.filter(p => p.id !== id);
  saveDB(db);
  if (highlightedId === id) highlightedId = null;
  renderList();
  redrawGraph();
}

function highlightNode(id) {
  if (highlightedId === id) {
    // Second click on the same problem → open drawer (same as clicking the node)
    openDrawer(id);
  } else {
    highlightedId = id;
    renderList();
    d3.selectAll('.node').classed('highlighted', d => d.id === highlightedId);
  }
}

// ─── D3 FORCE GRAPH ───────────────────────────────────────────────────────────
let simulation   = null;
let zoomBehavior = null;
let svgRoot      = null;

function getThreshold() {
  return parseFloat(document.getElementById('threshold').value);
}

function buildGraphData() {
  const nodes     = db.map(p => ({ ...p }));
  const simLinks  = [];
  const threshold = getThreshold();

  for (let i = 0; i < db.length; i++) {
    for (let j = i + 1; j < db.length; j++) {
      const embsA = db[i].tagEmbeddings;
      const embsB = db[j].tagEmbeddings;

      const hasEmbeddings = embsA && embsB &&
                            Object.keys(embsA).length > 0 &&
                            Object.keys(embsB).length > 0;

      let sim;
      if (hasEmbeddings) {
        sim = softJaccard(embsA, embsB);
      } else {
        const tagsA = new Set(db[i].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
        const tagsB = new Set(db[j].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
        const intersection = [...tagsA].filter(t => tagsB.has(t)).length;
        const union        = new Set([...tagsA, ...tagsB]).size;
        sim = union > 0 ? intersection / union : 0;
      }

      if (sim >= threshold) {
        simLinks.push({ source: db[i].id, target: db[j].id, sim, derived: false });
      }
    }
  }

  // Derived (buff/nerf) edges — one per child that has a parent in db
  const derivedLinks = [];
  if (showDerivedEdges) {
    for (const p of db) {
      if (p.derivedFromId && db.find(q => q.id === p.derivedFromId)) {
        derivedLinks.push({ source: p.derivedFromId, target: p.id, sim: 1, derived: true });
      }
    }
  }

  // All links together for the force simulation
  const links = [...simLinks, ...derivedLinks];
  return { nodes, links, simLinks, derivedLinks };
}

function redrawGraph() {
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  document.getElementById('emptyState').style.display =
    db.length === 0 ? 'flex' : 'none';
  if (db.length === 0) return;

  const { width, height } =
    document.querySelector('.graph-panel').getBoundingClientRect();
  svg.attr('viewBox', `0 0 ${width} ${height}`);

  const { nodes, links, simLinks, derivedLinks } = buildGraphData();

  // ── Zoom ────────────────────────────────────────────────────────────────────
  const g = svg.append('g');
  zoomBehavior = d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoomBehavior);
  svgRoot = svg;

  // ── Force simulation ────────────────────────────────────────────────────────
  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force('link',      d3.forceLink(links).id(d => d.id)
                           .distance(d => d.derived ? 80 : 120 * (1 - d.sim) + 60))
    .force('charge',    d3.forceManyBody().strength(-200))
    .force('center',    d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(30));

  // ── Similarity links ─────────────────────────────────────────────────────────
  const simLinkEl = g.append('g')
    .selectAll('line').data(simLinks).join('line')
      .attr('class',          'link')
      .attr('stroke-width',   d => 1 + d.sim * 5)
      .attr('stroke-opacity', d => 0.2 + d.sim * 0.6);

  // ── Derived (buff/nerf) links ────────────────────────────────────────────────
  const derivedLinkEl = g.append('g')
    .selectAll('line').data(derivedLinks).join('line')
      .attr('class', 'link link-derived')
      .attr('stroke-width',   2)
      .attr('stroke-opacity', 0.75)
      .attr('stroke-dasharray', '5,3');

  // ── Nodes ───────────────────────────────────────────────────────────────────
  const node = g.append('g')
    .selectAll('g').data(nodes).join('g')
      .attr('class', d => `node${d.id === highlightedId ? ' highlighted' : ''}`)
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag',  (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end',   (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

  node.append('circle')
    .attr('r', 16)
    .on('mouseover', (event, d) => {
      const tip        = document.getElementById('tooltip');
      const embCount   = Object.keys(d.tagEmbeddings || {}).length;
      const totalTags  = d.techniques.split(',').filter(t => t.trim()).length;
      const embLine    = embCount === totalTags
        ? `<span style="color:var(--accent);font-size:0.6rem">✓ ${embCount} tags embedded</span>`
        : `<span style="color:var(--muted);font-size:0.6rem">${embCount}/${totalTags} tags embedded</span>`;

      document.getElementById('tipId').textContent = `#${d.id}`;
      document.getElementById('tipContent').innerHTML =
        d.techniques.split(',')
          .map(t => `<span class="technique-tag">${t.trim()}</span>`).join('') +
        `<br>${embLine}` +
        `<br><span style="color:var(--muted);font-size:0.6rem;margin-top:0.3rem;display:block">click to view problem</span>`;
      tip.classList.add('visible');
      moveTip(event);
    })
    .on('mousemove', moveTip)
    .on('mouseout',  () => document.getElementById('tooltip').classList.remove('visible'))
    .on('click',     (e, d) => {
      document.getElementById('tooltip').classList.remove('visible');
      openDrawer(d.id);
    });

  // Numeric label
  node.append('text').text((d, i) => i + 1).attr('dy', 0);

  // ── Tick ────────────────────────────────────────────────────────────────────
  simulation.on('tick', () => {
    simLinkEl
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    derivedLinkEl
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

// ─── TOOLTIP POSITION ────────────────────────────────────────────────────────
function moveTip(event) {
  const tip = document.getElementById('tooltip');
  tip.style.left = (event.clientX + 14) + 'px';
  tip.style.top  = (event.clientY - 10) + 'px';
}

// ─── PROBLEM DRAWER ───────────────────────────────────────────────────────────
function openDrawer(id) {
  const problem = db.find(p => p.id === id);
  if (!problem) return;

  document.getElementById('drawerProblemId').textContent = `Problem #${id}`;
  document.getElementById('drawerDate').textContent =
    new Date(problem.addedAt).toLocaleString();

  // Tags line
  document.getElementById('drawerTags').innerHTML =
    problem.techniques.split(',')
      .map(t => `<span class="technique-tag">${t.trim()}</span>`).join('');

  // Summary — always present since it's fetched at add-time
  const bodyEl = document.getElementById('drawerBody');
  bodyEl.innerHTML = problem.summary
    ? renderMarkdownSummary(problem.summary)
    : `<span style="color:var(--muted);font-size:0.72rem">No summary available.</span>`;

  document.getElementById('drawerActions').innerHTML = `
    <button class="drawer-action-btn buff-btn" onclick="triggerModify(${id}, 'buff')">
      ⬆ Buff Problem
    </button>
    <button class="drawer-action-btn nerf-btn" onclick="triggerModify(${id}, 'nerf')">
      ⬇ Nerf Problem
    </button>
  `;

  document.getElementById('problemDrawer').classList.add('open');
  document.getElementById('drawerBackdrop').classList.add('open');

  highlightedId = id;
  renderList();
  d3.selectAll('.node').classed('highlighted', d => d.id === highlightedId);
}

/** Minimal markdown → HTML renderer for the summary (bold, inline code, paragraphs) */
function renderMarkdownSummary(md) {
  return md
    .split(/\n\n+/)
    .map(block => {
      const html = block
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.+?)`/g, '<code class="md-code">$1</code>')
        .replace(/\n/g, '<br>');
      return `<p class="summary-para">${html}</p>`;
    })
    .join('');
}

function closeDrawer() {
  document.getElementById('problemDrawer').classList.remove('open');
  document.getElementById('drawerBackdrop').classList.remove('open');
}

// ─── BUFF / NERF FLOW ─────────────────────────────────────────────────────────
async function triggerModify(id, mode) {
  const problem = db.find(p => p.id === id);
  if (!problem) return;

  const label   = mode === 'buff' ? '⬆ Buff' : '⬇ Nerf';
  const btn     = document.querySelector(`.${mode}-btn`);
  const allBtns = document.querySelectorAll('.drawer-action-btn');
  allBtns.forEach(b => b.disabled = true);
  btn.innerHTML = `<span class="spinner"></span>${label}ing…`;

  try {
    const result = await modifyProblem(problem.problemText, mode);
    if (result.trim().toLowerCase().startsWith("sorry i can't do this task")) {
      openModifyModal({ failed: true, mode });
    } else {
      openModifyModal({ failed: false, mode, result, originalId: id });
    }
  } catch (e) {
    openModifyModal({ failed: true, mode, errorMsg: e.message });
  } finally {
    allBtns.forEach(b => b.disabled = false);
    btn.innerHTML = mode === 'buff' ? '⬆ Buff Problem' : '⬇ Nerf Problem';
  }
}

// ─── MODIFY RESULT MODAL ──────────────────────────────────────────────────────
let _pendingModifiedText = null;
let _pendingOriginalId   = null;

function openModifyModal({ failed, mode, result, originalId, errorMsg }) {
  const modal   = document.getElementById('modifyModal');
  const title   = document.getElementById('modifyModalTitle');
  const body    = document.getElementById('modifyModalBody');
  const actions = document.getElementById('modifyModalActions');

  const modeLabel = mode === 'buff' ? '⬆ Buffed' : '⬇ Nerfed';

  if (failed) {
    _pendingModifiedText = null;
    _pendingOriginalId   = null;
    title.textContent = `${modeLabel} Problem — Failed`;
    title.style.color = 'var(--accent2)';
    body.innerHTML = `
      <div class="modify-fail">
        <div class="modify-fail-icon">✗</div>
        <div>The AI was unable to ${mode} this problem.</div>
        ${errorMsg ? `<div class="modify-fail-detail">${errorMsg}</div>` : ''}
      </div>`;
    actions.innerHTML = `<button class="btn btn-secondary" onclick="closeModifyModal()">Close</button>`;
  } else {
    _pendingModifiedText = result;
    _pendingOriginalId   = originalId;
    title.textContent = `${modeLabel} Problem`;
    title.style.color = mode === 'buff' ? 'var(--accent)' : 'var(--accent3)';
    body.innerHTML = `<pre class="modify-result-text">${escapeHtml(result)}</pre>`;
    actions.innerHTML = `
      <button class="btn btn-secondary" onclick="closeModifyModal()">Discard</button>
      <button class="btn btn-primary" onclick="addModifiedProblem()">+ Add to Graph</button>`;
  }

  modal.classList.add('open');
}

function closeModifyModal() {
  document.getElementById('modifyModal').classList.remove('open');
  _pendingModifiedText = null;
  _pendingOriginalId   = null;
}

async function addModifiedProblem() {
  const text     = _pendingModifiedText;
  const sourceId = _pendingOriginalId;
  if (!text) return;

  closeModifyModal();
  closeDrawer();

  document.getElementById('problemInput').value = text;
  await analyze(sourceId);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function handleModifyOverlayClick(e) {
  if (e.target === document.getElementById('modifyModal')) closeModifyModal();
}

// ─── DERIVED EDGES TOGGLE ─────────────────────────────────────────────────────
let showDerivedEdges = true;

function toggleDerivedEdges() {
  showDerivedEdges = !showDerivedEdges;
  const btn = document.getElementById('derivedToggleBtn');
  btn.classList.toggle('active', showDerivedEdges);
  redrawGraph();
}

// ─── PANEL COLLAPSE ───────────────────────────────────────────────────────────
function togglePanel(side) {
  const layout    = document.getElementById('layout');
  const btn       = document.getElementById(side === 'left' ? 'collapseLeft' : 'collapseRight');
  const cls       = side === 'left' ? 'left-collapsed' : 'right-collapsed';
  const collapsed = layout.classList.toggle(cls);

  if (side === 'left') {
    btn.textContent = collapsed ? '›' : '‹';
  } else {
    btn.textContent = collapsed ? '‹' : '›';
  }

  setTimeout(redrawGraph, 320);
}

// ─── ZOOM CONTROLS ───────────────────────────────────────────────────────────
function zoomIn()    { svgRoot && svgRoot.transition().call(zoomBehavior.scaleBy, 1.4); }
function zoomOut()   { svgRoot && svgRoot.transition().call(zoomBehavior.scaleBy, 0.7); }
function resetZoom() { svgRoot && svgRoot.transition().call(zoomBehavior.transform, d3.zoomIdentity); }

window.addEventListener('resize', redrawGraph);
// ─── BULK ADD ─────────────────────────────────────────────────────────────────

let _bulkRunning = false;

function openBulkModal() {
  document.getElementById('bulkModal').classList.add('open');
  document.getElementById('bulkProgress').style.display = 'none';
  document.getElementById('bulkInput').style.display = 'block';
  document.getElementById('bulkInput').value = '';
  document.getElementById('bulkLog').innerHTML = '';
  document.getElementById('bulkProgressBar').style.width = '0%';
  document.getElementById('bulkProgressLabel').textContent = '';
  document.getElementById('bulkStartBtn').disabled = false;
  document.getElementById('bulkStartBtn').innerHTML = '▶ Start Processing';
  document.getElementById('bulkModalFooter').innerHTML = `
    <button class="btn btn-secondary" onclick="closeBulkModal()">Cancel</button>
    <button class="btn btn-primary" id="bulkStartBtn" onclick="startBulkAdd()">▶ Start Processing</button>`;
}

function closeBulkModal() {
  if (_bulkRunning) return;   // don't allow close mid-run
  document.getElementById('bulkModal').classList.remove('open');
}

function handleBulkOverlayClick(e) {
  if (e.target === document.getElementById('bulkModal')) closeBulkModal();
}

async function startBulkAdd() {
  const raw = document.getElementById('bulkInput').value;
  const problems = raw
    .split(/^---$/m)
    .map(s => s.trim())
    .filter(Boolean);

  if (problems.length === 0) {
    alert('No problems found. Separate problems with --- on its own line.');
    return;
  }

  _bulkRunning = true;

  // Swap to progress UI
  document.getElementById('bulkInput').style.display = 'none';
  document.getElementById('bulkProgress').style.display = 'block';
  document.getElementById('bulkModalFooter').innerHTML =
    `<span style="font-size:0.65rem;color:var(--muted)">Processing — please wait…</span>`;

  const log      = document.getElementById('bulkLog');
  const bar      = document.getElementById('bulkProgressBar');
  const label    = document.getElementById('bulkProgressLabel');

  let succeeded = 0;
  let failed    = 0;

  function appendLog(text, type) {
    const line = document.createElement('div');
    line.className = `bulk-log-line ${type}`;
    line.textContent = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  for (let i = 0; i < problems.length; i++) {
    const pct = Math.round(((i) / problems.length) * 100);
    bar.style.width = pct + '%';
    label.textContent = `Processing ${i + 1} / ${problems.length}…`;

    const snippet = problems[i].slice(0, 60).replace(/\n/g, ' ') + (problems[i].length > 60 ? '…' : '');

    try {
      // 1. Analyze (techniques + summary in one request)
      const { techniques, summary } = await analyzeProblem(problems[i]);
      if (!techniques) throw new Error('No techniques returned');

      // 2. Embed tags
      const tags = techniques.split(',').map(t => t.trim()).filter(Boolean);
      let tagEmbeddings = {};
      try {
        tagEmbeddings = await embedTags(tags);
      } catch (e) {
        console.warn('Embedding failed for bulk item', i + 1, e.message);
      }

      // 3. Persist
      const id    = Date.now() + i;   // ensure unique even within same ms
      const entry = {
        id,
        techniques,
        tagEmbeddings,
        summary,
        problemText: summary || problems[i],
        addedAt: new Date().toISOString(),
        derivedFromId: null,
      };
      db.push(entry);
      saveDB(db);

      succeeded++;
      appendLog(`✓ [${i + 1}] ${snippet}`, 'ok');
    } catch (e) {
      failed++;
      appendLog(`✗ [${i + 1}] ${snippet} — ${e.message}`, 'err');
    }

    // Small delay to avoid hammering the API
    if (i < problems.length - 1) await new Promise(r => setTimeout(r, 300));
  }

  bar.style.width = '100%';
  label.textContent =
    `Done — ${succeeded} added, ${failed} failed out of ${problems.length} problems.`;
  label.style.color = failed > 0 ? 'var(--accent2)' : 'var(--accent)';

  renderList();
  redrawGraph();

  _bulkRunning = false;

  document.getElementById('bulkModalFooter').innerHTML =
    `<button class="btn btn-primary" onclick="closeBulkModal()">Close</button>`;
}


// ─── EXPORT AS ZIP ────────────────────────────────────────────────────────────

async function exportAllProblems() {
  if (db.length === 0) {
    alert('No problems to export.');
    return;
  }

  const btn = document.querySelector('.export-btn');
  btn.disabled = true;
  btn.textContent = '…';

  try {
    const zip = new JSZip();
    const threshold = getThreshold();

    // Build adjacency: id → Set of linked ids (similarity edges + derived edges)
    const adjacency = new Map(db.map(p => [p.id, new Set()]));

    // Similarity edges
    for (let i = 0; i < db.length; i++) {
      for (let j = i + 1; j < db.length; j++) {
        const embsA = db[i].tagEmbeddings;
        const embsB = db[j].tagEmbeddings;
        const hasEmb = embsA && embsB &&
                       Object.keys(embsA).length > 0 &&
                       Object.keys(embsB).length > 0;
        let sim;
        if (hasEmb) {
          sim = softJaccard(embsA, embsB);
        } else {
          const tA = new Set(db[i].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
          const tB = new Set(db[j].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
          const inter = [...tA].filter(t => tB.has(t)).length;
          const union = new Set([...tA, ...tB]).size;
          sim = union > 0 ? inter / union : 0;
        }
        if (sim >= threshold) {
          adjacency.get(db[i].id).add(db[j].id);
          adjacency.get(db[j].id).add(db[i].id);
        }
      }
    }

    // Derived edges
    for (const p of db) {
      if (p.derivedFromId && adjacency.has(p.derivedFromId)) {
        adjacency.get(p.id).add(p.derivedFromId);
        adjacency.get(p.derivedFromId).add(p.id);
      }
    }

    // Build each .md file
    for (const p of db) {
      const tags = p.techniques
        .split(',').map(t => t.trim()).filter(Boolean);

      // Line 1: tags as `#tag` tokens on one line
      const tagLine = tags.map(t => `\`${t}\``).join(', ');

      // Linked problem wikilinks
      const links = [...adjacency.get(p.id)]
        .map(linkedId => `[[${linkedId}]]`)
        .join('  ');

      const parts = [
        tagLine,
        '',
        p.problemText || '_(no problem text stored)_',
      ];

      if (links) {
        parts.push('', '---', '', links);
      }

      zip.file(`${p.id}.md`, parts.join('\n'));
    }

    const blob = await zip.generateAsync({ type: 'blob' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `cp_problems_${Date.now()}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Export failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '↓ zip';
  }
}

// ─── EXPORT AS OBSIDIAN CANVAS ────────────────────────────────────────────────

function exportCanvas() {
  if (db.length === 0) {
    alert('No problems to export.');
    return;
  }

  const btns = document.querySelectorAll('.export-btn');
  btns.forEach(b => b.disabled = true);

  try {
    const NODE_W    = 200;
    const NODE_H    = 300;
    const GAP       = 100;
    const STEP_X    = NODE_W + GAP;
    const STEP_Y    = NODE_H + GAP;

    // ── Spiral grid positions (col, row) outward from (0,0) ──────────────────
    // Generates positions in a square spiral: center first, then ring by ring.
    function spiralPositions(n) {
      const pos = [];
      if (n === 0) return pos;
      pos.push([0, 0]);
      let ring = 1;
      while (pos.length < n) {
        // Top side: row = -ring, col from -ring to +ring
        for (let c = -ring; c <= ring && pos.length < n; c++)
          pos.push([c, -ring]);
        // Right side: col = +ring, row from -ring+1 to +ring
        for (let r = -ring + 1; r <= ring && pos.length < n; r++)
          pos.push([ring, r]);
        // Bottom side: row = +ring, col from +ring-1 to -ring
        for (let c = ring - 1; c >= -ring && pos.length < n; c--)
          pos.push([c, ring]);
        // Left side: col = -ring, row from +ring-1 to -ring+1
        for (let r = ring - 1; r >= -ring + 1 && pos.length < n; r--)
          pos.push([-ring, r]);
        ring++;
      }
      return pos;
    }

    const positions = spiralPositions(db.length);

    // Map problem id → canvas node id (short hex string) and pixel position
    const nodeMap = new Map(); // id → { canvasId, x, y, cx, cy }
    function randomHex(len) {
      return Array.from({ length: len }, () =>
        Math.floor(Math.random() * 16).toString(16)).join('');
    }

    db.forEach((p, i) => {
      const [col, row] = positions[i];
      const x  = col * STEP_X - NODE_W / 2;   // top-left corner
      const y  = row * STEP_Y - NODE_H / 2;
      const cx = x + NODE_W / 2;              // center
      const cy = y + NODE_H / 2;
      nodeMap.set(p.id, { canvasId: randomHex(16), x, y, cx, cy });
    });

    // ── Build adjacency (same logic as exportAllProblems) ────────────────────
    const threshold = getThreshold();
    const adjacency = new Map(db.map(p => [p.id, new Set()]));

    for (let i = 0; i < db.length; i++) {
      for (let j = i + 1; j < db.length; j++) {
        const embsA = db[i].tagEmbeddings;
        const embsB = db[j].tagEmbeddings;
        const hasEmb = embsA && embsB &&
                       Object.keys(embsA).length > 0 &&
                       Object.keys(embsB).length > 0;
        let sim;
        if (hasEmb) {
          sim = softJaccard(embsA, embsB);
        } else {
          const tA   = new Set(db[i].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
          const tB   = new Set(db[j].techniques.split(',').map(t => t.trim().toLowerCase()).filter(Boolean));
          const inter = [...tA].filter(t => tB.has(t)).length;
          const union = new Set([...tA, ...tB]).size;
          sim = union > 0 ? inter / union : 0;
        }
        if (sim >= threshold) {
          adjacency.get(db[i].id).add(db[j].id);
          adjacency.get(db[j].id).add(db[i].id);
        }
      }
    }
    for (const p of db) {
      if (p.derivedFromId && adjacency.has(p.derivedFromId)) {
        adjacency.get(p.id).add(p.derivedFromId);
        adjacency.get(p.derivedFromId).add(p.id);
      }
    }

    // ── Pick best fromSide/toSide to minimise edge length ────────────────────
    const SIDES = ['top', 'bottom', 'left', 'right'];

    function sideAnchor(nx, ny, side) {
      // Returns the pixel coordinate of a side's midpoint anchor
      if (side === 'top')    return { x: nx + NODE_W / 2, y: ny };
      if (side === 'bottom') return { x: nx + NODE_W / 2, y: ny + NODE_H };
      if (side === 'left')   return { x: nx,               y: ny + NODE_H / 2 };
      /* right */            return { x: nx + NODE_W,      y: ny + NODE_H / 2 };
    }

    function bestSides(aId, bId) {
      const a = nodeMap.get(aId);
      const b = nodeMap.get(bId);
      let best = Infinity, fromSide = 'right', toSide = 'left';
      for (const fs of SIDES) {
        for (const ts of SIDES) {
          const fa = sideAnchor(a.x, a.y, fs);
          const tb = sideAnchor(b.x, b.y, ts);
          const d  = Math.hypot(fa.x - tb.x, fa.y - tb.y);
          if (d < best) { best = d; fromSide = fs; toSide = ts; }
        }
      }
      return { fromSide, toSide };
    }

    // ── Build canvas nodes ────────────────────────────────────────────────────
    const canvasNodes = db.map(p => {
      const { canvasId, x, y } = nodeMap.get(p.id);
      return {
        id:     canvasId,
        x,
        y,
        width:  NODE_W,
        height: NODE_H,
        type:   'file',
        file:   `${p.id}.md`,
      };
    });

    // ── Build canvas edges (two directed edges per undirected pair) ───────────
    const canvasEdges = [];
    const seen = new Set();

    for (const p of db) {
      for (const neighborId of adjacency.get(p.id)) {
        const pairKey = [p.id, neighborId].sort().join('-');
        if (seen.has(pairKey)) continue;
        seen.add(pairKey);

        const { fromSide, toSide } = bestSides(p.id, neighborId);
        const { fromSide: fs2, toSide: ts2 } = bestSides(neighborId, p.id);

        const nA = nodeMap.get(p.id);
        const nB = nodeMap.get(neighborId);

        // Edge A → B
        canvasEdges.push({
          id:       randomHex(16),
          fromNode: nA.canvasId,
          fromSide,
          toNode:   nB.canvasId,
          toSide,
        });
        // Edge B → A
        canvasEdges.push({
          id:       randomHex(16),
          fromNode: nB.canvasId,
          fromSide: fs2,
          toNode:   nA.canvasId,
          toSide:   ts2,
        });
      }
    }

    // ── Serialise and download ────────────────────────────────────────────────
    const canvas = { nodes: canvasNodes, edges: canvasEdges };
    const json   = JSON.stringify(canvas, null, '\t');
    const blob   = new Blob([json], { type: 'application/json' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href       = url;
    a.download   = 'graph.canvas';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

  } catch (e) {
    alert('Canvas export failed: ' + e.message);
  } finally {
    btns.forEach(b => b.disabled = false);
  }
}