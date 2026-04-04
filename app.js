// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = '';  // same-origin; change to 'http://localhost:5000' if needed

// ─── DATABASE (localStorage) ──────────────────────────────────────────────────
// Schema per entry:
//   { id, techniques, tagEmbeddings, problemText, addedAt }
//   tagEmbeddings: { "bitmask DP": [float, ...], "0-1 BFS": [...], ... }
const DB_KEY = 'cp_problems_v4';

function loadDB() {
  try {
    // Migrate from v3 → v4 if needed
    const legacy = localStorage.getItem('cp_problems_v3');
    if (legacy && !localStorage.getItem(DB_KEY)) {
      localStorage.setItem(DB_KEY, legacy);
    }
    const data = JSON.parse(localStorage.getItem(DB_KEY)) || [];
    // Migrate old entries that predate constraint / solution fields
    return data.map(p => ({
      timeComplex:  '',
      spaceComplex: '',
      special:      'None',
      feasibility:  'practical',
      solution:     null,
      ...p,
    }));
  }
  catch { return []; }
}
function saveDB(data) {
  localStorage.setItem(DB_KEY, JSON.stringify(data));
}

let db = loadDB();

// ─── DOM READY ────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('threshold');

  // Restore saved threshold
  const savedThreshold = localStorage.getItem('cp_threshold');
  if (savedThreshold !== null) {
    slider.value = savedThreshold;
  }
  document.getElementById('thresholdVal').textContent =
    parseFloat(slider.value).toFixed(2);

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    document.getElementById('thresholdVal').textContent = val;
    localStorage.setItem('cp_threshold', e.target.value);
    updateLinks();
  });
  renderList();
  redrawGraph();
});

// ─── FORCE PARAMETERS ────────────────────────────────────────────────────────
function getSpringLen()    { return 200; }
function getRepelRadius()  { return 240; }
function getSpringStr()    { return 0.3; }
function getNoEdgeLen()    { return 400; }
function getNoEdgeStr()    { return 0.002; }

// Short-range repulsion: pushes nodes apart only within repelRadius, falls off linearly.
// Unconnected distant nodes feel nothing — no global drift.
function forceShortRepel() {
  let nodes, radius = 120, strength = 1, _tick = 0;
  function force() {
    if (++_tick % 3 !== 0) return;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        if (dist >= radius) continue;
        const f = strength * (1 - dist / radius) / dist;
        a.vx -= dx * f; a.vy -= dy * f;
        b.vx += dx * f; b.vy += dy * f;
      }
    }
  }
  force.initialize = n => { nodes = n; };
  force.radius     = r => { radius = r; return force; };
  force.strength   = s => { strength = s; return force; };
  return force;
}

// Non-edge spring: for every pair NOT connected by an edge, apply a spring
// toward `restLen`. Repels when too close, attracts when too far — keeps
// unconnected nodes at a comfortable, adjustable spread.
function forceNonEdgeSpring() {
  let nodes, edgeSet = new Set(), restLen = 280, strength = 0.008, _tick = 0;
  function force() {
    if (++_tick % 3 !== 0) return;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const key = a.id < b.id ? `${a.id}|${b.id}` : `${b.id}|${a.id}`;
        if (edgeSet.has(key)) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = strength * (dist - restLen) / dist;
        a.vx += dx * f; a.vy += dy * f;
        b.vx -= dx * f; b.vy -= dy * f;
      }
    }
  }
  force.initialize = n => { nodes = n; };
  force.edgeSet    = s => { edgeSet = s; return force; };
  force.restLen    = r => { restLen = r; return force; };
  force.strength   = s => { strength = s; return force; };
  return force;
}


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
  return { feasibility: data.feasibility || 'practical', techniques: data.techniques, summary: data.summary, timeComplex: data.timeComplex, spaceComplex: data.spaceComplex, special: data.special };
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
 * Body:  { problem: string, mode: "buff" | "nerf", requirements?: string }
 * Reply: { result: string }
 */
async function modifyProblem(problem, mode, requirements = '') {
  const res  = await fetch(`${API_BASE}/api/modify`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      problem:      problem.problemText,
      summary:      problem.summary      || '',
      techniques:   problem.techniques   || '',
      timeComplex:  problem.timeComplex  || '',
      spaceComplex: problem.spaceComplex || '',
      special:      problem.special      || 'None',
      feasibility:  problem.feasibility  || 'practical',
      mode,
      requirements,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data.result;
}

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
    const { feasibility, techniques, summary, timeComplex, spaceComplex, special } = await analyzeProblem(text);
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
    const entry = { id, techniques, tagEmbeddings, summary, feasibility: feasibility || 'practical', timeComplex: timeComplex || '', spaceComplex: spaceComplex || '', special: special || 'None', solution: null, problemText: summary || text, addedAt: new Date().toISOString(), derivedFromId: derivedFromId || null };
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
  delete nodePositions[id];
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
let simulation     = null;
let zoomBehavior   = null;
let svgRoot        = null;
let nodePositions  = {};   // id → { x, y } — persists across redraws

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

// Live DOM references so updateLinks() can reach them without a full rebuild
let _simLinkEl    = null;
let _derivedLinkEl = null;
let _simNodes     = null;   // the node array currently bound to the simulation

// ── updateLinks: recompute edges and nudge simulation — no SVG teardown ──────
function updateLinks() {
  if (!simulation || !_simNodes) return;

  const { simLinks, derivedLinks } = buildGraphData();

  // Update edge DOM elements in-place
  const gSim = d3.select('#graph-svg g g:nth-child(1)');
  const gDer = d3.select('#graph-svg g g:nth-child(2)');

  _simLinkEl = gSim.selectAll('line').data(simLinks, d => `${d.source}-${d.target}`)
    .join('line')
      .attr('class',          'link')
      .attr('stroke-width',   d => 1 + d.sim * 5)
      .attr('stroke-opacity', d => 0.2 + d.sim * 0.6);

  _derivedLinkEl = gDer.selectAll('line').data(derivedLinks, d => `${d.source}-${d.target}`)
    .join('line')
      .attr('class', 'link link-derived')
      .attr('stroke-width',   2)
      .attr('stroke-opacity', 0.75)
      .attr('stroke-dasharray', '5,3');

  const allLinks = [...simLinks, ...derivedLinks];
  simulation.force('link').links(allLinks);
  simulation.force('link').strength(getSpringStr());

  // Refresh edge set for non-edge spring after link topology changes
  const newEdgeSet = new Set(allLinks.map(l => {
    const a = typeof l.source === 'object' ? l.source.id : l.source;
    const b = typeof l.target === 'object' ? l.target.id : l.target;
    return a < b ? `${a}|${b}` : `${b}|${a}`;
  }));
  simulation.force('nonEdge').edgeSet(newEdgeSet);

  // Tiny nudge so edges settle without throwing nodes around
  simulation.alpha(0.08).restart();
}

function redrawGraph() {
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  document.getElementById('emptyState').style.display =
    db.length === 0 ? 'flex' : 'none';
  if (db.length === 0) { simulation = null; _simNodes = null; return; }

  const { width, height } =
    document.querySelector('.graph-panel').getBoundingClientRect();
  svg.attr('viewBox', `0 0 ${width} ${height}`);

  const { nodes, links, simLinks, derivedLinks } = buildGraphData();

  // Seed positions from cache so nodes don't explode on redraw
  for (const n of nodes) {
    if (nodePositions[n.id]) {
      n.x = nodePositions[n.id].x;
      n.y = nodePositions[n.id].y;
    }
  }
  _simNodes = nodes;

  // ── Zoom ────────────────────────────────────────────────────────────────────
  const g = svg.append('g');
  zoomBehavior = d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoomBehavior);
  svgRoot = svg;

  // ── Force simulation ────────────────────────────────────────────────────────
  if (simulation) simulation.stop();
  const repel = forceShortRepel().radius(getRepelRadius()).strength(1);

  // Build edge set for non-edge spring (all linked pairs, sim + derived)
  const edgeSet = new Set(links.map(l => {
    const a = typeof l.source === 'object' ? l.source.id : l.source;
    const b = typeof l.target === 'object' ? l.target.id : l.target;
    return a < b ? `${a}|${b}` : `${b}|${a}`;
  }));
  const nonEdge = forceNonEdgeSpring()
    .edgeSet(edgeSet)
    .restLen(getNoEdgeLen())
    .strength(getNoEdgeStr());

  simulation = d3.forceSimulation(nodes)
    .alpha(0.3)
    .alphaMin(0)
    .alphaDecay(0.02)
    .velocityDecay(0.4)
    .force('link',      d3.forceLink(links).id(d => d.id)
                           .distance(d => {
                             const len = getSpringLen();
                             return d.derived ? len * 0.6 : len * (1 - d.sim * 0.6);
                           })
                           .strength(getSpringStr()))
    .force('repel',     repel)
    .force('nonEdge',   nonEdge)
    .force('center',    d3.forceCenter(width / 2, height / 2).strength(0.02))
    .force('collision', d3.forceCollide(22).strength(0.8));

  // ── Similarity links ─────────────────────────────────────────────────────────
  _simLinkEl = g.append('g')
    .selectAll('line').data(simLinks).join('line')
      .attr('class',          'link')
      .attr('stroke-width',   d => 1 + d.sim * 5)
      .attr('stroke-opacity', d => 0.2 + d.sim * 0.6);

  // ── Derived (buff/nerf) links ────────────────────────────────────────────────
  _derivedLinkEl = g.append('g')
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
      highlightNode(d.id);
    });

  // Numeric label
  node.append('text').text((d, i) => i + 1).attr('dy', 0);

  // ── Tick ────────────────────────────────────────────────────────────────────
  simulation.on('tick', () => {
    // Persist positions so redraws can seed from them
    for (const n of nodes) nodePositions[n.id] = { x: n.x, y: n.y };

    _simLinkEl
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    _derivedLinkEl
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

  // Complexity metadata — remove any previously injected block first
  document.querySelector('.drawer-constraints')?.remove();
  const feasibility = problem.feasibility || 'practical';
  const constraintsHtml = `
    <div class="drawer-constraints">
      <div class="constraint-row"><span class="constraint-label">Feasibility</span><span class="constraint-value feasibility-badge feasibility-${feasibility}">${feasibility}</span></div>
      <div class="constraint-row"><span class="constraint-label">Time complexity</span><span class="constraint-value">${renderMath(problem.timeComplex || 'N/A')}</span></div>
      <div class="constraint-row"><span class="constraint-label">Space complexity</span><span class="constraint-value">${renderMath(problem.spaceComplex || 'N/A')}</span></div>
      <div class="constraint-row"><span class="constraint-label">Additional constraints</span><span class="constraint-value">${renderMath(problem.special && problem.special !== 'None' ? problem.special : 'None')}</span></div>
    </div>`;
  document.querySelector('.drawer-tags-row').insertAdjacentHTML('afterend', constraintsHtml);

  // Summary — always present since it's fetched at add-time
  const bodyEl = document.getElementById('drawerBody');
  bodyEl.innerHTML = problem.summary
    ? renderMarkdownSummary(problem.summary)
    : `<span style="color:var(--muted);font-size:0.72rem">No summary available.</span>`;

  document.getElementById('drawerActions').innerHTML = `
    <button class="drawer-action-btn buff-btn" onclick="triggerModify(${id}, 'buff')">Buff</button>
    <button class="drawer-action-btn nerf-btn" onclick="triggerModify(${id}, 'nerf')">Nerf</button>
    <button class="drawer-action-btn remix-btn" onclick="triggerModify(${id}, 'remix')">Remix</button>
    <button class="drawer-action-btn solution-btn" id="solutionBtn" onclick="triggerSolution(${id})">Solution</button>
  `;

  // Solution section — show cached solution or a placeholder
  const existing = document.getElementById('drawerSolutionSection');
  if (existing) existing.remove();
  const solutionHtml = problem.solution
    ? `<div class="drawer-solution-section" id="drawerSolutionSection">
         <div class="drawer-solution-label">Solution</div>
         <div class="drawer-solution-body">${renderMarkdownSummary(problem.solution)}</div>
       </div>`
    : `<div class="drawer-solution-section drawer-solution-empty" id="drawerSolutionSection">
         <span>No solution yet — click <strong>💡 Solution</strong> to generate one.</span>
       </div>`;
  document.getElementById('drawerBody').insertAdjacentHTML('afterend', solutionHtml);

  document.getElementById('problemDrawer').classList.add('open');
  document.getElementById('drawerBackdrop').classList.add('open');

  highlightedId = id;
  renderList();
  d3.selectAll('.node').classed('highlighted', d => d.id === highlightedId);
}

/** Render $...$ and $$...$$ math using KaTeX, fallback to plain text.
 *  Non-math segments are HTML-escaped so raw text is safe to inject into innerHTML. */
function renderMath(s) {
  if (typeof katex === 'undefined') return escapeHtml(s).replace(/\$([^$]+)\$/g, '$1');

  // Split on math delimiters, alternating: text, math, text, math, ...
  // Pattern captures both $$...$$ and $...$
  const parts = s.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g);
  return parts.map((part, i) => {
    if (i % 2 === 0) {
      // Plain text — escape HTML
      return escapeHtml(part);
    }
    // Math span
    const isDisplay = part.startsWith('$$');
    const expr = isDisplay ? part.slice(2, -2).trim() : part.slice(1, -1).trim();
    try {
      return katex.renderToString(expr, { displayMode: isDisplay, throwOnError: false });
    } catch {
      return escapeHtml(part);
    }
  }).join('');
}

/** Strip $...$ delimiters for plain-text contexts (exports, labels). */
function stripDollar(s) {
  return s.replace(/\$([^$]+)\$/g, '$1');
}

/** Markdown → HTML renderer: headings, bullets, numbered lists, bold, code, paragraphs */
function renderMarkdownSummary(md) {
  if (!md) return '';
  function escBase(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function inlineFormat(s) {
    return renderMath(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g,   '<em>$1</em>')
      .replace(/`(.+?)`/g,     '<code class="md-code">$1</code>');
  }
  const lines  = md.split(/\r?\n/);
  const out    = [];
  let listType = null;
  let listBuf  = [];
  function flushList() {
    if (!listType) return;
    out.push(`<${listType} class="md-list">${listBuf.join('')}</${listType}>`);
    listType = null;
    listBuf  = [];
  }
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushList(); continue; }
    const hMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (hMatch) {
      flushList();
      const level = Math.min(hMatch[1].length + 2, 6);
      out.push(`<h${level} class="md-h">${inlineFormat(hMatch[2])}</h${level}>`);
      continue;
    }
    const ulMatch = line.match(/^[\-\*\+]\s+(.*)/);
    if (ulMatch) {
      if (listType === 'ol') flushList();
      listType = 'ul';
      listBuf.push(`<li>${inlineFormat(ulMatch[1])}</li>`);
      continue;
    }
    const olMatch = line.match(/^\d+\.\s+(.*)/);
    if (olMatch) {
      if (listType === 'ul') flushList();
      listType = 'ol';
      listBuf.push(`<li>${inlineFormat(olMatch[1])}</li>`);
      continue;
    }
    if (/^[-*_]{3,}$/.test(line.trim())) { flushList(); out.push('<hr class="md-hr">'); continue; }
    flushList();
    out.push(`<p class="summary-para">${inlineFormat(line)}</p>`);
  }
  flushList();
  return out.join('\n');
}

function closeDrawer() {
  document.getElementById('problemDrawer').classList.remove('open');
  document.getElementById('drawerBackdrop').classList.remove('open');
  document.getElementById('drawerSolutionSection')?.remove();
}

// ─── BUFF / NERF FLOW ─────────────────────────────────────────────────────────
function triggerModify(id, mode) {
  openModifyInputModal(id, mode);
}

function openModifyInputModal(id, mode) {
  const modal   = document.getElementById('modifyModal');
  const title   = document.getElementById('modifyModalTitle');
  const body    = document.getElementById('modifyModalBody');
  const actions = document.getElementById('modifyModalActions');

  const modeLabel = mode === 'buff' ? 'Buff Problem' : mode === 'nerf' ? 'Nerf Problem' : 'Remix Problem';
  const accent    = mode === 'buff' ? 'var(--accent)' : mode === 'nerf' ? 'var(--accent3)' : '#ffcc66';
  const modeVerb  = mode === 'remix' ? 'remixed' : mode + 'ed';

  title.textContent = modeLabel;
  title.style.color = accent;

  body.innerHTML = `
    <div class="modify-input-hint">
      Optionally describe any specific requirements for the ${modeVerb} problem.<br>
      Leave blank to let the AI decide freely.
    </div>
    <textarea id="modifyRequirementsInput" class="modify-requirements-textarea"
      placeholder="e.g. Make it require a segment tree. Keep N under 10^5. Add a query system…"></textarea>`;

  actions.innerHTML = `
    <button class="btn btn-secondary" onclick="closeModifyModal()">Cancel</button>
    <button class="btn btn-primary" id="modifyGenerateBtn" onclick="executeModify(${id}, '${mode}')">Generate</button>`;

  modal.classList.add('open');

  // Focus the textarea after transition
  setTimeout(() => document.getElementById('modifyRequirementsInput')?.focus(), 280);
}

async function executeModify(id, mode) {
  const problem = db.find(p => p.id === id);
  if (!problem) return;

  const requirements = (document.getElementById('modifyRequirementsInput')?.value || '').trim();

  const modal   = document.getElementById('modifyModal');
  const title   = document.getElementById('modifyModalTitle');
  const body    = document.getElementById('modifyModalBody');
  const actions = document.getElementById('modifyModalActions');

  const modeLabel = mode === 'buff' ? 'Buff Problem' : mode === 'nerf' ? 'Nerf Problem' : 'Remix Problem';
  const accent    = mode === 'buff' ? 'var(--accent)' : mode === 'nerf' ? 'var(--accent3)' : '#ffcc66';

  title.textContent = `${modeLabel} — Generating…`;
  title.style.color = accent;
  body.innerHTML    = `<div class="modify-generating"><span class="spinner"></span><span>Asking the AI…</span></div>`;
  actions.innerHTML = '';

  try {
    const result = await modifyProblem(problem, mode, requirements);
    if (result.trim().toLowerCase().startsWith("sorry i can't do this task")) {
      openModifyModal({ failed: true, mode });
    } else {
      openModifyModal({ failed: false, mode, result, originalId: id });
    }
  } catch (e) {
    openModifyModal({ failed: true, mode, errorMsg: e.message });
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

  const modeLabel = mode === 'buff' ? '⬆ Buffed' : mode === 'nerf' ? '⬇ Nerfed' : '⟳ Remixed';

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

// ─── SOLUTION FLOW ────────────────────────────────────────────────────────────

/**
 * POST /api/solution
 * Body:  { problem: string, techniques: string }
 * Reply: { solution: string }
 */
async function solveProblem(problemText, techniques, timeComplex, spaceComplex, special, feasibility) {
  const res  = await fetch(`${API_BASE}/api/solution`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ problem: problemText, techniques, timeComplex, spaceComplex, special, feasibility }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data.solution;
}

async function triggerSolution(id) {
  const problem = db.find(p => p.id === id);
  if (!problem) return;

  const btn     = document.getElementById('solutionBtn');
  const allBtns = document.querySelectorAll('.drawer-action-btn');
  allBtns.forEach(b => b.disabled = true);
  btn.innerHTML = `<span class="spinner"></span>Solving…`;

  // Update placeholder while loading
  const section = document.getElementById('drawerSolutionSection');
  if (section) {
    section.className = 'drawer-solution-section drawer-solution-loading';
    section.innerHTML = `<span class="spinner"></span><span style="color:var(--muted);font-size:0.72rem">Generating solution…</span>`;
  }

  try {
    const solution = await solveProblem(problem.problemText, problem.techniques, problem.timeComplex, problem.spaceComplex, problem.special, problem.feasibility);

    // Cache in DB
    problem.solution = solution;
    saveDB(db);

    // Render in drawer
    if (section) {
      section.className = 'drawer-solution-section';
      section.innerHTML = `
        <div class="drawer-solution-label">Solution</div>
        <div class="drawer-solution-body">${renderMarkdownSummary(solution)}</div>`;
    }
  } catch (e) {
    if (section) {
      section.className = 'drawer-solution-section drawer-solution-error';
      section.innerHTML = `<span style="color:var(--accent2);font-size:0.72rem">✗ ${escapeHtml(e.message)}</span>`;
    }
  } finally {
    allBtns.forEach(b => b.disabled = false);
    btn.innerHTML = 'Solution';
  }
}
let showDerivedEdges = true;

function toggleDerivedEdges() {
  showDerivedEdges = !showDerivedEdges;
  const btn = document.getElementById('derivedToggleBtn');
  btn.classList.toggle('active', showDerivedEdges);
  updateLinks();
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
  document.getElementById('bulkInput').style.display = 'block';
  document.getElementById('bulkInput').value = '';
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

  // Close the modal and show progress in the left panel
  document.getElementById('bulkModal').classList.remove('open');
  document.getElementById('panelLeft').querySelector('.panel-section').style.display = 'none';
  const progressPanel = document.getElementById('bulkProgressPanel');
  progressPanel.style.display = 'block';

  const log      = document.getElementById('bulkLog');
  const bar      = document.getElementById('bulkProgressBar');
  const label    = document.getElementById('bulkProgressLabel');
  log.innerHTML = '';
  bar.style.width = '0%';
  label.textContent = '';
  label.style.color = '';

  let succeeded = 0;
  let failed    = 0;

  function appendLog(text, type) {
    const line = document.createElement('div');
    line.className = `bulk-log-line ${type}`;
    line.textContent = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  const MAX_RETRIES = 3;
  const RETRY_BASE_DELAY = 2000; // ms, doubles each attempt

  async function withRetry(fn, label) {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        return await fn();
      } catch (e) {
        const isRetryable = /demand|quota|rate|limit|overload|unavailable|503|429/i.test(e.message);
        if (!isRetryable || attempt === MAX_RETRIES) throw e;
        const delay = RETRY_BASE_DELAY * Math.pow(2, attempt - 1);
        appendLog(`⟳ ${label} — retrying in ${delay / 1000}s (attempt ${attempt}/${MAX_RETRIES}): ${e.message}`, 'warn');
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  for (let i = 0; i < problems.length; i++) {
    const pct = Math.round(((i) / problems.length) * 100);
    bar.style.width = pct + '%';
    label.textContent = `Processing ${i + 1} / ${problems.length}…`;

    const snippet = problems[i].slice(0, 60).replace(/\n/g, ' ') + (problems[i].length > 60 ? '…' : '');

    try {
      // 1. Analyze with retry
      const { feasibility, techniques, summary, timeComplex, spaceComplex, special } =
        await withRetry(() => analyzeProblem(problems[i]), `[${i + 1}] analyze`);
      if (!techniques) throw new Error('No techniques returned');

      // 2. Embed tags with retry
      const tags = techniques.split(',').map(t => t.trim()).filter(Boolean);
      let tagEmbeddings = {};
      try {
        tagEmbeddings = await withRetry(() => embedTags(tags), `[${i + 1}] embed`);
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
        feasibility:  feasibility  || 'practical',
        timeComplex:  timeComplex  || '',
        spaceComplex: spaceComplex || '',
        special:      special      || 'None',
        solution:     null,
        problemText: problems[i],
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

  // Add dismiss button inside the progress panel
  const dismissBtn = document.createElement('button');
  dismissBtn.className = 'btn btn-secondary';
  dismissBtn.style.cssText = 'margin-top:0.8rem;width:100%;font-size:0.65rem';
  dismissBtn.textContent = 'Dismiss';
  dismissBtn.onclick = () => {
    document.getElementById('bulkProgressPanel').style.display = 'none';
    document.getElementById('panelLeft').querySelector('.panel-section').style.display = '';
  };
  document.getElementById('bulkProgressPanel').appendChild(dismissBtn);
}


// ─── EXPORT AS ZIP ────────────────────────────────────────────────────────────

// ─── SHARED EXPORT HELPER ─────────────────────────────────────────────────────

/** Build the markdown content for a single problem entry. */
function buildProblemMarkdown(p, adjacency) {
  const tags    = p.techniques.split(',').map(t => t.trim()).filter(Boolean);
  const tagLine = tags.map(t => `\`${t}\``).join(', ');
  const lines   = [
    tagLine,
    `Feasibility: ${p.feasibility || 'practical'}`,
    `Time complexity: ${p.timeComplex  || 'N/A'}`,
    `Space complexity: ${p.spaceComplex || 'N/A'}`,
    `Additional constraints: ${p.special && p.special !== 'None' ? p.special : 'None'}`,
    p.problemText  || '_(no problem text stored)_',
  ];
  if (p.solution) {
    lines.push('', '## Solution', '', p.solution);
  }
  const links = [...adjacency.get(p.id)].map(id => `[[${id}]]`).join('  ');
  if (links) lines.push('', '---', '', links);
  return lines.join('\n');
}

function buildAdjacency() {
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
  return adjacency;
}

async function exportAllProblems() {
  if (db.length === 0) { alert('No problems to export.'); return; }

  const btns = document.querySelectorAll('.export-btn');
  btns.forEach(b => b.disabled = true);

  try {
    const zip       = new JSZip();
    const adjacency = buildAdjacency();

    for (const p of db) {
      zip.file(`problems/${p.id}.md`, buildProblemMarkdown(p, adjacency));
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
    btns.forEach(b => b.disabled = false);
  }
}

// ─── EXPORT AS OBSIDIAN CANVAS ────────────────────────────────────────────────

function exportCanvas() {
  if (db.length === 0) { alert('No problems to export.'); return; }

  const btns = document.querySelectorAll('.export-btn');
  btns.forEach(b => b.disabled = true);

  try {
    const NODE_W = 200;
    const NODE_H = 300;
    const GAP    = 100;
    const STEP_X = NODE_W + GAP;
    const STEP_Y = NODE_H + GAP;

    // ── Spiral grid positions ─────────────────────────────────────────────────
    function spiralPositions(n) {
      const pos = [];
      if (n === 0) return pos;
      pos.push([0, 0]);
      let ring = 1;
      while (pos.length < n) {
        for (let c = -ring; c <= ring  && pos.length < n; c++) pos.push([c, -ring]);
        for (let r = -ring+1; r <= ring  && pos.length < n; r++) pos.push([ring, r]);
        for (let c = ring-1; c >= -ring  && pos.length < n; c--) pos.push([c, ring]);
        for (let r = ring-1; r >= -ring+1 && pos.length < n; r--) pos.push([-ring, r]);
        ring++;
      }
      return pos;
    }

    const positions = spiralPositions(db.length);

    function randomHex(len) {
      return Array.from({ length: len }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    }

    // id → { canvasId, x, y }
    const nodeMap = new Map();
    db.forEach((p, i) => {
      const [col, row] = positions[i];
      const x = col * STEP_X - NODE_W / 2;
      const y = row * STEP_Y - NODE_H / 2;
      nodeMap.set(p.id, { canvasId: randomHex(16), x, y });
    });

    // ── Best fromSide/toSide ──────────────────────────────────────────────────
    const SIDES = ['top', 'bottom', 'left', 'right'];

    function sideAnchor(x, y, side) {
      if (side === 'top')    return { x: x + NODE_W / 2, y };
      if (side === 'bottom') return { x: x + NODE_W / 2, y: y + NODE_H };
      if (side === 'left')   return { x,                  y: y + NODE_H / 2 };
      /* right */            return { x: x + NODE_W,      y: y + NODE_H / 2 };
    }

    function bestSides(aId, bId) {
      const a = nodeMap.get(aId);
      const b = nodeMap.get(bId);
      let best = Infinity, fromSide = 'right', toSide = 'left';
      for (const fs of SIDES) for (const ts of SIDES) {
        const fa = sideAnchor(a.x, a.y, fs);
        const tb = sideAnchor(b.x, b.y, ts);
        const d  = Math.hypot(fa.x - tb.x, fa.y - tb.y);
        if (d < best) { best = d; fromSide = fs; toSide = ts; }
      }
      return { fromSide, toSide };
    }

    // ── Adjacency ─────────────────────────────────────────────────────────────
    const adjacency = buildAdjacency();

    // ── Canvas nodes ──────────────────────────────────────────────────────────
    const canvasNodes = db.map(p => {
      const { canvasId, x, y } = nodeMap.get(p.id);
      return { id: canvasId, x, y, width: NODE_W, height: NODE_H,
               type: 'file', file: `problems/${p.id}.md` };
    });

    // ── Canvas edges (two directed edges per pair) ────────────────────────────
    const canvasEdges = [];
    const seen = new Set();

    for (const p of db) {
      for (const neighborId of adjacency.get(p.id)) {
        const pairKey = [p.id, neighborId].sort().join('-');
        if (seen.has(pairKey)) continue;
        seen.add(pairKey);

        const { fromSide, toSide }   = bestSides(p.id, neighborId);
        const { fromSide: fs2, toSide: ts2 } = bestSides(neighborId, p.id);
        const nA = nodeMap.get(p.id);
        const nB = nodeMap.get(neighborId);

        canvasEdges.push({ id: randomHex(16), fromNode: nA.canvasId, fromSide,  toNode: nB.canvasId, toSide  });
        canvasEdges.push({ id: randomHex(16), fromNode: nB.canvasId, fromSide: fs2, toNode: nA.canvasId, toSide: ts2 });
      }
    }

    // ── Build zip: problems/*.md + graph.canvas ───────────────────────────────
    const zip = new JSZip();

    for (const p of db) {
      zip.file(`problems/${p.id}.md`, buildProblemMarkdown(p, adjacency));
    }

    const canvas = { nodes: canvasNodes, edges: canvasEdges };
    zip.file('graph.canvas', JSON.stringify(canvas, null, '\t'));

    zip.generateAsync({ type: 'blob' }).then(blob => {
      const url = URL.createObjectURL(blob);
      const a   = document.createElement('a');
      a.href    = url;
      a.download = `cp_graph_${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

  } catch (e) {
    alert('Canvas export failed: ' + e.message);
  } finally {
    btns.forEach(b => b.disabled = false);
  }
}