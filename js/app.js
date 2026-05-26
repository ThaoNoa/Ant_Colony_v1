/**
 * app.js — SmartRoute v3 WebSocket Client
 *
 * Thay thế toàn bộ logic AI (aco.js, qlearning.js, ddpg_agent.js, ...)
 * Chỉ còn:
 *   - WebSocket connection tới Python backend
 *   - Nhận JSON mỗi generation → render Canvas + cập nhật UI
 *   - Gửi lệnh user (button clicks) lên backend
 */

import { Renderer }     from './renderer.js';
import { ChartManager } from './chart_manager.js';

// =====================================================================
// CONSTANTS
// =====================================================================
const WS_URL = `ws://${location.host}/ws`;

const STATE_FEATURE_LABELS = [
  'Cost norm', 'Improve rate', 'Stuck norm',
  'Avg phero', 'Std phero', 'Blocked ratio',
  'α norm', 'ρ norm', 'Gen progress', 'Entropy',
];

// =====================================================================
// DOM ELEMENTS
// =====================================================================
const mainCanvas     = document.getElementById('main-canvas');
const btnStartPause  = document.getElementById('btn-start-pause');
const btnBlockTool   = document.getElementById('btn-block-tool');
const btnClearTraffic= document.getElementById('btn-clear-traffic');
const btnReset       = document.getElementById('btn-reset');
const btnModeSwitch  = document.getElementById('btn-mode-switch');
const btnTrainFast   = document.getElementById('btn-train-fast');
const btnExportModel = document.getElementById('btn-export-model');
const btnImportModel = document.getElementById('btn-import-model');
const btnRunTestSuite= document.getElementById('btn-run-test-suite');
const fileImportPt   = document.getElementById('file-import-pt');

// Shared UI
const elGenNum   = document.getElementById('gen-number');
const elBestDist = document.getElementById('best-dist');
const elAlpha    = document.getElementById('param-alpha');
const elRho      = document.getElementById('param-rho');
const elEpsilon  = document.getElementById('param-epsilon');
const elAlgoLabel= document.getElementById('algo-label');

// Q-Learning panel
const qlPanel       = document.getElementById('ql-panel');
const elStateLabel  = document.getElementById('state-label');
const elStateBadge  = document.getElementById('state-badge');
const elAction      = document.getElementById('current-action');
const elReward      = document.getElementById('last-reward');
const elQTable      = document.getElementById('qtable-display');

// DDPG panel
const ddpgPanel     = document.getElementById('ddpg-panel');
const elGaugeAlpha  = document.getElementById('gauge-alpha');
const elGaugeRho    = document.getElementById('gauge-rho');
const elValAlpha    = document.getElementById('val-alpha');
const elValRho      = document.getElementById('val-rho');
const elCriticLoss  = document.getElementById('critic-loss');
const elActorLoss   = document.getElementById('actor-loss');
const elBufferFill  = document.getElementById('buffer-fill');
const elOuNoise     = document.getElementById('ou-noise');
const elDdpgReward  = document.getElementById('ddpg-reward');
const elStateVec    = document.getElementById('state-vec-display');

// Test Suite Modal
const testModal    = document.getElementById('test-modal');
const btnCloseModal= document.getElementById('btn-close-modal');
const testTableBody= document.getElementById('test-table-body');
let testCompareChart = null;

// =====================================================================
// APP STATE
// =====================================================================
let renderer = null;
let chartMgr = null;
let ws       = null;

let isRunning   = false;
let isBlockTool = false;
let mode        = 'ql';

// Virtual graph object (populated from server data, fed to renderer.js)
let vGraph = null;
let bestPath = null;
let bestCost = Infinity;
let generation = 0;

// =====================================================================
// VIRTUAL GRAPH — bridge between server JSON and renderer.js
// =====================================================================
function createVirtualGraph(data) {
  if (!data) return vGraph; // Keep old
  // Map Python snake_case → JS camelCase for renderer.js compatibility
  const nodes = (data.nodes || []).map(n => ({
    ...n,
    isDepot: n.is_depot ?? n.isDepot ?? false,
  }));
  return {
    nodes,
    pheromones:   data.pheromones || [],
    blockedEdges: new Set(data.blocked_edges || []),
    size:         data.size || nodes.length,
    isEdgeBlocked(i, j) {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      return this.blockedEdges.has(key);
    },
    distances: data.distances || [],
    costs:     data.costs     || [],
  };
}

// =====================================================================
// WEBSOCKET
// =====================================================================
function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[WS] Connected to Python backend');
    showToast('🔌 Đã kết nối server!');
  };

  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    handleServerMessage(msg);
  };

  ws.onclose = () => {
    showToast('⚠️ Mất kết nối server — thử kết nối lại sau 2s...');
    setTimeout(connectWS, 2000);
  };

  ws.onerror = (err) => {
    console.error('[WS] Error:', err);
  };
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

// =====================================================================
// SERVER MESSAGE HANDLER
// =====================================================================
function handleServerMessage(msg) {
  switch (msg.type) {

    case 'init':
      vGraph     = createVirtualGraph(msg.graph);
      mode       = msg.mode || 'ql';
      generation = 0;
      bestPath   = null;
      bestCost   = Infinity;
      applyModeUI(mode);
      renderFrame();
      break;

    case 'gen_update':
      generation = msg.generation ?? generation;
      bestPath   = msg.best_path  ?? null;
      bestCost   = msg.best_cost  ?? Infinity;
      if (msg.graph) vGraph = createVirtualGraph(msg.graph);

      // Update chart
      const dist = (bestCost != null && bestCost < 1e9) ? bestCost : 0;
      const reward = (msg.ql?.reward ?? msg.ddpg?.reward ?? 0);
      if (chartMgr) chartMgr.addDataPoint(generation, dist, reward);

      // Update UI
      if (msg.mode === 'ql' && msg.ql) updateUIQL(msg.ql);
      else if (msg.mode === 'ddpg' && msg.ddpg) updateUIDDPG(msg.ddpg);

      renderFrame();
      break;

    case 'mode_switch':
      mode = msg.mode || 'ql';
      applyModeUI(mode);
      if (chartMgr) chartMgr.reset();
      generation = 0;
      bestPath   = null;
      bestCost   = Infinity;
      if (elGenNum)   elGenNum.textContent  = '0';
      if (elBestDist) elBestDist.textContent = '—';
      break;

    case 'toast':
      showToast(msg.message || '');
      break;

    case 'train_start':
      if (btnTrainFast) {
        btnTrainFast.disabled    = true;
        btnTrainFast.textContent = '⏳ Đang train...';
      }
      break;

    case 'train_progress':
      if (elGenNum)   elGenNum.textContent  = msg.gen ?? '—';
      if (elBestDist) {
        elBestDist.textContent = msg.best_cost != null
          ? msg.best_cost.toFixed(1) : '—';
      }
      break;

    case 'train_done':
      if (btnTrainFast) {
        btnTrainFast.disabled    = false;
        btnTrainFast.textContent = '⚡ Train DDPG (Fast)';
      }
      break;

    case 'test_progress':
      if (btnRunTestSuite) {
        btnRunTestSuite.textContent = `Đang đánh giá... ${Math.round(msg.pct ?? 0)}%`;
      }
      break;

    case 'test_results':
      if (btnRunTestSuite) {
        btnRunTestSuite.disabled    = false;
        btnRunTestSuite.textContent = '📊 Run Test Suite';
      }
      showTestResults(msg.results || []);
      break;

    case 'model_export': {
      // Download .pt file
      const bytes  = atob(msg.data);
      const ab     = new ArrayBuffer(bytes.length);
      const view   = new Uint8Array(ab);
      for (let i = 0; i < bytes.length; i++) view[i] = bytes.charCodeAt(i);
      const blob   = new Blob([ab], { type: 'application/octet-stream' });
      const url    = URL.createObjectURL(blob);
      const a      = document.createElement('a');
      a.href       = url;
      a.download   = msg.filename || 'smartroute_ddpg.pt';
      a.click();
      URL.revokeObjectURL(url);
      break;
    }
  }
}

// =====================================================================
// RENDER
// =====================================================================
function renderFrame() {
  if (!renderer || !vGraph || vGraph.size === 0) return;
  renderer.render(vGraph, bestPath, { generation, bestCost: bestCost ?? 0 });
}

// =====================================================================
// UI UPDATE — Q-Learning
// =====================================================================
function updateUIQL(ql) {
  const STATE_CLASSES = { 0: 'state-ok', 1: 'state-stuck', 2: 'state-alert' };

  if (elStateLabel) elStateLabel.textContent = ql.state_label ?? 'BÌNH THƯỜNG';
  if (elStateBadge) {
    elStateBadge.className = `state-badge ${STATE_CLASSES[ql.state] ?? 'state-ok'}`;
  }
  if (elAlpha)   elAlpha.textContent   = (ql.alpha   ?? 1).toFixed(2);
  if (elRho)     elRho.textContent     = (ql.rho     ?? 0.1).toFixed(2);
  if (elEpsilon) elEpsilon.textContent = (ql.epsilon ?? 0.2).toFixed(3);
  if (elGenNum)  elGenNum.textContent  = generation;

  updateBestDistUI();

  if (elAction) elAction.textContent = ql.action_label ?? '—';
  if (elReward) {
    const r = ql.reward ?? 0;
    elReward.textContent = r >= 0 ? `+${r}` : `${r}`;
    elReward.className   = 'value ' + (r > 0 ? 'reward-pos' : r < 0 ? 'reward-neg' : '');
  }

  renderQTable(ql);
}

function renderQTable(ql) {
  if (!elQTable || !ql.q_table) return;
  const stateNames  = ['S0', 'S1', 'S2'];
  const actionNames = ['A0', 'A1', 'A2'];

  let html = '<table class="qtable"><thead><tr><th></th>';
  for (const a of actionNames) html += `<th>${a}</th>`;
  html += '</tr></thead><tbody>';

  for (let s = 0; s < 3; s++) {
    const isCurrent = s === ql.state;
    html += `<tr class="${isCurrent ? 'row-active' : ''}"><td class="qtable-state">${stateNames[s]}</td>`;
    const row = ql.q_table[s];
    const maxQ = Math.max(...row);
    for (let a = 0; a < 3; a++) {
      const q      = row[a];
      const isBest = Math.abs(q - maxQ) < 0.001 && isCurrent;
      const intens = Math.min(1, Math.max(0, q / 20));
      html += `<td class="qtable-cell ${isBest ? 'q-best' : ''}" style="background:rgba(74,158,255,${intens*0.5})">${q.toFixed(1)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  elQTable.innerHTML = html;
}

// =====================================================================
// UI UPDATE — DDPG
// =====================================================================
function updateUIDDPG(ddpg) {
  const ALPHA_MIN = 0.3, ALPHA_MAX = 2.5;
  const RHO_MIN   = 0.02, RHO_MAX  = 0.9;

  const alpha = ddpg.alpha ?? 1.0;
  const rho   = ddpg.rho   ?? 0.1;

  // Gauges
  const alphaPct = ((alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 100).toFixed(1);
  const rhoPct   = ((rho   - RHO_MIN)   / (RHO_MAX   - RHO_MIN)   * 100).toFixed(1);
  if (elGaugeAlpha) elGaugeAlpha.style.width = `${alphaPct}%`;
  if (elValAlpha)   elValAlpha.textContent   = alpha.toFixed(3);
  if (elGaugeRho)   elGaugeRho.style.width   = `${rhoPct}%`;
  if (elValRho)     elValRho.textContent     = rho.toFixed(3);

  // Training stats
  if (elCriticLoss) elCriticLoss.textContent = ddpg.critic_loss != null ? ddpg.critic_loss.toFixed(4) : '—';
  if (elActorLoss)  elActorLoss.textContent  = ddpg.actor_loss  != null ? ddpg.actor_loss.toFixed(4)  : '—';
  if (elBufferFill) {
    const bs = ddpg.buffer_size ?? 0;
    elBufferFill.textContent = `${bs}/5000`;
    elBufferFill.className   = bs >= 200 ? 'value reward-pos' : 'value';
  }
  if (elOuNoise)    elOuNoise.textContent = (ddpg.noise_level ?? 0.3).toFixed(3);
  if (elDdpgReward) {
    const r = ddpg.reward ?? 0;
    elDdpgReward.textContent = r >= 0 ? `+${r}` : `${r}`;
    elDdpgReward.className   = 'value ' + (r > 0 ? 'reward-pos' : r < 0 ? 'reward-neg' : '');
  }

  // Shared
  if (elAlpha)   elAlpha.textContent   = alpha.toFixed(3);
  if (elRho)     elRho.textContent     = rho.toFixed(3);
  if (elEpsilon) elEpsilon.textContent = (ddpg.noise_level ?? 0.3).toFixed(3);
  if (elGenNum)  elGenNum.textContent  = generation;

  updateBestDistUI();

  if (elStateVec && ddpg.state_vec) renderStateVector(ddpg.state_vec);
}

function renderStateVector(sv) {
  if (!elStateVec) return;
  let html = '';
  for (let i = 0; i < sv.length; i++) {
    const v   = sv[i];
    const pct = (Math.min(1, Math.max(0, v)) * 100).toFixed(1);
    const col = v > 0.7 ? '#ff6b6b' : v > 0.4 ? '#ffd700' : '#4a9eff';
    html += `<div class="sv-cell">
      <div class="sv-label">${STATE_FEATURE_LABELS[i]}</div>
      <div class="sv-bar-track"><div class="sv-bar-fill" style="width:${pct}%;background:${col}"></div></div>
      <div class="sv-val">${v.toFixed(2)}</div>
    </div>`;
  }
  elStateVec.innerHTML = html;
}

function updateBestDistUI() {
  if (elBestDist) {
    if (bestCost != null && bestCost < 1e9) {
      elBestDist.textContent = bestCost.toFixed(1);
    } else {
      elBestDist.textContent = '—';
    }
  }
}

// =====================================================================
// MODE UI
// =====================================================================
function applyModeUI(newMode) {
  mode = newMode;
  if (newMode === 'ddpg') {
    if (qlPanel)   qlPanel.classList.add('panel-hidden');
    if (ddpgPanel) ddpgPanel.classList.remove('panel-hidden');
    if (elAlgoLabel) elAlgoLabel.textContent = 'ACO + DDPG';
    btnModeSwitch.classList.remove('mode-ql');
    btnModeSwitch.classList.add('mode-ddpg');
    btnModeSwitch.querySelector('.mode-label').textContent = 'DDPG';
    btnModeSwitch.querySelector('.mode-arrow').textContent = '→ Q-Learning';
  } else {
    if (qlPanel)   qlPanel.classList.remove('panel-hidden');
    if (ddpgPanel) ddpgPanel.classList.add('panel-hidden');
    if (elAlgoLabel) elAlgoLabel.textContent = 'ACO + Q-Learning';
    btnModeSwitch.classList.remove('mode-ddpg');
    btnModeSwitch.classList.add('mode-ql');
    btnModeSwitch.querySelector('.mode-label').textContent = 'Q-Learning';
    btnModeSwitch.querySelector('.mode-arrow').textContent = '→ DDPG';
  }
}

// =====================================================================
// TEST RESULTS MODAL
// =====================================================================
function showTestResults(results) {
  testModal.classList.remove('hidden');
  testTableBody.innerHTML = '';

  const labels = [], qlData = [], ddpgData = [];

  for (const r of results) {
    labels.push(r.id);
    const qlErr   = ((r.ql.best_cost   - r.optimal) / r.optimal * 100);
    const ddpgErr = ((r.ddpg.best_cost - r.optimal) / r.optimal * 100);
    qlData.push(qlErr);
    ddpgData.push(ddpgErr);

    const qlErrStr   = qlErr   < 0.1 ? 'Optimal' : `+${qlErr.toFixed(2)}%`;
    const ddpgErrStr = ddpgErr < 0.1 ? 'Optimal' : `+${ddpgErr.toFixed(2)}%`;

    testTableBody.insertAdjacentHTML('beforeend', `
      <tr>
        <td>${r.name}</td>
        <td class="td-optimal">${r.optimal.toFixed(1)}</td>
        <td>${r.ql.best_cost.toFixed(1)}</td>
        <td class="${qlErr < 0.1 ? 'td-perfect' : 'td-error'}">${qlErrStr}</td>
        <td>${r.ddpg.best_cost.toFixed(1)}</td>
        <td class="${ddpgErr < 0.1 ? 'td-perfect' : 'td-error'}">${ddpgErrStr}</td>
      </tr>`);
  }

  const ctx = document.getElementById('test-compare-chart').getContext('2d');
  if (testCompareChart) testCompareChart.destroy();
  testCompareChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'QL-ACO Error %',  data: qlData,   backgroundColor: 'rgba(74,158,255,0.7)' },
        { label: 'DDPG Error %',    data: ddpgData,  backgroundColor: 'rgba(167,139,250,0.7)' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, title: { display: true, text: 'Error (%) vs Optimal' } } },
    },
  });
}

// =====================================================================
// CANVAS INTERACTION
// =====================================================================
function resizeCanvas() {
  const container = document.getElementById('canvas-container');
  if (!container) return;
  mainCanvas.width  = container.clientWidth;
  mainCanvas.height = container.clientHeight;
}

mainCanvas.addEventListener('click', (e) => {
  if (!isBlockTool || !vGraph || vGraph.size === 0) return;
  const rect = mainCanvas.getBoundingClientRect();
  const mx   = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
  const my   = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);

  const edge = renderer.findNearestEdge(vGraph, mx, my, 18);
  if (edge) {
    send({ type: 'block_edge', i: edge.i, j: edge.j });
  }
});

mainCanvas.addEventListener('mousemove', (e) => {
  if (!vGraph || !renderer) return;
  const rect = mainCanvas.getBoundingClientRect();
  const mx   = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
  const my   = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);
  if (isBlockTool) {
    const edge = renderer.findNearestEdge(vGraph, mx, my, 15);
    mainCanvas.style.cursor = edge ? 'crosshair' : 'default';
  } else {
    mainCanvas.style.cursor = 'default';
  }
});

window.addEventListener('resize', () => {
  resizeCanvas();
  renderFrame();
});

// =====================================================================
// BUTTON HANDLERS
// =====================================================================
btnStartPause.addEventListener('click', () => {
  isRunning = !isRunning;
  if (isRunning) {
    btnStartPause.textContent = '⏸ Tạm dừng';
    btnStartPause.classList.add('btn-pause');
    send({ type: 'start' });
  } else {
    btnStartPause.textContent = '▶ Bắt đầu';
    btnStartPause.classList.remove('btn-pause');
    send({ type: 'pause' });
  }
});

btnBlockTool.addEventListener('click', () => {
  isBlockTool = !isBlockTool;
  setBlockToolActive(isBlockTool);
});

if (btnClearTraffic) {
  btnClearTraffic.addEventListener('click', () => send({ type: 'clear_traffic' }));
}

btnReset.addEventListener('click', () => {
  isRunning   = false;
  isBlockTool = false;
  btnStartPause.textContent = '▶ Bắt đầu';
  btnStartPause.classList.remove('btn-pause');
  setBlockToolActive(false);
  send({ type: 'reset' });
});

btnModeSwitch.addEventListener('click', () => {
  const next = mode === 'ql' ? 'ddpg' : 'ql';
  isRunning   = false;
  btnStartPause.textContent = '▶ Bắt đầu';
  btnStartPause.classList.remove('btn-pause');
  send({ type: 'switch_mode', mode: next });
});

if (btnTrainFast) {
  btnTrainFast.addEventListener('click', () => {
    isRunning = false;
    btnStartPause.textContent = '▶ Bắt đầu';
    btnStartPause.classList.remove('btn-pause');
    send({ type: 'train_fast' });
  });
}

if (btnExportModel) {
  btnExportModel.addEventListener('click', () => send({ type: 'export_model' }));
}

if (btnImportModel) {
  btnImportModel.addEventListener('click', () => {
    if (fileImportPt) fileImportPt.click();
  });
}

if (fileImportPt) {
  fileImportPt.addEventListener('change', async (e) => {
    if (!e.target.files.length) return;
    const file   = e.target.files[0];
    const buffer = await file.arrayBuffer();
    const bytes  = new Uint8Array(buffer);
    // Convert to base64
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    const b64  = btoa(binary);
    send({ type: 'import_model', data: b64 });
    e.target.value = '';
  });
}

if (btnRunTestSuite) {
  btnRunTestSuite.addEventListener('click', () => {
    isRunning = false;
    btnStartPause.textContent = '▶ Bắt đầu';
    btnStartPause.classList.remove('btn-pause');
    btnRunTestSuite.disabled    = true;
    btnRunTestSuite.textContent = 'Đang đánh giá...';
    send({ type: 'run_test_suite' });
  });
}

if (btnCloseModal) {
  btnCloseModal.addEventListener('click', () => testModal.classList.add('hidden'));
}

// =====================================================================
// HELPERS
// =====================================================================
function setBlockToolActive(active) {
  isBlockTool = active;
  if (btnBlockTool) {
    btnBlockTool.classList.toggle('tool-active', active);
    btnBlockTool.textContent = active ? '🚫 Đang chọn cạnh...' : '🚧 Tạo Tắc Đường';
  }
  mainCanvas.style.cursor = active ? 'crosshair' : 'default';
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

// =====================================================================
// BOOTSTRAP
// =====================================================================
window.addEventListener('DOMContentLoaded', () => {
  resizeCanvas();

  renderer = new Renderer(mainCanvas);
  chartMgr = new ChartManager('line-chart');
  chartMgr.initialize();

  // Set initial mode UI
  applyModeUI('ql');

  // Connect WebSocket
  connectWS();

  // Animation loop (renders even when paused)
  function animLoop() {
    renderFrame();
    requestAnimationFrame(animLoop);
  }
  requestAnimationFrame(animLoop);
});
