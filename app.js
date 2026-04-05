/* ═══════════════════════════════════════════════
   GradeML — Student Performance Prediction
   app.js
═══════════════════════════════════════════════ */

'use strict';

/* ── Tab routing ──────────────────────────────── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'dataset' && !chartsBuilt.dist) buildDatasetCharts();
    if (tab === 'model'   && !chartsBuilt.algo) buildModelCharts();
  });
});

const chartsBuilt = { dist: false, algo: false };


/* ── Grade helpers ────────────────────────────── */
const GRADES = ['Fail', 'Average', 'Pass', 'Distinction'];

function gradeInfo(score) {
  if (score >= 85) return { label: 'Distinction', color: '#6ee7b7', bg: 'rgba(110,231,183,0.12)' };
  if (score >= 70) return { label: 'Pass',        color: '#38bdf8', bg: 'rgba(56,189,248,0.12)'  };
  if (score >= 50) return { label: 'Average',     color: '#fbbf24', bg: 'rgba(251,191,36,0.12)'  };
  return                  { label: 'Fail',        color: '#f87171', bg: 'rgba(248,113,113,0.12)' };
}


/* ── Score computation ────────────────────────── */
function computeScore(v) {
  const s =
    Math.min((v.study / 40) * 100, 100) * 0.22
    + ((v.attend - 50) / 50) * 100       * 0.20
    + ((v.gpa - 1) / 3) * 100            * 0.22
    + v.assign                            * 0.18
    + (v.sleep >= 6 && v.sleep <= 9 ? 1 : 0.6) * 100 * 0.08
    + ((v.parent - 1) / 3) * 100         * 0.05
    + v.internet * 100                    * 0.03
    + v.tutoring * 100                    * 0.03
    - (v.absences / 20) * 100            * 0.07
    + (v.extra / 2) * 100                * 0.02
    - v.job * 5;
  return Math.min(Math.max(Math.round(s), 0), 100);
}


/* ── Derived metrics ──────────────────────────── */
function computeDerived(v) {
  return {
    study_efficiency: ((v.assign / (v.study + 1))).toFixed(2),
    sleep_optimal:    (v.sleep >= 6 && v.sleep <= 9) ? 'Yes' : 'No',
    effort_score:     (v.study * 0.4 + v.attend * 0.4 + v.assign * 0.2).toFixed(1),
    support_index:    (v.parent + v.internet * 2 + v.tutoring * 3),
  };
}


/* ── Gauge drawing ────────────────────────────── */
let gaugeAnim = null;
let gaugeTarget = 0;
let gaugeCurrent = 0;

function drawGauge(canvas, score, color) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 14, r = 96;
  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#1e2a3a';
  ctx.stroke();

  // Fill
  const pct = score / 100;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, Math.PI + pct * Math.PI);
  ctx.strokeStyle = color;
  ctx.stroke();

  // Score text
  ctx.fillStyle = color;
  ctx.font = `500 28px 'DM Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.fillText(Math.round(score), cx, cy - 14);

  ctx.fillStyle = '#6b7585';
  ctx.font = `400 11px 'DM Sans', sans-serif`;
  ctx.fillText('/ 100', cx, cy + 4);
}

function animateGauge(canvas, target, color) {
  if (gaugeAnim) cancelAnimationFrame(gaugeAnim);
  const start = gaugeCurrent;
  const dur = 500;
  const t0 = performance.now();
  function step(t) {
    const p = Math.min((t - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    gaugeCurrent = start + (target - start) * ease;
    drawGauge(canvas, gaugeCurrent, color);
    if (p < 1) gaugeAnim = requestAnimationFrame(step);
  }
  gaugeAnim = requestAnimationFrame(step);
}


/* ── Prediction update ────────────────────────── */
function getValues() {
  return {
    study:    +document.getElementById('study').value,
    attend:   +document.getElementById('attend').value,
    gpa:      +document.getElementById('gpa').value,
    assign:   +document.getElementById('assign').value,
    absences: +document.getElementById('absences').value,
    sleep:    +document.getElementById('sleep').value,
    parent:   +document.getElementById('parent').value,
    internet: +document.getElementById('internet').value,
    tutoring: +document.getElementById('tutoring').value,
    extra:    +document.getElementById('extra').value,
    job:      +document.getElementById('job').value,
  };
}

function syncOutputs(v) {
  document.getElementById('study-out').value    = v.study;
  document.getElementById('attend-out').value   = v.attend;
  document.getElementById('gpa-out').value      = v.gpa.toFixed(1);
  document.getElementById('assign-out').value   = v.assign;
  document.getElementById('absences-out').value = v.absences;
  document.getElementById('sleep-out').value    = v.sleep;
}

function update() {
  const v = getValues();
  syncOutputs(v);

  const score = computeScore(v);
  const info  = gradeInfo(score);
  const gauge = document.getElementById('gaugeChart');

  // Score
  document.getElementById('result-score').textContent = score;
  document.getElementById('result-score').style.color = info.color;

  // Grade badge
  const badge = document.getElementById('result-grade');
  badge.textContent = info.label;
  badge.style.background = info.bg;
  badge.style.color = info.color;

  // Gauge
  animateGauge(gauge, score, info.color);

  // Advice
  const advice = {
    Distinction: 'Excellent academic profile — top-tier performance predicted.',
    Pass:        'Strong profile. A bit more study time could push to Distinction.',
    Average:     'Some areas need improvement. Focus on attendance and assignments.',
    Fail:        'High risk of underperformance. Early intervention recommended.',
  };
  document.getElementById('result-advice').textContent = advice[info.label];

  // Feature impact bars
  const impacts = [
    { name: 'Previous GPA',  pct: Math.round(((v.gpa - 1) / 3) * 100),        color: '#bc8cff' },
    { name: 'Study Time',    pct: Math.round((v.study / 40) * 100),            color: '#38bdf8' },
    { name: 'Attendance',    pct: Math.round(((v.attend - 50) / 50) * 100),    color: '#6ee7b7' },
    { name: 'Assignments',   pct: v.assign,                                    color: '#fbbf24' },
    { name: 'Sleep Quality', pct: v.sleep >= 6 && v.sleep <= 9 ? 100 : 55,    color: '#79c0ff' },
    { name: 'Low Absences',  pct: Math.round((1 - v.absences / 20) * 100),    color: '#f0883e' },
  ];

  document.getElementById('impact-list').innerHTML = impacts.map(f => `
    <div class="impact-row">
      <span class="impact-name">${f.name}</span>
      <div class="impact-bar-bg">
        <div class="impact-bar" style="width:${f.pct}%; background:${f.color};"></div>
      </div>
      <span class="impact-pct">${f.pct}%</span>
    </div>
  `).join('');

  // Derived metrics
  const d = computeDerived(v);
  document.getElementById('derived-grid').innerHTML = `
    <div class="derived-item">
      <div class="derived-val">${d.study_efficiency}</div>
      <div class="derived-name">Study Efficiency</div>
    </div>
    <div class="derived-item">
      <div class="derived-val">${d.sleep_optimal}</div>
      <div class="derived-name">Sleep Optimal</div>
    </div>
    <div class="derived-item">
      <div class="derived-val">${d.effort_score}</div>
      <div class="derived-name">Effort Score</div>
    </div>
    <div class="derived-item">
      <div class="derived-val">${d.support_index}</div>
      <div class="derived-name">Support Index</div>
    </div>
  `;
}

// Bind all inputs
document.querySelectorAll('input[type="range"], select').forEach(el => {
  el.addEventListener('input', update);
});


/* ── Dataset Charts ───────────────────────────── */
function buildDatasetCharts() {
  chartsBuilt.dist = true;

  // Score distribution
  new Chart(document.getElementById('distChart'), {
    type: 'bar',
    data: {
      labels: ['0–20', '21–40', '41–50', '51–60', '61–70', '71–80', '81–90', '91–100'],
      datasets: [{
        label: 'Students',
        data: [28, 42, 94, 187, 234, 398, 215, 89],
        backgroundColor: '#38bdf8',
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#6b7585', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#6b7585', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });

  // Grade donut
  new Chart(document.getElementById('gradeChart'), {
    type: 'doughnut',
    data: {
      labels: ['Distinction', 'Pass', 'Average', 'Fail'],
      datasets: [{
        data: [214, 812, 191, 70],
        backgroundColor: ['#6ee7b7', '#38bdf8', '#fbbf24', '#f87171'],
        borderWidth: 0,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#6b7585', font: { size: 11 }, boxWidth: 12, padding: 12 },
        },
      },
    },
  });

  // Sample table
  const rows = [
    ['S001', 20, '92%', 3.8, 8, '95%', 2,  91, 'Distinction'],
    ['S002', 12, '75%', 2.6, 6, '70%', 6,  62, 'Pass'],
    ['S003',  8, '60%', 1.9, 5, '55%', 14, 44, 'Fail'],
    ['S004', 16, '85%', 3.2, 7, '88%', 3,  77, 'Pass'],
    ['S005', 25, '98%', 3.9, 8, '100%',1,  96, 'Distinction'],
    ['S006', 10, '70%', 2.2, 9, '65%', 7,  53, 'Average'],
    ['S007', 18, '88%', 3.4, 7, '90%', 4,  82, 'Pass'],
    ['S008',  5, '55%', 1.5, 4, '40%', 18, 31, 'Fail'],
  ];

  const colorOf = g => ({ Distinction:'#6ee7b7', Pass:'#38bdf8', Average:'#fbbf24', Fail:'#f87171' }[g]);
  const bgOf    = g => ({ Distinction:'rgba(110,231,183,0.12)', Pass:'rgba(56,189,248,0.12)', Average:'rgba(251,191,36,0.12)', Fail:'rgba(248,113,113,0.12)' }[g]);

  document.querySelector('#sample-table tbody').innerHTML = rows.map(r => `
    <tr>
      <td style="font-family:var(--font-mono);color:var(--muted)">${r[0]}</td>
      <td>${r[1]} hrs</td>
      <td>${r[2]}</td>
      <td>${r[3]}</td>
      <td>${r[4]} hrs</td>
      <td>${r[5]}</td>
      <td>${r[6]}</td>
      <td style="font-family:var(--font-mono);font-weight:500">${r[7]}</td>
      <td><span class="grade-pill" style="background:${bgOf(r[8])};color:${colorOf(r[8])}">${r[8]}</span></td>
    </tr>
  `).join('');
}


/* ── Model Charts ─────────────────────────────── */
function buildModelCharts() {
  chartsBuilt.algo = true;

  const algos = [
    { name: 'Logistic Reg.',    acc: 78.6 },
    { name: 'Decision Tree',    acc: 81.1 },
    { name: 'KNN',              acc: 84.3 },
    { name: 'SVM',              acc: 88.7 },
    { name: 'Gradient Boosting',acc: 91.2 },
    { name: 'Random Forest',    acc: 92.5 },
  ];

  const algoColors = algos.map((a, i) => i === algos.length - 1 ? '#6ee7b7' : '#1e3a4a');

  new Chart(document.getElementById('algoChart'), {
    type: 'bar',
    data: {
      labels: algos.map(a => a.name),
      datasets: [{
        label: 'Test Accuracy (%)',
        data: algos.map(a => a.acc),
        backgroundColor: algoColors,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } },
      },
      scales: {
        x: {
          min: 70, max: 100,
          ticks: { color: '#6b7585', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: { ticks: { color: '#a0aec0', font: { size: 11 } }, grid: { display: false } },
      },
    },
  });

  // Feature importance
  const feats = [
    { name: 'Previous GPA',   imp: 22 },
    { name: 'Study Time',     imp: 20 },
    { name: 'Attendance',     imp: 19 },
    { name: 'Assignments',    imp: 17 },
    { name: 'Absences',       imp: 9  },
    { name: 'Sleep Hours',    imp: 7  },
    { name: 'Parental Edu.',  imp: 5  },
    { name: 'Support Index',  imp: 3  },
    { name: 'Effort Score',   imp: 3  },
  ].reverse();

  new Chart(document.getElementById('featChart'), {
    type: 'bar',
    data: {
      labels: feats.map(f => f.name),
      datasets: [{
        label: 'Importance',
        data: feats.map(f => f.imp),
        backgroundColor: '#38bdf8',
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } },
      },
      scales: {
        x: {
          ticks: { color: '#6b7585', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: { ticks: { color: '#a0aec0', font: { size: 11 } }, grid: { display: false } },
      },
    },
  });

  // Confusion matrix
  const labels = ['Fail', 'Average', 'Pass', 'Distinction'];
  const matrix = [
    [198, 12, 3,  0],
    [14,  149, 8, 0],
    [6,   12, 812, 9],
    [0,   1,  7,  214],
  ];

  const cmEl = document.getElementById('cm-grid');
  let html = '<div class="cm-hdr"></div>';
  labels.forEach(l => { html += `<div class="cm-hdr">${l}</div>`; });
  labels.forEach((rowLabel, r) => {
    html += `<div class="cm-side">${rowLabel}</div>`;
    labels.forEach((_, c) => {
      const isDiag = r === c;
      html += `<div class="cm-cell ${isDiag ? 'cm-correct' : 'cm-wrong'}">${matrix[r][c]}</div>`;
    });
  });
  cmEl.innerHTML = html;
}


/* ── Init ─────────────────────────────────────── */
update();
