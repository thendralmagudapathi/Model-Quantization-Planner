/* Canvas charts. Each draw function paints a chart and returns hover(x, y) → { text, … } | null. */

const C = {
  axis: '#cbd5e1', grid: '#f1f5f9', text: '#64748b', ink: '#0f172a',
  bar: '#93c5fd', barEdge: '#60a5fa', level: 'rgba(124,58,237,.5)', clip: 'rgba(220,38,38,.08)', clipLine: 'rgba(220,38,38,.6)',
  dot: 'rgba(37,99,235,.55)', diag: '#cbd5e1', err: '#fca5a5', errEdge: '#f87171', line: '#2563eb', best: '#16a34a',
  cur: '#2563eb', point: '#64748b',
};
const FONT = '11px Inter, "Segoe UI", system-ui, sans-serif';

export function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = canvas.clientHeight || 200;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;
  return { ctx, w, h };
}

export function fmtNum(v, digits = 3) {
  if (v === 0) return '0';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(digits > 2 ? 2 : 1).replace('e', 'e').replace('-', '−');
  return String(+v.toPrecision(digits)).replace('-', '−');
}

function niceStep(span, n) {
  const raw = span / Math.max(1, n), p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const s = niceStep(hi - lo, n), out = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + s * 1e-9; v += s) out.push(Math.abs(v) < s * 1e-9 ? 0 : v);
  return out;
}

// Axes + grid; returns coordinate mappers.
function frame(ctx, w, h, o) {
  const l = 46, r = 10, t = 8, b = 30;
  const P = { l, t, r: w - r, b: h - b, w: w - l - r, h: h - t - b };
  const X = (v) => P.l + ((v - o.xlo) / (o.xhi - o.xlo || 1)) * P.w;
  const Y = (v) => P.b - ((v - o.ylo) / (o.yhi - o.ylo || 1)) * P.h;
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.text; ctx.font = FONT;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of o.yticks || ticks(o.ylo, o.yhi, 4)) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.r, y); ctx.stroke();
    ctx.fillText(o.yfmt ? o.yfmt(v) : fmtNum(v, 2), P.l - 5, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of o.xticks || ticks(o.xlo, o.xhi, Math.max(3, Math.floor(P.w / 70)))) {
    const x = Math.round(X(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke();
    ctx.fillText(o.xfmt ? o.xfmt(v) : fmtNum(v, 2), x, P.b + 5);
  }
  ctx.strokeStyle = C.axis;
  ctx.beginPath(); ctx.moveTo(P.l + 0.5, P.t); ctx.lineTo(P.l + 0.5, P.b + 0.5); ctx.lineTo(P.r, P.b + 0.5); ctx.stroke();
  ctx.fillStyle = C.text;
  if (o.xlabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xlabel, P.r, h - 1); }
  if (o.ylabel) { ctx.save(); ctx.translate(10, P.t + P.h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.ylabel, 0, 0); ctx.restore(); }
  return { X, Y, P };
}

const logc = (c) => Math.log10(1 + c);

/* ---------- A. histogram + quantization levels ---------- */
export function drawHist(canvas, d, { log = true } = {}) {
  const { ctx, w, h } = prep(canvas);
  const counts = d.counts, nb = counts.length, total = counts.reduce((a, b) => a + b, 0);
  const span = d.hi - d.lo || 1, pad = span * 0.03;
  const xlo = d.lo - pad, xhi = d.hi + pad;
  const yv = (c) => (log ? logc(c) : c);
  let ymax = 0; for (const c of counts) ymax = Math.max(ymax, yv(c));
  const yticks = log ? Array.from({ length: Math.ceil(ymax) + 1 }, (_, i) => i).filter((i) => i <= ymax + 1e-9) : undefined;
  const { X, Y, P } = frame(ctx, w, h, { xlo, xhi, ylo: 0, yhi: ymax * 1.05 || 1, yticks, yfmt: log ? (v) => (v === 0 ? '0' : '10' + sup(v)) : (v) => fmtNum(v, 3), xlabel: 'weight value', ylabel: log ? 'count (log)' : 'count' });
  // clipped regions
  if (d.clip && Number.isFinite(d.clip[0])) {
    ctx.fillStyle = C.clip;
    if (d.clip[0] > d.lo) ctx.fillRect(P.l, P.t, X(d.clip[0]) - P.l, P.h);
    if (d.clip[1] < d.hi) ctx.fillRect(X(d.clip[1]), P.t, P.r - X(d.clip[1]), P.h);
  }
  const bw = span / nb;
  ctx.fillStyle = C.bar; ctx.strokeStyle = C.barEdge;
  for (let i = 0; i < nb; i++) {
    if (!counts[i]) continue;
    const x0 = X(d.lo + i * bw), x1 = X(d.lo + (i + 1) * bw), y = Y(yv(counts[i]));
    ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 0.5), P.b - y);
  }
  // levels
  const lv = (d.levels || []).filter((v) => v >= xlo && v <= xhi);
  ctx.strokeStyle = C.level; ctx.lineWidth = lv.length > 64 ? 0.6 : 1;
  for (const v of lv) { const x = Math.round(X(v)) + 0.5; ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke(); }
  if (d.clip && Number.isFinite(d.clip[0])) {
    ctx.strokeStyle = C.clipLine; ctx.setLineDash([4, 3]);
    for (const v of d.clip) if (v > xlo && v < xhi) { const x = Math.round(X(v)) + 0.5; ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke(); }
    ctx.setLineDash([]);
  }
  if (d.levels?.length) {
    const label = `${d.levels.length} levels${d.levelsNote ? ' ' + d.levelsNote : ''}`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.fillRect(P.l + 3, P.t + 1, ctx.measureText(label).width + 6, 15);
    ctx.fillStyle = '#7c3aed'; ctx.fillText(label, P.l + 6, P.t + 3);
  }
  ctx.lineWidth = 1;
  return (mx, my) => {
    if (mx < P.l || mx > P.r || my < P.t || my > P.b) return null;
    const v = xlo + ((mx - P.l) / P.w) * (xhi - xlo);
    const i = Math.floor((v - d.lo) / bw);
    const lines = [];
    if (i >= 0 && i < nb) lines.push(`Weights between ${fmtNum(d.lo + i * bw)} and ${fmtNum(d.lo + (i + 1) * bw)}: ${counts[i].toLocaleString('en-US')} (${((100 * counts[i]) / total).toFixed(2)}%)`);
    if (lv.length) {
      let best = lv[0]; for (const L of lv) if (Math.abs(L - v) < Math.abs(best - v)) best = L;
      lines.push(`Nearest stored value: ${fmtNum(best, 4)}`);
    }
    if (d.clip && (v < d.clip[0] || v > d.clip[1])) lines.push('Shaded: outside the clip range — these weights are cut to the edge value.');
    return lines.length ? { text: lines.join('\n') } : null;
  };
}
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '.': '·' };
const sup = (v) => String(Math.round(v)).split('').map((c) => SUP[c] ?? c).join('');

/* ---------- B. staircase ---------- */
export function drawStair(canvas, d) {
  const { ctx, w, h } = prep(canvas);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < d.x.length; i++) { lo = Math.min(lo, d.x[i], d.y[i]); hi = Math.max(hi, d.x[i], d.y[i]); }
  const pad = (hi - lo) * 0.04 || 1e-3;
  lo -= pad; hi += pad;
  const { X, Y, P } = frame(ctx, w, h, { xlo: lo, xhi: hi, ylo: lo, yhi: hi, xlabel: 'original weight', ylabel: 'stored (quantized)' });
  ctx.strokeStyle = C.diag; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(lo), Y(lo)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = C.dot;
  const px = new Float32Array(d.x.length), py = new Float32Array(d.x.length);
  for (let i = 0; i < d.x.length; i++) { px[i] = X(d.x[i]); py[i] = Y(d.y[i]); ctx.fillRect(px[i] - 1.25, py[i] - 1.25, 2.5, 2.5); }
  return (mx, my) => {
    if (mx < P.l || mx > P.r || my < P.t || my > P.b) return null;
    let bi = -1, bd = 144;
    for (let i = 0; i < px.length; i++) { const dd = (px[i] - mx) ** 2 + (py[i] - my) ** 2; if (dd < bd) { bd = dd; bi = i; } }
    if (bi < 0) return null;
    return { text: `Original: ${fmtNum(d.x[bi], 5)}\nStored:   ${fmtNum(d.y[bi], 5)}\nError:    ${fmtNum(d.x[bi] - d.y[bi], 3)}` };
  };
}

/* ---------- C. error histogram ---------- */
export function drawErr(canvas, d, { log = true } = {}) {
  const { ctx, w, h } = prep(canvas);
  const counts = d.counts, nb = counts.length, total = counts.reduce((a, b) => a + b, 0);
  const yv = (c) => (log ? logc(c) : c);
  let ymax = 0; for (const c of counts) ymax = Math.max(ymax, yv(c));
  const yticks = log ? Array.from({ length: Math.ceil(ymax) + 1 }, (_, i) => i).filter((i) => i <= ymax + 1e-9) : undefined;
  const { X, Y, P } = frame(ctx, w, h, { xlo: d.lo, xhi: d.hi, ylo: 0, yhi: ymax * 1.05 || 1, yticks, yfmt: log ? (v) => (v === 0 ? '0' : '10' + sup(v)) : (v) => fmtNum(v, 3), xlabel: 'error = original − stored', ylabel: log ? 'count (log)' : 'count' });
  const bw = (d.hi - d.lo) / nb;
  ctx.fillStyle = C.err;
  for (let i = 0; i < nb; i++) {
    if (!counts[i]) continue;
    const x0 = X(d.lo + i * bw), x1 = X(d.lo + (i + 1) * bw), y = Y(yv(counts[i]));
    ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 0.5), P.b - y);
  }
  ctx.strokeStyle = C.axis; ctx.beginPath(); ctx.moveTo(X(0) + 0.5, P.t); ctx.lineTo(X(0) + 0.5, P.b); ctx.stroke();
  return (mx, my) => {
    if (mx < P.l || mx > P.r || my < P.t || my > P.b) return null;
    const i = Math.floor(((mx - P.l) / P.w) * nb);
    if (i < 0 || i >= nb) return null;
    return { text: `Errors between ${fmtNum(d.lo + i * bw)} and ${fmtNum(d.lo + (i + 1) * bw)}: ${counts[i].toLocaleString('en-US')} weights (${((100 * counts[i]) / total).toFixed(2)}%)` };
  };
}

/* ---------- E. clipping trade-off ---------- */
export function drawClip(canvas, d, current) {
  const { ctx, w, h } = prep(canvas);
  if (!d) {
    ctx.fillStyle = C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Clipping does not apply to plain FP16 / BF16 casts.', w / 2, h / 2);
    return () => null;
  }
  const xs = d.ratios.map((r) => r * 100), ys = d.rmse;
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = (ymax - ymin) * 0.08 || ymax * 0.1 || 1e-6;
  const { X, Y, P } = frame(ctx, w, h, { xlo: 20, xhi: 100, ylo: Math.max(0, ymin - pad), yhi: ymax + pad, xfmt: (v) => v + '%', xlabel: 'clip at % of the max |weight|', ylabel: 'RMS error' });
  ctx.strokeStyle = C.line; ctx.lineWidth = 2; ctx.beginPath();
  xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i]))));
  ctx.stroke(); ctx.lineWidth = 1;
  const bx = X(xs[d.best]), by = Y(ys[d.best]);
  ctx.fillStyle = C.best; ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.fill();
  // best point label above it, "no clipping" label below the 100% point, so they never collide
  ctx.textAlign = bx > P.l + P.w * 0.7 ? 'right' : 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = C.best; ctx.font = '600 11px Inter, "Segoe UI", system-ui, sans-serif';
  ctx.fillText(`best: ${xs[d.best].toFixed(0)}%`, bx + (ctx.textAlign === 'left' ? 7 : -7), by - 7);
  ctx.font = FONT;
  if (d.best !== 0) {
    ctx.fillStyle = C.point; ctx.beginPath(); ctx.arc(X(100), Y(ys[0]), 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = C.text; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('no clipping', X(100) - 6, Math.min(P.b - 12, Y(ys[0]) + 6));
  }
  if (d.subset) { ctx.fillStyle = C.text; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`on a ${d.subset.toLocaleString('en-US')}-weight subset`, P.r - 4, P.t + 2); }
  return (mx, my) => {
    if (mx < P.l || mx > P.r || my < P.t || my > P.b) return null;
    let bi = 0; for (let i = 1; i < xs.length; i++) if (Math.abs(X(xs[i]) - mx) < Math.abs(X(xs[bi]) - mx)) bi = i;
    return { text: `Clip at ${xs[bi].toFixed(0)}% of the max:\nRMS error ${fmtNum(ys[bi], 3)} · SQNR ${d.sqnr[bi].toFixed(2)} dB${bi === d.best ? '\n← lowest error' : ''}${bi === 0 ? '\n(no clipping)' : ''}` };
  };
}

/* ---------- D. all formats ---------- */
export function drawCompare(canvas, list, current, subset = 0) {
  const { ctx, w, h } = prep(canvas);
  const cap = (s) => (Number.isFinite(s) ? s : 99);
  const pts = list.map((p) => ({ ...p, y: cap(p.sqnr) }));
  const all = current ? [...pts, { ...current, y: cap(current.sqnr) }] : pts;
  const ymax = Math.max(...all.map((p) => p.y)), ymin = Math.min(0, ...all.map((p) => p.y));
  const { X, Y, P } = frame(ctx, w, h, { xlo: 1.5, xhi: 17, ylo: ymin, yhi: ymax + 6, xticks: [2, 3, 4, 5, 6, 8, 10, 12, 14, 16], xlabel: 'bits per weight (incl. scales)', ylabel: 'SQNR (dB) — higher is better' });
  // 6 dB / bit guide through INT8
  const ref = pts.find((p) => p.label === 'INT8 per channel');
  if (ref) {
    ctx.strokeStyle = '#e2e8f0'; ctx.setLineDash([3, 4]); ctx.beginPath();
    ctx.moveTo(X(1.5), Y(ref.y - 6.02 * (ref.bits - 1.5))); ctx.lineTo(X(17), Y(ref.y + 6.02 * (17 - ref.bits))); ctx.stroke(); ctx.setLineDash([]);
  }
  const placed = [];
  for (const p of pts) {
    const x = X(p.bits), y = Y(p.y);
    ctx.fillStyle = C.point; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.font = '10.5px Inter, "Segoe UI", system-ui, sans-serif'; ctx.fillStyle = '#334155';
    const tw = ctx.measureText(p.label).width;
    let lx = x + 7, ly = y - 7;
    if (lx + tw > P.r) lx = x - 7 - tw;
    for (let k = 0; k < 6 && placed.some((q) => lx < q.x + q.w && lx + tw > q.x && Math.abs(ly - q.y) < 11); k++) ly += 12;
    placed.push({ x: lx, y: ly, w: tw });
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(p.label, lx, ly);
  }
  if (current) {
    const x = X(current.bits), y = Y(current.y);
    ctx.strokeStyle = C.cur; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = C.cur; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.font = FONT;
  if (subset) { ctx.fillStyle = C.text; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`grey points: measured on a ${subset.toLocaleString('en-US')}-weight subset`, P.r - 4, P.t + 2); }
  return (mx, my) => {
    let best = null, bd = 400;
    for (const p of all) { const dd = (X(p.bits) - mx) ** 2 + (Y(p.y) - my) ** 2; if (dd < bd) { bd = dd; best = p; } }
    if (!best) return null;
    const isCur = current && best === all[all.length - 1];
    return { text: `${isCur ? 'Your current setting: ' : ''}${best.label}\n${best.bits.toFixed(3)} bits per weight · SQNR ${Number.isFinite(best.sqnr) ? best.sqnr.toFixed(2) + ' dB' : 'exact (no error)'}` };
  };
}

/* ---------- F. 32 × 32 heatmaps ---------- */
function divColor(v, vmax) {
  const t = Math.max(-1, Math.min(1, v / (vmax || 1)));
  const a = Math.abs(t) ** 0.7;
  const [r, g, b] = t >= 0 ? [220, 38, 38] : [37, 99, 235];
  return `rgb(${Math.round(255 + (r - 255) * a)},${Math.round(255 + (g - 255) * a)},${Math.round(255 + (b - 255) * a)})`;
}
export function drawHeat(canvas, d) {
  const { ctx, w, h } = prep(canvas);
  const { nr, nc } = d;
  const gap = 18, top = 18;
  const cell = Math.max(2, Math.floor(Math.min((w - gap * 2 - 4) / (3 * nc), (h - top - 4) / nr)));
  const pw = cell * nc, ph = cell * nr;
  const x0 = Math.floor((w - (pw * 3 + gap * 2)) / 2);
  let vmax = 0, emax = 0;
  for (let k = 0; k < d.w.length; k++) { vmax = Math.max(vmax, Math.abs(d.w[k]), Math.abs(d.q[k])); emax = Math.max(emax, Math.abs(d.w[k] - d.q[k])); }
  const panels = [['original', (k) => d.w[k], vmax], ['quantized', (k) => d.q[k], vmax], ['error (own colour scale)', (k) => d.w[k] - d.q[k], emax]];
  const paint = (hk) => {
    ctx.clearRect(0, 0, w, h);
    panels.forEach(([label, val, m], pi) => {
      const px = x0 + pi * (pw + gap);
      ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(label, px, 2);
      for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
        ctx.fillStyle = divColor(val(r * nc + c), m);
        ctx.fillRect(px + c * cell, top + r * cell, cell, cell);
      }
      ctx.strokeStyle = C.axis; ctx.strokeRect(px + 0.5, top + 0.5, pw, ph);
      if (hk !== null) {
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
        ctx.strokeRect(px + (hk % nc) * cell + 0.75, top + Math.floor(hk / nc) * cell + 0.75, cell - 1.5, cell - 1.5);
        ctx.lineWidth = 1;
      }
    });
  };
  paint(null);
  let hl = null;
  return (mx, my) => {
    const pi = Math.floor((mx - x0) / (pw + gap));
    const lx = mx - x0 - pi * (pw + gap), ly = my - top;
    if (pi < 0 || pi > 2 || lx < 0 || lx >= pw || ly < 0 || ly >= ph) { if (hl !== null) { hl = null; paint(null); } return null; }
    const r = Math.floor(ly / cell), c = Math.floor(lx / cell), k = r * nc + c;
    if (hl !== k) { hl = k; paint(k); }
    return { k, row: d.r0 + r, col: d.c0 + c, text: `Row ${d.r0 + r}, column ${d.c0 + c}
Original ${fmtNum(d.w[k], 5)} → stored ${fmtNum(d.q[k], 5)}
Error ${fmtNum(d.w[k] - d.q[k], 3)}` };
  };
}

/* ---------- PNG export ---------- */
export function chartPNG(canvas, title, subtitle = '') {
  const pad = 16, head = subtitle ? 46 : 30;
  const out = document.createElement('canvas');
  out.width = canvas.width + pad * 2; out.height = canvas.height + pad * 2 + head;
  const ctx = out.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = C.ink; ctx.font = '600 16px Inter, "Segoe UI", system-ui, sans-serif'; ctx.textBaseline = 'top';
  ctx.fillText(title, pad, pad);
  if (subtitle) { ctx.fillStyle = C.text; ctx.font = '12px Inter, "Segoe UI", system-ui, sans-serif'; ctx.fillText(subtitle, pad, pad + 22); }
  ctx.drawImage(canvas, pad, pad + head);
  return out;
}

/** One tall PNG: title, text lines, then the charts in a 2-column grid. */
export function reportPNG(title, lines, charts) {
  const W = 1400, pad = 28, lh = 20;
  const colW = (W - pad * 3) / 2;
  const imgs = charts.map(({ canvas, title: t }) => ({ t, canvas, h: (canvas.height / canvas.width) * colW }));
  let gridH = 0;
  for (let i = 0; i < imgs.length; i += 2) gridH += Math.max(imgs[i].h, imgs[i + 1]?.h || 0) + 34;
  const H = pad * 2 + 40 + lines.length * lh + 16 + gridH;
  const out = document.createElement('canvas');
  out.width = W; out.height = H;
  const ctx = out.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = C.ink; ctx.font = '700 24px Inter, "Segoe UI", system-ui, sans-serif'; ctx.textBaseline = 'top';
  ctx.fillText(title, pad, pad);
  ctx.font = '14px Inter, "Segoe UI", system-ui, sans-serif'; ctx.fillStyle = '#334155';
  let y = pad + 40;
  for (const l of lines) { ctx.fillText(l, pad, y); y += lh; }
  y += 16;
  for (let i = 0; i < imgs.length; i += 2) {
    const row = [imgs[i], imgs[i + 1]].filter(Boolean);
    row.forEach((im, j) => {
      const x = pad + j * (colW + pad);
      ctx.fillStyle = C.ink; ctx.font = '600 14px Inter, "Segoe UI", system-ui, sans-serif'; ctx.fillText(im.t, x, y);
      ctx.drawImage(im.canvas, x, y + 22, colW, im.h);
    });
    y += Math.max(...row.map((r) => r.h)) + 34;
  }
  return out;
}
