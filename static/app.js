/**
 * Frontend Quantitative Terminal Controller for QQQ "Basket vs. Box" Arbitrage Scanner.
 * Connects via WebSocket (/ws/stream) for real-time 60ms-120ms microstructure updates,
 * renders WebGL 3D Price Gap Surface, PyTorch Lead-Lag analytics, and Live Trade Ticket.
 */

const MAX_HISTORY_POINTS = 75;
const timeHistory = [];
const qqqPriceHistory = [];
const boxPriceHistory = [];
const gapUsdHistory = [];

let surfaceMode = "temporal"; // "temporal" or "parametric"
let streamPaused = false;
let chartsInitialized = false;
let last3DRenderTs = 0;

// Format helpers
function fmtUsd(val, decimals = 2) {
  const num = Number(val) || 0;
  const sign = num < 0 ? "-" : "";
  return `${sign}$${Math.abs(num).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

function fmtSignedUsd(val, decimals = 2) {
  const num = Number(val) || 0;
  const sign = num > 0 ? "+" : num < 0 ? "-" : "";
  return `${sign}$${Math.abs(num).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

function fmtSignedBps(val, decimals = 2) {
  const num = Number(val) || 0;
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(decimals)} bps`;
}

// Initialize all Plotly charts with institutional dark theme
function initCharts() {
  const commonFont = { family: "JetBrains Mono, monospace", size: 10, color: "#94a3b8" };

  // 1. Dual Subplot: QQQ vs Synthetic Box ($) + Arbitrage Gap ($)
  const priceBoxLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(13,21,36,0.55)",
    margin: { l: 54, r: 18, t: 8, b: 26 },
    font: commonFont,
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0.01,
      y: 1.02,
      bgcolor: "rgba(11,17,30,0.8)",
      font: { size: 10, color: "#e2e8f0" },
    },
    grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
    xaxis: {
      showticklabels: false,
      gridcolor: "rgba(148,163,184,0.08)",
      zeroline: false,
      domain: [0, 1],
    },
    yaxis: {
      title: { text: "Price ($)", font: { size: 10 } },
      gridcolor: "rgba(148,163,184,0.1)",
      tickprefix: "$",
      domain: [0.36, 1.0],
    },
    xaxis2: {
      gridcolor: "rgba(148,163,184,0.08)",
      tickfont: { size: 9 },
      domain: [0, 1],
    },
    yaxis2: {
      title: { text: "Gap ($)", font: { size: 10 } },
      gridcolor: "rgba(148,163,184,0.1)",
      zeroline: true,
      zerolinecolor: "rgba(255,255,255,0.25)",
      tickprefix: "$",
      domain: [0.0, 0.28],
    },
    uirevision: "keep_price_box",
  };

  const priceBoxTraces = [
    {
      x: [],
      y: [],
      name: "QQQ ETF Market ($)",
      type: "scatter",
      mode: "lines",
      line: { color: "#f59e0b", width: 2.3 },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: [],
      y: [],
      name: "Synthetic Stock Box ($)",
      type: "scatter",
      mode: "lines",
      line: { color: "#06b6d4", width: 2.0, dash: "dot" },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: [],
      y: [],
      name: "Price Gap (QQQ - Box)",
      type: "bar",
      marker: { color: [] },
      xaxis: "x2",
      yaxis: "y2",
    },
  ];

  Plotly.newPlot("chart-price-box", priceBoxTraces, priceBoxLayout, {
    displayModeBar: false,
    responsive: true,
  });

  // 2. PyTorch Lead-Lag Cross-Correlation Profile
  const corrLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(11,17,30,0.45)",
    margin: { l: 42, r: 12, t: 10, b: 32 },
    font: commonFont,
    xaxis: {
      title: { text: "Lag τ (ms) — Positive = NVDA Leads QQQ", font: { size: 9.5 } },
      gridcolor: "rgba(148,163,184,0.09)",
      zeroline: true,
      zerolinecolor: "rgba(255,255,255,0.28)",
    },
    yaxis: {
      title: { text: "Cross-Corr (r)", font: { size: 9.5 } },
      range: [-0.45, 1.0],
      gridcolor: "rgba(148,163,184,0.09)",
      zeroline: true,
      zerolinecolor: "rgba(255,255,255,0.2)",
    },
    uirevision: "keep_corr",
  };

  Plotly.newPlot(
    "chart-lead-lag-corr",
    [
      {
        x: [],
        y: [],
        type: "scatter",
        mode: "lines+markers",
        fill: "tozeroy",
        fillcolor: "rgba(245, 158, 11, 0.14)",
        line: { color: "#f59e0b", width: 2.2, shape: "spline" },
        marker: { size: 6, color: "#fbbf24" },
      },
    ],
    corrLayout,
    { displayModeBar: false, responsive: true }
  );

  // 3. PyTorch Temporal Attention Weights by Lag (ms)
  const attnLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(11,17,30,0.45)",
    margin: { l: 42, r: 12, t: 10, b: 32 },
    font: commonFont,
    xaxis: {
      title: { text: "Historical Lookback Lag (ms)", font: { size: 9.5 } },
      gridcolor: "rgba(148,163,184,0.08)",
    },
    yaxis: {
      title: { text: "Attention Wt", font: { size: 9.5 } },
      range: [0, 0.35],
      gridcolor: "rgba(148,163,184,0.08)",
    },
    uirevision: "keep_attn",
  };

  Plotly.newPlot(
    "chart-attention-weights",
    [
      {
        x: [],
        y: [],
        type: "bar",
        marker: {
          color: "#a855f7",
          line: { color: "#c084fc", width: 1 },
        },
      },
    ],
    attnLayout,
    { displayModeBar: false, responsive: true }
  );

  // 4. 3D Arbitrage Price Gap Surface
  const surface3DLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, t: 6, b: 0 },
    font: commonFont,
    scene: {
      bgcolor: "rgba(8, 13, 24, 0.85)",
      xaxis: {
        title: "Lead-Lag τ (ms)",
        gridcolor: "#1e293b",
        zerolinecolor: "#475569",
        showbackground: true,
        backgroundcolor: "rgba(15, 23, 42, 0.65)",
      },
      yaxis: {
        title: "Rolling Tick Window",
        gridcolor: "#1e293b",
        zerolinecolor: "#475569",
        showbackground: true,
        backgroundcolor: "rgba(15, 23, 42, 0.65)",
      },
      zaxis: {
        title: "Price Gap ($ = QQQ − Box)",
        gridcolor: "#1e293b",
        zerolinecolor: "#f59e0b",
        showbackground: true,
        backgroundcolor: "rgba(15, 23, 42, 0.8)",
      },
      camera: {
        eye: { x: 1.55, y: -1.65, z: 0.95 },
      },
    },
    uirevision: "keep_3d_camera",
  };

  const surfaceTrace = {
    type: "surface",
    x: [-200, -150, -100, -50, 0, 50, 100, 150, 200, 250],
    y: Array.from({ length: 24 }, (_, i) => i),
    z: Array.from({ length: 24 }, () => Array(10).fill(0)),
    colorscale: [
      [0.0, "#10b981"],
      [0.35, "#06b6d4"],
      [0.5, "#1e293b"],
      [0.65, "#f59e0b"],
      [1.0, "#f43f5e"],
    ],
    cmin: -0.5,
    cmax: 0.5,
    colorbar: {
      title: { text: "Gap ($)", side: "right", font: { size: 10, color: "#94a3b8" } },
      thickness: 12,
      len: 0.75,
      tickprefix: "$",
      tickfont: { size: 9, color: "#94a3b8" },
    },
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "#ffffff",
        project: { z: true },
      },
    },
  };

  Plotly.newPlot("chart-surface-3d", [surfaceTrace], surface3DLayout, {
    displayModeBar: false,
    responsive: true,
  });

  chartsInitialized = true;
}

// Render complete telemetry update from backend snapshot
function renderDashboard(data) {
  if (!data || !data.market) return;
  if (!chartsInitialized) initCharts();

  const mkt = data.market;
  const ll = data.lead_lag;
  const arb = data.arbitrage;
  const ticket = arb.ticket;
  const econ = ticket.economics;

  // 1. Update Top Hero Strip
  const qqqPriceEl = document.getElementById("kpi-qqq-price");
  const qqqRetEl = document.getElementById("kpi-qqq-ret");
  const boxPriceEl = document.getElementById("kpi-box-price");
  const boxRetEl = document.getElementById("kpi-box-ret");
  const gapUsdEl = document.getElementById("kpi-gap-usd");
  const gapBpsEl = document.getElementById("kpi-gap-bps");
  const zScoreEl = document.getElementById("kpi-zscore");
  const arbDirEl = document.getElementById("kpi-arb-direction");
  const gapCardEl = document.getElementById("kpi-gap-card");

  qqqPriceEl.textContent = fmtUsd(mkt.qqq.price, 2);
  qqqRetEl.textContent = fmtSignedBps(mkt.qqq.tick_ret_bps, 2);
  qqqRetEl.className = `kpi-delta mono ${mkt.qqq.tick_ret_bps >= 0 ? "pos" : "neg"}`;
  document.getElementById("kpi-qqq-bbo").textContent = `Bid ${fmtUsd(mkt.qqq.bid)} · Ask ${fmtUsd(mkt.qqq.ask)}`;

  boxPriceEl.textContent = fmtUsd(mkt.box.price, 2);
  boxRetEl.textContent = fmtSignedBps(mkt.box.tick_ret_bps, 2);
  boxRetEl.className = `kpi-delta mono ${mkt.box.tick_ret_bps >= 0 ? "pos" : "neg"}`;

  const gapUsd = mkt.spread.gap_usd;
  gapUsdEl.textContent = fmtSignedUsd(gapUsd, 4);
  gapUsdEl.className = `kpi-value mono ${gapUsd >= 0 ? "accent-amber" : "accent-emerald"}`;
  gapBpsEl.textContent = `(${fmtSignedBps(mkt.spread.gap_bps, 2)})`;
  zScoreEl.textContent = `Z: ${arb.z_score >= 0 ? "+" : ""}${arb.z_score.toFixed(2)}σ`;

  if (gapUsd >= 0.08) {
    arbDirEl.textContent = "QQQ OVERPRICED → SELL QQQ / BUY CHEAPER STOCKS";
    gapCardEl.style.borderColor = "#f59e0b";
  } else if (gapUsd <= -0.08) {
    arbDirEl.textContent = "QQQ UNDERPRICED → BUY CHEAP QQQ / SHORT STOCKS";
    gapCardEl.style.borderColor = "#10b981";
  } else {
    arbDirEl.textContent = "NEAR PARITY EQUILIBRIUM (MONITORING LEAD-LAG)";
    gapCardEl.style.borderColor = "#2a3e63";
  }

  document.getElementById("active-shock-pill").textContent = mkt.active_shock_label;
  document.getElementById("source-mode-badge").textContent = mkt.data_source_mode;
  document.getElementById("tick-counter-badge").textContent = `Tick #${mkt.tick_count}`;

  // 2. Push to rolling chart history
  const tsLabel = new Date(mkt.timestamp * 1000).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }) + `.${String(mkt.tick_count % 10)}`;

  if (timeHistory.length === 0 || timeHistory[timeHistory.length - 1] !== tsLabel) {
    timeHistory.push(tsLabel);
    qqqPriceHistory.push(mkt.qqq.price);
    boxPriceHistory.push(mkt.box.price);
    gapUsdHistory.push(gapUsd);

    if (timeHistory.length > MAX_HISTORY_POINTS) {
      timeHistory.shift();
      qqqPriceHistory.shift();
      boxPriceHistory.shift();
      gapUsdHistory.shift();
    }
  } else {
    qqqPriceHistory[qqqPriceHistory.length - 1] = mkt.qqq.price;
    boxPriceHistory[boxPriceHistory.length - 1] = mkt.box.price;
    gapUsdHistory[gapUsdHistory.length - 1] = gapUsd;
  }

  const barColors = gapUsdHistory.map((g) =>
    g >= 0.1 ? "#f59e0b" : g <= -0.1 ? "#10b981" : "#475569"
  );

  Plotly.update(
    "chart-price-box",
    {
      x: [timeHistory, timeHistory, timeHistory],
      y: [qqqPriceHistory, boxPriceHistory, gapUsdHistory],
      "marker.color": [null, null, barColors],
    },
    {},
    [0, 1, 2]
  );

  // 3. Populate Constituent Box Replication Table
  const tbody = document.getElementById("constituent-table-body");
  tbody.innerHTML = mkt.constituents
    .map((c) => {
      const retCls = c.tick_ret_bps > 0 ? "pos" : c.tick_ret_bps < 0 ? "neg" : "";
      const isNvda = c.symbol === "NVDA";
      return `
        <tr>
          <td class="mono"><strong>${c.symbol}</strong> ${isNvda ? '<span class="badge badge-amber" style="font-size:9px;padding:1px 4px;">LEAD</span>' : ""}</td>
          <td>${c.name}</td>
          <td class="num mono"><strong>${fmtUsd(c.price)}</strong></td>
          <td class="num mono text-muted">${fmtUsd(c.bid)} / ${fmtUsd(c.ask)}</td>
          <td class="num mono ${retCls}">${fmtSignedBps(c.tick_ret_bps, 2)}</td>
          <td class="num mono">${c.qqq_weight_pct.toFixed(2)}%</td>
          <td class="num mono">${c.box_norm_weight_pct.toFixed(2)}%</td>
          <td class="num mono">${c.shares_per_qqq.toFixed(4)}</td>
          <td class="num mono accent-cyan"><strong>${fmtUsd(c.box_contribution_usd)}</strong></td>
        </tr>
      `;
    })
    .join("");

  document.getElementById("foot-raw-wt").textContent = `${mkt.box.top5_raw_qqq_weight_pct.toFixed(2)}%`;
  document.getElementById("foot-box-sum").textContent = fmtUsd(mkt.box.price);

  // 4. Update Panel 2: PyTorch Lead-Lag Predictor
  document.getElementById("pytorch-loss-badge").textContent = `Huber Loss: ${ll.model_loss_ema.toFixed(4)}`;
  document.getElementById("pytorch-latency-badge").textContent = `Infer: ${ll.inference_latency_ms.toFixed(2)} ms`;

  document.getElementById("ll-optimal-lag").textContent = `${ll.optimal_lag_ms} ms Head Start`;
  document.getElementById("ll-peak-corr").textContent = `Peak Cross-Corr: r = +${ll.lead_lag_corr_peak.toFixed(3)}`;

  const nvdaImpEl = document.getElementById("ll-nvda-impulse");
  nvdaImpEl.textContent = fmtSignedBps(ll.nvda_impulse_bps, 2);
  nvdaImpEl.className = `ll-value mono ${ll.nvda_impulse_bps >= 0 ? "pos" : "neg"}`;
  document.getElementById("ll-unreacted-gap").textContent = `Unreacted QQQ Lag: ${fmtSignedUsd(ll.qqq_unreacted_gap_usd, 4)}`;

  const predDeltaEl = document.getElementById("ll-pred-delta");
  predDeltaEl.textContent = `${fmtSignedBps(ll.predicted_qqq_delta_bps, 2)} (${fmtSignedUsd(ll.predicted_qqq_delta_usd, 3)})`;
  predDeltaEl.className = `ll-value mono ${ll.predicted_qqq_delta_bps >= 0 ? "pos" : "neg"}`;
  document.getElementById("ll-pred-target").textContent = `Target QQQ: ${fmtUsd(ll.predicted_qqq_target_price, 2)}`;

  const sigBadgeEl = document.getElementById("ll-signal-badge");
  sigBadgeEl.textContent = ll.signal_direction.replace("_", " ");
  sigBadgeEl.className = `ll-value mono ${
    ll.signal_direction === "BULLISH_LEAD"
      ? "pos"
      : ll.signal_direction === "BEARISH_LEAD"
      ? "neg"
      : "accent-cyan"
  }`;

  document.getElementById("ll-conf-bar").style.width = `${ll.confidence}%`;
  document.getElementById("ll-conf-text").textContent = `Confidence: ${ll.confidence.toFixed(1)}%`;

  // Update Lead-Lag Cross-Correlation Curve
  const lagsX = ll.cross_corr_profile.map((d) => d.lag_ms);
  const corrY = ll.cross_corr_profile.map((d) => d.correlation);
  Plotly.update("chart-lead-lag-corr", { x: [lagsX], y: [corrY] }, {}, [0]);

  // Update PyTorch Temporal Attention Bar Chart
  const attnX = ll.attention_by_lag_ms.map((d) => `${d.lag_ms}ms`);
  const attnY = ll.attention_by_lag_ms.map((d) => d.weight);
  Plotly.update("chart-attention-weights", { x: [attnX], y: [attnY] }, {}, [0]);

  // Constituent Lead Attribution Strip
  const attrStrip = document.getElementById("constituent-attribution-strip");
  attrStrip.innerHTML = Object.entries(ll.constituent_lead_scores)
    .map(([sym, score]) => {
      const pct = (score * 100).toFixed(1);
      return `
        <div class="attr-chip mono">
          <span><strong>${sym}</strong> Lead Share</span>
          <span class="${sym === "NVDA" ? "accent-amber" : "accent-cyan"}"><strong>${pct}%</strong></span>
        </div>
      `;
    })
    .join("");

  // 5. Update Panel 3: 3D Price Gap Surface (throttled to ~4 FPS for smooth WebGL interaction)
  const nowMs = performance.now();
  if (nowMs - last3DRenderTs > 220) {
    last3DRenderTs = nowMs;
    const surf = arb.surface_3d;
    if (surfaceMode === "temporal") {
      Plotly.update(
        "chart-surface-3d",
        {
          x: [surf.x_lags_ms],
          y: [surf.y_time_labels],
          z: [surf.z_gap_matrix],
        },
        {
          "scene.xaxis.title": "Lead-Lag Horizon τ (ms)",
          "scene.yaxis.title": "Rolling Tick Time",
          "scene.zaxis.title": "Price Gap ($ = QQQ − Box)",
        },
        [0]
      );
    } else {
      const pSurf = surf.parametric;
      Plotly.update(
        "chart-surface-3d",
        {
          x: [pSurf.x_nvda_shocks_pct.map((s) => `${s >= 0 ? "+" : ""}${s}%`)],
          y: [pSurf.y_reaction_lags_ms.map((m) => `${m}ms`)],
          z: [pSurf.z_gap_matrix],
        },
        {
          "scene.xaxis.title": "NVDA Instant Shock (%)",
          "scene.yaxis.title": "QQQ Reaction Lag (ms)",
          "scene.zaxis.title": "Implied Price Gap ($)",
        },
        [0]
      );
    }
  }

  // 6. Update Panel 4: Live Arbitrage Trade Ticket & Economics
  const statusBadgeEl = document.getElementById("ticket-status-badge");
  statusBadgeEl.textContent = ticket.status_badge;
  statusBadgeEl.className = `badge mono ${ticket.is_executable ? "badge-emerald" : "badge-amber"}`;

  document.getElementById("ticket-strategy-type").textContent =
    ticket.strategy_type === "SHORT_QQQ_LONG_BASKET"
      ? "CONVERSION BOX ARB // SELL OVERPRICED ETF vs. BUY STOCKS"
      : "REVERSAL BOX ARB // BUY CHEAP ETF vs. SHORT STOCKS";

  document.getElementById("ticket-headline").textContent = ticket.headline;

  const qqqPill = document.getElementById("ticket-qqq-pill");
  qqqPill.textContent = `${ticket.qqq_leg.side} ${ticket.qqq_leg.shares.toLocaleString()} QQQ @ ${fmtUsd(ticket.qqq_leg.price)} (${fmtUsd(ticket.qqq_leg.notional_usd, 0)})`;
  qqqPill.className = `leg-pill ${ticket.qqq_leg.side.includes("SELL") ? "leg-sell" : "leg-buy"}`;

  const boxPill = document.getElementById("ticket-box-pill");
  boxPill.textContent = `${ticket.box_summary.side} 5-STOCK BOX @ ${fmtUsd(ticket.box_summary.implied_price_per_qqq)} (${fmtUsd(ticket.box_summary.total_notional_usd, 0)})`;
  boxPill.className = `leg-pill ${ticket.box_summary.side.includes("SELL") ? "leg-sell" : "leg-buy"}`;

  document.getElementById("ticket-lot-note").textContent = `Per ${ticket.qqq_lot_size.toLocaleString()} QQQ Lot`;

  // Populate Ticket Multi-Leg Basket Order Rows (QQQ leg + 5 Constituent legs)
  const qqqRowHtml = `
    <tr style="background: rgba(245, 158, 11, 0.08);">
      <td class="mono"><strong>QQQ (ETF)</strong></td>
      <td class="mono ${ticket.qqq_leg.side.includes("SELL") ? "neg" : "pos"}"><strong>${ticket.qqq_leg.side}</strong></td>
      <td class="num mono">${ticket.qqq_leg.shares.toLocaleString()}.00</td>
      <td class="num mono">${fmtUsd(ticket.qqq_leg.price)}</td>
      <td class="num mono"><strong>${fmtUsd(ticket.qqq_leg.notional_usd)}</strong></td>
    </tr>
  `;

  const stockRowsHtml = ticket.basket_legs
    .map((leg) => {
      const sideCls = leg.side.includes("SELL") ? "neg" : "pos";
      return `
        <tr>
          <td class="mono"><strong>${leg.symbol}</strong> <span class="text-muted">(${leg.box_weight_pct.toFixed(1)}%)</span></td>
          <td class="mono ${sideCls}"><strong>${leg.side}</strong></td>
          <td class="num mono">${leg.exact_shares.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td class="num mono">${fmtUsd(leg.price)}</td>
          <td class="num mono">${fmtUsd(leg.notional_usd)}</td>
        </tr>
      `;
    })
    .join("");

  document.getElementById("ticket-legs-body").innerHTML = qqqRowHtml + stockRowsHtml;

  // Populate Economics Waterfall
  document.getElementById("econ-gap-share").textContent = `${fmtSignedUsd(econ.gap_per_share_usd, 4)} / sh`;
  document.getElementById("econ-gross-spread").textContent = `+${fmtUsd(econ.gross_spread_profit_usd)}`;
  document.getElementById("econ-lead-alpha").textContent = `+${fmtUsd(econ.lead_lag_alpha_usd)}`;
  document.getElementById("econ-slippage").textContent = `-${fmtUsd(econ.slippage_cost_usd)}`;
  document.getElementById("econ-reg-fees").textContent = `-${fmtUsd(econ.exchange_reg_fees_usd)}`;
  document.getElementById("econ-borrow-fees").textContent = `-${fmtUsd(econ.borrow_clearing_usd)}`;

  const netProfitEl = document.getElementById("econ-net-profit");
  netProfitEl.textContent = `${fmtSignedUsd(econ.net_profit_usd, 2)} (${fmtSignedBps(econ.net_margin_bps, 2)})`;
  netProfitEl.className = `econ-total-val ${econ.net_profit_usd >= 0 ? "pos" : "neg"}`;

  // Update Executed Blotter
  renderBlotter(arb.blotter_summary);
}

function renderBlotter(blotter) {
  if (!blotter) return;
  document.getElementById("blotter-count").textContent = blotter.total_boxes_executed;
  document.getElementById("blotter-gross").textContent = fmtUsd(blotter.cumulative_gross_pnl_usd);
  const netEl = document.getElementById("blotter-net");
  netEl.textContent = fmtSignedUsd(blotter.cumulative_net_pnl_usd);
  netEl.className = blotter.cumulative_net_pnl_usd >= 0 ? "pos" : "neg";

  const tbody = document.getElementById("blotter-table-body");
  if (!blotter.recent_trades || blotter.recent_trades.length === 0) {
    return;
  }

  tbody.innerHTML = blotter.recent_trades
    .map((t) => {
      const netCls = t.net_profit_usd >= 0 ? "pos" : "neg";
      const modeBadge =
        t.trigger_source === "AUTO_ARB_BOT"
          ? '<span class="badge badge-cyan">BOT</span>'
          : '<span class="badge badge-amber">MANUAL</span>';
      return `
        <tr>
          <td class="mono">${t.trade_id}</td>
          <td class="mono text-muted">${t.timestamp}</td>
          <td>${modeBadge}</td>
          <td class="mono"><strong>${t.qqq_side} ${t.qqq_shares.toLocaleString()} QQQ</strong></td>
          <td class="num mono">${fmtUsd(t.qqq_price)} vs ${fmtUsd(t.box_price)}</td>
          <td class="num mono">${fmtSignedUsd(t.gap_per_share_usd, 3)}</td>
          <td class="num mono neg">-${fmtUsd(t.fees_usd)}</td>
          <td class="num mono ${netCls}"><strong>${fmtSignedUsd(t.net_profit_usd)}</strong></td>
          <td class="num mono text-muted">${t.fill_latency_us} μs</td>
        </tr>
      `;
    })
    .join("");
}

// Interactive Controls & API Actions
async function triggerShock(shockType) {
  try {
    const resp = await fetch("/api/shock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shock_type: shockType }),
    });
    const data = await resp.json();
    if (data.snapshot) {
      renderDashboard(data.snapshot);
    }
  } catch (err) {
    console.error("Error triggering market shock:", err);
  }
}

async function executeArbitrageTicket() {
  const btn = document.getElementById("btn-execute-arb");
  btn.textContent = "✓ LOCKED IN BOX ARB!";
  setTimeout(() => {
    btn.textContent = "⚡ EXECUTE ARBITRAGE BOX";
  }, 650);

  try {
    const resp = await fetch("/api/execute", { method: "POST" });
    const data = await resp.json();
    if (data.blotter_summary) {
      renderBlotter(data.blotter_summary);
    }
  } catch (err) {
    console.error("Error executing arbitrage ticket:", err);
  }
}

async function updateTicketConfig() {
  const lotSize = parseInt(document.getElementById("lot-size-select").value, 10);
  const autoArb = document.getElementById("auto-arb-checkbox").checked;

  try {
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        qqq_lot_size: lotSize,
        auto_arb_enabled: autoArb,
      }),
    });
  } catch (err) {
    console.error("Error updating ticket config:", err);
  }
}

async function togglePauseStream() {
  streamPaused = !streamPaused;
  const btn = document.getElementById("btn-pause-stream");
  const dot = document.getElementById("ws-status-dot");
  const txt = document.getElementById("ws-status-text");

  if (streamPaused) {
    btn.textContent = "▶ Resume Stream";
    btn.className = "btn btn-emerald";
    dot.style.background = "#f59e0b";
    txt.textContent = "STREAM FROZEN FOR INSPECTION";
  } else {
    btn.textContent = "⏸ Freeze Tick";
    btn.className = "btn btn-outline";
    dot.style.background = "#10b981";
    txt.textContent = "LIVE 60ms MICROSTRUCTURE STREAM";
  }

  try {
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_paused: streamPaused }),
    });
  } catch (err) {
    console.error("Error toggling stream pause:", err);
  }
}

async function syncYahooPrices(normalizeTo500 = true) {
  try {
    const resp = await fetch("/api/sync_live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ normalize_to_500: normalizeTo500 }),
    });
    const data = await resp.json();
    if (data.snapshot) {
      renderDashboard(data.snapshot);
    }
  } catch (err) {
    console.error("Error syncing Yahoo Finance prices:", err);
  }
}

function setSurfaceMode(mode) {
  surfaceMode = mode;
  document.getElementById("btn-surf-temporal").className =
    mode === "temporal" ? "btn btn-xs btn-active" : "btn btn-xs btn-outline";
  document.getElementById("btn-surf-parametric").className =
    mode === "parametric" ? "btn btn-xs btn-active" : "btn btn-xs btn-outline";
  last3DRenderTs = 0;
}

// WebSocket Connection Lifecycle
function connectStream() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${proto}://${window.location.host}/ws/stream`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    const dot = document.getElementById("ws-status-dot");
    if (!streamPaused) {
      dot.style.background = "#10b981";
    }
  };

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      renderDashboard(payload);
    } catch (e) {
      console.error("Failed to parse stream payload:", e);
    }
  };

  ws.onclose = () => {
    const dot = document.getElementById("ws-status-dot");
    dot.style.background = "#f43f5e";
    setTimeout(connectStream, 1500);
  };
}

// Guided Popup Tour & Interactive Walkthrough Engine
const TOUR_STEPS = [
  {
    badge: "STEP 1 OF 5 // BASKET vs. BOX PARITY",
    title: "1. Real-Time QQQ vs. Constituent Stock Box Tracker",
    body: "QQQ is an ETF holding Nasdaq-100 stocks. Panel 01 continuously sums the weighted live prices of NVDA (24.8%), AAPL (24.1%), MSFT (22.6%), AMZN (15.0%), and META (13.5%) to compute the Synthetic Stock Box NAV ($500.00) and compares it tick-by-tick against the QQQ market price.",
    formula: "Synthetic Box ($) = Σ (Shares_i × Price_i) | Spread ($) = QQQ − Box",
    panelId: "panel-tracker",
    shock: null,
  },
  {
    badge: "STEP 2 OF 5 // JANE STREET $0.45 ARBITRAGE SCENARIO",
    title: "2. Locking in the $500.45 QQQ vs. $500.00 Stock Box Gap",
    body: "During fast market sweeps, QQQ surges to $500.45 while its underlying constituent stocks only sum to $500.00. Market makers instantly SELL SHORT the overpriced QQQ ETF at $500.45 and BUY the cheaper individual stocks at $500.00 to lock in the +$0.45/share spread.",
    formula: "QQQ ($500.45) − Stock Box ($500.00) = +$0.4500/sh (+9.00 bps Gross Edge)",
    panelId: "panel-tracker",
    shock: "JANE_STREET_500_45",
  },
  {
    badge: "STEP 3 OF 5 // PYTORCH MILLISECOND LEAD-LAG PREDICTOR",
    title: "3. Detecting NVDA Breakouts 100ms Before QQQ Reacts",
    body: "Our PyTorch TemporalCrossAttentionNet (Conv1D + GRU + Cross-Lag Attention) monitors 20 lookback ticks across all 5 stocks. When NVDA jumps +1.85%, the neural network detects the positive cross-correlation peak (τ* = 100ms head start) and predicts ΔQQQ before the ETF catches up.",
    formula: "PyTorch Attention: α_τ = Softmax(Q_t · K_{t-τ}^T / √d) → Predicts ΔQQQ (+200ms)",
    panelId: "panel-pytorch",
    shock: "NVDA_BREAKOUT",
  },
  {
    badge: "STEP 4 OF 5 // 3D ARBITRAGE PRICE GAP SURFACE",
    title: "4. WebGL 3D Price Gap Topology (Time × Lag τ & NVDA Shocks)",
    body: "Panel 03 renders the live 3D arbitrage basis surface Z = QQQ − Box ($). Amber/Crimson ridges (Z > +$0.12) reveal overpriced QQQ conversion opportunities, while Emerald troughs (Z < -$0.12) reveal underpriced QQQ reversal opportunities across reaction lag horizons.",
    formula: "Surface Z(t, τ) = Gap(t)·e^(-τ/180ms) + Neural_Lead_Wave(τ)",
    panelId: "panel-surface3d",
    shock: null,
  },
  {
    badge: "STEP 5 OF 5 // LIVE ARBITRAGE TRADE TICKET & FEE WATERFALL",
    title: "5. Multi-Leg Basket Execution & Net Profit After Fees",
    body: "Panel 04 calculates the exact share quantities to trade for QQQ and all 5 constituent stocks, deducts institutional DMA midpoint slippage (-0.28 bps), SEC/TAF/exchange fees (-0.09 bps), and borrow costs (-0.04 bps), and logs atomic multi-leg fills in the Realized PnL Blotter.",
    formula: "Net Profit ($) = Gross_Box_Spread + PyTorch_Lead_Alpha − (Slippage + SEC_TAF + Borrow)",
    panelId: "panel-ticket",
    shock: "JANE_STREET_500_45",
  },
];

let currentTourStep = 0;
let tourAutoplayTimer = null;

function clearPanelSpotlights() {
  ["panel-tracker", "panel-pytorch", "panel-surface3d", "panel-ticket"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("panel-tour-spotlight");
  });
}

async function startGuidedTour(stepIndex = 0, autoplay = false) {
  currentTourStep = ((stepIndex % TOUR_STEPS.length) + TOUR_STEPS.length) % TOUR_STEPS.length;
  const step = TOUR_STEPS[currentTourStep];

  const popup = document.getElementById("tour-popup-card");
  popup.classList.remove("hidden");

  document.getElementById("tour-step-badge").textContent = step.badge;
  document.getElementById("tour-popup-title").textContent = step.title;
  document.getElementById("tour-popup-body").textContent = step.body;
  document.getElementById("tour-popup-formula").textContent = step.formula;

  const dots = document.querySelectorAll("#tour-dots-container .tour-dot");
  dots.forEach((d, idx) => {
    d.className = idx === currentTourStep ? "tour-dot active" : "tour-dot";
  });

  clearPanelSpotlights();
  const targetPanel = document.getElementById(step.panelId);
  if (targetPanel) {
    targetPanel.classList.add("panel-tour-spotlight");
  }

  if (step.shock) {
    await triggerShock(step.shock);
  }

  if (autoplay) {
    startTourAutoplayTimer();
  }
}

function nextTourStep() {
  startGuidedTour(currentTourStep + 1, Boolean(tourAutoplayTimer));
}

function prevTourStep() {
  startGuidedTour(currentTourStep - 1, Boolean(tourAutoplayTimer));
}

function startTourAutoplayTimer() {
  stopTourAutoplayTimer();
  const btn = document.getElementById("btn-tour-autoplay");
  if (btn) {
    btn.textContent = "⏸ Playing Tour";
    btn.className = "btn btn-xs btn-emerald";
  }
  tourAutoplayTimer = setInterval(() => {
    currentTourStep = (currentTourStep + 1) % TOUR_STEPS.length;
    startGuidedTour(currentTourStep, true);
  }, 4200);
}

function stopTourAutoplayTimer() {
  if (tourAutoplayTimer) {
    clearInterval(tourAutoplayTimer);
    tourAutoplayTimer = null;
  }
  const btn = document.getElementById("btn-tour-autoplay");
  if (btn) {
    btn.textContent = "▶ Auto-Play";
    btn.className = "btn btn-xs btn-amber";
  }
}

function toggleTourAutoplay() {
  if (tourAutoplayTimer) {
    stopTourAutoplayTimer();
  } else {
    startTourAutoplayTimer();
  }
}

function closeGuidedTour() {
  stopTourAutoplayTimer();
  clearPanelSpotlights();
  const popup = document.getElementById("tour-popup-card");
  if (popup) popup.classList.add("hidden");
}

// Flowcharts, Video Walkthrough & Annotated Screenshots Modal
function openDocsModal(defaultTab = "video") {
  const modal = document.getElementById("docs-modal");
  if (modal) modal.classList.remove("hidden");
  switchDocsTab(defaultTab);
}

function closeDocsModal() {
  const modal = document.getElementById("docs-modal");
  if (modal) modal.classList.add("hidden");
}

function switchDocsTab(tabName) {
  ["video", "flowcharts", "screenshots"].forEach((t) => {
    const pane = document.getElementById(`docs-tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (pane) pane.className = t === tabName ? "docs-tab-pane" : "docs-tab-pane hidden";
    if (btn) btn.className = t === tabName ? "docs-tab active" : "docs-tab";
  });
}

function seekVideoChapter(seconds) {
  const vid = document.getElementById("walkthrough-video-el");
  if (vid) {
    vid.currentTime = seconds;
    vid.play().catch(() => {});
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  const params = new URLSearchParams(window.location.search);
  try {
    const res = await fetch("/api/snapshot");
    const snap = await res.json();
    renderDashboard(snap);
  } catch (e) {
    console.warn("Initial snapshot fetch skipped:", e);
  }
  if (!params.has("snapshot_only")) {
    connectStream();
  }

  // Support URL query parameters (?tour_step=1..5 or ?modal=flowcharts)
  if (params.has("tour_step")) {
    const stepNum = parseInt(params.get("tour_step"), 10) - 1;
    setTimeout(() => startGuidedTour(Math.max(0, stepNum), false), 200);
  } else if (params.has("modal")) {
    setTimeout(() => openDocsModal(params.get("modal") || "video"), 200);
  }
});

