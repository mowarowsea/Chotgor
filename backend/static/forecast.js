/*
 * 予報パネルのチャート描画（vanilla JS・依存なし）。
 *
 * #forecast-data の JSON（services/timeline/forecast.build_forecast の出力）を読み、
 *   1. 週間カレンダー（日行×24h帯 — 予定・伏せ枠・シーン・行動権スロット）
 *   2. 圧力予報（3系列ライン・クロスヘア tooltip）
 *   3. 意図圧予報（≤4系列＋閾値破線＋問い合わせ予報点★）
 *   4. 発火散布（日×時刻ドット）
 *   5. リズム波形（単系列・now マーカー）
 * を SVG / DOM で組み立てる。系列色は CSS トークン --ch-viz-1〜4 を参照し、
 * ライト/ダークのテーマ切替に自動追従する。
 */
(function () {
  "use strict";

  var dataEl = document.getElementById("forecast-data");
  if (!dataEl) return;
  var DATA = JSON.parse(dataEl.textContent);
  var NOW = new Date(DATA.now);

  /* ---- 共通ヘルパ ---- */

  // ISO（ローカル naive）→ Date
  function d(iso) { return new Date(iso); }

  // "MM/DD HH:MM" 表示
  function fmt(date) {
    function z(n) { return (n < 10 ? "0" : "") + n; }
    return z(date.getMonth() + 1) + "/" + z(date.getDate()) + " " + z(date.getHours()) + ":" + z(date.getMinutes());
  }

  // CSS 変数参照（SVG の属性値に var() を書けない場所用）
  function cssVar(name) {
    return "var(" + name + ")";
  }

  // SVG 要素生成
  function svgEl(tag, attrs) {
    var el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // 一日の中の位置（0〜1）。24h 帯のレイアウトに使う
  function dayFrac(date) {
    return (date.getHours() * 60 + date.getMinutes()) / (24 * 60);
  }

  /* ---- 1. 週間カレンダー ---- */

  var STATE_COLOR = {
    OnTime: "--ch-viz-4",
    active: "--ch-viz-1",
    busy: "--ch-viz-2",
    offline: "--ch-t4",
  };

  function renderCalendar() {
    var host = document.getElementById("viz-calendar");
    if (!host) return;
    var cal = DATA.calendar;
    var wrap = document.createElement("div");
    wrap.className = "ch-viz-cal";

    cal.days.forEach(function (dayIso, di) {
      var dayStart = new Date(dayIso + "T00:00:00");
      var dayEnd = new Date(dayStart.getTime() + 86400000);
      var row = document.createElement("div");
      row.className = "ch-viz-cal__row";
      var label = document.createElement("div");
      label.className = "ch-viz-cal__label";
      var youbi = ["日", "月", "火", "水", "木", "金", "土"][dayStart.getDay()];
      label.textContent = (dayStart.getMonth() + 1) + "/" + dayStart.getDate() + " " + youbi;
      var strip = document.createElement("div");
      strip.className = "ch-viz-cal__strip";

      // 6時間ごとのグリッド線（最初の行だけ時刻ラベルも置く）
      for (var h = 0; h <= 24; h += 6) {
        var line = document.createElement("div");
        line.className = "ch-viz-cal__grid";
        line.style.left = (h / 24 * 100) + "%";
        strip.appendChild(line);
        if (di === 0 && h < 24) {
          var hl = document.createElement("span");
          hl.className = "ch-viz-cal__hour";
          hl.style.left = (h / 24 * 100) + "%";
          hl.textContent = h + "時";
          strip.appendChild(hl);
        }
      }

      // 予定エントリ帯（この日と重なるものをクリップして描く）
      cal.entries.forEach(function (e) {
        var s = d(e.start_at), t = d(e.end_at);
        if (t <= dayStart || s >= dayEnd) return;
        var left = Math.max(0, (Math.max(s, dayStart) - dayStart) / 86400000);
        var right = Math.min(1, (Math.min(t, dayEnd) - dayStart) / 86400000);
        var bar = document.createElement("div");
        bar.className = "ch-viz-cal__entry" + (e.is_seed ? " ch-viz-cal__entry--seed" : "");
        bar.style.left = (left * 100) + "%";
        bar.style.width = (Math.max(0.004, right - left) * 100) + "%";
        if (!e.is_seed) {
          bar.style.background = cssVar(STATE_COLOR[e.state] || "--ch-t4");
          bar.style.opacity = String(0.35 + 0.55 * (e.occupancy || 0));
        }
        if (e.status === "cancelled") bar.style.opacity = "0.15";
        bar.title = (e.is_seed ? "③伏せ枠: " : "") + (e.label || "(無題)") +
          "\n" + fmt(s) + " – " + fmt(t) +
          "\nstate=" + e.state + " 占有圧=" + e.occupancy +
          " " + e.source + "/" + e.origin + " [" + e.status + "]";
        strip.appendChild(bar);
      });

      // うつつシーン起動予定（●）
      cal.scenes.forEach(function (sc) {
        var at = d(sc.fire_at);
        if (at < dayStart || at >= dayEnd) return;
        var dot = document.createElement("div");
        dot.className = "ch-viz-cal__scene";
        dot.style.left = (dayFrac(at) * 100) + "%";
        dot.title = "うつつシーン予定: " + sc.label + "\n起動 " + fmt(at);
        strip.appendChild(dot);
      });

      // 行動権評価スロット（下段の目盛）
      cal.action_slots.forEach(function (sl) {
        var at = d(sl.eval_at);
        if (at < dayStart || at >= dayEnd) return;
        var tick = document.createElement("span");
        tick.className = "ch-viz-cal__tick";
        tick.style.left = (dayFrac(at) * 100) + "%";
        var sym = { fires: "▲", quiet: "△", unavailable: "×", past: "·" }[sl.forecast] || "·";
        var col = {
          fires: "--ch-viz-4", quiet: "--ch-t3",
          unavailable: "--ch-danger", past: "--ch-t4",
        }[sl.forecast] || "--ch-t4";
        tick.textContent = sym;
        tick.style.color = cssVar(col);
        tick.title = "行動権評価 " + fmt(at) + "\n予報: " + sl.forecast;
        strip.appendChild(tick);
      });

      // now マーカー（今日の行のみ）
      if (NOW >= dayStart && NOW < dayEnd) {
        var nowLine = document.createElement("div");
        nowLine.className = "ch-viz-cal__now";
        nowLine.style.left = (dayFrac(NOW) * 100) + "%";
        nowLine.title = "いま " + fmt(NOW);
        strip.appendChild(nowLine);
      }

      row.appendChild(label);
      row.appendChild(strip);
      wrap.appendChild(row);
    });
    host.appendChild(wrap);
  }

  /* ---- 2/3. ラインチャート（クロスヘア tooltip 付き） ---- */

  // opts: { grid: [Date], series: [{label, colorVar, values}], threshold, stars: [{at, label}] }
  function lineChart(hostId, opts) {
    var host = document.getElementById(hostId);
    if (!host) return;
    var W = 860, H = 230, padL = 34, padR = 10, padT = 10, padB = 22;
    var grid = opts.grid;
    var x0 = grid[0].getTime(), x1 = grid[grid.length - 1].getTime();
    function X(t) { return padL + (t - x0) / (x1 - x0) * (W - padL - padR); }
    function Y(v) { return padT + (1 - v) * (H - padT - padB); }

    var figure = document.createElement("div");
    figure.className = "ch-viz-figure";

    // 凡例（2系列以上で常設）
    if (opts.series.length >= 2) {
      var legend = document.createElement("div");
      legend.className = "ch-viz-legend";
      opts.series.forEach(function (s) {
        var item = document.createElement("span");
        var chip = document.createElement("span");
        chip.className = "ch-viz-chip";
        chip.style.background = cssVar(s.colorVar);
        chip.style.marginLeft = "0";
        item.appendChild(chip);
        item.appendChild(document.createTextNode(" " + s.label));
        legend.appendChild(item);
      });
      figure.appendChild(legend);
    }

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: "100%" });
    svg.style.display = "block";

    // 控えめな水平グリッドと y ラベル
    [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
      svg.appendChild(svgEl("line", {
        x1: padL, y1: Y(v), x2: W - padR, y2: Y(v),
        stroke: cssVar("--ch-sep"), "stroke-width": 1,
      }));
      var lab = svgEl("text", {
        x: padL - 6, y: Y(v) + 3.5, "text-anchor": "end",
        "font-size": 10, fill: cssVar("--ch-t3"), "font-family": "var(--ch-fm)",
      });
      lab.textContent = v.toFixed(2).replace(/\.?0+$/, "") || "0";
      svg.appendChild(lab);
    });

    // x 目盛（12時間ごと）
    var tickMs = 12 * 3600 * 1000;
    var firstTick = Math.ceil(x0 / tickMs) * tickMs;
    for (var t = firstTick; t <= x1; t += tickMs) {
      var td = new Date(t);
      svg.appendChild(svgEl("line", {
        x1: X(t), y1: H - padB, x2: X(t), y2: H - padB + 3,
        stroke: cssVar("--ch-t4"), "stroke-width": 1,
      }));
      var xl = svgEl("text", {
        x: X(t), y: H - padB + 14, "text-anchor": "middle",
        "font-size": 10, fill: cssVar("--ch-t3"), "font-family": "var(--ch-fm)",
      });
      xl.textContent = (td.getMonth() + 1) + "/" + td.getDate() + (td.getHours() === 0 ? "" : " " + td.getHours() + "時");
      svg.appendChild(xl);
    }

    // now マーカー（グリッドが過去を含むとき）
    if (NOW.getTime() > x0 && NOW.getTime() < x1) {
      svg.appendChild(svgEl("line", {
        x1: X(NOW.getTime()), y1: padT, x2: X(NOW.getTime()), y2: H - padB,
        stroke: cssVar("--ch-danger"), "stroke-width": 1.5, "stroke-dasharray": "2 3",
      }));
    }

    // 閾値の破線
    if (opts.threshold != null) {
      svg.appendChild(svgEl("line", {
        x1: padL, y1: Y(opts.threshold), x2: W - padR, y2: Y(opts.threshold),
        stroke: cssVar("--ch-t2"), "stroke-width": 1.5, "stroke-dasharray": "5 4",
      }));
      var thLab = svgEl("text", {
        x: W - padR, y: Y(opts.threshold) - 4, "text-anchor": "end",
        "font-size": 10, fill: cssVar("--ch-t2"), "font-family": "var(--ch-fm)",
      });
      thLab.textContent = "閾値 " + opts.threshold;
      svg.appendChild(thLab);
    }

    // 系列（2px ライン＋末端の直接ラベル）
    opts.series.forEach(function (s) {
      var path = "";
      s.values.forEach(function (v, i) {
        path += (i === 0 ? "M" : "L") + X(grid[i].getTime()).toFixed(1) + " " + Y(v).toFixed(1);
      });
      svg.appendChild(svgEl("path", {
        d: path, fill: "none", stroke: cssVar(s.colorVar),
        "stroke-width": 2, "stroke-linejoin": "round",
      }));
    });

    // 問い合わせ予報点（★）
    (opts.stars || []).forEach(function (st) {
      var star = svgEl("text", {
        x: X(st.at.getTime()), y: Y(opts.threshold != null ? opts.threshold : 1) - 8,
        "text-anchor": "middle", "font-size": 13, fill: cssVar("--ch-viz-4"),
      });
      star.textContent = "★";
      var title = svgEl("title", {});
      title.textContent = "問い合わせ予報 " + fmt(st.at) + "\n" + st.label;
      star.appendChild(title);
      svg.appendChild(star);
    });

    // クロスヘア＋ツールチップ
    var cross = svgEl("line", {
      x1: 0, y1: padT, x2: 0, y2: H - padB,
      stroke: cssVar("--ch-t3"), "stroke-width": 1, visibility: "hidden",
    });
    svg.appendChild(cross);
    var tip = document.createElement("div");
    tip.className = "ch-viz-tip";
    figure.appendChild(tip);

    svg.addEventListener("mousemove", function (ev) {
      var rect = svg.getBoundingClientRect();
      var mx = (ev.clientX - rect.left) / rect.width * W;
      var ti = Math.round((mx - padL) / (W - padL - padR) * (grid.length - 1));
      if (ti < 0 || ti >= grid.length) { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; return; }
      var gx = X(grid[ti].getTime());
      cross.setAttribute("x1", gx); cross.setAttribute("x2", gx);
      cross.setAttribute("visibility", "visible");
      var html = "<strong>" + fmt(grid[ti]) + "</strong>";
      opts.series.forEach(function (s) {
        html += "<br><span class=\"ch-viz-chip\" style=\"background:" + cssVar(s.colorVar) + ";margin-left:0\"></span> "
          + s.label + ": " + s.values[ti].toFixed(2);
      });
      tip.innerHTML = html;
      tip.style.display = "block";
      var px = gx / W * rect.width;
      tip.style.left = Math.min(px + 12, rect.width - tip.offsetWidth - 4) + "px";
      tip.style.top = "28px";
    });
    svg.addEventListener("mouseleave", function () {
      cross.setAttribute("visibility", "hidden");
      tip.style.display = "none";
    });

    figure.appendChild(svg);
    host.appendChild(figure);
  }

  /* ---- 4. 発火散布（日×時刻） ---- */

  var SCHED_COLOR = {
    action: "--ch-viz-4",
    usual_days: "--ch-viz-2",
    sudden_event: "--ch-viz-3",
    escrow_delivery: "--ch-viz-1",
    weekly_schedule: "--ch-t3",
  };

  function renderScatter() {
    var host = document.getElementById("viz-scatter");
    if (!host) return;
    var days = 14;
    var W = 400, H = 200, padL = 44, padR = 8, padT = 6, padB = 18;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: "100%" });
    svg.style.display = "block";

    var today = new Date(NOW.getFullYear(), NOW.getMonth(), NOW.getDate());
    function rowY(dayOffset) { // 0=今日（最下段）
      return padT + (days - 1 - dayOffset) / (days - 1) * (H - padT - padB - 8) + 4;
    }
    function colX(hourFrac) {
      return padL + hourFrac / 24 * (W - padL - padR);
    }

    // x 目盛（6時間ごと）と行ラベル（1日おき）
    for (var h = 0; h <= 24; h += 6) {
      svg.appendChild(svgEl("line", {
        x1: colX(h), y1: padT, x2: colX(h), y2: H - padB,
        stroke: cssVar("--ch-sep"), "stroke-width": 1,
      }));
      var xl = svgEl("text", {
        x: colX(h), y: H - 5, "text-anchor": "middle",
        "font-size": 9, fill: cssVar("--ch-t3"), "font-family": "var(--ch-fm)",
      });
      xl.textContent = h + "時";
      svg.appendChild(xl);
    }
    for (var i = 0; i < days; i += 2) {
      var day = new Date(today.getTime() - i * 86400000);
      var yl = svgEl("text", {
        x: padL - 6, y: rowY(i) + 3, "text-anchor": "end",
        "font-size": 9, fill: cssVar("--ch-t3"), "font-family": "var(--ch-fm)",
      });
      yl.textContent = (day.getMonth() + 1) + "/" + day.getDate();
      svg.appendChild(yl);
    }

    DATA.variance.fired_scatter.forEach(function (f) {
      var at = d(f.at);
      var offset = Math.floor((today - new Date(at.getFullYear(), at.getMonth(), at.getDate())) / 86400000);
      if (offset < 0 || offset >= days) return;
      var dot = svgEl("circle", {
        cx: colX(at.getHours() + at.getMinutes() / 60), cy: rowY(offset),
        r: 4, fill: cssVar(SCHED_COLOR[f.scheduler] || "--ch-t3"),
        stroke: cssVar("--ch-bg"), "stroke-width": 1.5,
      });
      var title = svgEl("title", {});
      title.textContent = f.scheduler + " " + fmt(at);
      dot.appendChild(title);
      svg.appendChild(dot);
    });

    if (!DATA.variance.fired_scatter.length) {
      var empty = svgEl("text", {
        x: W / 2, y: H / 2, "text-anchor": "middle",
        "font-size": 11, fill: cssVar("--ch-t3"), "font-family": "var(--ch-fm)",
      });
      empty.textContent = "直近14日の発火実績はまだありません";
      svg.appendChild(empty);
    }
    host.appendChild(svg);
  }

  /* ---- 5. リズム波形 ---- */

  function renderRhythm() {
    var host = document.getElementById("viz-rhythm");
    if (!host) return;
    var r = DATA.variance.rhythm;
    var grid = r.grid.map(d);
    // 振幅が小さいので [min,max] を少し広げた帯で描く（0-1 固定だと平坦に見える）
    var lo = Math.min.apply(null, r.values), hi = Math.max.apply(null, r.values);
    var span = Math.max(0.05, hi - lo);
    var norm = r.values.map(function (v) { return (v - (lo - span * 0.1)) / (span * 1.2); });
    lineChart("viz-rhythm", {
      grid: grid,
      series: [{ label: "リズム成分", colorVar: "--ch-viz-3", values: norm }],
    });
    var note = document.createElement("div");
    note.className = "ch-hint";
    note.textContent = "実レンジ " + lo.toFixed(3) + " 〜 " + hi.toFixed(3) + "（表示は正規化）";
    host.appendChild(note);
  }

  /* ---- 組み立て ---- */

  renderCalendar();

  var pf = DATA.pressure_forecast;
  var pfGrid = pf.grid.map(d);
  lineChart("viz-pressure", {
    grid: pfGrid,
    series: [
      { label: "社会圧", colorVar: "--ch-viz-1", values: pf.social },
      { label: "退屈圧", colorVar: "--ch-viz-2", values: pf.boredom },
      { label: "体調圧", colorVar: "--ch-viz-3", values: pf.body },
    ],
  });

  if (pf.intents.length) {
    // 色は意図の同一性に固定する（表示順＝圧の高い順とは独立。ランクで塗り替えない）
    var idsSorted = pf.intents.map(function (it) { return it.intent_id; }).sort();
    lineChart("viz-intent", {
      grid: pfGrid,
      series: pf.intents.map(function (it) {
        var ci = (idsSorted.indexOf(it.intent_id) % 4) + 1;
        return {
          label: it.description.length > 24 ? it.description.slice(0, 24) + "…" : it.description,
          colorVar: "--ch-viz-" + ci,
          values: it.series,
        };
      }),
      threshold: pf.threshold,
      stars: pf.fire_points.map(function (fp) {
        return { at: d(fp.at), label: fp.descriptions.join(" / ") };
      }),
    });
  }

  renderScatter();
  renderRhythm();
})();
