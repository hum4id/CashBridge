/* CashBridge UI helpers: forecast chart, ask bar, what-if simulations. */
(function () {
  "use strict";

  /* Palette (closed set — dark-adjusted tokens from style.css). */
  var BLUE = "#8AB4F8";
  var GAIN = "#81C995";
  var LOSS = "#F28B82";
  var INK2 = "rgba(255,255,255,0.60)";

  function fmtIDR(value) {
    var n = Math.round(Number(value) || 0);
    return "IDR " + n.toLocaleString("en-US");
  }

  function esc(text) {
    var div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function chip(label, kind) {
    return '<span class="chip ' + (kind || "") + '">' + esc(label) + "</span>";
  }

  function skeleton(lines) {
    var widths = ["w80", "w100", "w60"];
    var html = "";
    for (var i = 0; i < lines; i++) {
      html += '<div class="skeleton ' + widths[i % widths.length] + '"></div>';
    }
    return html;
  }

  /* ------------------------------------------------------------- chart */
  function initForecastChart() {
    var canvas = document.getElementById("forecastChart");
    var data = chartData();
    if (!canvas || !data) return;
    if (typeof Chart === "undefined") {
      // Offline fallback: show the minimum balances as text, no chart.
      var box = canvas.parentNode;
      box.innerHTML = "<p class='muted small'>Chart library not reachable offline. " +
        "Lowest projected (low): <strong>" + esc(fmtIDR(Math.min.apply(null, data.low))) + "</strong>, " +
        "expected: <strong>" + esc(fmtIDR(Math.min.apply(null, data.mid))) + "</strong>, " +
        "high: <strong>" + esc(fmtIDR(Math.min.apply(null, data.high))) + "</strong>.</p>";
      return;
    }
    Chart.defaults.font.family =
      "'Roboto',system-ui,sans-serif";
    Chart.defaults.color = INK2;
    new Chart(canvas, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          { label: "Best likely (P90)", data: data.high, borderColor: GAIN, borderWidth: 1.5,
            fill: { target: 2, above: "rgba(255,255,255,0.05)" }, tension: 0.3, pointRadius: 0 },
          { label: "Most likely (median)", data: data.mid, borderColor: BLUE, borderWidth: 2.5,
            fill: false, tension: 0.3, pointRadius: 0 },
          { label: "Worst likely (P10)", data: data.low, borderColor: LOSS, borderWidth: 1.5,
            fill: false, tension: 0.3, pointRadius: 0 },
          { label: "Safety threshold", data: data.labels.map(function () { return data.safety; }),
            borderColor: INK2, borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 14, color: INK2 } } },
        scales: {
          y: { ticks: { color: INK2, callback: function (v) { return Math.round(v / 1e6) + "M"; } },
               grid: { color: "rgba(255,255,255,0.08)" } },
          x: { ticks: { color: INK2 }, grid: { display: false } }
        }
      }
    });
  }

  /* --------------------------------------------------------------- ask */
  function renderAnswer(payload) {
    var answer = payload.answer || {};
    var ok = payload.validation && payload.validation.ok;
    var html = '<div class="answer">';
    html += "<div>" + chip("engine: " + payload.engine) + " " +
      (ok ? chip("grounded in computed numbers", "gain") : chip("grounding issues", "loss")) + "</div>";
    if (payload.note) html += "<p class='muted small'>" + esc(payload.note) + "</p>";
    html += "<p class='lead'>" + esc(answer.summary) + "</p>";
    html += "<p>" + esc(answer.why) + "</p>";
    if (answer.next_steps && answer.next_steps.length) {
      html += "<ol class='steps'>";
      answer.next_steps.forEach(function (step) { html += "<li>" + esc(step) + "</li>"; });
      html += "</ol>";
    }
    if (answer.key_figures && answer.key_figures.length) {
      html += "<table class='figures'><tr><th>Figure</th><th>Value</th><th>Source</th></tr>";
      answer.key_figures.forEach(function (figure) {
        html += "<tr><td>" + esc(figure.label) + "</td><td>" + esc(fmtIDR(figure.value)) +
          "</td><td class='muted'>" + esc(figure.source) + "</td></tr>";
      });
      html += "</table>";
    }
    if (payload.label && payload.label.text) {
      html += "<p class='ai-label' data-ai-generated='" + (payload.label.ai_generated ? "true" : "false") +
        "'>" + esc(payload.label.text) + "</p>";
    }
    if (ok === false && payload.validation.issues && payload.validation.issues.length) {
      html += "<p class='danger-text small'>" + esc(payload.validation.issues[0]) + "</p>";
    }
    html += "</div>";
    return html;
  }

  function wireAsk() {
    var form = document.getElementById("ask-form");
    var out = document.getElementById("ask-result");
    var panel = document.getElementById("ask-panel");
    if (!form || !out) return;
    var input = form.question;
    var send = document.getElementById("ask-send");
    var close = document.getElementById("ask-close");

    function syncSend() {
      var has = input.value.trim().length > 0;
      if (send) {
        send.disabled = !has;
        send.classList.toggle("enabled", has);
      }
    }
    input.addEventListener("input", syncSend);
    syncSend();
    if (close && panel) {
      close.addEventListener("click", function () { panel.hidden = true; });
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var question = input.value.trim();
      if (!question) return;
      if (panel) panel.hidden = false;
      out.innerHTML = skeleton(3);
      fetch(form.dataset.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken() },
        body: JSON.stringify({ question: question })
      }).then(function (response) {
        return response.json().then(function (data) { return { status: response.status, data: data }; });
      }).then(function (result) {
        if (result.status !== 200) {
          var detail = (result.data && result.data.detail) || {};
          out.innerHTML = "<p class='danger-text'>" + esc(detail.reason || "Request failed") + "</p>";
          if (detail.code) out.innerHTML = "<p class='danger-text small'>" + esc(detail.code) + "</p>" + out.innerHTML;
          return;
        }
        out.innerHTML = renderAnswer(result.data);
      }).catch(function (error) {
        out.innerHTML = "<p class='danger-text'>" + esc(String(error)) + "</p>";
      });
    });
  }

  function chartData() {
    var el = document.getElementById("chart-data");
    try { return el ? JSON.parse(el.textContent) : null; } catch (e) { return null; }
  }

  function csrfToken() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  /* ----------------------------------------------------------- simulate */
  function adjustmentsFor(form) {
    var preset = form.preset.value;
    if (preset === "accelerate") return { payout_shift_days: -3 };
    if (preset === "restock") return { inventory_delay_days: 7 };
    if (preset === "cut") return { category_factors: { other_opex: 0.6 } };
    var days = Math.min(14, Math.max(1, parseInt(form.shift.value, 10) || 3));
    return { payout_shift_days: -days };
  }

  function wireSimulate() {
    var form = document.getElementById("simulate-form");
    var out = document.getElementById("simulate-result");
    if (!form || !out) return;
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      out.innerHTML = skeleton(2);
      fetch(form.dataset.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken() },
        body: JSON.stringify({ adjustments: adjustmentsFor(form) })
      }).then(function (response) {
        return response.json().then(function (data) { return { status: response.status, data: data }; });
      }).then(function (result) {
        if (result.status !== 200) {
          var detail = (result.data && result.data.detail) || {};
          out.innerHTML = "<p class='danger-text'>" + esc(detail.reason || "Simulation failed") + "</p>";
          return;
        }
        var d = result.data;
        var badge = d.clears_safety ? chip("clears the safety line", "gain") : chip("still below safety", "loss-soft");
        out.innerHTML = "<div class='answer'>" +
          "<p class='simline'>Lowest projected balance: <span class='muted'>before</span> " +
          esc(fmtIDR(d.baseline_min_low)) + " &rarr; <span class='simbig'>" + esc(fmtIDR(d.simulated_min_low)) + "</span></p>" +
          "<p class='simline'>Improvement: <strong>" + esc(fmtIDR(d.improvement)) + "</strong> &nbsp; " + badge + "</p>" +
          "<p class='muted small'>What-if simulation only, not stored as a forecast. " +
          "Applied: <code>" + esc(JSON.stringify(d.applied_adjustments)) + "</code></p></div>";
      }).catch(function (error) {
        out.innerHTML = "<p class='danger-text'>" + esc(String(error)) + "</p>";
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initForecastChart();
    wireAsk();
    wireSimulate();
  });
})();
