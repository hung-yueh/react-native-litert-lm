/* litert·lm docs — highlighting, copy buttons, mobile nav */
(function () {
  "use strict";

  /* ── Tiny TypeScript/shell highlighter ─────────────────────────────── */
  var KEYWORDS =
    /\b(import|from|export|const|let|var|function|return|await|async|if|else|switch|case|break|new|try|catch|finally|throw|typeof|interface|type|extends|implements|readonly|class|for|of|in|while|default|undefined|null|true|false|void|as)\b/g;
  var TYPES =
    /\b(string|number|boolean|Promise|ArrayBuffer|Float64Array|Record|Array|Error|LLMConfig|ExecuteOptions|ThinkingOptions|MultimodalPart|StreamEvent|StreamEventCallback|StreamChannel|StreamEventParser|LiteRTLMInstance|CreateLLMOptions|ConversationHandle|ConversationOptions|MemoryEstimate|MemoryEstimateInputs|MemoryForecast|MemoryForecastInputs|MemoryBudget|MemoryUsage|MemorySnapshot|MemoryTracker|MemoryTrackerSummary|MemoryWarningLevel|MemoryVerdict|BudgetLevel|GenerationStats|ToolDefinition|Message|Backend|Role|PartType|ActivationDataType|ModelDownloadOptions|ModelFile|UseModelConfig|UseModelResult|LiteRTLMError|MemoryError|LiteRTLMErrorCode|TokenCallback)\b/g;

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlight(src) {
    var out = "";
    var i = 0;
    var n = src.length;
    while (i < n) {
      var ch = src[i];
      // line comment
      if (ch === "/" && src[i + 1] === "/") {
        var e = src.indexOf("\n", i);
        if (e === -1) e = n;
        out += '<span class="tok-com">' + esc(src.slice(i, e)) + "</span>";
        i = e;
        continue;
      }
      // block comment
      if (ch === "/" && src[i + 1] === "*") {
        var e2 = src.indexOf("*/", i + 2);
        e2 = e2 === -1 ? n : e2 + 2;
        out += '<span class="tok-com">' + esc(src.slice(i, e2)) + "</span>";
        i = e2;
        continue;
      }
      // shell comment (only when line starts with #)
      if (ch === "#" && (i === 0 || src[i - 1] === "\n")) {
        var e3 = src.indexOf("\n", i);
        if (e3 === -1) e3 = n;
        out += '<span class="tok-com">' + esc(src.slice(i, e3)) + "</span>";
        i = e3;
        continue;
      }
      // strings
      if (ch === '"' || ch === "'" || ch === "`") {
        var q = ch;
        var j = i + 1;
        while (j < n && src[j] !== q) {
          if (src[j] === "\\") j++;
          j++;
        }
        j = Math.min(j + 1, n);
        out += '<span class="tok-str">' + esc(src.slice(i, j)) + "</span>";
        i = j;
        continue;
      }
      // identifiers / numbers / other — accumulate until next special char
      var k = i;
      while (k < n && !'"\'`'.includes(src[k]) && !(src[k] === "/" && (src[k + 1] === "/" || src[k + 1] === "*")) && !(src[k] === "#" && (k === 0 || src[k - 1] === "\n"))) {
        k++;
      }
      var chunk = esc(src.slice(i, k));
      chunk = chunk.replace(/\b(0x[\da-fA-F]+|\d[\d_]*(\.\d+)?)\b/g, '<span class="tok-num">$1</span>');
      chunk = chunk.replace(KEYWORDS, '<span class="tok-kw">$1</span>');
      chunk = chunk.replace(TYPES, '<span class="tok-ty">$1</span>');
      chunk = chunk.replace(/([A-Za-z_$][\w$]*)(\()/g, '<span class="tok-fn">$1</span>$2');
      out += chunk;
      i = k;
    }
    return out;
  }

  document.querySelectorAll(".code pre code").forEach(function (block) {
    if (block.dataset.plain !== undefined) return;
    block.innerHTML = highlight(block.textContent);
  });

  /* ── Copy buttons ──────────────────────────────────────────────────── */
  document.querySelectorAll(".code").forEach(function (box) {
    var pre = box.querySelector("pre");
    if (!pre) return;
    var head = box.querySelector(".code-head");
    if (!head) {
      head = document.createElement("div");
      head.className = "code-head";
      head.innerHTML = "<span></span>";
      box.insertBefore(head, pre);
    }
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "copy";
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.textContent).then(function () {
        btn.textContent = "copied";
        setTimeout(function () { btn.textContent = "copy"; }, 1400);
      });
    });
    head.appendChild(btn);
  });

  /* ── Heading anchors ───────────────────────────────────────────────── */
  document.querySelectorAll(".doc h2[id], .doc h3[id]").forEach(function (h) {
    var a = document.createElement("a");
    a.className = "anchor";
    a.href = "#" + h.id;
    a.textContent = "#";
    a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  /* ── Mobile menu ───────────────────────────────────────────────────── */
  var menuBtn = document.querySelector(".menu-btn");
  var nav = document.querySelector(".top nav");
  if (menuBtn && nav) {
    menuBtn.addEventListener("click", function () {
      nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", nav.classList.contains("open"));
    });
  }
})();
