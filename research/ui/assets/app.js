"use strict";

// Everything rendered here comes from documents retrieved on the open web.
// It is inserted as text, never as markup: `el()` sets textContent, and there
// is no innerHTML anywhere in this file. That is the whole XSS story.

const state = { investigation: null, view: "overview", cache: new Map() };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path) {
  if (state.cache.has(path)) return state.cache.get(path);
  const response = await fetch(path, { headers: { accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  state.cache.set(path, payload);
  return payload;
}

function scoped(suffix) {
  return `/api/investigations/${encodeURIComponent(state.investigation)}${suffix}`;
}

function tag(text, className) {
  return el("span", { class: `tag ${className || String(text).replace(/\s+/g, "_")}` }, text);
}

function idLink(id, onClick) {
  return el("button", { class: "link", onclick: () => onClick(id) }, id);
}

// ---------------------------------------------------------------- detail
function showDetail(nodes) {
  const body = document.getElementById("detail-body");
  body.replaceChildren(...nodes);
  document.getElementById("detail").hidden = false;
}

async function showDocument(id) {
  const doc = await api(`/api/documents/${encodeURIComponent(id)}`);
  const nodes = [
    el("h2", {}, doc.title || doc.id),
    el("p", { class: "meta" }, `${doc.id} · ${doc.type} · ${doc.publisher || doc.provider}` +
      (doc.published ? ` · ${doc.published.slice(0, 10)}` : "")),
  ];
  if (doc.retracted) nodes.push(el("p", { class: "warning" }, "Reported as retracted."));
  if (doc.duplicate_of || doc.derived_from) {
    nodes.push(el("p", { class: "warning" },
      `Not independent: ${doc.duplicate_relation} of ${doc.duplicate_of || doc.derived_from}. ` +
      (doc.duplicate_reason || "")));
  }
  if (doc.url) nodes.push(el("p", {}, el("a", { href: doc.url, rel: "noreferrer noopener", target: "_blank" }, doc.url)));
  if (doc.doi) nodes.push(el("p", { class: "meta" }, `doi: ${doc.doi}`));
  if (doc.authors && doc.authors.length) nodes.push(el("p", { class: "meta" }, doc.authors.join(", ")));
  nodes.push(el("p", { class: "meta" }, `provenance: ${doc.provenance || "—"}`));
  nodes.push(el("p", { class: "meta" }, `content hash: ${doc.content_hash}`));
  if (doc.claims.length) {
    nodes.push(el("h3", {}, "Bears on"));
    nodes.push(el("ul", {}, doc.claims.map((link) =>
      el("li", {}, `${link.claim_id} — ${link.stance}`))));
  }
  nodes.push(el("h3", {}, "Content (external, untrusted)"));
  nodes.push(el("pre", {}, doc.text || "(no text stored)"));
  showDetail(nodes);
}

async function showClaim(id) {
  const { claims } = await api(scoped("/claims"));
  const claim = claims.find((entry) => entry.claim_id === id);
  if (!claim) return;
  const nodes = [
    el("h2", {}, claim.text),
    el("p", {}, tag(claim.status), claim.explanation),
  ];
  if (claim.possibly_superseded) {
    nodes.push(el("p", { class: "warning" },
      "The contradicting evidence is more recent than any supporting evidence."));
  }
  for (const stance of ["supports", "contradicts", "qualifies", "mentions"]) {
    const links = claim.links.filter((link) => link.stance === stance);
    if (!links.length) continue;
    nodes.push(el("h3", {}, `${stance} (${links.length})`));
    for (const link of links) {
      const item = [el("p", {}, idLink(link.document_id, showDocument))];
      if (link.excerpt) item.push(el("blockquote", {}, `"${link.excerpt}"`));
      if (link.analysis) item.push(el("p", { class: "analysis" }, `analysis: ${link.analysis}`));
      nodes.push(el("div", {}, ...item));
    }
  }
  if (claim.gaps.length) {
    nodes.push(el("h3", {}, "What would strengthen this"));
    nodes.push(el("ul", { class: "gaps" }, claim.gaps.map((gap) => el("li", {}, gap))));
  }
  showDetail(nodes);
}

// ---------------------------------------------------------------- views
const views = {
  async overview() {
    const data = await api(scoped(""));
    const counts = data.counts;
    const stats = el("div", { class: "stats" },
      [["documents", "documents"], ["independent_sources", "independent sources"],
       ["claims", "claims"], ["entities", "entities"], ["tasks", "tasks"],
       ["searches", "searches"], ["failed_fetches", "failed fetches"]]
        .map(([key, label]) => el("div", { class: "stat" },
          el("b", {}, counts[key]), el("span", {}, label))));

    const nodes = [
      el("h2", {}, data.question),
      el("p", {}, tag(data.status),
        data.stop_reason ? tag(data.stop_reason) : null,
        data.stop_detail || ""),
      data.brief ? el("p", { class: "meta" }, data.brief) : null,
      stats,
      el("h2", {}, "Stopping rules"),
      el("table", {},
        el("tbody", {}, Object.entries(data.stopping_rules).map(([name, value]) =>
          el("tr", {}, el("th", {}, name.replace(/_/g, " ")), el("td", {}, String(value)))))),
      el("h2", {}, "Budget"),
      el("table", {},
        el("thead", {}, el("tr", {}, ["resource", "used", "limit", "remaining"].map((h) => el("th", {}, h)))),
        el("tbody", {}, Object.entries(data.budget).map(([name, value]) =>
          el("tr", {}, el("td", {}, name.replace(/_/g, " ")),
            el("td", { class: "num" }, value.used),
            el("td", { class: "num" }, value.limit),
            el("td", { class: "num" }, value.remaining))))),
    ];
    return nodes.filter(Boolean);
  },

  async tasks() {
    const { tasks } = await api(scoped("/tasks"));
    const byParent = new Map();
    for (const task of tasks) {
      const key = task.parent || "";
      if (!byParent.has(key)) byParent.set(key, []);
      byParent.get(key).push(task);
    }
    const render = (parent) => el("ul", { class: "tree" },
      (byParent.get(parent) || []).map((task) => el("li", {},
        el("div", { class: "card" },
          el("h3", {}, `${task.id} · ${task.role}`),
          el("p", {}, tag(task.status), tag(task.operation), `depth ${task.depth}`),
          el("p", {}, task.objective),
          task.summary ? el("p", { class: "meta" }, task.summary) : null,
          task.error ? el("p", { class: "warning" }, task.error) : null,
          task.evidence.length
            ? el("p", { class: "meta" }, "evidence: ",
                ...task.evidence.map((id) => [idLink(id, showDocument), " "]).flat())
            : null,
          task.claims.length
            ? el("p", { class: "meta" }, "claims: ",
                ...task.claims.map((id) => [idLink(id, showClaim), " "]).flat())
            : null),
        render(task.id))));
    return [el("h2", {}, `Research tasks (${tasks.length})`), render("")];
  },

  async claims() {
    const { claims } = await api(scoped("/claims"));
    if (!claims.length) return [el("p", {}, "No claims recorded.")];
    return [
      el("h2", {}, `Claims (${claims.length})`),
      ...claims.map((claim) => el("div", { class: "card" },
        el("h3", {}, el("button", { class: "link", onclick: () => showClaim(claim.claim_id) },
          `${claim.claim_id} — ${claim.text}`)),
        el("p", {}, tag(claim.status), claim.explanation),
        el("p", { class: "meta" },
          `${claim.support.independent_sources} independent supporting, ` +
          `${claim.contradiction.independent_sources} contradicting` +
          (claim.support.documents.length > claim.support.independent_sources
            ? ` (${claim.support.documents.length} documents, copies counted once)` : "")),
        claim.possibly_superseded
          ? el("p", { class: "warning" }, "contradicting evidence is newer than the support")
          : null)),
    ];
  },

  async graph() {
    const { nodes, edges } = await api(scoped("/graph"));
    if (!nodes.length) return [el("p", {}, "Nothing to draw yet.")];

    // A deterministic two-column layout: claims on the left, the evidence
    // they rest on to the right. Stable across reloads, which a force layout
    // would not be.
    const claims = nodes.filter((node) => node.kind === "claim");
    const evidence = nodes.filter((node) => node.kind === "evidence");
    const rowHeight = 34;
    const claimX = 330;
    const evidenceX = 620;
    const width = 1180;
    const height = Math.max(claims.length, evidence.length, 1) * rowHeight + 50;
    const position = new Map();
    claims.forEach((node, index) => position.set(node.id, { x: claimX, y: 32 + index * rowHeight }));
    evidence.forEach((node, index) => position.set(node.id, { x: evidenceX, y: 32 + index * rowHeight }));

    // Characters that fit beside each column at 11px, so nothing is clipped.
    const fit = (text, pixels) => {
      const limit = Math.floor(pixels / 5.9);
      return text.length > limit ? text.slice(0, limit - 1) + "\u2026" : text;
    };

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", String(height));

    const draw = (name, attrs) => {
      const node = document.createElementNS(svgNS, name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
      return node;
    };

    for (const edge of edges) {
      const from = position.get(edge.source);
      const to = position.get(edge.target);
      if (!from || !to) continue;
      if (edge.kind === "cites") {
        // Both ends sit in the evidence column, so a straight line would be
        // invisible. Bow it out to the right instead.
        const bulge = 40 + Math.abs(to.y - from.y) / 6;
        svg.append(draw("path", {
          d: `M ${from.x} ${from.y} Q ${from.x + bulge} ${(from.y + to.y) / 2} ${to.x} ${to.y}`,
          class: "edge-cites", fill: "none", "stroke-width": 1,
        }));
      } else {
        svg.append(draw("line", {
          x1: from.x, y1: from.y, x2: to.x, y2: to.y,
          class: `edge-${edge.kind}`, "stroke-width": 1.6,
        }));
      }
    }

    for (const node of nodes) {
      const point = position.get(node.id);
      if (!point) continue;
      const group = draw("g", { class: "node" });
      const isClaim = node.kind === "claim";
      group.append(draw("circle", {
        cx: point.x, cy: point.y, r: isClaim ? 7 : 5,
        class: isClaim ? "node-claim" : `node-evidence${node.retracted ? " retracted" : ""}`,
      }));
      const suffix = node.copies && node.copies.length
        ? ` (+${node.copies.length} cop${node.copies.length === 1 ? "y" : "ies"})` : "";
      const label = draw("text", {
        x: isClaim ? point.x - 13 : point.x + 12,
        y: point.y + 4,
        ...(isClaim ? { "text-anchor": "end" } : {}),
      });
      label.textContent = fit(node.label + suffix, isClaim ? claimX - 20 : width - evidenceX - 20);
      group.append(label);
      const title = draw("title", {});
      title.textContent = `${node.id}: ${node.label}${suffix}`;
      group.append(title);
      group.addEventListener("click", () => (isClaim ? showClaim(node.id) : showDocument(node.id)));
      svg.append(group);
    }

    const legend = el("div", { class: "legend" },
      // Class only: the page forbids inline styles, including its own.
      ...["supports", "contradicts", "qualifies", "mentions", "cites"].map((kind) =>
        el("span", {}, el("i", { class: `swatch-${kind}` }), kind)));

    return [
      el("h2", {}, `Claim and evidence graph (${claims.length} claims, ${evidence.length} sources)`),
      el("p", { class: "meta" },
        "Copies are folded into the document they copy: each evidence node is one independent source."),
      legend,
      svg,
    ];
  },

  async entities() {
    const { entities } = await api(scoped("/entities"));
    if (!entities.length) return [el("p", {}, "No entities registered.")];
    return [
      el("h2", {}, `Entities (${entities.length})`),
      el("table", {},
        el("thead", {}, el("tr", {}, ["id", "name", "type", "identifiers", "documents", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, entities.map((entity) => el("tr", {},
          el("td", {}, entity.id),
          el("td", {}, entity.name),
          el("td", {}, entity.type),
          el("td", {}, entity.identifiers.join(", ") || "—"),
          el("td", { class: "num" }, entity.documents.length),
          el("td", {}, entity.review_needed ? tag("name-only match", "warn") : ""))))),
    ];
  },

  async timeline() {
    const { timeline } = await api(scoped("/timeline"));
    if (!timeline.length) return [el("p", {}, "Nothing dated yet.")];
    return [
      el("h2", {}, "Timeline"),
      el("p", { class: "meta" },
        "Events are asserted and evidence-backed. Publication entries are one per independent source."),
      el("table", {},
        el("thead", {}, el("tr", {}, ["date", "precision", "kind", "what", "evidence"].map((h) => el("th", {}, h)))),
        el("tbody", {}, timeline.map((entry) => el("tr", {},
          el("td", {}, entry.date || "undated"),
          el("td", {}, entry.precision),
          el("td", {}, entry.kind),
          el("td", {}, entry.description),
          el("td", {}, ...entry.evidence.map((id) => [idLink(id, showDocument), " "]).flat()))))),
    ];
  },

  async sources() {
    const { sources } = await api(scoped("/sources"));
    if (!sources.length) return [el("p", {}, "No sources held.")];
    const copies = sources.reduce((total, source) => total + source.copies.length, 0);
    return [
      el("h2", {}, `Sources (${sources.length} independent)`),
      el("p", { class: "meta" }, copies
        ? `${copies} further documents are copies, syndicated versions or rewrites of these and are not counted separately.`
        : "Each source counted once."),
      ...sources.map((source) => el("div", { class: "card" },
        el("h3", {}, el("button", { class: "link", onclick: () => showDocument(source.id) },
          `${source.id} — ${source.title || source.citation}`)),
        el("p", { class: "meta" }, `${source.type} · ${source.citation}` +
          (source.published ? ` · ${source.published}` : "")),
        source.url ? el("p", { class: "meta" }, source.url) : null,
        el("p", { class: "meta" }, `retrieved: ${source.provenance || "—"}`),
        source.copies.length
          ? el("p", { class: "meta" }, `also seen as: ` +
              source.copies.map((copy) => `${copy.id} (${copy.relation} via ${copy.source})`).join(", "))
          : null)),
    ];
  },

  async questions() {
    const { open_questions: questions } = await api(scoped("/questions"));
    if (!questions.length) return [el("p", {}, "No claim has an outstanding gap.")];
    return [
      el("h2", {}, `Open questions (${questions.length})`),
      ...questions.map((question) => el("div", { class: "card" },
        el("h3", {}, el("button", { class: "link", onclick: () => showClaim(question.claim_id) },
          `${question.claim_id} — ${question.text}`)),
        el("p", {}, tag(question.status), question.explanation),
        el("ul", { class: "gaps" }, question.gaps.map((gap) => el("li", {}, gap))))),
    ];
  },

  async activity() {
    const { searches, fetches } = await api(scoped("/activity"));
    return [
      el("h2", {}, `Searches (${searches.length})`),
      el("table", {},
        el("thead", {}, el("tr", {}, ["id", "provider", "family", "results", "status", "query", "why"].map((h) => el("th", {}, h)))),
        el("tbody", {}, searches.map((search) => el("tr", {},
          el("td", {}, search.id),
          el("td", {}, search.provider),
          el("td", {}, search.family),
          el("td", { class: "num" }, search.result_count),
          el("td", {}, search.status === "ok" ? tag("ok", "completed") : tag(search.status, "failed")),
          el("td", {}, search.query_text),
          el("td", { class: "meta" }, search.objective || ""))))),
      el("h2", {}, `Fetches (${fetches.length})`),
      el("table", {},
        el("thead", {}, el("tr", {}, ["id", "ok", "status", "bytes", "document", "url", "error"].map((h) => el("th", {}, h)))),
        el("tbody", {}, fetches.map((fetch) => el("tr", {},
          el("td", {}, fetch.id),
          el("td", {}, fetch.ok ? "yes" : "no"),
          el("td", { class: "num" }, fetch.status_code ?? ""),
          el("td", { class: "num" }, fetch.bytes ?? ""),
          el("td", {}, fetch.document_id ? idLink(fetch.document_id, showDocument) : ""),
          el("td", {}, fetch.url),
          el("td", { class: "meta" }, fetch.error || ""))))),
    ];
  },
};

// ---------------------------------------------------------------- shell
async function render() {
  const content = document.getElementById("content");
  content.replaceChildren(el("p", { class: "meta" }, "Loading…"));
  try {
    const nodes = await views[state.view]();
    content.replaceChildren(...nodes);
  } catch (error) {
    content.replaceChildren(el("p", { class: "warning" }, String(error.message || error)));
  }
}

function selectView(view) {
  state.view = view;
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.view === view);
  }
  render();
}

async function start() {
  document.getElementById("detail-close").addEventListener("click", () => {
    document.getElementById("detail").hidden = true;
  });
  for (const button of document.querySelectorAll("#tabs button")) {
    button.addEventListener("click", () => selectView(button.dataset.view));
  }

  const { investigations } = await api("/api/investigations");
  const picker = document.getElementById("investigation-picker");
  picker.replaceChildren(...investigations.map((investigation) =>
    el("option", { value: investigation.id },
      `${investigation.id} — ${investigation.question.slice(0, 70)}`)));
  picker.addEventListener("change", () => {
    state.investigation = picker.value;
    state.cache.clear();
    render();
  });

  if (!investigations.length) {
    document.getElementById("content").replaceChildren(
      el("p", {}, "No investigations yet. Run one:"),
      el("pre", {}, "research --offline investigate \"your question\""));
    return;
  }
  state.investigation = investigations[0].id;
  document.getElementById("status-line").textContent =
    `${investigations.length} investigation${investigations.length === 1 ? "" : "s"}`;
  render();
}

start();
