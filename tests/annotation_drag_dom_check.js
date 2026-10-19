"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const appJs = path.join(__dirname, "..", "hitl_features", "web", "static", "app.js");
const GRAPH = {left: 100, top: 50, width: 700, height: 460};

function Element(tag) {
  this.tagName = String(tag).toUpperCase();
  this.children = [];
  this.parentElement = null;
  this.style = {};
  this.dataset = {};
  this.attributes = {};
  this._listeners = {};
  this._on = {};
  this.className = "";
  this.id = "";
  this.hidden = false;
  this.draggable = false;
  this.value = "";
  this.disabled = false;
  this.textContent = "";
  this.src = "";
  this.alt = "";
  const self = this;
  Object.defineProperty(this, "innerHTML", {
    configurable: true,
    get: function () { return self._html || ""; },
    set: function (html) {
      self._html = String(html);
      self.children = [];
      parseInto(self, self._html);
    }
  });
  ["dragover", "drop", "dragenter"].forEach(function (type) {
    Object.defineProperty(self, "on" + type, {
      configurable: true,
      get: function () { return self._on[type] || null; },
      set: function (fn) { self._on[type] = fn; }
    });
  });
}

Element.prototype.appendChild = function (child) {
  child.parentElement = this;
  this.children.push(child);
  return child;
};

Element.prototype.addEventListener = function (type, fn) {
  (this._listeners[type] = this._listeners[type] || []).push(fn);
};

Element.prototype.setAttribute = function (name, value) {
  this.attributes[name] = String(value);
  if (name === "class") {
    this.className = String(value);
  }
};

Element.prototype.getAttribute = function (name) {
  return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
};

Element.prototype.getBoundingClientRect = function () {
  if (this.id === "graph" || this.tagName === "SVG" || this.className.indexOf("edges-svg") >= 0) {
    return {
      left: GRAPH.left,
      top: GRAPH.top,
      width: GRAPH.width,
      height: GRAPH.height,
      right: GRAPH.left + GRAPH.width,
      bottom: GRAPH.top + GRAPH.height
    };
  }
  if (this.className.split(/\s+/).indexOf("category") >= 0) {
    const x = parseFloat(this.style.left) || 0;
    const y = parseFloat(this.style.top) || 0;
    const left = GRAPH.left + x - 60;
    const top = GRAPH.top + y - 32;
    const width = 120;
    const height = 64;
    return {left: left, top: top, width: width, height: height, right: left + width, bottom: top + height};
  }
  return {left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0};
};

Element.prototype.querySelectorAll = function (selector) {
  const out = [];
  walk(this, selector, out);
  return out;
};

Element.prototype.querySelector = function (selector) {
  return this.querySelectorAll(selector)[0] || null;
};

Element.prototype.contains = function (other) {
  let node = other;
  while (node) {
    if (node === this) {
      return true;
    }
    node = node.parentElement;
  }
  return false;
};

Element.prototype.replaceWith = function () {};

function matches(el, selector) {
  if (!el.tagName) {
    return false;
  }
  if (selector[0] === ".") {
    return el.className.split(/\s+/).indexOf(selector.slice(1)) >= 0;
  }
  return el.tagName === selector.toUpperCase();
}

function walk(el, selector, out) {
  el.children.forEach(function (child) {
    if (matches(child, selector)) {
      out.push(child);
    }
    walk(child, selector, out);
  });
}

function parseInto(parent, html) {
  const re = /<\/?([a-zA-Z0-9]+)[^>]*>|([^<]+)/g;
  const stack = [parent];
  let match;
  while ((match = re.exec(html))) {
    if (match[2]) {
      if (!match[2].trim()) {
        continue;
      }
      const text = new Element("#text");
      text.textContent = match[2];
      stack[stack.length - 1].appendChild(text);
      continue;
    }
    if (match[0][1] === "/") {
      stack.pop();
      continue;
    }
    const el = new Element(match[1]);
    stack[stack.length - 1].appendChild(el);
    stack.push(el);
  }
}

function createElement(tag) {
  return new Element(tag);
}

const document = {
  createElement: createElement,
  createElementNS: function (_ns, tag) { return createElement(tag); },
  createTextNode: function (text) {
    const node = createElement("#text");
    node.textContent = text;
    return node;
  },
  getElementById: function (id) {
    return findId(document.body, id);
  },
  body: null
};

function findId(el, id) {
  if (!el) {
    return null;
  }
  if (el.id === id) {
    return el;
  }
  for (let i = 0; i < el.children.length; i++) {
    const found = findId(el.children[i], id);
    if (found) {
      return found;
    }
  }
  return null;
}

function mount(id, tag) {
  const el = createElement(tag || "div");
  el.id = id;
  document.body.appendChild(el);
  return el;
}

document._listeners = {};
document._on = {};
document.addEventListener = function (type, fn) {
  (this._listeners[type] = this._listeners[type] || []).push(fn);
};
document.removeEventListener = function (type, fn) {
  this._listeners[type] = (this._listeners[type] || []).filter(function (item) { return item !== fn; });
};
document.body = createElement("body");
[
  ["status", "p"],
  ["open-session", "button"],
  ["feature-path", "input"],
  ["img-dir", "input"],
  ["pool-size", "input"],
  ["seed", "input"],
  ["out-dim", "input"],
  ["round-size", "input"],
  ["epochs", "input"],
  ["recommend", "button"],
  ["mechanism", "select"],
  ["target-category", "select"],
  ["confirm", "button"],
  ["reset-mapper", "button"],
  ["add-class", "button"],
  ["train-classification", "button"],
  ["train-ranking", "button"],
  ["pref-strength", "input"],
  ["task-result", "pre"],
  ["workbench", "main"],
  ["images", "div"],
  ["graph", "div"],
  ["edges", "div"],
  ["crumbs", "div"],
  ["distances", "table"],
  ["task-classes", "div"]
].forEach(function (pair) { mount(pair[0], pair[1]); });

const calls = [];
const prompts = [];
const snap0 = {
  id: "sess",
  mechanisms: ["random"],
  all_categories: [],
  categories: [],
  strengths: [],
  images: [
    {index: 11, role: "unplaced"},
    {index: 22, role: "unplaced"}
  ],
  breadcrumb: [],
  distances: [],
  can_confirm: false,
  task_draft: {classes: [], order: [], strength: 1}
};
const snap1 = {
  id: "sess",
  mechanisms: ["random"],
  all_categories: [{id: "c1", name: "Spiral", parent_id: null}],
  categories: [{id: "c1", name: "Spiral", parent_id: null, count: 1}],
  strengths: [],
  images: [{index: 22, role: "unplaced"}],
  breadcrumb: [],
  distances: [],
  can_confirm: false,
  task_draft: {classes: [], order: [], strength: 1}
};
const snap2 = {
  id: "sess",
  mechanisms: ["random"],
  all_categories: [{id: "c1", name: "Spiral", parent_id: null}],
  categories: [{id: "c1", name: "Spiral", parent_id: null, count: 2}],
  strengths: [],
  images: [],
  breadcrumb: [],
  distances: [],
  can_confirm: true,
  task_draft: {classes: [], order: [], strength: 1}
};

function fetch(url, opts) {
  const body = JSON.parse((opts && opts.body) || "{}");
  calls.push({url: String(url), body: body});
  let payload = {snapshot: snap0};
  if (String(url).endsWith("/place")) {
    payload = body.new_name ? {snapshot: snap1} : {snapshot: snap2};
  }
  return Promise.resolve({
    ok: true,
    statusText: "OK",
    json: function () { return Promise.resolve(payload); }
  });
}

function DataTransfer() {
  this._data = {};
  this.dropEffect = "none";
  this.effectAllowed = "uninitialized";
}
DataTransfer.prototype.setData = function (type, value) {
  this._data[type] = String(value);
};
DataTransfer.prototype.getData = function (type) {
  return Object.prototype.hasOwnProperty.call(this._data, type) ? this._data[type] : "";
};

function dispatch(target, type, extra) {
  const event = {
    type: type,
    target: target,
    bubbles: true,
    defaultPrevented: false,
    clientX: extra && extra.clientX || 0,
    clientY: extra && extra.clientY || 0,
    dataTransfer: extra && extra.dataTransfer || null,
    preventDefault: function () { this.defaultPrevented = true; },
    stopPropagation: function () { this._stop = true; }
  };
  const chain = [];
  let node = target;
  while (node) {
    chain.push(node);
    node = node.parentElement;
  }
  for (let i = 0; i < chain.length && !event._stop; i++) {
    event.currentTarget = chain[i];
    const listeners = (chain[i]._listeners[type] || []).slice();
    for (let j = 0; j < listeners.length; j++) {
      listeners[j].call(chain[i], event);
      if (event._stop) {
        return event;
      }
    }
    if (chain[i]._on[type]) {
      chain[i]._on[type].call(chain[i], event);
    }
  }
  return event;
}

const sandbox = {
  console: console,
  document: document,
  fetch: fetch,
  setTimeout: setTimeout,
  clearTimeout: clearTimeout,
  Promise: Promise,
  Number: Number,
  JSON: JSON,
  Math: Math,
  String: String
};
sandbox.window = sandbox;
sandbox.prompt = function (message) {
  prompts.push(String(message));
  return prompts.length === 1 ? "Spiral" : null;
};
sandbox.window.prompt = sandbox.prompt;
sandbox.confirm = function () { return true; };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appJs, "utf8"), sandbox, {filename: appJs});

function settle() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

function indexes() {
  return document.getElementById("images").querySelectorAll(".thumb").map(function (el) {
    return el.dataset.index;
  });
}

async function main() {
  const failures = [];
  function check(cond, message) {
    if (!cond) {
      failures.push(message);
    }
  }
  dispatch(document.getElementById("open-session"), "click");
  await settle();
  await settle();
  const status = document.getElementById("status").textContent;
  check(indexes().join(",") === "11,22", "open session did not list both samples (" + indexes().join(",") + ") status=" + status);

  const graph = document.getElementById("graph");
  const svg = graph.querySelector("svg");
  check(!!svg, "graph did not draw the edge overlay");
  if (svg) {
    const overlayEvent = {
      defaultPrevented: false,
      preventDefault: function () { this.defaultPrevented = true; },
      dataTransfer: new DataTransfer()
    };
    if (typeof svg.ondragover !== "function") {
      failures.push("edge overlay has no dragover handler, so a drop on the canvas never fires");
    } else {
      svg.ondragover(overlayEvent);
      check(overlayEvent.defaultPrevented, "edge overlay dragover did not preventDefault");
    }
  }

  const emptyTransfer = new DataTransfer();
  const firstThumb = document.getElementById("images").querySelector(".thumb");
  dispatch(firstThumb.querySelector("img") || firstThumb, "dragstart", {dataTransfer: emptyTransfer});
  const graphRect = graph.getBoundingClientRect();
  dispatch(svg || graph, "drop", {
    dataTransfer: emptyTransfer,
    clientX: graphRect.left + 30,
    clientY: graphRect.bottom - 24
  });
  await settle();
  await settle();
  const created = calls.filter(function (call) { return call.url.endsWith("/place"); });
  check(created.length === 1, "drop on empty overlay did not create a category; places=" + JSON.stringify(created));
  if (created[0]) {
    check(created[0].body.sample_index === 11, "empty drop placed the wrong sample " + JSON.stringify(created[0].body));
    check(created[0].body.new_name === "Spiral", "empty drop did not name the new category " + JSON.stringify(created[0].body));
  }
  const shown = graph.querySelector(".category");
  check(!!shown && shown.querySelector("strong") && shown.querySelector("strong").textContent === "Spiral",
    "new category was not shown on the graph");
  check(indexes().indexOf("11") < 0, "placed sample 11 is still waiting in the left list: " + indexes().join(","));

  const card = graph.querySelector(".category");
  if (card) {
    const rect = card.getBoundingClientRect();
    const onDrawnCard = rect.left + 8;
    const wrongBoxLeft = graphRect.left + (parseFloat(card.style.left) || 0);
    check(onDrawnCard < wrongBoxLeft, "test point is not on the margin-shifted part of the drawn category");
    const secondThumb = document.getElementById("images").querySelector(".thumb");
    const ontoTransfer = new DataTransfer();
    dispatch(secondThumb.querySelector("img") || secondThumb, "dragstart", {dataTransfer: ontoTransfer});
    dispatch(svg || graph, "drop", {
      dataTransfer: ontoTransfer,
      clientX: onDrawnCard,
      clientY: rect.top + rect.height / 2
    });
    await settle();
    await settle();
  }
  const places = calls.filter(function (call) { return call.url.endsWith("/place"); });
  check(places.length === 2, "second drop did not place the sample; places=" + JSON.stringify(places));
  if (places[1]) {
    check(places[1].body.sample_index === 22, "category drop placed the wrong sample " + JSON.stringify(places[1].body));
    check(places[1].body.category_id === "c1", "drop on the drawn category did not add the image to it " + JSON.stringify(places[1].body));
    check(!places[1].body.new_name, "drop on the drawn category created another category " + JSON.stringify(places[1].body));
  }
  check(prompts.length === 1, "expected one new-category prompt, got " + prompts.length);
  const spiral = graph.querySelector(".category");
  check(!!spiral && spiral.querySelector("span") && spiral.querySelector("span").textContent === "2 张",
    "category did not show the added image");
  check(indexes().length === 0, "left list still has waiting samples: " + indexes().join(","));

  mount("score-board", "div");
  graph.clientWidth = 700;
  graph.clientHeight = 460;
  const edgeSnap = {
    id: "sess",
    mechanisms: ["random"],
    all_categories: [
      {id: "c1", name: "Spiral", parent_id: null},
      {id: "c2", name: "Smooth", parent_id: null}
    ],
    categories: [
      {id: "c1", name: "Spiral", parent_id: null, count: 1, x: 0.25, y: 0.3, recent: [], representative: []},
      {id: "c2", name: "Smooth", parent_id: null, count: 1, x: 0.7, y: 0.6, recent: [], representative: []}
    ],
    strengths: [{a: "c1", b: "c2", a_name: "Spiral", b_name: "Smooth", strength: 1}],
    images: [],
    breadcrumb: [],
    distances: [],
    can_confirm: true,
    task_draft: {classes: [], order: [], strength: 1},
    tasks: [],
    score_board: {
      kind: "classification",
      name: "Classification 1",
      bins: [{low: 0, high: 1, count: 1, images: [{index: 11, score: 0.9, label: "Spiral", in_graph: true}]}]
    }
  };
  sandbox.fetch = function (url, opts) {
    const body = JSON.parse((opts && opts.body) || "{}");
    calls.push({url: String(url), body: body});
    const snap = JSON.parse(JSON.stringify(edgeSnap));
    if (String(url).indexOf("/position") >= 0 && snap.categories[0]) {
      snap.categories[0].x = body.x;
      snap.categories[0].y = body.y;
    }
    return Promise.resolve({
      ok: true,
      statusText: "OK",
      json: function () { return Promise.resolve({snapshot: snap}); }
    });
  };
  dispatch(document.getElementById("recommend"), "click");
  await settle();
  await settle();
  const placedCards = graph.querySelectorAll(".category");
  const line = graph.querySelector("line");
  check(placedCards.length === 2, "saved positions did not draw both cards");
  check(line && Math.abs(parseFloat(line.getAttribute("x1")) - 175) < 0.1,
    "edge did not start at the saved card position " + (line && line.getAttribute("x1")));
  dispatch(placedCards[0], "mousedown", {clientX: 100, clientY: 100, button: 0});
  dispatch(document, "mousemove", {clientX: 180, clientY: 150, button: 0});
  check(Math.abs(parseFloat(placedCards[0].style.left) - 255) < 0.1,
    "card did not follow the pointer, left=" + placedCards[0].style.left);
  check(Math.abs(parseFloat(line.getAttribute("x1")) - parseFloat(placedCards[0].style.left)) < 0.1,
    "edge did not follow the moved card");
  dispatch(document, "mouseup", {clientX: 180, clientY: 150, button: 0});
  await settle();
  await settle();
  const moved = calls.filter(function (call) { return String(call.url).indexOf("/position") >= 0; });
  check(moved.length === 1, "dragging a card did not save its position");
  const thumb = document.getElementById("score-board").querySelector(".score-thumb");
  check(thumb && thumb.className.indexOf("in-graph") >= 0, "an image already in the graph has no distinct marker");
  check(thumb && thumb.draggable === true, "score-bin image is not draggable");
  const scoreTransfer = new DataTransfer();
  dispatch(thumb, "dragstart", {dataTransfer: scoreTransfer});
  check(scoreTransfer.getData("text/plain") === "11", "score drag did not carry the catalog index");

  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
  console.log("annotation drag-drop dom check ok");
}

main().catch(function (error) {
  console.error(error && error.stack || error);
  process.exit(1);
});
