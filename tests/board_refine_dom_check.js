"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const appJs = path.join(__dirname, "..", "hitl_features", "web", "static", "app.js");

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
    }
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
Element.prototype.removeEventListener = function (type, fn) {
  this._listeners[type] = (this._listeners[type] || []).filter(function (item) { return item !== fn; });
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
  if (this.tagName === "IMG") {
    return {left: 10, top: 20, width: 36, height: 36, right: 46, bottom: 56};
  }
  return {left: 0, top: 0, width: 120, height: 80, right: 120, bottom: 80};
};
Element.prototype.querySelectorAll = function (selector) {
  const out = [];
  walk(this, selector, out);
  return out;
};
Element.prototype.querySelector = function (selector) {
  return this.querySelectorAll(selector)[0] || null;
};
Element.prototype.closest = function (selector) {
  const options = String(selector).split(/\s*,\s*/);
  let node = this;
  while (node) {
    for (let i = 0; i < options.length; i++) {
      if (matches(node, options[i])) {
        return node;
      }
    }
    node = node.parentElement;
  }
  return null;
};
Element.prototype.cloneNode = function () {
  return new Element(this.tagName);
};

function matches(el, selector) {
  if (!el || !el.tagName) {
    return false;
  }
  if (selector[0] === ".") {
    return el.className.split(/\s+/).indexOf(selector.slice(1)) >= 0;
  }
  return el.tagName === selector.toUpperCase();
}

function walk(el, selector, out) {
  (el.children || []).forEach(function (child) {
    if (matches(child, selector)) {
      out.push(child);
    }
    walk(child, selector, out);
  });
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
  getElementById: function (id) { return findId(document.body, id); },
  title: ""
};

function findId(el, id) {
  if (!el) {
    return null;
  }
  if (el.id === id) {
    return el;
  }
  for (let i = 0; i < (el.children || []).length; i++) {
    const found = findId(el.children[i], id);
    if (found) {
      return found;
    }
  }
  return null;
}

document.body = createElement("body");
document.body.dataset = {};

function mount(id, tag) {
  const el = createElement(tag || "div");
  el.id = id;
  document.body.appendChild(el);
  return el;
}

[
  ["status", "p"],
  ["lang-toggle", "button"],
  ["magnifier-toggle", "button"],
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
  ["add-class", "button"],
  ["clear-tasks", "button"],
  ["confirm", "button"],
  ["pref-strength", "input"],
  ["images", "div"],
  ["graph", "div"],
  ["edges", "div"],
  ["task-classes", "div"],
  ["list-recommend", "div"],
  ["well-recommend", "section"],
  ["loupe", "div"],
  ["ghost", "div"]
].forEach(function (pair) { mount(pair[0], pair[1]); });
const loupeInner = createElement("div");
loupeInner.id = "loupe-inner";
document.getElementById("loupe").appendChild(loupeInner);
document.getElementById("loupe").hidden = true;
document.getElementById("well-recommend").className = "well";
document.getElementById("well-recommend").setAttribute("data-well", "recommend");

document._listeners = {};
document.addEventListener = function (type, fn) {
  (this._listeners[type] = this._listeners[type] || []).push(fn);
};
document.removeEventListener = function (type, fn) {
  this._listeners[type] = (this._listeners[type] || []).filter(function (item) { return item !== fn; });
};

const store = {};
const calls = [];
let pointTarget = null;
const snap = {
  id: "sess",
  mechanisms: ["random", "certain"],
  frozen: true,
  shot_size: "small",
  edge_mode: "strength",
  recommend_set: [],
  all_categories: [
    {id: "c1", name: "Spiral", parent_id: null, count: 2, color: "#2f5d73"},
    {id: "c2", name: "Smooth", parent_id: null, count: 1, color: "#8a5a2a"}
  ],
  categories: [
    {id: "c1", name: "Spiral", parent_id: null, count: 2, color: "#2f5d73", x: 0.25, y: 0.3, recent: [11], representative: []},
    {id: "c2", name: "Smooth", parent_id: null, count: 1, color: "#8a5a2a", x: 0.7, y: 0.6, recent: [], representative: []}
  ],
  strengths: [{a: "c1", b: "c2", a_name: "Spiral", b_name: "Smooth", strength: 1, distance: 0.5}],
  images: [{index: 11, role: "unplaced"}, {index: 22, role: "unplaced"}],
  breadcrumb: [],
  distances: [],
  can_confirm: false,
  task_draft: {classes: [], order: [], strength: 1}
};

function fetch(url, opts) {
  const body = JSON.parse((opts && opts.body) || "{}");
  calls.push({url: String(url), body: body});
  if (String(url).indexOf("/task-draft") >= 0) {
    snap.task_draft = {
      classes: body.classes || [],
      order: body.order || [],
      strength: body.strength
    };
  }
  if (String(url).indexOf("/ui") >= 0 && body.recommend_ids) {
    snap.recommend_set = body.recommend_ids.slice();
  }
  if (String(url).indexOf("/ui") >= 0 && Object.prototype.hasOwnProperty.call(body, "frozen")) {
    snap.frozen = !!body.frozen;
  }
  const payload = JSON.parse(JSON.stringify({snapshot: snap}));
  return Promise.resolve({
    ok: true,
    statusText: "OK",
    json: function () { return Promise.resolve(payload); }
  });
}

function dispatch(target, type, extra) {
  const event = {
    type: type,
    target: target,
    key: extra && extra.key || "",
    button: extra && extra.button || 0,
    clientX: extra && extra.clientX || 0,
    clientY: extra && extra.clientY || 0,
    preventDefault: function () {},
    stopPropagation: function () { this._stop = true; }
  };
  const chain = [];
  let node = target;
  while (node) {
    chain.push(node);
    node = node.parentElement;
  }
  for (let i = 0; i < chain.length && !event._stop; i++) {
    (chain[i]._listeners[type] || []).slice().forEach(function (fn) {
      fn.call(chain[i], event);
    });
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
  String: String,
  localStorage: {
    getItem: function (key) {
      return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null;
    },
    setItem: function (key, value) { store[key] = String(value); }
  }
};
sandbox.window = sandbox;
sandbox.prompt = function () { return "Broad"; };
sandbox.window.prompt = sandbox.prompt;
sandbox.confirm = function () { return true; };
document.elementFromPoint = function () { return pointTarget; };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appJs, "utf8"), sandbox, {filename: appJs});

function settle() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

function key(target, name) {
  const event = {type: "keydown", target: target, key: name};
  (document._listeners.keydown || []).forEach(function (fn) { fn.call(document, event); });
}

async function main() {
  const failures = [];
  function check(cond, message) {
    if (!cond) {
      failures.push(message);
    }
  }
  const mag = document.getElementById("magnifier-toggle");
  check(mag.getAttribute("aria-pressed") !== "true", "magnifier should start off");
  check(mag.textContent.indexOf("Z") >= 0, "magnifier control should name Z, got " + mag.textContent);
  key(document.body, "z");
  check(mag.getAttribute("aria-pressed") === "true", "Z did not turn the magnifier on");
  key(document.body, "m");
  key(document.body, "M");
  check(mag.getAttribute("aria-pressed") === "true", "M changed the magnifier");
  const field = createElement("input");
  key(field, "z");
  check(mag.getAttribute("aria-pressed") === "true", "Z inside an input toggled the magnifier");
  key(document.body, "Z");
  check(mag.getAttribute("aria-pressed") === "false", "Z did not turn the magnifier off");

  dispatch(document.getElementById("open-session"), "click");
  await settle();
  await settle();
  const shots = document.getElementById("images").querySelectorAll(".thumb");
  const blanks = document.getElementById("images").querySelectorAll(".shot-empty");
  check(shots.length === 2, "small grid did not keep both images");
  check(blanks.length === 7, "small grid did not keep empty cells, blanks=" + blanks.length);

  dispatch(document.getElementById("add-class"), "click");
  await settle();
  await settle();
  const draftCalls = calls.filter(function (call) { return call.url.indexOf("/task-draft") >= 0; });
  check(draftCalls.length === 1, "new class was not saved");
  check(draftCalls[0] && draftCalls[0].body.classes.length === 1, "empty class was not created alone");
  check(draftCalls[0] && draftCalls[0].body.classes[0].category_ids.length === 0, "new class should start empty");
  const taskBox = document.getElementById("task-classes").querySelector(".task-class");
  check(!!taskBox && taskBox.querySelector("strong").textContent === "Broad", "empty task class is not visible");
  check(taskBox && !taskBox.querySelector(".tag"), "empty task class already shows a category tag");

  const card = document.getElementById("graph").querySelector(".category");
  pointTarget = taskBox;
  dispatch(card, "mousedown", {clientX: 10, clientY: 10, button: 0});
  dispatch(document, "mousemove", {clientX: 80, clientY: 40});
  dispatch(document, "mouseup", {clientX: 80, clientY: 40});
  await settle();
  await settle();
  const afterDrop = calls.filter(function (call) { return call.url.indexOf("/task-draft") >= 0; });
  const membership = afterDrop[afterDrop.length - 1].body.classes;
  check(membership.length === 1, "dropping a category created another task class: " + JSON.stringify(membership));
  check(membership[0].category_ids.join(",") === "c1", "category was not added to the empty class: " + JSON.stringify(membership));
  const tagged = document.getElementById("task-classes").querySelector(".task-class");
  const tag = tagged && tagged.querySelector(".tag");
  check(!!tag, "task class tag is missing");
  check(tag && tag.querySelector(".tag-name").textContent === "Spiral", "tag is missing the category name");
  check(tag && tag.querySelector(".tag-count").textContent === "2 张", "tag is missing the sample count: " + (tag && tag.querySelector(".tag-count").textContent));
  check(tag && tag.style.background === "#2f5d73", "tag is missing its color: " + (tag && tag.style.background));
  check(tag && tag.querySelector(".tag-x").textContent === "×", "tag is missing remove");

  const second = document.getElementById("graph").querySelectorAll(".category")[1];
  pointTarget = document.getElementById("task-classes").querySelector(".task-class");
  dispatch(second, "mousedown", {clientX: 10, clientY: 10, button: 0});
  dispatch(document, "mousemove", {clientX: 90, clientY: 50});
  dispatch(document, "mouseup", {clientX: 90, clientY: 50});
  await settle();
  await settle();
  const both = calls.filter(function (call) { return call.url.indexOf("/task-draft") >= 0; });
  const joined = both[both.length - 1].body.classes;
  check(joined.length === 1 && joined[0].category_ids.join(",") === "c1,c2",
    "one task class should hold both categories: " + JSON.stringify(joined));

  pointTarget = document.getElementById("well-recommend");
  const again = document.getElementById("graph").querySelector(".category");
  dispatch(again, "mousedown", {clientX: 10, clientY: 10, button: 0});
  dispatch(document, "mousemove", {clientX: 30, clientY: 80});
  dispatch(document, "mouseup", {clientX: 30, clientY: 80});
  await settle();
  await settle();
  const recTag = document.getElementById("list-recommend").querySelector(".tag");
  check(!!recTag, "recommend tag is missing");
  check(recTag && recTag.querySelector(".tag-name").textContent === "Spiral", "recommend tag has no name");
  check(recTag && recTag.querySelector(".tag-count").textContent === "2 张", "recommend tag has no count");
  check(recTag && recTag.style.background === "#2f5d73", "recommend tag has no color");
  check(recTag && recTag.querySelector("button").textContent === "×", "recommend tag has no remove");

  const hit = document.getElementById("graph").querySelectorAll("line").filter(function (line) {
    return line.getAttribute("data-hit") === "1";
  })[0];
  dispatch(hit, "click");
  const selected = document.getElementById("edges").querySelectorAll(".is-selected");
  check(selected.length === 1, "clicking an edge did not highlight its row");
  dispatch(hit, "click");
  check(document.getElementById("edges").querySelectorAll(".is-selected").length === 0,
    "clicking the same edge did not clear the highlight");

  key(document.getElementById("graph").querySelector(".preview-shot img") || document.body, "z");
  const preview = document.getElementById("graph").querySelector(".preview-shot");
  const img = preview && preview.querySelector("img");
  check(!!img, "category card has no representative image");
  check(preview && !preview.querySelector("button"), "category card still has a per-image delete control");
  const move = {type: "mousemove", target: img, clientX: 28, clientY: 36};
  (document._listeners.mousemove || []).forEach(function (fn) { fn.call(document, move); });
  check(document.getElementById("loupe").hidden === false, "magnifier ignored a representative image");

  dispatch(document.getElementById("lang-toggle"), "click");
  check(store["hitl-board-lang"] === "en", "language choice was not stored");
  check(document.getElementById("lang-toggle").textContent === "中文", "language button did not offer Chinese");
  check(document.getElementById("open-session").textContent === "Open session", "chrome stayed in Chinese");

  const uiBefore = calls.filter(function (call) {
    return call.url.indexOf("/ui") >= 0 && Object.prototype.hasOwnProperty.call(call.body, "frozen");
  }).length;
  key(document.body, "f");
  await settle();
  await settle();
  const freezeCalls = calls.filter(function (call) {
    return call.url.indexOf("/ui") >= 0 && Object.prototype.hasOwnProperty.call(call.body, "frozen");
  });
  check(freezeCalls.length === uiBefore + 1 && freezeCalls[freezeCalls.length - 1].body.frozen === false,
    "F did not unfreeze");
  key(createElement("input"), "F");
  key(createElement("textarea"), "f");
  key(createElement("select"), "f");
  await settle();
  const freezeAfter = calls.filter(function (call) {
    return call.url.indexOf("/ui") >= 0 && Object.prototype.hasOwnProperty.call(call.body, "frozen");
  });
  check(freezeAfter.length === freezeCalls.length, "F inside a field toggled freeze");

  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
}

main().catch(function (error) {
  console.error(error && error.stack || error);
  process.exit(1);
});
