(function () {
  "use strict";
  var sessionId = null;
  var snapshot = null;
  var statusEl = document.getElementById("status");
  var page = 0;
  var magnifierOn = false;
  var selectedEdge = null;
  var pinnedSlots = null;
  var viewMode = "recommend";
  var rightMode = "mapper";
  var umapLoading = false;
  var umapError = "";
  var centerMode = "graph";
  var umapLayout = null;
  var umapToken = null;
  var umapColorKey = null;
  var umapPinnedPoint = null;
  var resultView = "saved";
  var labeledPick = null;
  var busy = false;
  var edgeSort = {key: "pair", dir: 1};
  var edgePage = 0;
  var edgeTableOpen = true;
  var lastFrac = "train";
  var pendingCsv = "";
  var importHeaders = [];
  var importLabelPicks = [];
  var EDGE_PAGE = 8;
  var SHOT_GAP = 8;
  var CAPTION_H = 22;
  var lang = readLang();
  var STR = {
    zh: {
      title: "星系标注",
      "feature": "特征文件",
      "img": "图片目录",
      "pool": "搜索池大小",
      "seed": "随机种子",
      "dim": "输出维度",
      "round": "每轮张数",
      "epochs": "训练轮数",
      "project-name": "项目名称",
      "open-session": "创建项目",
      "saved-projects": "已保存项目",
      "load-project": "打开项目",
      "name-required": "请填写项目名称",
      "name-bad": "名称不能包含 | 或路径分隔符",
      "name-taken": "已有同名项目",
      "choose-project": "请选择一个已保存的项目",
      "freeze-off": "解冻",
      "freeze-on": "冻结",
      "freeze-label": "卡片是否冻结",
      "edge-strength": "分离强度",
      "edge-distance": "平均距离",
      "edge-label": "连线粗细的含义",
      "import-label": "导入 CSV",
      "import-labels": "导入并训练",
      "export-labels": "导出",
      "rec-title": "本轮推荐",
      "shot-small": "小图",
      "shot-large": "大图",
      "shot-label": "推荐图大小",
      "mode": "方式",
      "scope-factor": "倍数",
      "recommend": "推荐",
      "mode-recommend": "推荐",
      "mode-labeled": "当前类别",
      "mode-recycle": "回收站",
      "mode-mapper": "特征训练",
      "mode-task": "分类/排序",
      "mapper-size": "映射宽度",
      "head-size": "模型大小",
      "show-labeled": "当前类别",
      "show-recommend": "推荐",
      "mapper-reset": "映射宽度已更换，需要重新训练。",
      "busy": "上一项还在进行",
      "recycle-sample": "移入回收站",
      "recycle-count": "回收站 {n}",
      "page-prev": "上一页",
      "page-next": "下一页",
      "rec-well": "推荐类别",
      "select-all": "选中全部",
      "clear-rec": "移除全部",
      "graph-title": "类别图",
      "mode-graph": "类别图",
      "mode-umap": "UMAP",
      "center-label": "中心视图",
      "source-mapped": "映射",
      "source-raw": "原始",
      "source-label": "特征",
      "umap-raw": "原始",
      "umap-mapped": "映射",
      "umap-note": "灰点是候选池，彩色点是已标注样本",
      "umap-loading": "正在计算 UMAP…",
      "umap-read-error": "没能读出 UMAP 结果",
      "umap-empty": "UMAP 没有返回坐标",
      "name-bar": "名称不能包含 |",
      "confirm": "确认并训练映射",
      "reset-mapper": "重置映射",
      "task-title": "分类 / 排序",
      "add-class": "新建空类",
      "clear-tasks": "移除全部",
      "strength": "排序强度",
      "train-classification": "训练分类",
      "train-ranking": "训练排序",
      "show-results": "显示分类结果",
      "run-predict": "运行预测",
      "result-count": "结果张数",
      "leaf-only": "只看没有子类的样本",
      "mapper-tags": "全部类别",
      "rename-class": "改名",
      "rename-task": "结果名称",
      "delete-task": "删除",
      "confirm-delete-task": "删除这个结果？",
      "import-title": "选择列",
      "import-note": "先选索引列，再按层级顺序选择一个或多个标签列。确认后才会导入并训练。",
      "import-index": "索引列",
      "import-add-label": "添加标签列",
      "import-confirm": "导入并训练",
      "import-cancel": "取消",
      "import-level": "标签列",
      "import-need-columns": "CSV 至少需要索引列和一个标签列",
      "import-bad-columns": "标签列不能重复，也不能与索引列相同",
      "import-open-session": "请先创建或打开项目，再导入 CSV",
      "frac-train": "训练",
      "frac-val": "验证",
      "frac-test": "测试",
      "split-train": "训练",
      "split-val": "验证",
      "split-test": "测试",
      "raw-acc": "原始准确率",
      "mapped-acc": "映射准确率",
      "raw-auc": "原始 ROC AUC",
      "mapped-auc": "映射 ROC AUC",
      "raw-pair": "原始成对准确率",
      "mapped-pair": "映射成对准确率",
      "edge-hide": "隐藏边表",
      "edge-show": "显示边表",
      "score-title": "分数箱",
      "score-note": "底边是各分数段的大图。放大镜打开时，悬停可以放大。",
      "magnifier-off": "放大镜关 · Z",
      "magnifier-on": "放大镜开 · Z",
      "freeze-hint-on": "已冻结：卡片停在原地。从卡片任意位置拖进推荐类别，或拖进右侧的任务类。",
      "freeze-hint-off": "已解冻：拖动卡片改变位置，连线会跟着走。",
      "edge-hint-strength": "线越粗，两类越该分开。节点边线是其子类边的平均分离强度。",
      "edge-hint-distance": "线越粗，映射后两类离得越远。节点边线是类内平均映射距离。",
      "graph-note-strength": "线宽表示分离强度",
      "graph-note-distance": "线宽表示映射距离",
      "empty-recommend": "槽里还没有类别。冻结后，从卡片任意处拖到这里。空槽表示使用当前层的全部类别。",
      "empty-task": "先新建空类，再把图上的类别拖进该类。",
      "page-empty": "还没有推荐",
      "page-recycle": "回收站是空的",
      "page-label": "第 {page} / {pages} 页",
      "images-unit": " 张",
      "remove-sample": "移出",
      "root": "根",
      "score-rank": "排序分数",
      "score-class": "分类分数",
      "col-round": "轮次",
      "within": "类内",
      "between": "类间",
      "loss": "损失",
      "edge-pair": "类别对",
      "edge-value": "数值",
      "up": "上移",
      "down": "下移",
      "remove-class": "移除这类",
      "mech-random": "随机",
      "mech-diversity": "多样",
      "mech-unknown": "未知类别",
      "mech-certain": "最明确",
      "mech-uncertain": "最不明确",
      "new-category": "新类别名称",
      "rename-category": "类别名称",
      "task-class-name": "任务类名称",
      "strength-prompt": "分离强度，范围 0 到 2",
      "confirm-train": "用当前类别和分离强度训练特征映射？",
      "confirm-reset": "重置特征映射，下次从头训练？",
      "trained-round": "已训练轮次 ",
      "choose-csv": "请选择含 index、label_1、label_2… 的 CSV",
      "imported": "已导入 {n} 张图，共 {c} 个类别。"
    },
    en: {
      title: "Galaxy labeling",
      "feature": "Feature file",
      "img": "Image directory",
      "pool": "Pool size",
      "seed": "Seed",
      "dim": "Output dimension",
      "round": "Round size",
      "epochs": "Epochs",
      "project-name": "Project name",
      "open-session": "Create project",
      "saved-projects": "Saved projects",
      "load-project": "Open project",
      "name-required": "Enter a project name",
      "name-bad": "A name cannot contain | or a path separator",
      "name-taken": "A project with this name already exists",
      "choose-project": "Choose a saved project",
      "freeze-off": "Unfrozen",
      "freeze-on": "Frozen",
      "freeze-label": "Whether cards are frozen",
      "edge-strength": "Separation",
      "edge-distance": "Mean distance",
      "edge-label": "What the stroke thickness means",
      "import-label": "Import CSV",
      "import-labels": "Import and train",
      "export-labels": "Export",
      "rec-title": "This round",
      "shot-small": "Small",
      "shot-large": "Large",
      "shot-label": "Recommendation image size",
      "mode": "Mode",
      "scope-factor": "×n",
      "recommend": "Recommend",
      "mode-recommend": "Suggest",
      "mode-labeled": "Current",
      "mode-recycle": "Recycle",
      "mode-mapper": "Features",
      "mode-task": "Classes",
      "mapper-size": "Map width",
      "head-size": "Model size",
      "show-labeled": "Current",
      "show-recommend": "Recommend",
      "mapper-reset": "Map width changed. Train the map again.",
      "busy": "Another job is still running",
      "recycle-sample": "Recycle",
      "recycle-count": "Bin {n}",
      "page-prev": "Prev",
      "page-next": "Next",
      "rec-well": "Categories",
      "select-all": "All",
      "clear-rec": "Clear",
      "graph-title": "Category graph",
      "mode-graph": "Graph",
      "mode-umap": "UMAP",
      "center-label": "Center view",
      "source-mapped": "Mapped",
      "source-raw": "Raw",
      "source-label": "Features",
      "umap-raw": "Raw",
      "umap-mapped": "Mapped",
      "umap-note": "Gray points are the candidate pool; colored points are labeled",
      "umap-loading": "Computing UMAP…",
      "umap-read-error": "Could not read the UMAP result",
      "umap-empty": "UMAP returned no coordinates",
      "name-bar": "A name cannot contain |",
      "confirm": "Train map",
      "reset-mapper": "Reset map",
      "task-title": "Classification / ranking",
      "add-class": "New class",
      "clear-tasks": "Clear",
      "strength": "Ranking strength",
      "train-classification": "Train classifier",
      "train-ranking": "Train ranker",
      "show-results": "Show results",
      "run-predict": "Predict",
      "result-count": "Result count",
      "leaf-only": "No subclass only",
      "mapper-tags": "All categories",
      "rename-class": "Rename",
      "rename-task": "Result name",
      "delete-task": "Delete",
      "confirm-delete-task": "Delete this result?",
      "import-title": "Choose columns",
      "import-note": "Choose the index column, then one or more label columns in level order. Import starts only after you confirm.",
      "import-index": "Index column",
      "import-add-label": "Add label column",
      "import-confirm": "Import and train",
      "import-cancel": "Cancel",
      "import-level": "Label column",
      "import-need-columns": "The CSV needs an index column and at least one label column",
      "import-bad-columns": "Label columns must be distinct and different from the index",
      "import-open-session": "Create or open a project before importing a CSV",
      "frac-train": "Train",
      "frac-val": "Val",
      "frac-test": "Test",
      "split-train": "Train",
      "split-val": "Val",
      "split-test": "Test",
      "raw-acc": "Raw accuracy",
      "mapped-acc": "Mapped accuracy",
      "raw-auc": "Raw ROC AUC",
      "mapped-auc": "Mapped ROC AUC",
      "raw-pair": "Raw pairwise accuracy",
      "mapped-pair": "Mapped pairwise accuracy",
      "edge-hide": "Hide edges",
      "edge-show": "Show edges",
      "score-title": "Score bins",
      "score-note": "The bottom row shows a large image for each score bin. Hover magnifies while the magnifier is on.",
      "magnifier-off": "Magnifier off · Z",
      "magnifier-on": "Magnifier on · Z",
      "freeze-hint-on": "Frozen: cards stay put. Drag a card into the recommend well or into a task class.",
      "freeze-hint-off": "Unfrozen: drag a card to move it. Edges follow.",
      "edge-hint-strength": "A thicker edge means the two classes should separate. A node outline is the mean separation of its child edges.",
      "edge-hint-distance": "A thicker edge means the two classes are farther apart after mapping. A node outline is the mean within-node mapped distance.",
      "graph-note-strength": "Edge width is separation strength",
      "graph-note-distance": "Edge width is mapped distance",
      "empty-recommend": "No categories in this well. While frozen, drag a card here. An empty well uses every category at this level.",
      "empty-task": "Create an empty class, then drag graph categories into that class.",
      "page-empty": "No recommendations yet",
      "page-recycle": "The recycle bin is empty",
      "page-label": "Page {page} / {pages}",
      "images-unit": " images",
      "remove-sample": "Remove",
      "root": "Root",
      "score-rank": "Ranking score",
      "score-class": "Classification score",
      "col-round": "Round",
      "within": "Within",
      "between": "Between",
      "loss": "Loss",
      "edge-pair": "Pair",
      "edge-value": "Value",
      "up": "Up",
      "down": "Down",
      "remove-class": "Remove class",
      "mech-random": "Random",
      "mech-diversity": "Diversity",
      "mech-unknown": "Unknown",
      "mech-certain": "Certain",
      "mech-uncertain": "Uncertain",
      "new-category": "New category name",
      "rename-category": "Category name",
      "task-class-name": "Task class name",
      "strength-prompt": "Relation strength from 0 to 2",
      "confirm-train": "Train the feature map with the current categories and relation strengths?",
      "confirm-reset": "Reset the feature map and train from scratch next time?",
      "trained-round": "Trained round ",
      "choose-csv": "Choose a CSV with index, label_1, label_2, ...",
      "imported": "Imported {n} images into {c} categories."
    }
  };
  var TEXT_IDS = {
    "i18n-title": "title",
    "i18n-feature": "feature",
    "i18n-img": "img",
    "i18n-pool": "pool",
    "i18n-seed": "seed",
    "i18n-dim": "dim",
    "i18n-round": "round",
    "i18n-epochs": "epochs",
    "i18n-project-name": "project-name",
    "open-session": "open-session",
    "i18n-saved-projects": "saved-projects",
    "load-project": "load-project",
    "freeze-off": "freeze-off",
    "freeze-on": "freeze-on",
    "edge-strength": "edge-strength",
    "edge-distance": "edge-distance",
    "i18n-import-label": "import-label",
    "import-labels": "import-labels",
    "export-labels": "export-labels",
    "i18n-rec-title": "rec-title",
    "shot-small": "shot-small",
    "shot-large": "shot-large",
    "i18n-mode": "mode",
    "i18n-scope-factor": "scope-factor",
    "recommend": "recommend",
    "mode-recommend": "mode-recommend",
    "mode-labeled": "mode-labeled",
    "mode-recycle": "mode-recycle",
    "mode-mapper": "mode-mapper",
    "mode-task": "mode-task",
    "i18n-mapper-size": "mapper-size",
    "i18n-head-size": "head-size",
    "page-prev": "page-prev",
    "page-next": "page-next",
    "i18n-rec-well": "rec-well",
    "i18n-select-all": "select-all",
    "i18n-clear-rec": "clear-rec",
    "i18n-graph-title": "graph-title",
    "mode-graph": "mode-graph",
    "mode-umap": "mode-umap",
    "source-mapped": "source-mapped",
    "source-raw": "source-raw",
    "umap-raw-label": "umap-raw",
    "umap-mapped-label": "umap-mapped",
    "confirm": "confirm",
    "reset-mapper": "reset-mapper",
    "i18n-task-title": "task-title",
    "add-class": "add-class",
    "clear-tasks": "clear-tasks",
    "i18n-strength": "strength",
    "train-classification": "train-classification",
    "train-ranking": "train-ranking",
    "show-results": "show-results",
    "run-predict": "run-predict",
    "i18n-result-count": "result-count",
    "i18n-leaf-only": "leaf-only",
    "i18n-mapper-tags": "mapper-tags",
    "i18n-import-title": "import-title",
    "i18n-import-note": "import-note",
    "i18n-import-index": "import-index",
    "import-add-label": "import-add-label",
    "import-confirm": "import-confirm",
    "import-cancel": "import-cancel",
    "i18n-frac-train": "frac-train",
    "i18n-frac-val": "frac-val",
    "i18n-frac-test": "frac-test",
    "i18n-score-title": "score-title",
    "i18n-score-note": "score-note"
  };

  function readLang() {
    try {
      if (window.localStorage) {
        var saved = window.localStorage.getItem("hitl-board-lang");
        if (saved === "en" || saved === "zh") {
          return saved;
        }
      }
    } catch (err) {
      return "zh";
    }
    return "zh";
  }

  function t(key) {
    var pack = STR[lang] || STR.zh;
    if (Object.prototype.hasOwnProperty.call(pack, key)) {
      return pack[key];
    }
    return STR.zh[key] || key;
  }

  function setStatus(text) {
    if (statusEl) {
      statusEl.textContent = text || "";
    }
  }

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body || {})
    }).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) {
          throw new Error(payload.error || response.statusText);
        }
        return payload;
      });
    });
  }

  var JOB_IDS = [
    "recommend", "confirm", "reset-mapper", "train-classification", "train-ranking",
    "show-results", "run-predict",
    "import-labels", "export-labels", "import-confirm", "mapper-size", "head-size",
    "mode-umap", "source-mapped", "source-raw"
  ];

  function formatCount(value) {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function paintBusy(layer, progress) {
    if (!layer) {
      return;
    }
    var base = layer.id === "busy-export" ? "busy-top" : "busy-layer";
    var known = progress && Number(progress.total) > 0;
    layer.className = known ? base : base + " is-indeterminate";
    var fill = layer.querySelector ? layer.querySelector(".busy-fill") : null;
    if (!fill) {
      return;
    }
    if (known) {
      var pct = Math.max(0, Math.min(100, (100 * Number(progress.done)) / Number(progress.total)));
      fill.style.width = pct + "%";
    } else {
      fill.style.width = "";
    }
  }

  function setBusy(on, region, progress) {
    busy = !!on;
    JOB_IDS.forEach(function (id) {
      var node = byId(id);
      if (node) {
        node.disabled = busy;
      }
    });
    ["busy-rec", "busy-task", "busy-graph", "busy-export"].forEach(function (id) {
      var layer = byId(id);
      if (!layer) {
        return;
      }
      var show = busy && id === region;
      layer.hidden = !show;
      if (show) {
        paintBusy(layer, progress);
      }
    });
    if (!busy) {
      syncIdleControls();
    }
  }

  function syncIdleControls() {
    var recycle = viewMode === "recycle";
    ["mechanism", "recommend"].forEach(function (id) {
      var node = byId(id);
      if (node) {
        node.disabled = recycle;
      }
    });
    var row = byId("recommend-row");
    if (row) {
      row.className = recycle ? "rec-tools is-off" : "rec-tools";
    }
    var confirm = byId("confirm");
    if (confirm && snapshot) {
      confirm.disabled = !snapshot.can_confirm;
    }
    ["page-prev", "page-next", "reset-mapper", "train-classification", "train-ranking",
      "import-labels", "export-labels", "import-confirm", "mapper-size", "head-size",
      "mode-umap", "source-raw"].forEach(function (id) {
      var node = byId(id);
      if (node && id !== "confirm") {
        node.disabled = false;
      }
    });
    syncFeatureSource();
  }

  function readPayload(response, region) {
    if (response.body && response.body.getReader) {
      return readStream(response, region);
    }
    return response.json().then(function (payload) {
      if (!response.ok) {
        throw new Error((payload && payload.error) || response.statusText);
      }
      return payload;
    });
  }

  function readStream(response, region) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var finalPayload = null;
    var failed = null;
    function consume(chunk) {
      buffer += chunk;
      var lines = buffer.split("\n");
      buffer = lines.pop();
      lines.forEach(function (line) {
        if (!line.trim()) {
          return;
        }
        var msg = JSON.parse(line);
        if (msg.event === "progress") {
          setBusy(true, region, {done: msg.done, total: msg.total});
        } else if (msg.event === "error") {
          failed = new Error(msg.error || "failed");
        } else if (msg.event === "done") {
          finalPayload = msg;
        }
      });
    }
    function pump() {
      return reader.read().then(function (result) {
        if (result.done) {
          if (buffer.trim()) {
            consume("\n");
          }
          if (failed) {
            throw failed;
          }
          if (!finalPayload) {
            throw new Error("empty response");
          }
          return finalPayload;
        }
        consume(decoder.decode(result.value, {stream: true}));
        if (failed) {
          throw failed;
        }
        return pump();
      });
    }
    return pump();
  }

  function postJob(url, body, region) {
    if (busy) {
      return Promise.reject(new Error(t("busy")));
    }
    setBusy(true, region, null);
    var sent = {};
    var source = body || {};
    Object.keys(source).forEach(function (key) {
      sent[key] = source[key];
    });
    sent.stream = true;
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(sent)
    }).then(function (response) {
      return readPayload(response, region);
    }).finally(function () {
      setBusy(false);
    });
  }

  function apply(payload) {
    snapshot = payload.snapshot;
    sessionId = snapshot.id;
    var workbench = document.getElementById("workbench");
    if (workbench) {
      workbench.hidden = false;
    }
    render();
    setStatus(snapshot.name || "");
  }

  function fail(error) {
    var message = error && error.message ? error.message : String(error);
    if (message === "project name is required") {
      message = t("name-required");
    } else if (message === "project name cannot contain | or a path separator") {
      message = t("name-bad");
    } else if (message === "a project with this name already exists") {
      message = t("name-taken");
    }
    setStatus(message);
  }

  function refreshProjects(selectId) {
    var menu = byId("saved-projects");
    if (!menu || typeof fetch !== "function") {
      return Promise.resolve();
    }
    return fetch("/api/sessions").then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) {
          throw new Error((payload && payload.error) || response.statusText);
        }
        return payload;
      });
    }).then(function (payload) {
      var projects = (payload && payload.projects) || [];
      while (menu.firstChild) {
        menu.removeChild(menu.firstChild);
      }
      projects.forEach(function (item) {
        var option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.name || item.id;
        menu.appendChild(option);
      });
      if (selectId) {
        menu.value = selectId;
      }
    }).catch(function () {
      /* The list is filled again the next time the page opens. */
    });
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function on(id, type, handler) {
    var node = byId(id);
    if (node) {
      node.addEventListener(type, handler);
    }
  }

  on("open-session", "click", function () {
    var nameInput = byId("project-name");
    var projectName = nameInput ? String(nameInput.value || "").trim() : "";
    if (nameInput) {
      if (!projectName) {
        setStatus(t("name-required"));
        return;
      }
      if (projectName.indexOf("|") >= 0 || projectName.indexOf("/") >= 0 || projectName.indexOf("\\") >= 0) {
        setStatus(t("name-bad"));
        return;
      }
    }
    var body = {
      feature_path: byId("feature-path").value,
      img_dir: byId("img-dir").value,
      pool_size: Number(byId("pool-size").value),
      seed: Number(byId("seed").value),
      out_dim: Number(byId("out-dim").value),
      round_size: Number(byId("round-size").value),
      epochs: Number(byId("epochs").value)
    };
    if (nameInput) {
      body.name = projectName;
    }
    post("/api/sessions", body).then(function (payload) {
      apply(payload);
      var created = payload && payload.snapshot ? payload.snapshot.id : "";
      return refreshProjects(created);
    }).catch(fail);
  });

  on("load-project", "click", function () {
    var menu = byId("saved-projects");
    if (!menu || !menu.value) {
      setStatus(t("choose-project"));
      return;
    }
    fetch("/api/sessions/" + encodeURIComponent(menu.value)).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) {
          throw new Error((payload && payload.error) || response.statusText);
        }
        return payload;
      });
    }).then(apply).catch(fail);
  });

  on("recommend", "click", function () {
    if (busy || viewMode === "recycle") {
      return;
    }
    var stayLabeled = viewMode === "labeled";
    if (!stayLabeled) {
      viewMode = "recommend";
      labeledPick = null;
    }
    pinnedSlots = null;
    page = 0;
    var mechanism = byId("mechanism");
    var request = {
      mechanism: mechanism ? mechanism.value : "random",
      category_ids: (snapshot && snapshot.recommend_set) || [],
      within_labeled: stayLabeled
    };
    var factor = byId("scope-factor");
    if (factor && String(factor.value).trim() !== "") {
      request.scope_factor = Number(factor.value);
    }
    postJob("/api/sessions/" + sessionId + "/recommend", request, "busy-rec").then(function (payload) {
      if (stayLabeled) {
        labeledPick = payload && payload.chosen ? payload.chosen.slice() : [];
      }
      apply(payload);
    }).catch(fail);
  });

  on("confirm", "click", function () {
    if (busy) {
      return;
    }
    if (!window.confirm(t("confirm-train"))) {
      return;
    }
    rightMode = "mapper";
    renderRightMode();
    postJob("/api/sessions/" + sessionId + "/confirm", {}, "busy-task").then(function (payload) {
      apply(payload);
      setStatus(t("trained-round") + payload.distance.round);
    }).catch(fail);
  });

  on("reset-mapper", "click", function () {
    if (!window.confirm(t("confirm-reset"))) {
      return;
    }
    post("/api/sessions/" + sessionId + "/reset-mapper", {confirm: true}).then(apply).catch(fail);
  });

  on("train-classification", "click", function () {
    if (!snapshot || busy) {
      return;
    }
    viewMode = "recommend";
    pinnedSlots = null;
    rightMode = "task";
    renderRightMode();
    postJob("/api/sessions/" + sessionId + "/tasks/classification", {
      classes: snapshot.task_draft.classes,
      epochs: 40,
      fractions: readFractions(),
      features: selectedFeatures()
    }, "busy-task").then(apply).catch(fail);
  });

  on("train-ranking", "click", function () {
    if (!snapshot || busy) {
      return;
    }
    viewMode = "recommend";
    pinnedSlots = null;
    rightMode = "task";
    renderRightMode();
    postJob("/api/sessions/" + sessionId + "/tasks/ranking", {
      classes: snapshot.task_draft.classes,
      order: snapshot.task_draft.order,
      strength: Number(byId("pref-strength").value),
      epochs: 40,
      features: selectedFeatures()
    }, "busy-task").then(apply).catch(fail);
  });

  on("show-results", "click", function () {
    if (!snapshot || !hasSavedResults()) {
      return;
    }
    viewMode = "results";
    resultView = "saved";
    pinnedSlots = null;
    page = 0;
    renderImages();
    renderChrome();
  });

  on("run-predict", "click", function () {
    if (!snapshot || busy || !hasActiveTask()) {
      return;
    }
    viewMode = "results";
    resultView = "pool";
    pinnedSlots = null;
    page = 0;
    postJob("/api/sessions/" + sessionId + "/tasks/project", {
      count: resultLimit()
    }, "busy-task").then(apply).catch(fail);
  });

  on("edge-table-toggle", "click", function () {
    edgeTableOpen = !edgeTableOpen;
    renderEdges();
  });

  ["frac-train", "frac-val", "frac-test"].forEach(function (id) {
    on(id, "input", function () { lastFrac = id.replace("frac-", ""); });
    on(id, "change", function () {
      lastFrac = id.replace("frac-", "");
      applyFractionEdit();
    });
  });

  on("label-file", "change", function () {
    var file = byId("label-file") && byId("label-file").files && byId("label-file").files[0];
    if (!file) {
      return;
    }
    readImportFile(file);
  });

  on("import-labels", "click", function () {
    var file = byId("label-file") && byId("label-file").files && byId("label-file").files[0];
    if (!file) {
      setStatus(t("choose-csv"));
      return;
    }
    if (pendingCsv) {
      openImportDialog();
      return;
    }
    readImportFile(file);
  });

  on("import-add-label", "click", function () {
    importLabelPicks.push("");
    renderImportRows();
  });

  on("import-cancel", "click", function () {
    var dialog = byId("import-dialog");
    if (dialog) {
      dialog.hidden = true;
    }
  });

  on("import-panel", "submit", function (event) {
    event.preventDefault();
    confirmImport();
  });

  function showList(mode) {
    viewMode = mode;
    labeledPick = null;
    pinnedSlots = null;
    page = 0;
    renderImages();
    renderChrome();
  }

  on("mode-recommend", "click", function () { showList("recommend"); });
  on("mode-labeled", "click", function () { showList("labeled"); });
  on("mode-recycle", "click", function () { showList("recycle"); });
  on("show-labeled", "click", function () {
    showList(viewMode === "labeled" ? "recommend" : "labeled");
  });
  on("mode-mapper", "click", function () {
    rightMode = "mapper";
    renderRightMode();
  });
  on("mode-task", "click", function () {
    rightMode = "task";
    renderRightMode();
  });
  on("mapper-size", "change", function () {
    if (!snapshot || busy) {
      return;
    }
    var node = byId("mapper-size");
    var hidden = node ? Number(node.value) : 0;
    if (!hidden || hidden === Number(snapshot.mapper_hidden)) {
      return;
    }
    post("/api/sessions/" + sessionId + "/mapper-size", {hidden_dim: hidden}).then(function (payload) {
      apply(payload);
      setStatus(t("mapper-reset"));
    }).catch(fail);
  });
  on("head-size", "change", function () {
    if (!snapshot || busy) {
      return;
    }
    var node = byId("head-size");
    var picked = null;
    (snapshot.head_sizes || []).forEach(function (option) {
      if (node && option.id === node.value) {
        picked = option;
      }
    });
    if (!picked) {
      return;
    }
    post("/api/sessions/" + sessionId + "/head-size", {hidden: picked.hidden}).then(apply).catch(fail);
  });
  on("source-mapped", "click", function () { setFeatureSource("mapped"); });
  on("source-raw", "click", function () { setFeatureSource("raw"); });
  on("mode-graph", "click", function () {
    centerMode = "graph";
    renderCenter();
  });
  on("mode-umap", "click", function () {
    if (busy) {
      return;
    }
    centerMode = "umap";
    renderCenter();
  });

  on("result-count", "change", function () {
    if (viewMode !== "results") {
      return;
    }
    pinnedSlots = null;
    page = 0;
    if (resultView === "pool" && hasActiveTask()) {
      postJob("/api/sessions/" + sessionId + "/tasks/project", {
        count: resultLimit()
      }, "busy-task").then(apply).catch(fail);
      return;
    }
    renderImages();
  });

  on("leaf-only", "change", function () {
    if (viewMode === "labeled") {
      pinnedSlots = null;
      page = 0;
      renderImages();
    }
  });

  on("export-labels", "click", function () {
    if (!sessionId || busy) {
      return;
    }
    setBusy(true, "busy-export", null);
    fetch("/api/sessions/" + sessionId + "/labels.csv").then(function (response) {
      if (!response.ok) {
        throw new Error(response.statusText || "export failed");
      }
      if (!response.blob) {
        window.location = "/api/sessions/" + sessionId + "/labels.csv";
        return null;
      }
      return response.blob();
    }).then(function (blob) {
      if (!blob || typeof URL === "undefined" || !URL.createObjectURL) {
        return;
      }
      var url = URL.createObjectURL(blob);
      var link = document.createElement("a");
      link.href = url;
      link.download = "labels.csv";
      if (document.body) {
        document.body.appendChild(link);
      }
      if (link.click) {
        link.click();
      }
      if (link.parentElement) {
        link.parentElement.removeChild(link);
      }
      URL.revokeObjectURL(url);
    }).catch(fail).finally(function () {
      setBusy(false);
    });
  });

  on("page-prev", "click", function () {
    pinnedSlots = null;
    page -= 1;
    renderImages();
  });

  on("page-next", "click", function () {
    pinnedSlots = null;
    page += 1;
    renderImages();
  });

  on("magnifier-toggle", "click", function () {
    setMagnifier(!magnifierOn);
  });

  ["freeze-off", "freeze-on"].forEach(function (id) {
    on(id, "click", function () {
      saveUi({frozen: id === "freeze-on"});
    });
  });

  on("edge-strength", "click", function () { saveUi({edge_mode: "strength"}); });
  on("edge-distance", "click", function () { saveUi({edge_mode: "distance"}); });
  on("shot-small", "click", function () {
    page = 0;
    pinnedSlots = null;
    saveUi({shot_size: "small"});
  });
  on("shot-large", "click", function () {
    page = 0;
    pinnedSlots = null;
    saveUi({shot_size: "large"});
  });

  on("lang-toggle", "click", function () {
    setLanguage(lang === "zh" ? "en" : "zh");
  });

  on("add-class", "click", function () {
    if (!snapshot) {
      return;
    }
    var name = typedName("task-class-name");
    if (!name) {
      return;
    }
    var classes = taskClasses();
    classes.push({name: name, category_ids: []});
    saveDraft(classes);
  });

  on("clear-tasks", "click", function () {
    if (!snapshot) {
      return;
    }
    saveDraft([]);
  });

  document.addEventListener("keydown", function (event) {
    var tag = event.target && event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }
    if (event.key === "z" || event.key === "Z") {
      setMagnifier(!magnifierOn);
    }
    if ((event.key === "f" || event.key === "F") && snapshot) {
      saveUi({frozen: !snapshot.frozen});
    }
  });

  document.addEventListener("mousemove", function (event) {
    if (!magnifierOn || !event.target || !event.target.closest) {
      return;
    }
    var plate = event.target.closest(".plate, .score-thumb, .preview-shot");
    if (plate) {
      showLoupe(plate, event);
    } else {
      hideLoupe();
    }
  });

  function setMagnifier(on) {
    magnifierOn = !!on;
    if (document.body && document.body.dataset) {
      document.body.dataset.magnifier = magnifierOn ? "on" : "off";
    }
    var button = byId("magnifier-toggle");
    if (button) {
      button.setAttribute("aria-pressed", magnifierOn ? "true" : "false");
      button.textContent = magnifierOn ? t("magnifier-on") : t("magnifier-off");
    }
    if (!magnifierOn) {
      hideLoupe();
    }
  }

  function saveUi(patch) {
    if (!sessionId) {
      return;
    }
    post("/api/sessions/" + sessionId + "/ui", patch).then(apply).catch(fail);
  }

  function mappedFeaturesAllowed() {
    return !!(snapshot && snapshot.mapper_trained);
  }

  function selectedFeatures() {
    if (!mappedFeaturesAllowed()) {
      return "raw";
    }
    return snapshot && snapshot.feature_source === "raw" ? "raw" : "mapped";
  }

  function syncFeatureSource() {
    var mapped = byId("source-mapped");
    if (mapped) {
      mapped.disabled = busy || !mappedFeaturesAllowed();
    }
    var source = selectedFeatures();
    setPressed("source-mapped", source === "mapped");
    setPressed("source-raw", source !== "mapped");
  }

  function setFeatureSource(source) {
    if (!snapshot || !sessionId || busy) {
      return;
    }
    if (source === "mapped" && !mappedFeaturesAllowed()) {
      return;
    }
    if (selectedFeatures() === source) {
      return;
    }
    post("/api/sessions/" + sessionId + "/feature-source", {features: source}).then(apply).catch(fail);
  }

  function typedName(promptKey, current) {
    var name = current === undefined ? window.prompt(t(promptKey)) : window.prompt(t(promptKey), current);
    if (!name) {
      return null;
    }
    if (String(name).indexOf("|") >= 0) {
      setStatus(t("name-bar"));
      return null;
    }
    return name;
  }

  function renderCenter() {
    var showUmap = centerMode === "umap";
    var graph = byId("graph");
    var crumbs = byId("crumbs");
    var board = byId("umap-board");
    if (graph) {
      graph.hidden = showUmap;
    }
    if (crumbs) {
      crumbs.hidden = showUmap;
    }
    if (board) {
      board.hidden = !showUmap;
    }
    setPressed("mode-graph", !showUmap);
    setPressed("mode-umap", showUmap);
    paintUmapStatus();
    if (showUmap) {
      drawUmaps();
      maybeRefreshUmap();
    }
  }

  function paintUmapStatus() {
    var node = byId("umap-status");
    if (!node) {
      return;
    }
    if (centerMode !== "umap") {
      node.hidden = true;
      return;
    }
    if (umapLoading) {
      node.hidden = false;
      node.className = "umap-status";
      node.textContent = t("umap-loading");
      return;
    }
    if (umapError) {
      node.hidden = false;
      node.className = "umap-status is-error";
      node.textContent = umapError;
      return;
    }
    node.hidden = true;
    node.textContent = "";
  }

  function umapFailure(error) {
    umapLoading = false;
    var message = error && error.message ? String(error.message) : "";
    if (!message || message === "empty response" || message.indexOf("Unexpected") === 0 || message.indexOf("JSON") >= 0) {
      umapError = t("umap-read-error");
    } else {
      umapError = message;
    }
    paintUmapStatus();
    fail(error);
  }

  function labelColorKey(snap) {
    return ((snap && snap.label_colors) || []).map(function (item) {
      return item.index + ":" + item.color;
    }).join(",");
  }

  function maybeRefreshUmap() {
    if (centerMode !== "umap" || busy || umapLoading || !sessionId || !snapshot) {
      return;
    }
    var colorKey = labelColorKey(snapshot);
    if (umapLayout && umapToken === snapshot.layout_token && umapColorKey === colorKey && !umapError) {
      return;
    }
    umapError = "";
    umapLoading = true;
    paintUmapStatus();
    postJob("/api/sessions/" + sessionId + "/umap", {}, "busy-graph").then(function (payload) {
      umapLoading = false;
      var layout = payload && payload.umap;
      if (!layout || !layout.raw || !layout.mapped) {
        umapError = t("umap-empty");
        paintUmapStatus();
        if (payload && payload.snapshot) {
          apply(payload);
        }
        return;
      }
      umapLayout = layout;
      umapToken = layout.token;
      umapColorKey = labelColorKey(payload.snapshot || snapshot);
      umapError = "";
      apply(payload);
    }).catch(umapFailure);
  }

  function drawUmaps() {
    drawUmapPane("umap-raw", umapLayout && umapLayout.raw);
    drawUmapPane("umap-mapped", umapLayout && umapLayout.mapped);
  }

  function drawUmapPane(id, points) {
    var svg = byId(id);
    if (!svg) {
      return;
    }
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    var rows = points || [];
    if (!rows.length) {
      return;
    }
    var minX = Infinity;
    var maxX = -Infinity;
    var minY = Infinity;
    var maxY = -Infinity;
    rows.forEach(function (point) {
      var x = Number(point.x);
      var y = Number(point.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      if (x < minX) {
        minX = x;
      }
      if (x > maxX) {
        maxX = x;
      }
      if (y < minY) {
        minY = y;
      }
      if (y > maxY) {
        maxY = y;
      }
    });
    if (!Number.isFinite(minX)) {
      return;
    }
    var spanX = maxX - minX || 1;
    var spanY = maxY - minY || 1;
    var pad = 8;
    var width = 100;
    var height = 100;
    svg.setAttribute("viewBox", "0 0 " + (width + pad * 2) + " " + (height + pad * 2));
    function place(point) {
      var x = Number(point.x);
      var y = Number(point.y);
      var px = pad + width / 2;
      var py = pad + height / 2;
      if (Number.isFinite(x) && maxX !== minX) {
        px = pad + ((x - minX) / spanX) * width;
      }
      if (Number.isFinite(y) && maxY !== minY) {
        py = pad + (1 - (y - minY) / spanY) * height;
      }
      return {x: px, y: py};
    }
    function add(point, fill, radius) {
      var pos = place(point);
      var node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      node.setAttribute("cx", String(pos.x));
      node.setAttribute("cy", String(pos.y));
      node.setAttribute("r", String(radius));
      node.setAttribute("fill", fill);
      if (!point || !point.color) {
        node.setAttribute("pointer-events", "none");
      }
      svg.appendChild(node);
      return node;
    }
    var background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    background.setAttribute("x", "0");
    background.setAttribute("y", "0");
    background.setAttribute("width", "100%");
    background.setAttribute("height", "100%");
    background.setAttribute("fill", "transparent");
    background.addEventListener("click", function () {
      umapPinnedPoint = null;
      hideUmapDetail();
    });
    svg.appendChild(background);
    rows.forEach(function (point) {
      add(point, "rgba(90,100,112,0.22)", 0.85);
    });
    rows.forEach(function (point) {
      if (!point.color) {
        return;
      }
      add(point, point.color, 1.35);
      var pos = place(point);
      var hit = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      hit.setAttribute("class", "umap-hit");
      hit.setAttribute("cx", String(pos.x));
      hit.setAttribute("cy", String(pos.y));
      hit.setAttribute("r", "4");
      hit.setAttribute("fill", "transparent");
      hit.addEventListener("mouseenter", function () {
        showUmapDetail(point);
      });
      hit.addEventListener("mouseleave", function () {
        if (umapPinnedPoint && umapPinnedPoint.index === point.index) {
          return;
        }
        if (umapPinnedPoint) {
          showUmapDetail(umapPinnedPoint);
          return;
        }
        hideUmapDetail();
      });
      hit.addEventListener("click", function (event) {
        if (event.stopPropagation) {
          event.stopPropagation();
        }
        umapPinnedPoint = point;
        showUmapDetail(point);
      });
      svg.appendChild(hit);
    });
  }

  function showUmapDetail(point) {
    var box = byId("umap-detail");
    if (!box || !point) {
      return;
    }
    box.hidden = false;
    var img = byId("umap-detail-img");
    if (img) {
      img.alt = String(point.index);
      img.src = "/media/" + sessionId + "/" + point.index;
    }
    var name = byId("umap-detail-name");
    if (name) {
      name.textContent = point.category ? point.category : ("#" + point.index);
    }
  }

  function hideUmapDetail() {
    var box = byId("umap-detail");
    if (box) {
      box.hidden = true;
    }
  }

  function render() {
    if (document.body && document.body.dataset) {
      document.body.dataset.layout = "board";
      document.body.dataset.frozen = snapshot.frozen ? "true" : "false";
      document.body.dataset.shot = snapshot.shot_size || "small";
    }
    renderMechanisms();
    renderModelSizes();
    renderRightMode();
    renderChrome();
    renderImages();
    renderGraph();
    renderCenter();
    renderEdges();
    renderCrumbs();
    renderDistances();
    renderScoreBoard();
    renderWells();
    renderTaskClasses();
    renderTaskList();
    renderMapperTags();
    renderTaskSummary();
    var confirm = byId("confirm");
    if (confirm) {
      confirm.disabled = busy || !snapshot.can_confirm;
    }
    var strength = byId("pref-strength");
    if (strength && snapshot.task_draft) {
      strength.value = snapshot.task_draft.strength;
    }
  }

  function renderMechanisms() {
    var mechanism = byId("mechanism");
    if (!mechanism) {
      return;
    }
    var previous = mechanism.value;
    mechanism.innerHTML = "";
    (snapshot.mechanisms || []).forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = t("mech-" + name);
      mechanism.appendChild(option);
    });
    if (previous) {
      mechanism.value = previous;
    }
  }

  function renderRightMode() {
    setPressed("mode-mapper", rightMode === "mapper");
    setPressed("mode-task", rightMode === "task");
    var mapper = byId("panel-mapper");
    var task = byId("panel-task");
    if (mapper) {
      mapper.hidden = rightMode !== "mapper";
    }
    if (task) {
      task.hidden = rightMode !== "task";
    }
  }

  function renderModelSizes() {
    if (!snapshot) {
      return;
    }
    fillSizeSelect(
      "mapper-size",
      snapshot.mapper_sizes || [],
      String(snapshot.mapper_hidden || ""),
      function (option) {
        return option.hidden_dim + " · " + formatCount(option.parameters);
      },
      "hidden_dim"
    );
    var head = snapshot.head_hidden || [];
    fillSizeSelect(
      "head-size",
      snapshot.head_sizes || [],
      head.length ? head[0] + "x" + head[1] : "",
      function (option) {
        return option.hidden[0] + "×" + option.hidden[1] + " · " + formatCount(option.parameters);
      },
      "id"
    );
  }

  function fillSizeSelect(id, options, selected, label, valueKey) {
    var node = byId(id);
    if (!node) {
      return;
    }
    var signature = options.map(function (option) {
      var value = valueKey === "id" ? option.id : option.hidden_dim;
      return value + ":" + option.parameters;
    }).join("|") + "@" + selected;
    if (node.getAttribute && node.getAttribute("data-built") === signature) {
      return;
    }
    node.innerHTML = "";
    options.forEach(function (option) {
      var item = document.createElement("option");
      item.value = valueKey === "id" ? option.id : String(option.hidden_dim);
      item.textContent = label(option);
      node.appendChild(item);
    });
    if (selected) {
      node.value = selected;
    }
    if (node.setAttribute) {
      node.setAttribute("data-built", signature);
    }
  }

  function renderChrome() {
    var frozen = !!snapshot.frozen;
    var edgeMode = snapshot.edge_mode || "distance";
    var shot = snapshot.shot_size || "small";
    setPressed("freeze-off", !frozen);
    setPressed("freeze-on", frozen);
    setPressed("edge-strength", edgeMode === "strength");
    setPressed("edge-distance", edgeMode === "distance");
    setPressed("shot-small", shot !== "large");
    setPressed("shot-large", shot === "large");
    var freezeHint = byId("freeze-hint");
    if (freezeHint) {
      freezeHint.textContent = frozen ? t("freeze-hint-on") : t("freeze-hint-off");
    }
    var edgeHint = byId("edge-hint");
    if (edgeHint) {
      edgeHint.textContent = edgeMode === "strength" ? t("edge-hint-strength") : t("edge-hint-distance");
    }
    var note = byId("graph-note");
    if (note) {
      if (centerMode === "umap") {
        note.textContent = t("umap-note");
      } else {
        note.textContent = edgeMode === "strength" ? t("graph-note-strength") : t("graph-note-distance");
      }
    }
    syncFeatureSource();
    setPressed("mode-graph", centerMode !== "umap");
    setPressed("mode-umap", centerMode === "umap");
    var emptyRecommend = byId("empty-recommend");
    if (emptyRecommend && emptyRecommend.hidden) {
      emptyRecommend.textContent = t("empty-recommend");
    } else if (emptyRecommend) {
      emptyRecommend.textContent = t("empty-recommend");
    }
    var emptyTask = byId("empty-task");
    if (emptyTask) {
      emptyTask.textContent = t("empty-task");
    }
    var labeled = byId("show-labeled");
    if (labeled) {
      var showing = viewMode === "labeled";
      labeled.setAttribute("aria-pressed", showing ? "true" : "false");
      labeled.textContent = showing ? t("show-recommend") : t("show-labeled");
    }
    setPressed("mode-recommend", viewMode === "recommend");
    setPressed("mode-labeled", viewMode === "labeled");
    setPressed("mode-recycle", viewMode === "recycle");
    var leafRow = byId("leaf-only-row");
    if (leafRow) {
      leafRow.hidden = viewMode !== "labeled";
    }
    var factor = byId("scope-factor");
    if (factor && document.activeElement !== factor) {
      factor.value = String(snapshot.scope_factor || 10);
    }
    renderRightMode();
    if (!busy) {
      syncIdleControls();
    }
    var recycleNote = byId("recycle-note");
    if (recycleNote && snapshot.recycle) {
      recycleNote.textContent = t("recycle-count").replace("{n}", snapshot.recycle.count);
    }
  }

  function setPressed(id, on) {
    var node = byId(id);
    if (node) {
      node.setAttribute("aria-checked", on ? "true" : "false");
    }
  }

  function byScoreDesc(rows) {
    var scored = rows.some(function (image) {
      return Number.isFinite(Number(image.score));
    });
    if (!scored) {
      return rows;
    }
    return rows.slice().sort(function (a, b) {
      var as = Number(a.score);
      var bs = Number(b.score);
      var aOk = Number.isFinite(as);
      var bOk = Number.isFinite(bs);
      if (aOk && bOk && as !== bs) {
        return bs - as;
      }
      return Number(a.index) - Number(b.index);
    });
  }

  function roundImages() {
    var unplaced = (snapshot.images || []).filter(function (image) {
      return image.role === "unplaced";
    });
    if (unplaced.length) {
      return byScoreDesc(unplaced);
    }
    return (snapshot.images || []).slice();
  }

  function gridShape(root) {
    var large = (snapshot.shot_size || "small") === "large";
    var cols = large ? 1 : 3;
    var minRows = large ? 1 : 3;
    var rect = root.getBoundingClientRect ? root.getBoundingClientRect() : {width: 0, height: 0};
    var width = root.clientWidth || rect.width || 0;
    var height = root.clientHeight || rect.height || 0;
    var rows = minRows;
    if (width > 0 && height > 0) {
      var imageWidth = width / cols;
      var rowHeight = imageWidth + CAPTION_H + SHOT_GAP;
      var fitted = Math.floor((height + SHOT_GAP) / rowHeight);
      if (fitted > rows) {
        rows = fitted;
      }
    }
    return {cols: cols, rows: rows, size: cols * rows};
  }

  function resultLimit() {
    var node = byId("result-count");
    var value = node ? Math.floor(Number(node.value)) : 9;
    if (!Number.isFinite(value) || value < 1) {
      return 1;
    }
    return value;
  }

  function onlyLeaves() {
    var node = byId("leaf-only");
    if (!node) {
      return true;
    }
    return !!node.checked;
  }

  function hasActiveTask() {
    return (snapshot.tasks || []).some(function (task) { return task.active; });
  }

  function hasSavedResults() {
    var summary = snapshot.task_summary;
    if (!summary) {
      return false;
    }
    if (summary.kind === "classification") {
      return Array.isArray(summary.inference);
    }
    if (summary.kind === "ranking") {
      return !!(snapshot.score_board && snapshot.score_board.bins);
    }
    return false;
  }

  function boardImages(board) {
    var rows = [];
    ((board && board.bins) || []).forEach(function (bin) {
      (bin.images || []).forEach(function (image) {
        rows.push({
          index: image.index,
          role: "result",
          score: Number(image.score),
          label: image.label || null,
          color: image.color || null
        });
      });
    });
    return rows;
  }

  function savedResultRows() {
    var summary = snapshot.task_summary;
    if (summary && summary.kind === "classification" && Array.isArray(summary.inference)) {
      return summary.inference.map(function (item) {
        return {
          index: item.index,
          role: "result",
          score: Number(item.score),
          label: item.label || null,
          color: item.color || null
        };
      });
    }
    if (summary && summary.kind === "ranking") {
      return boardImages(snapshot.score_board);
    }
    return [];
  }

  function labeledItems() {
    var rows = (snapshot.labeled_images || []).filter(function (image) {
      if (!onlyLeaves()) {
        return true;
      }
      var parent = snapshot.view_parent;
      if (!parent) {
        return !image.parent_id;
      }
      return image.category_id === parent;
    });
    if (!labeledPick || !labeledPick.length) {
      return rows;
    }
    var byIndex = {};
    var rest = [];
    rows.forEach(function (image) {
      byIndex[image.index] = image;
    });
    var ordered = [];
    labeledPick.forEach(function (index) {
      if (byIndex[index]) {
        ordered.push(byIndex[index]);
        delete byIndex[index];
      }
    });
    rows.forEach(function (image) {
      if (byIndex[image.index]) {
        rest.push(image);
      }
    });
    return ordered.concat(rest);
  }

  function imageSource() {
    if (viewMode === "results") {
      var rows = resultView === "pool" && snapshot.projection && snapshot.projection.images
        ? snapshot.projection.images.map(function (item) {
          return {
            index: item.index,
            role: "result",
            score: Number(item.score),
            label: item.label || null,
            color: item.color || null
          };
        })
        : savedResultRows();
      return byScoreDesc(rows).slice(0, resultLimit());
    }
    if (viewMode === "labeled") {
      return labeledItems();
    }
    if (viewMode === "recycle") {
      var bin = snapshot.recycle || {};
      return (bin.images || []).slice();
    }
    return roundImages();
  }

  function renderImages() {
    var root = byId("images");
    if (!root || !snapshot) {
      return;
    }
    var items = imageSource();
    var byIndex = {};
    items.forEach(function (image) {
      byIndex[image.index] = image;
    });
    var shape = gridShape(root);
    var size = shape.size;
    var pages = Math.max(1, Math.ceil(items.length / size));
    if (page >= pages) {
      page = pages - 1;
    }
    if (page < 0) {
      page = 0;
    }
    root.style.gridTemplateColumns = "repeat(" + shape.cols + ", minmax(0, 1fr))";
    if (!pinnedSlots || pinnedSlots.length !== size) {
      var start = page * size;
      pinnedSlots = [];
      for (var i = 0; i < size; i++) {
        pinnedSlots.push(items[start + i] ? items[start + i].index : null);
      }
    } else {
      var kept = pinnedSlots.map(function (index) {
        return index !== null && byIndex[index] ? index : null;
      });
      var visible = {};
      kept.forEach(function (index) {
        if (index !== null) {
          visible[index] = true;
        }
      });
      var queue = [];
      items.forEach(function (image) {
        if (!visible[image.index]) {
          queue.push(image.index);
        }
      });
      var next = 0;
      pinnedSlots = kept.map(function (index) {
        if (index !== null) {
          return index;
        }
        if (next < queue.length) {
          return queue[next++];
        }
        return null;
      });
    }
    root.innerHTML = "";
    pinnedSlots.forEach(function (index) {
      if (index === null || !byIndex[index]) {
        root.appendChild(emptyShot());
      } else {
        root.appendChild(shotCard(byIndex[index]));
      }
    });
    var note = byId("page-note");
    if (note) {
      note.textContent = items.length
        ? t("page-label").replace("{page}", page + 1).replace("{pages}", pages)
        : (viewMode === "recycle" ? t("page-recycle") : t("page-empty"));
    }
    var show = byId("show-results");
    if (show) {
      show.disabled = busy || !hasSavedResults();
    }
    var predict = byId("run-predict");
    if (predict) {
      predict.disabled = busy || !hasActiveTask();
    }
  }

  function emptyShot() {
    var fig = document.createElement("div");
    fig.className = "shot-empty";
    var plate = document.createElement("div");
    plate.className = "plate";
    var caption = document.createElement("div");
    caption.className = "shot-caption";
    fig.appendChild(plate);
    fig.appendChild(caption);
    return fig;
  }

  function shotCard(image) {
    var fig = document.createElement("figure");
    fig.className = "shot";
    var card = document.createElement("div");
    card.className = "thumb";
    card.draggable = true;
    card.dataset.index = String(image.index);
    var plate = document.createElement("div");
    plate.className = "plate";
    var img = document.createElement("img");
    img.alt = String(image.index);
    img.draggable = false;
    img.src = "/media/" + sessionId + "/" + image.index;
    plate.appendChild(img);
    card.appendChild(plate);
    card.addEventListener("dragstart", function (event) {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("text/plain", String(image.index));
    });
    fig.appendChild(card);
    var border = image.color || image.prediction_color;
    if (border) {
      card.style.boxShadow = "inset 0 0 0 4px " + border;
    }
    var caption = document.createElement("figcaption");
    var name = document.createElement("span");
    name.className = "cap-name";
    if (image.label && typeof image.score === "number" && Number.isFinite(image.score)) {
      name.textContent = image.label + "  " + image.score.toFixed(2);
    } else if (typeof image.score === "number" && Number.isFinite(image.score)) {
      name.textContent = "#" + image.index + "  " + image.score.toFixed(2);
    } else {
      name.textContent = image.prediction ? image.prediction : ("#" + image.index);
    }
    caption.appendChild(name);
    if (image.role === "waiting") {
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-sample";
      remove.textContent = t("remove-sample");
      remove.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        removeSample(image.index);
      });
      caption.appendChild(remove);
    }
    if (image.role === "labeled") {
      var recycle = document.createElement("button");
      recycle.type = "button";
      recycle.className = "remove-sample";
      recycle.textContent = t("recycle-sample");
      recycle.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        post("/api/sessions/" + sessionId + "/recycle", {
          sample_index: image.index
        }).then(apply).catch(fail);
      });
      caption.appendChild(recycle);
    }
    fig.appendChild(caption);
    return fig;
  }

  function renderCrumbs() {
    var root = byId("crumbs");
    if (!root) {
      return;
    }
    root.innerHTML = "";
    var rootButton = document.createElement("button");
    rootButton.type = "button";
    rootButton.textContent = t("root");
    rootButton.addEventListener("click", function () {
      labeledPick = null;
      pinnedSlots = null;
      page = 0;
      post("/api/sessions/" + sessionId + "/view", {category_id: null}).then(apply).catch(fail);
    });
    root.appendChild(rootButton);
    (snapshot.breadcrumb || []).forEach(function (crumb) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = crumb.name;
      button.addEventListener("click", function () {
        labeledPick = null;
        pinnedSlots = null;
        page = 0;
        post("/api/sessions/" + sessionId + "/view", {category_id: crumb.id}).then(apply).catch(fail);
      });
      root.appendChild(button);
    });
  }

  function allowDrop(event) {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function drawnCategoryAt(graph, clientX, clientY) {
    var cards = graph.querySelectorAll(".category");
    for (var i = cards.length - 1; i >= 0; i--) {
      var rect = cards[i].getBoundingClientRect();
      if (clientX >= rect.left && clientX < rect.right && clientY >= rect.top && clientY < rect.bottom) {
        return cards[i];
      }
    }
    return null;
  }

  function strokeWidth(edge) {
    var mode = snapshot.edge_mode || "distance";
    var value = mode === "distance" ? edge.distance : edge.strength;
    if (value === null || value === undefined) {
      return 2;
    }
    return Math.max(1, 2 + Number(value) * 6);
  }

  function outlineWidth(cat) {
    var mode = snapshot.edge_mode || "distance";
    var value = mode === "distance" ? cat.outline_distance : cat.outline_strength;
    if (value === null || value === undefined) {
      return 1;
    }
    return Math.max(1, 1 + Number(value) * 4);
  }

  function renderGraph() {
    var graph = byId("graph");
    if (!graph) {
      return;
    }
    graph.innerHTML = "";
    graph.ondragenter = allowDrop;
    graph.ondragover = allowDrop;
    graph.ondrop = function (event) {
      var index = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
      if (!index) {
        return;
      }
      event.preventDefault();
      var card = drawnCategoryAt(graph, event.clientX, event.clientY);
      if (card) {
        event.stopPropagation();
        placeImage(Number(index), card.dataset.id, null);
        return;
      }
      var name = typedName("new-category");
      if (!name) {
        return;
      }
      placeImage(Number(index), null, name);
    };
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "edges-svg");
    svg.setAttribute("pointer-events", "none");
    svg.ondragenter = allowDrop;
    svg.ondragover = allowDrop;
    graph.appendChild(svg);
    var count = snapshot.categories.length;
    var centers = {};
    var size = graphSize(graph);
    function redrawLines() {
      svg.querySelectorAll("line").forEach(function (line) {
        var from = centers[line.getAttribute("data-a")];
        var to = centers[line.getAttribute("data-b")];
        if (!from || !to) {
          return;
        }
        line.setAttribute("x1", from.x);
        line.setAttribute("y1", from.y);
        line.setAttribute("x2", to.x);
        line.setAttribute("y2", to.y);
      });
    }
    snapshot.categories.forEach(function (cat, i) {
      var card = document.createElement("div");
      card.className = "cat-card category";
      card.dataset.id = cat.id;
      var tagColor = cat.color || "#2f5d73";
      card.style["--c"] = tagColor;
      card.style.borderWidth = outlineWidth(cat) + "px";
      card.style.borderLeftWidth = "6px";
      card.style.borderColor = tagColor;
      card.style.borderLeftColor = tagColor;
      var point = categoryPoint(cat, i, count, size);
      centers[cat.id] = point;
      card.style.left = point.x + "px";
      card.style.top = point.y + "px";
      var top = document.createElement("div");
      top.className = "card-top";
      var accent = document.createElement("i");
      accent.className = "accent";
      var title = document.createElement("strong");
      title.textContent = cat.name;
      top.appendChild(accent);
      top.appendChild(title);
      var countLine = document.createElement("span");
      countLine.className = "card-count";
      countLine.textContent = cat.count + t("images-unit");
      card.appendChild(top);
      card.appendChild(countLine);
      appendPreview(card, cat.recent || [], cat.representative || []);
      card.addEventListener("mousedown", function (event) {
        if (event.button && event.button !== 0) {
          return;
        }
        if (snapshot.frozen) {
          startTagDrag(event, cat, card);
          return;
        }
        if (dragBlocked(event.target, card)) {
          return;
        }
        event.preventDefault();
        var originX = event.clientX;
        var originY = event.clientY;
        var startLeft = parseFloat(card.style.left) || 0;
        var startTop = parseFloat(card.style.top) || 0;
        var moved = false;
        function onMove(moveEvent) {
          var dx = moveEvent.clientX - originX;
          var dy = moveEvent.clientY - originY;
          if (Math.abs(dx) + Math.abs(dy) > 3) {
            moved = true;
          }
          var next = clampPoint(startLeft + dx, startTop + dy, graphSize(graph));
          centers[cat.id] = next;
          card.style.left = next.x + "px";
          card.style.top = next.y + "px";
          redrawLines();
        }
        function onUp() {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          if (!moved) {
            return;
          }
          card._suppressClick = true;
          var current = graphSize(graph);
          post("/api/sessions/" + sessionId + "/position", {
            category_id: cat.id,
            x: (parseFloat(card.style.left) || 0) / current.width,
            y: (parseFloat(card.style.top) || 0) / current.height
          }).then(apply).catch(fail);
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
      card.addEventListener("dragover", function (event) { event.preventDefault(); });
      card.addEventListener("drop", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var index = event.dataTransfer.getData("text/plain");
        if (!index) {
          return;
        }
        placeImage(Number(index), cat.id, null);
      });
      var clickTimer = null;
      card.addEventListener("dblclick", function () {
        clearTimeout(clickTimer);
        viewMode = "labeled";
        labeledPick = null;
        pinnedSlots = null;
        page = 0;
        post("/api/sessions/" + sessionId + "/view", {category_id: cat.id}).then(apply).catch(fail);
      });
      card.addEventListener("click", function () {
        if (card._suppressClick) {
          card._suppressClick = false;
          return;
        }
        clearTimeout(clickTimer);
        clickTimer = setTimeout(function () {
          var name = typedName("rename-category", cat.name);
          if (!name || name === cat.name) {
            return;
          }
          post("/api/sessions/" + sessionId + "/rename", {
            category_id: cat.id,
            name: name
          }).then(apply).catch(fail);
        }, 250);
      });
      graph.appendChild(card);
    });
    (snapshot.strengths || []).forEach(function (edge) {
      if (!centers[edge.a] || !centers[edge.b]) {
        return;
      }
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("data-a", edge.a);
      line.setAttribute("data-b", edge.b);
      line.setAttribute("x1", centers[edge.a].x);
      line.setAttribute("y1", centers[edge.a].y);
      line.setAttribute("x2", centers[edge.b].x);
      line.setAttribute("y2", centers[edge.b].y);
      line.setAttribute("stroke", "#243140");
      line.setAttribute("stroke-width", String(strokeWidth(edge)));
      line.setAttribute("pointer-events", "none");
      svg.appendChild(line);
      var hit = document.createElementNS("http://www.w3.org/2000/svg", "line");
      hit.setAttribute("class", "edge-hit");
      hit.setAttribute("data-hit", "1");
      hit.setAttribute("data-a", edge.a);
      hit.setAttribute("data-b", edge.b);
      hit.setAttribute("x1", centers[edge.a].x);
      hit.setAttribute("y1", centers[edge.a].y);
      hit.setAttribute("x2", centers[edge.b].x);
      hit.setAttribute("y2", centers[edge.b].y);
      hit.setAttribute("stroke", "transparent");
      hit.setAttribute("stroke-width", "14");
      hit.setAttribute("pointer-events", "stroke");
      hit.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        toggleEdge(edge.a, edge.b);
      });
      svg.appendChild(hit);
    });
    paintEdgeSelection();
  }

  function placeImage(index, categoryId, newName) {
    var body = {sample_index: index};
    if (newName) {
      body.new_name = newName;
    } else {
      body.category_id = categoryId;
    }
    post("/api/sessions/" + sessionId + "/place", body).then(apply).catch(fail);
  }

  function startTagDrag(event, cat, card) {
    event.preventDefault();
    var originX = event.clientX;
    var originY = event.clientY;
    var moved = false;
    showGhost(cat, event.clientX, event.clientY);
    function onMove(moveEvent) {
      if (Math.abs(moveEvent.clientX - originX) + Math.abs(moveEvent.clientY - originY) > 5) {
        moved = true;
      }
      moveGhost(moveEvent.clientX, moveEvent.clientY);
      markWell(moveEvent.clientX, moveEvent.clientY);
    }
    function onUp(upEvent) {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      var target = moved ? markWell(upEvent.clientX, upEvent.clientY) : null;
      hideGhost();
      if (!target) {
        return;
      }
      if (target.className.indexOf("task-class") >= 0) {
        dropOnTaskClass(Number(target.getAttribute("data-class-index")), cat.id);
        return;
      }
      dropOnWell(target.getAttribute("data-well"), cat.id);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function showGhost(cat, x, y) {
    var ghost = byId("ghost");
    if (!ghost) {
      return;
    }
    ghost.hidden = false;
    ghost.textContent = cat.name + " · " + categoryCount(cat) + t("images-unit");
    ghost.style.background = cat.color || "#2f5d73";
    ghost.style.left = x + "px";
    ghost.style.top = y + "px";
    if (document.body) {
      document.body.className += document.body.className.indexOf("is-tag-dragging") >= 0 ? "" : " is-tag-dragging";
    }
  }

  function moveGhost(x, y) {
    var ghost = byId("ghost");
    if (!ghost) {
      return;
    }
    ghost.style.left = x + "px";
    ghost.style.top = y + "px";
  }

  function hideGhost() {
    var ghost = byId("ghost");
    if (ghost) {
      ghost.hidden = true;
    }
    if (document.body) {
      document.body.className = document.body.className.replace(" is-tag-dragging", "");
    }
    hideWellMarks();
  }

  function markWell(x, y) {
    hideWellMarks();
    if (!document.elementFromPoint) {
      return null;
    }
    var hit = document.elementFromPoint(x, y);
    var taskClass = hit && hit.closest ? hit.closest(".task-class") : null;
    var well = taskClass ? null : (hit && hit.closest ? hit.closest(".well") : null);
    var target = taskClass || well;
    if (target && target.className.indexOf("is-over") < 0) {
      target.className += " is-over";
    }
    return target;
  }

  function hideWellMarks() {
    var recommend = byId("well-recommend");
    if (recommend) {
      recommend.className = recommend.className.replace(" is-over", "");
    }
    var root = byId("task-classes");
    if (!root || !root.querySelectorAll) {
      return;
    }
    root.querySelectorAll(".task-class").forEach(function (box) {
      box.className = box.className.replace(" is-over", "");
    });
  }

  function dropOnWell(key, categoryId) {
    if (key !== "recommend") {
      return;
    }
    var ids = (snapshot.recommend_set || []).slice();
    if (ids.indexOf(categoryId) < 0) {
      ids.push(categoryId);
    }
    saveUi({recommend_ids: ids});
  }

  function dropOnTaskClass(classIndex, categoryId) {
    var classes = taskClasses();
    if (!classes[classIndex]) {
      return;
    }
    classes.forEach(function (item, index) {
      if (index === classIndex) {
        return;
      }
      item.category_ids = item.category_ids.filter(function (id) { return id !== categoryId; });
    });
    if (classes[classIndex].category_ids.indexOf(categoryId) < 0) {
      classes[classIndex].category_ids.push(categoryId);
    }
    saveDraft(classes);
  }

  function taskClasses() {
    return ((snapshot.task_draft && snapshot.task_draft.classes) || []).map(function (item) {
      return {name: item.name, category_ids: (item.category_ids || []).slice()};
    });
  }

  function saveDraft(classes) {
    post("/api/sessions/" + sessionId + "/task-draft", {
      classes: classes,
      order: classes.map(function (item) { return item.name; }),
      strength: Number((byId("pref-strength") && byId("pref-strength").value) || 1)
    }).then(apply).catch(fail);
  }

  function currentIds() {
    return (snapshot.categories || []).map(function (cat) { return cat.id; });
  }

  function onWellAction(key, act) {
    if (key !== "recommend") {
      return;
    }
    if (act === "clear") {
      saveUi({recommend_ids: []});
      return;
    }
    saveUi({recommend_ids: currentIds()});
  }

  function removeTag(key, categoryId, classIndex) {
    if (key === "recommend") {
      saveUi({
        recommend_ids: (snapshot.recommend_set || []).filter(function (id) { return id !== categoryId; })
      });
      return;
    }
    var classes = taskClasses();
    if (!classes[classIndex]) {
      return;
    }
    classes[classIndex].category_ids = classes[classIndex].category_ids.filter(function (id) {
      return id !== categoryId;
    });
    saveDraft(classes);
  }

  function renderWells() {
    var list = byId("list-recommend");
    var empty = byId("empty-recommend");
    if (!list) {
      return;
    }
    var tags = (snapshot.recommend_set || []).map(function (id) {
      return categoryRecord(id);
    }).filter(function (cat) { return !!cat; });
    list.innerHTML = "";
    if (empty) {
      empty.hidden = tags.length !== 0;
      empty.textContent = t("empty-recommend");
    }
    tags.forEach(function (cat) {
      list.appendChild(categoryTag(cat, function () {
        removeTag("recommend", cat.id);
      }));
    });
  }

  function renderTaskClasses() {
    var root = byId("task-classes");
    if (!root) {
      return;
    }
    var classes = taskClasses();
    var empty = byId("empty-task");
    if (empty) {
      empty.hidden = classes.length !== 0;
      empty.textContent = t("empty-task");
    }
    root.innerHTML = "";
    classes.forEach(function (taskClass, classIndex) {
      var box = document.createElement("section");
      box.className = "task-class";
      box.setAttribute("data-class-index", String(classIndex));
      var head = document.createElement("div");
      head.className = "task-class-head";
      var title = document.createElement("strong");
      title.textContent = taskClass.name;
      var up = document.createElement("button");
      up.type = "button";
      up.textContent = t("up");
      up.addEventListener("click", function () { moveClass(classIndex, -1); });
      var down = document.createElement("button");
      down.type = "button";
      down.textContent = t("down");
      down.addEventListener("click", function () { moveClass(classIndex, 1); });
      var renameClass = document.createElement("button");
      renameClass.type = "button";
      renameClass.textContent = t("rename-class");
      renameClass.addEventListener("click", function () {
        var name = typedName("task-class-name", taskClass.name);
        if (!name || name === taskClass.name) {
          return;
        }
        var next = taskClasses();
        next[classIndex].name = name;
        saveDraft(next);
      });
      var removeClass = document.createElement("button");
      removeClass.type = "button";
      removeClass.className = "class-delete";
      removeClass.textContent = t("delete-task");
      removeClass.setAttribute("aria-label", t("remove-class"));
      removeClass.addEventListener("click", function () {
        var next = taskClasses();
        next.splice(classIndex, 1);
        saveDraft(next);
      });
      var actions = document.createElement("div");
      actions.className = "task-class-actions";
      actions.appendChild(up);
      actions.appendChild(down);
      actions.appendChild(renameClass);
      actions.appendChild(removeClass);
      head.appendChild(title);
      head.appendChild(actions);
      var tags = document.createElement("div");
      tags.className = "tag-list";
      (taskClass.category_ids || []).forEach(function (id) {
        var cat = categoryRecord(id);
        if (!cat) {
          return;
        }
        tags.appendChild(categoryTag(cat, function () {
          removeTag("task", id, classIndex);
        }));
      });
      box.appendChild(head);
      box.appendChild(tags);
      root.appendChild(box);
    });
  }

  function moveClass(index, delta) {
    var classes = taskClasses();
    var next = index + delta;
    if (next < 0 || next >= classes.length) {
      return;
    }
    var swap = classes[next];
    classes[next] = classes[index];
    classes[index] = swap;
    saveDraft(classes);
  }

  function categoryTag(cat, onRemove) {
    var row = document.createElement("div");
    row.className = "tag";
    var color = cat.color || "#2f5d73";
    row.style.background = color;
    row.style.color = "#fff";
    row.style["--c"] = color;
    var name = document.createElement("span");
    name.className = "tag-name";
    name.textContent = cat.name;
    var count = document.createElement("span");
    count.className = "tag-count";
    count.textContent = categoryCount(cat) + t("images-unit");
    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "tag-x";
    remove.textContent = "×";
    remove.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      onRemove();
    });
    row.appendChild(name);
    row.appendChild(count);
    row.appendChild(remove);
    return row;
  }

  function categoryCount(cat) {
    if (cat && typeof cat.count === "number") {
      return cat.count;
    }
    return 0;
  }

  function categoryRecord(id) {
    var lists = [snapshot.categories || [], snapshot.all_categories || []];
    for (var i = 0; i < lists.length; i++) {
      for (var j = 0; j < lists[i].length; j++) {
        if (lists[i][j].id === id) {
          return lists[i][j];
        }
      }
    }
    return null;
  }

  function appendPreview(card, recent, representative) {
    if (!recent.length && !representative.length) {
      return;
    }
    var strip = document.createElement("div");
    strip.className = "preview-strip";
    recent.forEach(function (index) { strip.appendChild(previewShot(index)); });
    if (recent.length && representative.length) {
      var bar = document.createElement("span");
      bar.className = "preview-bar";
      strip.appendChild(bar);
    }
    representative.forEach(function (index) { strip.appendChild(previewShot(index)); });
    card.appendChild(strip);
  }

  function previewShot(index) {
    var wrap = document.createElement("span");
    wrap.className = "preview-shot";
    wrap.appendChild(previewImage(index));
    return wrap;
  }

  function previewImage(index) {
    var img = document.createElement("img");
    img.alt = String(index);
    img.draggable = false;
    img.src = "/media/" + sessionId + "/" + index;
    return img;
  }

  function removeSample(index) {
    post("/api/sessions/" + sessionId + "/unassign", {
      sample_index: Number(index)
    }).then(apply).catch(fail);
  }

  function renderScoreBoard() {
    var root = byId("score-board");
    if (!root) {
      return;
    }
    root.innerHTML = "";
    var board = snapshot.score_board;
    if (!board || !board.bins) {
      return;
    }
    var title = document.createElement("h3");
    var kindLabel = board.kind === "ranking" ? t("score-rank") : t("score-class");
    title.textContent = board.name ? board.name + " — " + kindLabel : kindLabel;
    root.appendChild(title);
    var row = document.createElement("div");
    row.className = "score-bins";
    board.bins.forEach(function (bin) {
      var column = document.createElement("div");
      column.className = "score-bin";
      var heading = document.createElement("div");
      heading.textContent = Number(bin.low).toFixed(2) + " – " + Number(bin.high).toFixed(2);
      column.appendChild(heading);
      (bin.images || []).forEach(function (image) {
        var thumb = document.createElement("div");
        thumb.className = "score-thumb" + (image.in_graph ? " in-graph" : "");
        thumb.draggable = true;
        var plate = document.createElement("div");
        plate.className = "plate";
        plate.appendChild(previewImage(image.index));
        thumb.appendChild(plate);
        thumb.addEventListener("dragstart", function (event) {
          event.dataTransfer.effectAllowed = "copy";
          event.dataTransfer.setData("text/plain", String(image.index));
        });
        if (image.label) {
          var caption = document.createElement("div");
          caption.textContent = image.label;
          thumb.appendChild(caption);
        }
        column.appendChild(thumb);
      });
      row.appendChild(column);
    });
    root.appendChild(row);
  }

  function graphSize(graph) {
    var rect = graph.getBoundingClientRect();
    return {
      width: graph.clientWidth || rect.width || 700,
      height: graph.clientHeight || rect.height || 460
    };
  }

  function categoryPoint(cat, index, count, size) {
    if (typeof cat.x === "number" && typeof cat.y === "number") {
      return clampPoint(cat.x * size.width, cat.y * size.height, size);
    }
    var angle = -Math.PI / 2 + (count ? (index * 2 * Math.PI) / count : 0);
    var radius = Math.min(size.width, size.height) * 0.28;
    return {
      x: size.width / 2 + radius * Math.cos(angle),
      y: size.height / 2 + radius * Math.sin(angle)
    };
  }

  function clampPoint(x, y, size) {
    return {
      x: Math.max(0, Math.min(size.width, x)),
      y: Math.max(0, Math.min(size.height, y))
    };
  }

  function dragBlocked(target, card) {
    var node = target;
    while (node && node !== card) {
      var tag = String(node.tagName || "").toUpperCase();
      if (tag === "BUTTON" || tag === "IMG") {
        return true;
      }
      node = node.parentElement;
    }
    return false;
  }

  function renderTaskList() {
    var root = byId("saved-tasks");
    if (!root) {
      return;
    }
    root.innerHTML = "";
    (snapshot.tasks || []).forEach(function (task) {
      var wrap = document.createElement("div");
      wrap.className = "saved-task";
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = task.name;
      if (task.active) {
        button.className = "active-task";
        button.setAttribute("aria-pressed", "true");
      }
      button.addEventListener("click", function () {
        post("/api/sessions/" + sessionId + "/tasks/select", {task_id: task.id}).then(apply).catch(fail);
      });
      var tools = document.createElement("div");
      tools.className = "saved-task-tools";
      var rename = document.createElement("button");
      rename.type = "button";
      rename.textContent = t("rename-class");
      rename.addEventListener("click", function () {
        var name = typedName("rename-task", task.name);
        if (!name || name === task.name) {
          return;
        }
        post("/api/sessions/" + sessionId + "/tasks/rename", {
          task_id: task.id,
          name: name
        }).then(apply).catch(fail);
      });
      var remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = t("delete-task");
      remove.addEventListener("click", function () {
        if (!window.confirm(t("confirm-delete-task"))) {
          return;
        }
        post("/api/sessions/" + sessionId + "/tasks/delete", {
          task_id: task.id
        }).then(apply).catch(fail);
      });
      tools.appendChild(rename);
      tools.appendChild(remove);
      wrap.appendChild(button);
      wrap.appendChild(tools);
      root.appendChild(wrap);
    });
  }

  function renderMapperTags() {
    var root = byId("mapper-tags");
    if (!root || !snapshot) {
      return;
    }
    root.innerHTML = "";
    (snapshot.all_categories || []).forEach(function (cat) {
      var row = document.createElement("div");
      row.className = "tag source-tag";
      var color = cat.color || "#2f5d73";
      row.style.background = color;
      row.style.color = "#fff";
      row.style["--c"] = color;
      var name = document.createElement("span");
      name.className = "tag-name";
      name.textContent = cat.name;
      row.appendChild(name);
      row.addEventListener("mousedown", function (event) {
        if (event.button && event.button !== 0) {
          return;
        }
        startTagDrag(event, cat, row);
      });
      root.appendChild(row);
    });
  }

  function edgeKey(a, b) {
    return String(a) + "\t" + String(b);
  }

  function toggleEdge(a, b) {
    var key = edgeKey(a, b);
    if (selectedEdge === key) {
      selectedEdge = null;
    } else {
      selectedEdge = key;
      var rows = sortedEdges();
      var at = -1;
      rows.forEach(function (edge, index) {
        if (edgeKey(edge.a, edge.b) === key) {
          at = index;
        }
      });
      if (at >= 0) {
        edgePage = Math.floor(at / EDGE_PAGE);
      }
      edgeTableOpen = true;
    }
    renderEdges();
  }

  function edgeLabel(edge) {
    return edge.a_name + " — " + edge.b_name;
  }

  function edgeMetric(edge) {
    var raw = (snapshot.edge_mode || "distance") === "distance" ? edge.distance : edge.strength;
    if (raw === null || raw === undefined || raw === "") {
      return null;
    }
    var number = Number(raw);
    return Number.isFinite(number) ? number : null;
  }

  function sortedEdges() {
    var rows = (snapshot.strengths || []).slice();
    var dir = edgeSort.dir;
    rows.sort(function (a, b) {
      if (edgeSort.key === "value") {
        var av = edgeMetric(a);
        var bv = edgeMetric(b);
        if (av === null && bv === null) {
          return 0;
        }
        if (av === null) {
          return 1;
        }
        if (bv === null) {
          return -1;
        }
        if (av === bv) {
          return edgeLabel(a).localeCompare(edgeLabel(b));
        }
        return (av - bv) * dir;
      }
      return edgeLabel(a).localeCompare(edgeLabel(b)) * dir;
    });
    return rows;
  }

  function paintEdgeSelection() {
    var root = byId("edges");
    if (root && root.querySelectorAll) {
      root.querySelectorAll(".edge-stat").forEach(function (row) {
        var on = selectedEdge === edgeKey(row.getAttribute("data-a"), row.getAttribute("data-b"));
        row.className = on ? "edge-stat is-selected" : "edge-stat";
      });
    }
    var graph = byId("graph");
    if (!graph || !graph.querySelectorAll) {
      return;
    }
    graph.querySelectorAll("line").forEach(function (line) {
      if (line.getAttribute("data-hit") === "1") {
        return;
      }
      var on = selectedEdge === edgeKey(line.getAttribute("data-a"), line.getAttribute("data-b"));
      line.setAttribute("stroke", on ? "#c45911" : "#243140");
    });
  }

  function renderEdges() {
    var toggle = byId("edge-table-toggle");
    if (toggle) {
      toggle.textContent = edgeTableOpen ? t("edge-hide") : t("edge-show");
      toggle.setAttribute("aria-pressed", edgeTableOpen ? "true" : "false");
    }
    var root = byId("edges");
    if (!root || !snapshot) {
      return;
    }
    root.hidden = !edgeTableOpen;
    root.innerHTML = "";
    if (!edgeTableOpen) {
      paintEdgeSelection();
      return;
    }
    var rows = sortedEdges();
    var pages = Math.max(1, Math.ceil(rows.length / EDGE_PAGE));
    if (edgePage >= pages) {
      edgePage = pages - 1;
    }
    if (edgePage < 0) {
      edgePage = 0;
    }
    var table = document.createElement("table");
    table.className = "edge-stats";
    var head = document.createElement("tr");
    head.appendChild(edgeHeader(t("edge-pair"), "pair"));
    head.appendChild(edgeHeader(t("edge-value"), "value"));
    table.appendChild(head);
    rows.slice(edgePage * EDGE_PAGE, edgePage * EDGE_PAGE + EDGE_PAGE).forEach(function (edge) {
      var row = document.createElement("tr");
      row.className = "edge-stat";
      row.setAttribute("data-a", edge.a);
      row.setAttribute("data-b", edge.b);
      var pair = document.createElement("td");
      pair.textContent = edgeLabel(edge);
      var metric = edgeMetric(edge);
      var value = document.createElement("td");
      value.textContent = metric === null ? "" : metric.toFixed(2);
      row.appendChild(pair);
      row.appendChild(value);
      row.addEventListener("click", function () {
        var next = window.prompt(t("strength-prompt"), String(edge.strength));
        if (next === null) {
          return;
        }
        post("/api/sessions/" + sessionId + "/strength", {
          a: edge.a,
          b: edge.b,
          strength: Number(next)
        }).then(apply).catch(fail);
      });
      table.appendChild(row);
    });
    root.appendChild(table);
    if (pages > 1) {
      var bar = document.createElement("div");
      bar.className = "edge-pages";
      var prev = document.createElement("button");
      prev.type = "button";
      prev.textContent = t("page-prev");
      prev.disabled = edgePage === 0;
      prev.addEventListener("click", function () {
        edgePage -= 1;
        renderEdges();
      });
      var note = document.createElement("span");
      note.textContent = t("page-label").replace("{page}", edgePage + 1).replace("{pages}", pages);
      var next = document.createElement("button");
      next.type = "button";
      next.textContent = t("page-next");
      next.disabled = edgePage >= pages - 1;
      next.addEventListener("click", function () {
        edgePage += 1;
        renderEdges();
      });
      bar.appendChild(prev);
      bar.appendChild(note);
      bar.appendChild(next);
      root.appendChild(bar);
    }
    paintEdgeSelection();
  }

  function edgeHeader(label, key) {
    var th = document.createElement("th");
    var button = document.createElement("button");
    button.type = "button";
    button.className = "edge-sort";
    button.textContent = label;
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (edgeSort.key === key) {
        edgeSort.dir = -edgeSort.dir;
      } else {
        edgeSort.key = key;
        edgeSort.dir = 1;
      }
      edgePage = 0;
      renderEdges();
    });
    th.appendChild(button);
    return th;
  }

  function renderDistances() {
    var table = byId("distances");
    if (!table) {
      return;
    }
    table.innerHTML = "";
    var head = document.createElement("tr");
    [t("col-round"), t("within"), t("between"), t("loss")].forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      head.appendChild(th);
    });
    table.appendChild(head);
    (snapshot.distances || []).forEach(function (row) {
      var tr = document.createElement("tr");
      [row.round, format(row.mean_within), format(row.mean_between), format(row.loss)].forEach(function (value) {
        var td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
  }

  function format(value) {
    if (value === null || value === undefined) {
      return "";
    }
    return Number(value).toFixed(3);
  }

  function formatMetric(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return "";
    }
    return Number(value).toFixed(3);
  }

  function formatSummary(summary) {
    if (!summary) {
      return "";
    }
    if (summary.kind === "ranking") {
      var rankLines = ["n=" + summary.n];
      if (formatMetric(summary.raw_pairwise_accuracy)) {
        rankLines.push(t("raw-pair") + " " + formatMetric(summary.raw_pairwise_accuracy));
      }
      if (formatMetric(summary.mapped_pairwise_accuracy)) {
        rankLines.push(t("mapped-pair") + " " + formatMetric(summary.mapped_pairwise_accuracy));
      }
      return rankLines.join("\n");
    }
    if (summary.kind !== "classification") {
      return "";
    }
    var fractions = summary.fractions || {};
    var lines = [
      t("frac-train") + " " + Number(fractions.train).toFixed(2)
        + "  " + t("frac-val") + " " + Number(fractions.val).toFixed(2)
        + "  " + t("frac-test") + " " + Number(fractions.test).toFixed(2)
    ];
    ["train", "val", "test"].forEach(function (name) {
      var split = (summary.splits && summary.splits[name]) || {n: 0};
      var parts = [t("split-" + name), "n=" + split.n];
      if (split.n) {
        if (formatMetric(split.raw_accuracy)) {
          parts.push(t("raw-acc") + " " + formatMetric(split.raw_accuracy));
        }
        if (formatMetric(split.mapped_accuracy)) {
          parts.push(t("mapped-acc") + " " + formatMetric(split.mapped_accuracy));
        }
        if (formatMetric(split.raw_roc_auc)) {
          parts.push(t("raw-auc") + " " + formatMetric(split.raw_roc_auc));
        }
        if (formatMetric(split.mapped_roc_auc)) {
          parts.push(t("mapped-auc") + " " + formatMetric(split.mapped_roc_auc));
        }
      }
      lines.push(parts.join("  "));
    });
    return lines.join("\n");
  }

  function renderTaskSummary() {
    var node = byId("task-result");
    if (!node) {
      return;
    }
    node.textContent = formatSummary(snapshot && snapshot.task_summary);
  }

  function basisPoints(id) {
    var node = byId(id);
    var value = node ? Number(node.value) : 0;
    if (!Number.isFinite(value)) {
      return 0;
    }
    return Math.round(value * 100);
  }

  function writeFractions(train, val, test) {
    var values = [train, val, test];
    ["frac-train", "frac-val", "frac-test"].forEach(function (id, index) {
      var node = byId(id);
      if (node) {
        node.value = (values[index] / 100).toFixed(2);
      }
    });
  }

  function applyFractionEdit() {
    var train = basisPoints("frac-train");
    var val = Math.max(0, basisPoints("frac-val"));
    var test = Math.max(0, basisPoints("frac-test"));
    if (train < 1) {
      train = 1;
    }
    if (train > 100) {
      train = 100;
    }
    if (lastFrac === "val") {
      if (val > 100 - train) {
        val = 100 - train;
      }
      test = 100 - train - val;
    } else if (lastFrac === "test") {
      if (test > 100 - train) {
        test = 100 - train;
      }
      val = 100 - train - test;
    } else {
      var rest = 100 - train;
      var denom = val + test;
      if (denom <= 0) {
        val = rest;
        test = 0;
      } else {
        val = Math.round(rest * val / denom);
        test = rest - val;
        if (test < 0) {
          val += test;
          test = 0;
        }
      }
    }
    writeFractions(train, val, test);
  }

  function readFractions() {
    var train = byId("frac-train");
    var val = byId("frac-val");
    var test = byId("frac-test");
    if (!train || !val || !test) {
      return {train: 0.6, val: 0.2, test: 0.2};
    }
    applyFractionEdit();
    return {
      train: Number(train.value),
      val: Number(val.value),
      test: Number(test.value)
    };
  }

  function showLoupe(plate, event) {
    var loupe = byId("loupe");
    var inner = byId("loupe-inner");
    var source = plate.querySelector ? plate.querySelector("img") : null;
    if (!loupe || !inner || !source) {
      return;
    }
    var rect = source.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return;
    }
    var zoom = 2.5;
    var lens = 180;
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    var clone = source.cloneNode(true);
    clone.style.width = (rect.width * zoom) + "px";
    clone.style.height = (rect.height * zoom) + "px";
    clone.style.left = (lens / 2 - x * zoom) + "px";
    clone.style.top = (lens / 2 - y * zoom) + "px";
    inner.innerHTML = "";
    inner.appendChild(clone);
    var left = event.clientX + 18;
    var top = event.clientY - lens - 16;
    if (top < 8) {
      top = event.clientY + 20;
    }
    loupe.style.left = left + "px";
    loupe.style.top = top + "px";
    loupe.hidden = false;
  }

  function hideLoupe() {
    var loupe = byId("loupe");
    if (loupe) {
      loupe.hidden = true;
    }
  }

  function applyStaticText() {
    Object.keys(TEXT_IDS).forEach(function (id) {
      var node = byId(id);
      if (node) {
        node.textContent = t(TEXT_IDS[id]);
      }
    });
    var labels = {
      "freeze-group": "freeze-label",
      "edge-group": "edge-label",
      "shot-group": "shot-label",
      "center-modes": "center-label",
      "feature-source": "source-label"
    };
    Object.keys(labels).forEach(function (id) {
      var node = byId(id);
      if (node) {
        node.setAttribute("aria-label", t(labels[id]));
      }
    });
    if (document.documentElement) {
      document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    }
    try {
      document.title = t("title");
    } catch (err) {
      /* The drag harness has no title. */
    }
    var toggle = byId("lang-toggle");
    if (toggle) {
      toggle.textContent = lang === "zh" ? "English" : "中文";
    }
    setMagnifier(magnifierOn);
  }

  function setLanguage(next) {
    lang = next === "en" ? "en" : "zh";
    try {
      if (window.localStorage) {
        window.localStorage.setItem("hitl-board-lang", lang);
      }
    } catch (err) {
      /* Keep the choice for this page view. */
    }
    applyStaticText();
    if (snapshot) {
      render();
    }
  }

  function csvHeader(text) {
    var line = String(text || "").split(/\r?\n/)[0] || "";
    return parseCsvRow(line).filter(function (name) { return name.length; });
  }

  function parseCsvRow(line) {
    var out = [];
    var cur = "";
    var quoted = false;
    for (var i = 0; i < line.length; i++) {
      var ch = line.charAt(i);
      if (quoted) {
        if (ch === '"') {
          if (line.charAt(i + 1) === '"') {
            cur += '"';
            i += 1;
          } else {
            quoted = false;
          }
        } else {
          cur += ch;
        }
      } else if (ch === '"') {
        quoted = true;
      } else if (ch === ",") {
        out.push(cur.trim());
        cur = "";
      } else {
        cur += ch;
      }
    }
    out.push(cur.trim());
    return out;
  }

  function readImportFile(file) {
    if (!sessionId) {
      setStatus(t("import-open-session"));
      return;
    }
    var reader = new FileReader();
    reader.onload = function () {
      pendingCsv = String(reader.result || "");
      importHeaders = csvHeader(pendingCsv);
      if (importHeaders.length < 2) {
        setStatus(t("import-need-columns"));
        return;
      }
      openImportDialog();
    };
    reader.onerror = function () {
      setStatus(t("choose-csv"));
    };
    reader.readAsText(file);
  }

  function openImportDialog() {
    var index = byId("import-index");
    if (!index || !importHeaders.length) {
      return;
    }
    index.innerHTML = "";
    importHeaders.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      index.appendChild(option);
    });
    var indexName = importHeaders.indexOf("index") >= 0 ? "index" : importHeaders[0];
    index.value = indexName;
    var levels = importHeaders.filter(function (name) {
      return /^label_[1-9][0-9]*$/.test(name);
    }).sort(function (a, b) {
      return Number(a.slice(6)) - Number(b.slice(6));
    });
    if (!levels.length) {
      levels = importHeaders.filter(function (name) { return name !== indexName; }).slice(0, 1);
    }
    importLabelPicks = levels.slice();
    renderImportRows();
    var dialog = byId("import-dialog");
    if (dialog) {
      dialog.hidden = false;
    }
  }

  function renderImportRows() {
    var root = byId("import-label-rows");
    if (!root) {
      return;
    }
    root.innerHTML = "";
    importLabelPicks.forEach(function (picked, rowIndex) {
      var row = document.createElement("div");
      row.className = "import-label-row";
      var label = document.createElement("span");
      label.textContent = t("import-level") + " " + (rowIndex + 1);
      var select = document.createElement("select");
      importHeaders.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
      if (picked) {
        select.value = picked;
      }
      select.addEventListener("change", function () {
        importLabelPicks[rowIndex] = select.value;
      });
      var up = document.createElement("button");
      up.type = "button";
      up.textContent = t("up");
      up.disabled = rowIndex === 0;
      up.addEventListener("click", function () {
        var swap = importLabelPicks[rowIndex - 1];
        importLabelPicks[rowIndex - 1] = select.value;
        importLabelPicks[rowIndex] = swap;
        renderImportRows();
      });
      var down = document.createElement("button");
      down.type = "button";
      down.textContent = t("down");
      down.disabled = rowIndex === importLabelPicks.length - 1;
      down.addEventListener("click", function () {
        var swap = importLabelPicks[rowIndex + 1];
        importLabelPicks[rowIndex + 1] = select.value;
        importLabelPicks[rowIndex] = swap;
        renderImportRows();
      });
      var remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.disabled = importLabelPicks.length === 1;
      remove.addEventListener("click", function () {
        importLabelPicks.splice(rowIndex, 1);
        renderImportRows();
      });
      row.appendChild(label);
      row.appendChild(select);
      row.appendChild(up);
      row.appendChild(down);
      row.appendChild(remove);
      root.appendChild(row);
      importLabelPicks[rowIndex] = select.value;
    });
  }

  function confirmImport() {
    if (!sessionId || !pendingCsv) {
      return;
    }
    var index = byId("import-index");
    var indexColumn = index ? index.value : "";
    var labels = importLabelPicks.slice();
    var seen = {};
    var bad = !indexColumn || !labels.length;
    labels.forEach(function (name) {
      if (!name || name === indexColumn || seen[name]) {
        bad = true;
      }
      seen[name] = true;
    });
    if (bad) {
      setStatus(t("import-bad-columns"));
      return;
    }
    var dialog = byId("import-dialog");
    if (dialog) {
      dialog.hidden = true;
    }
    postJob("/api/sessions/" + sessionId + "/import-labels", {
      csv: pendingCsv,
      index_column: indexColumn,
      label_columns: labels
    }, "busy-graph").then(function (payload) {
      apply(payload);
      var imported = payload.imported || {};
      setStatus(t("imported").replace("{n}", imported.n_assigned).replace("{c}", imported.n_categories));
    }).catch(fail);
  }

  bindStaticWellButtons();
  setLanguage(lang);
  refreshProjects();

  function bindStaticWellButtons() {
    ["well-recommend"].forEach(function (id) {
      var well = byId(id);
      if (!well || !well.querySelectorAll) {
        return;
      }
      well.querySelectorAll("button").forEach(function (button) {
        var act = button.getAttribute && button.getAttribute("data-act");
        if (!act) {
          return;
        }
        button.addEventListener("click", function () {
          onWellAction(button.getAttribute("data-well"), act);
        });
      });
    });
  }
})();
