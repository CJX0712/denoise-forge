/* _uicheck.js：最小 DOM stub，把 index.html 的 UI 脚本拉起来跑一遍，点遍所有控件 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const uiS = html.match(/<script id="ui">([\s\S]*?)<\/script>/);
if (!eng || !uiS) { console.log("FAIL 找不到 engine/ui 脚本块"); process.exit(1); }

let pass = 0, fail = 0; const fails = [];
function T(n, c, note) { if (c) { pass++; console.log("PASS " + n + (note ? "  " + note : "")); } else { fail++; fails.push(n); console.log("FAIL " + n + (note ? "  " + note : "")); } }

/* ---- DOM stub ---- */
let timerCalls = 0;
function makeCtx2d(el) {
  const rec = { calls: [], fillStyle: "", strokeStyle: "", font: "", lineWidth: 1 };
  const push = (n) => rec.calls.push(n);
  return {
    _rec: rec,
    clearRect: () => push("clearRect"), fillRect: () => push("fillRect"),
    beginPath: () => push("beginPath"), moveTo: () => push("moveTo"), lineTo: () => push("lineTo"),
    stroke: () => push("stroke"), fill: () => push("fill"), fillText: () => push("fillText"),
    arc: () => push("arc"), closePath: () => push("closePath"),
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => push("putImageData"),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    drawImage: () => push("drawImage"), save: () => { }, restore: () => { }, translate: () => { }, scale: () => { },
    set fillStyle(v) { rec.fillStyle = v; }, get fillStyle() { return rec.fillStyle; },
    set strokeStyle(v) { rec.strokeStyle = v; }, get strokeStyle() { return rec.strokeStyle; },
    set font(v) { rec.font = v; }, get font() { return rec.font; },
    set lineWidth(v) { rec.lineWidth = v; }, get lineWidth() { return rec.lineWidth; }
  };
}
function makeEl(tag, id) {
  const el = {
    tagName: (tag || "div").toUpperCase(), id: id || "", children: [], _ctx: null,
    value: "", textContent: "", className: "", disabled: false,
    clientWidth: 520, clientHeight: 160, width: 0, height: 0, scrollTop: 0, scrollHeight: 0,
    style: {}, classList: { add() { }, remove() { }, toggle() { } },
    appendChild(c) { el.children.push(c); return c; },
    removeChild(c) { const i = el.children.indexOf(c); if (i >= 0) el.children.splice(i, 1); },
    addEventListener() { }, removeEventListener() { },
    getContext() { if (!el._ctx) el._ctx = makeCtx2d(el); return el._ctx; },  // 必须缓存
    getBoundingClientRect() { return { width: 520, height: 160, left: 0, top: 0 }; }
  };
  // innerHTML 必须是真 setter：赋空串要清空 children（否则 UI 用 innerHTML='' 清列表会失效）
  let _html = "";
  Object.defineProperty(el, "innerHTML", {
    get() { return _html; },
    set(v) { _html = String(v); if (_html === "") el.children.length = 0; },
    enumerable: true, configurable: true
  });
  return el;
}
const els = {};
function getEl(id) { if (!els[id]) { els[id] = makeEl(id === "lossCv" || id === "schCv" || id === "trajCv" ? "canvas" : "div", id); if (els[id].tagName === "CANVAS") els[id].clientWidth = 520; } return els[id]; }
// select 初始值
getEl("dsSel").value = "center";
getEl("stepsIn").value = "10";
getEl("batchIn").value = "8";
getEl("lrIn").value = "0.002";

const document = {
  readyState: "complete",
  getElementById: (id) => getEl(id),
  createElement: (t) => makeEl(t),
  addEventListener: () => { },
  querySelector: () => null,
  querySelectorAll: () => [],
  body: makeEl("body")
};

const ctx = {
  console, Math, Object, Array, JSON, isFinite, isNaN, Infinity, NaN,
  Float64Array, Int32Array, Uint8Array, Uint8ClampedArray, Number, String, Boolean, Error,
  document: document,
  setTimeout: (fn) => { if (timerCalls++ < 20000) fn(); return timerCalls; },
  clearTimeout: () => { },
  requestAnimationFrame: (fn) => { if (timerCalls++ < 20000) fn(); return timerCalls; }
};
ctx.globalThis = ctx; ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(eng[1], ctx, { filename: "engine.js" });
T("引擎脚本加载成功", !!ctx.IDF);

let uiOk = true, uiErr = "";
try { vm.runInContext(uiS[1], ctx, { filename: "ui.js" }); } catch (e) { uiOk = false; uiErr = e.message; }
T("UI 脚本加载无异常", uiOk, uiErr);
T("UI 导出 __DNF", !!(ctx.__DNF));
if (!uiOk || !ctx.__DNF) { console.log("=== " + pass + " / " + (pass + fail) + " FAIL"); process.exit(1); }

const $ = (id) => getEl(id);
function click(id) { const e = $(id); if (typeof e.onclick === "function") { e.onclick(); return true; } return false; }

/* 1 调度图一开始就该画过 */
T("启动时绘制了 ᾱ 调度图", !!$("schCv")._ctx && $("schCv")._ctx._rec.calls.includes("stroke"));

/* 2 引擎自检 */
click("btnTest");
const sum = $("testSum").innerHTML, out = $("testOut").innerHTML;
const badLines = String(out).split('<div class="selftest">').filter(s => s.indexOf("✗") >= 0)
  .map(s => s.replace(/<[^>]+>/g, "").trim());
fs.writeFileSync(path.join(__dirname, "_selftest.txt"), String(out).replace(/<[^>]+>/g, "\n").replace(/\n{2,}/g, "\n"), "utf8");
T("自检面板有输出", out.length > 0);
T("自检全部通过（无 ✗）", out.indexOf("✗") < 0, String(sum).replace(/<[^>]+>/g, "") + (badLines.length ? " 失败项: " + badLines.join(" | ") : ""));

/* 3 训练（步数已设为 10，setTimeout 同步执行） */
click("btnTrain");
T("训练推进了步数", ctx.__DNF.st.t >= 10, "步数=" + ctx.__DNF.st.t);
T("训练产生了 loss 曲线", ctx.__DNF.st.curve.length >= 10, "点数=" + ctx.__DNF.st.curve.length);
T("训练后重绘了 loss 曲线", !!$("lossCv")._ctx && $("lossCv")._ctx._rec.calls.includes("stroke"));
T("统计栏已更新", /步/.test($("statStep").innerHTML), String($("statStep").innerHTML).replace(/<[^>]+>/g, ""));

/* 4 ancestral 采样 + 轨迹动画 */
click("btnSample");
T("采样生成了样本画布", $("samples").children.length === 4, "画布数=" + $("samples").children.length);
T("轨迹画布已绘制", !!$("trajCv")._ctx && $("trajCv")._ctx._rec.calls.includes("putImageData"));
T("轨迹标签已更新", /t = /.test($("trajLab").innerHTML), String($("trajLab").innerHTML).replace(/<[^>]+>/g, ""));

/* 5 DDIM 采样 */
click("btnDDIM");
T("DDIM 采样生成了样本", $("samples").children.length === 4, "画布数=" + $("samples").children.length);

/* 6 数据集切换 */
$("dsSel").value = "rand";
if (typeof $("dsSel").onchange === "function") $("dsSel").onchange();
T("切换数据集后重建数据", ctx.__DNF.st.data && ctx.__DNF.st.data.length === 256);

/* 7 重置 */
click("btnReset");
T("重置后步数归零", ctx.__DNF.st.t === 0);

/* 8 停止 */
$("stepsIn").value = "10";
click("btnTrain");
click("btnStop");
T("停止标志生效", ctx.__DNF.st.stop === true);

console.log("=== " + pass + " / " + (pass + fail) + (fail ? " FAIL — " + fails.join(", ") : " ALL GREEN"));
fs.writeFileSync(path.join(__dirname, "_uicheck.log"), "PASS " + pass + " / " + (pass + fail) + "\n" + (fail ? "FAIL " + fails.join(", ") : "ALL GREEN") + "\n");
process.exit(fail ? 1 : 0);
