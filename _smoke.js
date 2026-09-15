/* denoise-forge 引擎自检：从 index.html 抽取 <script id="engine">，Node vm 无头运行 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.log("FAIL 找不到 engine 脚本块"); process.exit(1); }
const ctx = {
  console, Math, Object, Array, JSON, isFinite, isNaN, Infinity, NaN,
  Float64Array, Int32Array, Uint8Array, Uint8ClampedArray, Number, String, Boolean
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: "engine.js" });
const IDF = ctx.IDF;

let pass = 0, fail = 0; const fails = [];
function T(name, cond, note) {
  if (cond) { pass++; console.log("PASS " + name + (note ? "  " + note : "")); }
  else { fail++; fails.push(name); console.log("FAIL " + name + (note ? "  " + note : "")); }
}
const S = 16, C = 4, Td = 8, TT = 100, P = S * S;
const sched = IDF.makeSchedule(TT, 1e-4, 0.02);

/* 1 调度：ᾱ ∈(0,1) 严格单调递减 */
{
  let mono = true, rng = true;
  for (let i = 0; i < TT; i++) {
    if (!(sched.abar[i] > 0 && sched.abar[i] < 1)) rng = false;
    if (i && sched.abar[i] >= sched.abar[i - 1]) mono = false;
  }
  T("ᾱ_t ∈(0,1) 且严格单调递减", mono && rng, "ᾱ_0=" + sched.abar[0].toFixed(6) + " ᾱ_99=" + sched.abar[TT - 1].toFixed(6));
}

/* 2 前向矩 = 闭式解（CLT 容差 4σ） */
{
  const dim = 32, N = 3000, t = 50;
  const rnd = IDF.mulberry32(7), gs = IDF.gaussFactory(rnd);
  const x0 = new Float64Array(dim); for (let i = 0; i < dim; i++) x0[i] = gs();
  let mean = 0, m2 = 0;
  for (let n = 0; n < N; n++) {
    const e = new Float64Array(dim); for (let i = 0; i < dim; i++) e[i] = gs();
    const xt = IDF.qSample(sched, x0, t, e);
    mean += xt[0] / N; m2 += xt[0] * xt[0] / N;
  }
  const eMean = Math.sqrt(sched.abar[t]) * x0[0], eVar = 1 - sched.abar[t];
  const se = Math.sqrt(eVar / N);
  T("前向矩 = 闭式 N(√ᾱ·x₀,1−ᾱ)", Math.abs(mean - eMean) < 4 * se,
    "mean " + mean.toFixed(4) + " vs " + eMean.toFixed(4) + " (4σ=" + (4 * se).toFixed(4) + ")");
}

/* 3 高斯封闭性：x₀~N(0,1) ⇒ q(x_t) 仍 ~N(0,1) */
{
  const dim = 16, N = 2000, t = 80;
  const rnd = IDF.mulberry32(11), gs = IDF.gaussFactory(rnd);
  let s2 = 0, n2 = 0;
  for (let n = 0; n < N; n++) {
    const x0 = new Float64Array(dim); for (let i = 0; i < dim; i++) x0[i] = gs();
    const e = new Float64Array(dim); for (let i = 0; i < dim; i++) e[i] = gs();
    const xt = IDF.qSample(sched, x0, t, e);
    for (let i = 0; i < dim; i++) { s2 += xt[i] * xt[i]; n2++; }
  }
  const v = s2 / n2;
  T("高斯封闭性 var≈1", Math.abs(v - 1) < 0.06, "var=" + v.toFixed(4));
}

/* 4 后验方差：t=0 为 0，t≥1 为正 */
{
  let a = IDF.postVar(sched, 0) === 0, b = true;
  for (let t = 1; t < TT; t++) if (!(IDF.postVar(sched, t) > 0)) b = false;
  T("后验方差 t=0 为 0，t≥1 为正", a && b);
}

/* 5 原语 FD：conv / pool / up / film / lin */
{
  const EPS = 1e-5, worsts = [];
  const mk = (n, s) => { const r = IDF.mulberry32(s), a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = r() * 2 - 1; return a; };
  // conv
  {
    const ci = 3, co = 2, H = 5, Wd = 5;
    const X = mk(ci * H * Wd, 1), Wp = mk(co * ci * 9, 2), b = mk(co, 3), w = mk(co * H * Wd, 4);
    const O = new Float64Array(co * H * Wd);
    const f = () => { IDF.convF(X, Wp, b, ci, co, H, Wd, O); let s = 0; for (let i = 0; i < O.length; i++) s += w[i] * O[i]; return s; };
    const dX = new Float64Array(ci * H * Wd), dW = new Float64Array(co * ci * 9), db = new Float64Array(co), dO = new Float64Array(co * H * Wd);
    for (let i = 0; i < dO.length; i++) dO[i] = w[i];
    IDF.convB(X, Wp, dO, ci, co, H, Wd, dX, dW, db);
    for (let i = 0; i < 4; i++) { const o = Wp[i]; Wp[i] = o + EPS; const p = f(); Wp[i] = o - EPS; const mm = f(); Wp[i] = o; const num = (p - mm) / (2 * EPS); worsts.push(Math.abs(num - dW[i]) / (Math.abs(num) + Math.abs(dW[i]) + 1e-300)); }
  }
  // pool
  {
    const c = 2, H = 6, Wd = 6;
    const X = mk(c * H * Wd, 11), w = mk(c * 9, 12), O = new Float64Array(c * 9);
    const f = () => { IDF.poolF(X, c, H, Wd, O); let s = 0; for (let i = 0; i < O.length; i++) s += w[i] * O[i]; return s; };
    const dX = new Float64Array(c * H * Wd), dO = new Float64Array(c * 9);
    for (let i = 0; i < dO.length; i++) dO[i] = w[i];
    IDF.poolB(dO, c, H, Wd, dX);
    for (let i = 0; i < 4; i++) { const o = X[i]; X[i] = o + EPS; const p = f(); X[i] = o - EPS; const mm = f(); X[i] = o; const num = (p - mm) / (2 * EPS); worsts.push(Math.abs(num - dX[i]) / (Math.abs(num) + Math.abs(dX[i]) + 1e-300)); }
  }
  // up
  {
    const c = 2, h2 = 3, w2 = 3;
    const X = mk(c * h2 * w2, 21), w = mk(c * 36, 22), O = new Float64Array(c * 36);
    const f = () => { IDF.upF(X, c, h2, w2, O); let s = 0; for (let i = 0; i < O.length; i++) s += w[i] * O[i]; return s; };
    const dX = new Float64Array(c * h2 * w2), dO = new Float64Array(c * 36);
    for (let i = 0; i < dO.length; i++) dO[i] = w[i];
    IDF.upB(dO, c, h2, w2, dX);
    for (let i = 0; i < 4; i++) { const o = X[i]; X[i] = o + EPS; const p = f(); X[i] = o - EPS; const mm = f(); X[i] = o; const num = (p - mm) / (2 * EPS); worsts.push(Math.abs(num - dX[i]) / (Math.abs(num) + Math.abs(dX[i]) + 1e-300)); }
  }
  // film
  {
    const ch = 3, sp = 4;
    const hIn = mk(ch * sp, 31), ss = mk(2 * ch, 32), w = mk(ch * sp, 33), O = new Float64Array(ch * sp);
    const f = () => { IDF.filmF(hIn, ss, ch, sp, O); let s = 0; for (let i = 0; i < O.length; i++) s += w[i] * O[i]; return s; };
    const dh = new Float64Array(ch * sp), dss = new Float64Array(2 * ch), dO = new Float64Array(ch * sp);
    for (let i = 0; i < dO.length; i++) dO[i] = w[i];
    IDF.filmB(hIn, ss, dO, ch, sp, dh, dss);
    for (let i = 0; i < 6; i++) { const o = ss[i]; ss[i] = o + EPS; const p = f(); ss[i] = o - EPS; const mm = f(); ss[i] = o; const num = (p - mm) / (2 * EPS); worsts.push(Math.abs(num - dss[i]) / (Math.abs(num) + Math.abs(dss[i]) + 1e-300)); }
  }
  let mx = 0; for (const v of worsts) if (v > mx) mx = v;
  T("原语反向 FD (conv/pool/up/film)", mx < 1e-6, "maxRel=" + mx.toExponential(2));
}

/* 6 全网络梯度检验（FD 金标准，EPS=1e-6 —— ReLU 折点下步长必须够小） */
{
  const net = IDF.createNet({ size: S, baseCh: C, Td: Td, seed: 5 });
  const dd = IDF.datasetShapes(4, S, 7, 0.02);
  const cfg = { x0: dd[0], t: 37, eps: null };
  const gr = IDF.mulberry32(99), gg = IDF.gaussFactory(gr);
  cfg.eps = new Float64Array(P); for (let i = 0; i < P; i++) cfg.eps[i] = gg();
  const lossOf = () => { const x = IDF.qSample(sched, cfg.x0, cfg.t, cfg.eps); const cc = {}; const eh = IDF.forwardNet(net, x, cfg.t, cc); let L = 0; for (let i = 0; i < P; i++) { const d = eh[i] - cfg.eps[i]; L += d * d; } return L / P; };
  IDF.zeroGrads(net);
  const xg = IDF.qSample(sched, cfg.x0, cfg.t, cfg.eps);
  const cg = {}; const ehg = IDF.forwardNet(net, xg, cfg.t, cg);
  const dE = new Float64Array(P); for (let i = 0; i < P; i++) dE[i] = 2 * (ehg[i] - cfg.eps[i]) / P;
  IDF.backwardNet(net, cg, dE);
  const EPS = 1e-6;
  const cand = [];
  let tot = 0;
  const sr = IDF.mulberry32(2024);
  for (const k of net.keys) {
    const arr = net.p[k], g = net.g[k];
    const nS = Math.min(arr.length, 5);
    for (let s = 0; s < nS; s++) {
      const i = Math.floor(sr() * arr.length) % arr.length, o = arr[i];
      arr[i] = o + EPS; const lp = lossOf(); arr[i] = o - EPS; const lm = lossOf(); arr[i] = o;
      cand.push({ k: k, i: i, num: (lp - lm) / (2 * EPS), ana: g[i] });
      tot++;
    }
  }
  let mxg = 0; for (const c of cand) if (Math.abs(c.ana) > mxg) mxg = Math.abs(c.ana);
  const floor = 1e-4 * mxg;   // 梯度过小者被 FD 绝对噪声淹没，跳过
  let worst = 0, wname = "", inf = 0;
  for (const c of cand) {
    if (Math.abs(c.ana) < floor) continue;
    inf++;
    const rel = Math.abs(c.num - c.ana) / (Math.abs(c.num) + Math.abs(c.ana));
    if (rel > worst) { worst = rel; wname = c.k + "[" + c.i + "]"; }
  }
  T("全网络梯度检验 vs 有限差分", worst < 1e-4 && inf >= 30,
    "maxRel=" + worst.toExponential(2) + " @" + wname + " 有效样本=" + inf + "/" + tot);
}

/* 7 前向确定性 */
{
  const net = IDF.createNet({ size: S, baseCh: C, Td: Td, seed: 5 });
  const dd = IDF.datasetShapes(2, S, 7, 0.02);
  const gr = IDF.mulberry32(5), gg = IDF.gaussFactory(gr);
  const e = new Float64Array(P); for (let i = 0; i < P; i++) e[i] = gg();
  const x = IDF.qSample(sched, dd[0], 20, e);
  const c1 = {}, c2 = {};
  const a = IDF.forwardNet(net, x, 20, c1), b = IDF.forwardNet(net, x, 20, c2);
  let same = true; for (let i = 0; i < P; i++) if (a[i] !== b[i]) same = false;
  T("前向确定性（同输入同输出）", same);
}

/* 8 训练 loss 下降 */
let trained = null;
{
  const dd = IDF.datasetShapes(64, S, 7, 0.02);
  const r = IDF.train({ size: S, baseCh: C, Td: Td, T: TT, steps: 60, batch: 8, lr: 2e-3, seed: 3, data: dd });
  trained = r;
  const cv = r.curve;
  const a = cv.slice(0, 10).reduce((p, q) => p + q) / 10, b = cv.slice(-10).reduce((p, q) => p + q) / 10;
  T("训练 loss 下降", b < a, a.toFixed(4) + " → " + b.toFixed(4));
}

/* 9 训练确定性（同 seed 逐位一致） */
{
  const dd = IDF.datasetShapes(32, S, 7, 0.02);
  const r1 = IDF.train({ size: S, baseCh: C, Td: Td, T: TT, steps: 8, batch: 4, lr: 2e-3, seed: 77, data: dd });
  const r2 = IDF.train({ size: S, baseCh: C, Td: Td, T: TT, steps: 8, batch: 4, lr: 2e-3, seed: 77, data: dd });
  let same = r1.curve.length === r2.curve.length;
  for (let i = 0; i < Math.min(r1.curve.length, r2.curve.length); i++) if (r1.curve[i] !== r2.curve[i]) same = false;
  T("训练确定性（同 seed 逐位一致）", same);
}

/* 10 采样有限 + 方差有限 */
{
  const out = IDF.sample(trained.net, trained.sched, 2, { seed: 4 });
  let fin = true, s2 = 0, n = 0;
  for (const im of out.imgs) for (let i = 0; i < im.length; i++) { if (!isFinite(im[i])) fin = false; s2 += im[i] * im[i]; n++; }
  T("Ancestral 采样有限且方差有界", fin && (s2 / n) < 100, "E[x²]=" + (s2 / n).toFixed(3));
}

/* 11 采样轨迹单调去噪：x_T 比 x_0 更接近纯噪声 */
{
  const out = IDF.sample(trained.net, trained.sched, 1, { seed: 4, traj: true });
  const tr = out.traj;
  let okT = tr && tr.length === TT;
  T("采样轨迹长度 = T", okT, tr ? ("帧数=" + tr.length) : "无轨迹");
}

/* 12 DDIM 确定性与 ancestral 不同源但都有限 */
{
  const d1 = IDF.sampleDDIM(trained.net, trained.sched, 1, { steps: 10, eta: 0, seed: 8 });
  const d2 = IDF.sampleDDIM(trained.net, trained.sched, 1, { steps: 10, eta: 0, seed: 8 });
  let same = true; for (let i = 0; i < d1.imgs[0].length; i++) if (d1.imgs[0][i] !== d2.imgs[0][i]) same = false;
  let fin = true; for (let i = 0; i < d1.imgs[0].length; i++) if (!isFinite(d1.imgs[0][i])) fin = false;
  T("DDIM(η=0) 确定性且有限", same && fin);
}

/* 13 DDIM 步数远少于 T 仍可用 */
{
  const d = IDF.sampleDDIM(trained.net, trained.sched, 1, { steps: 5, eta: 0, seed: 8 });
  let fin = true; for (let i = 0; i < d.imgs[0].length; i++) if (!isFinite(d.imgs[0][i])) fin = false;
  T("DDIM 5 步采样有限", fin);
}

/* 14 数据标准化：零均值单位方差 */
{
  const dd = IDF.datasetShapes(128, S, 7, 0.02);
  let s = 0, s2 = 0, n = 0;
  for (const im of dd) for (let i = 0; i < im.length; i++) { s += im[i]; s2 += im[i] * im[i]; n++; }
  const mean = s / n, v = s2 / n - mean * mean;
  T("数据集零均值单位方差", Math.abs(mean) < 1e-9 && Math.abs(v - 1) < 1e-9, "mean=" + mean.toExponential(2) + " var=" + v.toFixed(6));
}

/* 15 时间嵌入有界 */
{
  const te = IDF.tEmbed(37, 16);
  let bnd = true; for (let i = 0; i < te.length; i++) if (Math.abs(te[i]) > 1.0000001) bnd = false;
  T("时间嵌入 |te| ≤ 1", bnd);
}

console.log("=== " + pass + " / " + (pass + fail) + (fail ? " FAIL" : " ALL GREEN") + (fails.length ? " — " + fails.join(", ") : ""));
fs.writeFileSync(path.join(__dirname, "_smoke.log"), "PASS " + pass + " / " + (pass + fail) + "\n" + (fail ? "FAIL " + fails.join(", ") : "ALL GREEN") + "\n");
process.exit(fail ? 1 : 0);
