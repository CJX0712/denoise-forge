/* _probe.js：训练到能出形状，把真实样本与生成样本打成 ASCII 人工读数，并统计模式覆盖 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console, Math, Object, Array, JSON, isFinite, isNaN, Infinity, NaN, Float64Array, Int32Array, Uint8Array, Uint8ClampedArray, Number, String, Boolean };
ctx.globalThis = ctx; vm.createContext(ctx); vm.runInContext(m[1], ctx, { filename: "engine.js" });
const IDF = ctx.IDF;

const S = 16, C = 8, Td = 16, T = 100, P = S * S;
const STEPS = +(process.argv[2] || 900);
let out = "";
const say = (s) => { out += s + "\n"; console.log(s); };

const t0 = Date.now();
const data = IDF.datasetShapes(200, S, 7, 0.02);
const r = IDF.train({ size: S, baseCh: C, Td: Td, T: T, steps: STEPS, batch: 8, lr: 2e-3, seed: 3, data: data });
const cv = r.curve;
say("训练 " + STEPS + " 步，耗时 " + ((Date.now() - t0) / 1000).toFixed(1) + "s");
say("loss：首 " + cv[0].toFixed(4) + " → 末 " + cv[cv.length - 1].toFixed(4)
  + " | 前20均 " + (cv.slice(0, 20).reduce((a, b) => a + b) / 20).toFixed(4)
  + " → 末20均 " + (cv.slice(-20).reduce((a, b) => a + b) / 20).toFixed(4));

const mom = (arr) => { let s = 0, n = 0; for (const im of arr) for (let i = 0; i < im.length; i++) { s += im[i] * im[i]; n++; } return s / n; };
const gen = IDF.sample(r.net, r.sched, 8, { seed: 11 });
const ddim = IDF.sampleDDIM(r.net, r.sched, 4, { steps: 20, eta: 0, seed: 11 });
say("数据 E[x²]=" + mom(data).toFixed(3) + " | 生成(ancestral) E[x²]=" + mom(gen.imgs).toFixed(3) + " | 生成(DDIM) E[x²]=" + mom(ddim.imgs).toFixed(3));

const chars = " .:-=+*#%@";
function render(im) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < im.length; i++) { if (im[i] < mn) mn = im[i]; if (im[i] > mx) mx = im[i]; }
  const L = [];
  for (let y = 0; y < S; y++) {
    let s = "";
    for (let x = 0; x < S; x++) { const v = (im[y * S + x] - mn) / (mx - mn + 1e-9); s += chars[Math.min(9, Math.floor(v * 10))]; }
    L.push(s);
  }
  return L.join("\n");
}
const names = ["圆", "方", "环", "十字", "三角"];
say("\n===== 真实样本（5 类模板）=====");
for (let k = 0; k < 5; k++) { say("--- " + names[k] + " ---"); say(render(data[k])); }

say("\n===== 生成样本（ancestral, T=" + T + " 步）=====");
for (let k = 0; k < 8; k++) { say("--- #" + k + " ---"); say(render(gen.imgs[k])); }

say("\n===== 生成样本（DDIM, 20 步, η=0）=====");
for (let k = 0; k < 4; k++) { say("--- #" + k + " ---"); say(render(ddim.imgs[k])); }

/* 模式覆盖：每个生成样本按 MSE 最近邻归到 5 类模板 */
const tmpl = [0, 1, 2, 3, 4].map(k => data[k]);
const cnt = [0, 0, 0, 0, 0]; const cntD = [0, 0, 0, 0, 0];
function assign(im) {
  let best = 0, bv = Infinity;
  for (let k = 0; k < 5; k++) { let s = 0; for (let i = 0; i < P; i++) { const d = im[i] - tmpl[k][i]; s += d * d; } if (s < bv) { bv = s; best = k; } }
  return best;
}
for (const im of gen.imgs) cnt[assign(im)]++;
for (const im of ddim.imgs) cntD[assign(im)]++;
say("\n===== 模式覆盖（生成样本最近邻归类）=====");
say("ancestral: " + cnt.map((c, i) => names[i] + "=" + c).join(" "));
say("DDIM     : " + cntD.map((c, i) => names[i] + "=" + c).join(" "));
say("覆盖类别数：ancestral=" + cnt.filter(c => c > 0).length + "/5  DDIM=" + cntD.filter(c => c > 0).length + "/5");

/* 结构度：相邻像素差的均值 —— 纯噪声高、平滑块低、有边缘的形状居中偏高但成片 */
const rough = (im) => { let s = 0, n = 0; for (let y = 0; y < S; y++) for (let x = 0; x < S - 1; x++) { s += Math.abs(im[y * S + x] - im[y * S + x + 1]); n++; } return s / n; };
let rd = 0; for (let k = 0; k < 5; k++) rd += rough(data[k]) / 5;
let rg = 0; for (const im of gen.imgs) rg += rough(im) / gen.imgs.length;
say("\n相邻像素差均值：真实=" + rd.toFixed(4) + " 生成=" + rg.toFixed(4) + "（同量级说明学到了边缘结构而非纯噪声）");

fs.writeFileSync(path.join(__dirname, "_probe.txt"), out, "utf8");
say("\n已写出 _probe.txt");
