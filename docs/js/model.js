// Inference for the classifier exported by ecglab/export_web.py (docs/data/model.json).
// Supports multinomial logistic regression, a ReLU multilayer perceptron, and
// histogram gradient-boosted trees, the model types the Python side can export.

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function standardize(x, scaler) {
  return scaler ? x.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]) : x;
}

function dense(x, W, b, relu) {
  // W is [inputs][outputs] (sklearn coefs_ layout)
  const out = b.slice();
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], row = W[i];
    for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
  }
  return relu ? out.map((v) => (v > 0 ? v : 0)) : out;
}

function treeValue(tree, x) {
  // tree = {f: feature (-1 leaf), t: threshold or leaf value, l: left, r: right, m: missing-goes-left}
  let node = 0;
  while (tree.f[node] >= 0) {
    const v = x[tree.f[node]];
    const left = Number.isNaN(v) ? tree.m[node] === 1 : v <= tree.t[node];
    node = left ? tree.l[node] : tree.r[node];
  }
  return tree.t[node];
}

/** Class probabilities for one feature row. */
export function predictProba(model, row) {
  if (model.type === "logistic_regression") {
    const x = standardize(row, model.scaler);
    return softmax(model.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * x[i], model.intercept[k])));
  }
  if (model.type === "mlp") {
    let h = standardize(row, model.scaler);
    model.layers.forEach((L, i) => { h = dense(h, L.W, L.b, i < model.layers.length - 1); });
    return softmax(h);
  }
  if (model.type === "gradient_boosting") {
    const raw = model.baseline.slice();
    for (const iteration of model.trees) iteration.forEach((tree, k) => { raw[k] += treeValue(tree, row); });
    return softmax(raw);
  }
  throw new Error(`unknown model type ${model.type}`);
}
