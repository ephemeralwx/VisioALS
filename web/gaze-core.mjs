const MODEL_NAMES = ["lr", "poly", "svr", "knn"];

function transpose(matrix) {
  return matrix[0].map((_, i) => matrix.map((row) => row[i]));
}

function multiply(a, b) {
  return a.map((row) => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
}

function solve(matrix, vector) {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  const n = vector.length;
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    if (Math.abs(a[column][column]) < 1e-10) throw new Error("singular gaze calibration");
    const divisor = a[column][column];
    for (let j = column; j <= n; j += 1) a[column][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let j = column; j <= n; j += 1) a[row][j] -= factor * a[column][j];
    }
  }
  return a.map((row) => row[n]);
}

function fitRows(rows, outputs, alpha = 0) {
  const xt = transpose(rows);
  const xtx = multiply(xt, rows);
  // The final column is the intercept. sklearn does not penalize it.
  for (let i = 0; i < xtx.length - 1; i += 1) xtx[i][i] += alpha;
  const fitOutput = (index) => solve(xtx, multiply(xt, outputs.map((value) => [value[index]])).map((value) => value[0]));
  return { x: fitOutput(0), y: fitOutput(1) };
}

function fitScaler(features) {
  const mean = [0, 1].map((axis) => features.reduce((sum, feature) => sum + feature[axis], 0) / features.length);
  const scale = [0, 1].map((axis) => {
    const variance = features.reduce((sum, feature) => sum + (feature[axis] - mean[axis]) ** 2, 0) / features.length;
    return Math.sqrt(variance) || 1;
  });
  return { mean, scale };
}

function scaleFeature(feature, scaler) {
  return [(feature[0] - scaler.mean[0]) / scaler.scale[0], (feature[1] - scaler.mean[1]) / scaler.scale[1]];
}

function linearBasis(feature) {
  return [feature[0], feature[1], 1];
}

function polynomialBasis(feature) {
  return [feature[0], feature[1], feature[0] ** 2, feature[0] * feature[1], feature[1] ** 2, 1];
}

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function knnPredict(feature, samples, k = 5) {
  const nearest = samples
    .map((sample) => ({ sample, distance: Math.hypot(feature[0] - sample.feature[0], feature[1] - sample.feature[1]) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
  const total = nearest.reduce((sum, item) => sum + 1 / (item.distance + 0.00001), 0);
  return nearest.reduce((result, item) => {
    const weight = 1 / (item.distance + 0.00001) / total;
    result[0] += item.sample.target[0] * weight;
    result[1] += item.sample.target[1] * weight;
    return result;
  }, [0, 0]);
}

function rbfPredict(feature, samples) {
  const nearest = samples
    .map((sample) => ({ sample, distance: Math.hypot(feature[0] - sample.feature[0], feature[1] - sample.feature[1]) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 28);
  const sigma = Math.max(0.05, nearest[Math.min(12, nearest.length - 1)].distance);
  let x = 0; let y = 0; let weightSum = 0;
  nearest.forEach(({ sample, distance }) => {
    const weight = Math.exp(-(distance ** 2) / (2 * sigma ** 2));
    x += sample.target[0] * weight;
    y += sample.target[1] * weight;
    weightSum += weight;
  });
  return weightSum ? [x / weightSum, y / weightSum] : [0, 0];
}

export function trainModels(samples) {
  const features = samples.map((sample) => sample.feature);
  const outputs = samples.map((sample) => sample.target);
  const scaler = fitScaler(features);
  const scaledSamples = samples.map((sample) => ({ feature: scaleFeature(sample.feature, scaler), target: sample.target }));
  const scaledFeatures = scaledSamples.map((sample) => sample.feature);
  return {
    scaler,
    // Desktop: StandardScaler + unregularized least squares.
    lr: fitRows(scaledFeatures.map(linearBasis), outputs, 0),
    // Desktop: StandardScaler + degree-2 PolynomialFeatures + Ridge(alpha=1).
    poly: fitRows(scaledFeatures.map(polynomialBasis), outputs, 1),
    samples: scaledSamples,
  };
}

export function predictModels(feature, models) {
  const scaled = scaleFeature(feature, models.scaler);
  const linear = linearBasis(scaled);
  const polynomial = polynomialBasis(scaled);
  return {
    lr: [dot(linear, models.lr.x), dot(linear, models.lr.y)],
    poly: [dot(polynomial, models.poly.x), dot(polynomial, models.poly.y)],
    svr: rbfPredict(scaled, models.samples),
    knn: knnPredict(scaled, models.samples),
  };
}

class OneEuroFilter {
  constructor(minCutoff = 1, beta = 0.007, derivativeCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.reset();
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  update(value, dt) {
    if (this.previous === null) { this.previous = value; return value; }
    const derivativeAlpha = this.alpha(this.derivativeCutoff, dt);
    const derivative = (value - this.previous) / dt;
    const filteredDerivative = derivativeAlpha * derivative + (1 - derivativeAlpha) * this.previousDerivative;
    const alpha = this.alpha(this.minCutoff + this.beta * Math.abs(filteredDerivative), dt);
    const filtered = alpha * value + (1 - alpha) * this.previous;
    this.previous = filtered;
    this.previousDerivative = filteredDerivative;
    return filtered;
  }

  reset() { this.previous = null; this.previousDerivative = 0; }
}

class OneEuroFilter2D {
  constructor(minCutoff, beta) {
    this.x = new OneEuroFilter(minCutoff, beta);
    this.y = new OneEuroFilter(minCutoff, beta);
  }

  update(x, y, dt) { return [this.x.update(x, dt), this.y.update(y, dt)]; }
  reset() { this.x.reset(); this.y.reset(); }
}

export class GazePrediction {
  constructor() {
    this.inputFilter = new OneEuroFilter2D(1.5, 0.01);
    this.outputFilter = new OneEuroFilter2D(0.8, 0.005);
    this.reset();
  }

  reset() {
    this.modelPositions = { lr: null, poly: null, svr: null, knn: null };
    this.ensemble = null;
    this.inputFilter.reset();
    this.outputFilter.reset();
  }

  update(feature, models, dt) {
    const filteredFeature = this.inputFilter.update(feature[0], feature[1], Math.max(dt, 0.001));
    const raw = predictModels(filteredFeature, models);
    MODEL_NAMES.forEach((name) => {
      const previous = this.modelPositions[name];
      const target = raw[name];
      this.modelPositions[name] = previous
        ? [Math.trunc(previous[0] * 0.85 + target[0] * 0.15), Math.trunc(previous[1] * 0.85 + target[1] * 0.15)]
        : [Math.trunc(target[0]), Math.trunc(target[1])];
    });
    const weights = { lr: 0.35, poly: 0.25, svr: 0.25, knn: 0.15 };
    const combined = [0, 0];
    Object.entries(weights).forEach(([name, weight]) => {
      combined[0] += this.modelPositions[name][0] * weight;
      combined[1] += this.modelPositions[name][1] * weight;
    });
    const filteredOutput = this.outputFilter.update(combined[0], combined[1], Math.max(dt, 0.001));
    this.ensemble = [Math.trunc(filteredOutput[0]), Math.trunc(filteredOutput[1])];
    return { positions: this.modelPositions, ensemble: this.ensemble };
  }
}

export function extractFeatures(landmarks, mode) {
  const faceCx = (landmarks[454].x + landmarks[234].x) / 2;
  const faceCy = (landmarks[10].y + landmarks[152].y) / 2;
  const faceW = Math.abs(landmarks[454].x - landmarks[234].x);
  const faceH = Math.abs(landmarks[10].y - landmarks[152].y);
  if (faceW < 1e-6 || faceH < 1e-6) return null;
  if (mode === "head") {
    return [(landmarks[454].z - landmarks[234].z) / (faceW + 1e-6), (landmarks[4].y - faceCy) / (faceH + 1e-6)];
  }
  // Match desktop exactly: 468 and 473 are MediaPipe's iris-center landmarks.
  const eyeX = (landmarks[468].x + landmarks[473].x) / 2;
  const eyeY = (landmarks[468].y + landmarks[473].y) / 2;
  return [(eyeX - faceCx) / faceW, (eyeY - faceCy) / faceH];
}
