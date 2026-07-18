import test from "node:test";
import assert from "node:assert/strict";

import { extractFeatures, predictModels, trainModels } from "../../web/gaze-core.mjs";

function calibrationSamples() {
  const samples = [];
  for (let ix = -3; ix <= 3; ix += 1) {
    for (let iy = -3; iy <= 3; iy += 1) {
      const x = 0.012 + ix * 0.0015;
      const y = -0.018 + iy * 0.0012;
      samples.push({
        feature: [x, y],
        target: [500 + ix * 115 + ix * ix * 13, 380 + iy * 82 - iy * iy * 9],
      });
    }
  }
  return samples;
}

test("tiny normalized eye features retain full red and green prediction range", () => {
  const models = trainModels(calibrationSamples());
  const low = predictModels([0.0075, -0.0216], models);
  const high = predictModels([0.0165, -0.0144], models);

  assert.ok(high.lr[0] - low.lr[0] > 500, "linear/red prediction was damped");
  assert.ok(high.poly[0] - low.poly[0] > 500, "polynomial/green prediction was collapsed");
  assert.ok(high.poly[1] - low.poly[1] > 300, "polynomial/green vertical prediction was collapsed");
});

test("eye features use the same MediaPipe iris-center landmarks as desktop", () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0, y: 0, z: 0 }));
  landmarks[234] = { x: 0.2, y: 0.5, z: 0 };
  landmarks[454] = { x: 0.8, y: 0.5, z: 0 };
  landmarks[10] = { x: 0.5, y: 0.2, z: 0 };
  landmarks[152] = { x: 0.5, y: 0.8, z: 0 };
  landmarks[468] = { x: 0.44, y: 0.47, z: 0 };
  landmarks[473] = { x: 0.56, y: 0.53, z: 0 };
  // Ring points deliberately disagree; these must not influence the feature.
  for (const index of [469, 470, 471, 472, 474, 475, 476, 477]) landmarks[index] = { x: 0.9, y: 0.9, z: 0 };

  const feature = extractFeatures(landmarks, "eye");
  assert.ok(Math.abs(feature[0]) < 1e-12);
  assert.ok(Math.abs(feature[1]) < 1e-12);
});
