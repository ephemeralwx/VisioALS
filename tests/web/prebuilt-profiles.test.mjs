import assert from "node:assert/strict";
import test from "node:test";

import { getPrebuiltProfile, PREBUILT_PROFILES } from "../../web/prebuilt-profiles.mjs";

test("British Person is available as a complete pre-built profile", () => {
  assert.equal(PREBUILT_PROFILES.some((profile) => profile.id === "british-person"), true);

  const profile = getPrebuiltProfile("british-person");
  assert.equal(profile.name, "British Person");
  assert.equal(profile.builtInId, "british-person");
  assert.match(profile.summary, /colloquial British English/);
  assert.ok(profile.samples.length >= 10);
  assert.match(profile.samples.join("\n"), /Unexpected item in the bagging area/);
  assert.match(profile.samples.join("\n"), /what's the point, eh\?/i);
});

test("unknown pre-built profile ids return null", () => {
  assert.equal(getPrebuiltProfile("missing"), null);
});
