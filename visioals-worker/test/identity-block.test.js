import test from "node:test";
import assert from "node:assert/strict";

import { buildIdentityBlock } from "../src/index.js";

test("writing samples are explicitly isolated from answer subject matter", () => {
  const block = buildIdentityBlock(
    "Formal, emphatic prose",
    ["Writing is astronomically hard."],
    null,
  );

  assert.match(block, /style evidence only/i);
  assert.match(block, /not evidence of what the patient thinks/i);
  assert.match(block, /Never copy a sample's topic/i);
});

test("long samples are capped so one essay cannot dominate the prompt", () => {
  const marker = "x".repeat(1000);
  const block = buildIdentityBlock(null, [marker], null);

  assert.ok(block.includes("x".repeat(400)));
  assert.ok(!block.includes("x".repeat(401)));
});
