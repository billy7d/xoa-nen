import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_ZOOM,
  MIN_ZOOM,
  normalizeWheelDelta,
  zoomViewAt,
} from "../src/components/canvasNavigation.ts";

test("normalizes pixel, line, and page wheel deltas", () => {
  assert.deepEqual(normalizeWheelDelta(2, -3, 0, 800), { x: 2, y: -3 });
  assert.deepEqual(normalizeWheelDelta(2, -3, 1, 800), { x: 32, y: -48 });
  assert.deepEqual(normalizeWheelDelta(2, -3, 2, 800), { x: 1600, y: -2400 });
});

test("zoom keeps the preview point beneath the pointer stationary", () => {
  const viewport = { width: 1000, height: 700 };
  const pointer = { x: 730, y: 210 };
  const current = { zoom: 1.4, pan: { x: 45, y: -28 } };
  const next = zoomViewAt(current, pointer, viewport, 1.18);

  const beforeX = (pointer.x - viewport.width / 2 - current.pan.x) / current.zoom;
  const beforeY = (pointer.y - viewport.height / 2 - current.pan.y) / current.zoom;
  const afterX = (pointer.x - viewport.width / 2 - next.pan.x) / next.zoom;
  const afterY = (pointer.y - viewport.height / 2 - next.pan.y) / next.zoom;
  assert.ok(Math.abs(beforeX - afterX) < 1e-10);
  assert.ok(Math.abs(beforeY - afterY) < 1e-10);
});

test("zoom is clamped at viewer limits", () => {
  const viewport = { width: 1000, height: 700 };
  const pointer = { x: 500, y: 350 };
  assert.equal(zoomViewAt({ zoom: 1, pan: pointer }, pointer, viewport, 100).zoom, MAX_ZOOM);
  assert.equal(zoomViewAt({ zoom: 1, pan: pointer }, pointer, viewport, 0.001).zoom, MIN_ZOOM);
});
