export interface NavigationPoint {
  x: number;
  y: number;
}

export interface NavigationView {
  zoom: number;
  pan: NavigationPoint;
}

export const MIN_ZOOM = 0.15;
export const MAX_ZOOM = 32;

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function normalizeWheelDelta(
  deltaX: number,
  deltaY: number,
  deltaMode: number,
  pageSize: number,
): NavigationPoint {
  const multiplier = deltaMode === 1 ? 16 : deltaMode === 2 ? pageSize : 1;
  return { x: deltaX * multiplier, y: deltaY * multiplier };
}

export function zoomViewAt(
  current: NavigationView,
  pointer: NavigationPoint,
  viewport: { width: number; height: number },
  factor: number,
): NavigationView {
  const nextZoom = clamp(current.zoom * factor, MIN_ZOOM, MAX_ZOOM);
  if (nextZoom === current.zoom) return current;

  // Keep the image pixel below the pointer stationary while scaling around it.
  const appliedFactor = nextZoom / current.zoom;
  const offsetX = pointer.x - viewport.width / 2;
  const offsetY = pointer.y - viewport.height / 2;
  return {
    zoom: nextZoom,
    pan: {
      x: offsetX - appliedFactor * (offsetX - current.pan.x),
      y: offsetY - appliedFactor * (offsetY - current.pan.y),
    },
  };
}
