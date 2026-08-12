import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { isTauriRuntime, localAssetUrl } from "../bridge";
import type { ProjectPayload, ToolMode } from "../types";
import {
  clamp,
  normalizeWheelDelta,
  zoomViewAt,
  type NavigationView,
} from "./canvasNavigation";

interface Point {
  x: number;
  y: number;
}

interface Props {
  project: ProjectPayload;
  tool: ToolMode;
  radius: number;
  background: "checker" | "white" | "black" | "garment" | "custom";
  backgroundColor?: string;
  disabled?: boolean;
  foregroundPoints: Point[];
  onBrush: (points: Point[]) => Promise<void>;
  onWand: (point: Point) => Promise<void>;
  onSubject: (point: Point) => Promise<void>;
  onProtect: (point: Point, append: boolean) => Promise<void>;
}

interface ViewTransform {
  scale: number;
  originX: number;
  originY: number;
}

interface WebKitGestureEvent extends Event {
  clientX: number;
  clientY: number;
  scale: number;
}

interface MacosMagnifyPayload {
  magnification: number;
  x: number;
  y: number;
}

const checkerSize = 12;

export default function EditorCanvas({
  project,
  tool,
  radius,
  background,
  backgroundColor = "#263a58",
  disabled,
  foregroundPoints,
  onBrush,
  onWand,
  onSubject,
  onProtect,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const transformRef = useRef<ViewTransform>({ scale: 1, originX: 0, originY: 0 });
  const drawRef = useRef<() => void>(() => undefined);
  const viewRef = useRef<NavigationView>({ zoom: 1, pan: { x: 0, y: 0 } });
  const frameRef = useRef<number | null>(null);
  const gestureRef = useRef<{ lastScale: number; active: boolean }>({ lastScale: 1, active: false });
  const hoverRef = useRef(false);
  const pointerRef = useRef<{
    id: number;
    mode: "pan" | "brush";
    last: Point;
    points: Point[];
  } | null>(null);
  const [view, setView] = useState<NavigationView>(viewRef.current);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [loading, setLoading] = useState(true);
  const { zoom, pan } = view;

  const scheduleView = useCallback((next: NavigationView) => {
    viewRef.current = next;
    if (frameRef.current !== null) return;
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      setView(viewRef.current);
    });
  }, []);

  const resetView = useCallback(() => {
    scheduleView({ zoom: 1, pan: { x: 0, y: 0 } });
  }, [scheduleView]);

  const zoomAt = useCallback((clientX: number, clientY: number, factor: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !Number.isFinite(factor) || factor <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const pointer = {
      x: Number.isFinite(clientX) ? clientX - rect.left : rect.width / 2,
      y: Number.isFinite(clientY) ? clientY - rect.top : rect.height / 2,
    };
    scheduleView(zoomViewAt(viewRef.current, pointer, rect, factor));
  }, [scheduleView]);

  const draw = useCallback(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!host || !canvas) return;

    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (background === "checker") {
      ctx.fillStyle = "#d8d8d8";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#bcbcbc";
      for (let y = 0; y < height; y += checkerSize) {
        for (let x = 0; x < width; x += checkerSize) {
          if ((x / checkerSize + y / checkerSize) % 2 === 0) {
            ctx.fillRect(x, y, checkerSize, checkerSize);
          }
        }
      }
    } else {
      // Màu tùy chọn giúp soi halo POD-clean trên các phông khác nhau.
      ctx.fillStyle = background === "white" ? "#fff" : background === "black" ? "#080808" : background === "custom" ? backgroundColor : "#263a58";
      ctx.fillRect(0, 0, width, height);
    }

    if (!image || !image.complete || image.naturalWidth === 0) return;
    const fit = Math.min(width / image.naturalWidth, height / image.naturalHeight) * 0.93;
    const scale = Math.max(0.01, fit * zoom);
    const displayWidth = image.naturalWidth * scale;
    const displayHeight = image.naturalHeight * scale;
    const originX = (width - displayWidth) / 2 + pan.x;
    const originY = (height - displayHeight) / 2 + pan.y;
    transformRef.current = { scale, originX, originY };

    ctx.imageSmoothingEnabled = zoom < 6;
    ctx.drawImage(image, originX, originY, displayWidth, displayHeight);
    ctx.strokeStyle = "rgba(255,255,255,.42)";
    ctx.lineWidth = 1;
    ctx.strokeRect(originX - 0.5, originY - 0.5, displayWidth + 1, displayHeight + 1);

    const drawBox = (bbox: [number, number, number, number], color: string, dashed: boolean) => {
      const [x0, y0, x1, y1] = bbox;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}22`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dashed ? [6, 4] : []);
      const left = originX + (x0 / project.width) * displayWidth;
      const top = originY + (y0 / project.height) * displayHeight;
      const boxWidth = ((x1 - x0) / project.width) * displayWidth;
      const boxHeight = ((y1 - y0) / project.height) * displayHeight;
      ctx.fillRect(left, top, boxWidth, boxHeight);
      ctx.strokeRect(left, top, boxWidth, boxHeight);
      ctx.restore();
    };
    for (const region of project.process?.review_regions ?? []) {
      drawBox(region.bbox, "#f2c14f", true);
    }
    if (tool === "subject") {
      for (const subject of project.process?.subjects ?? []) {
        drawBox(subject.bbox, subject.selected ? "#61e59e" : "#ff7185", false);
      }
    }
    for (const point of foregroundPoints) {
      const x = originX + (point.x / project.width) * displayWidth;
      const y = originY + (point.y / project.height) * displayHeight;
      ctx.save();
      ctx.strokeStyle = "#55f19a";
      ctx.fillStyle = "rgba(15, 73, 45, .78)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 12, y);
      ctx.lineTo(x + 12, y);
      ctx.moveTo(x, y - 12);
      ctx.lineTo(x, y + 12);
      ctx.stroke();
      ctx.restore();
    }

    if (cursor && (tool === "keep" || tool === "remove")) {
      const previewRatio = image.naturalWidth / project.width;
      const brushRadius = radius * previewRatio * scale;
      ctx.beginPath();
      ctx.arc(cursor.x, cursor.y, Math.max(2, brushRadius), 0, Math.PI * 2);
      ctx.strokeStyle = tool === "keep" ? "#56f09b" : "#ff6078";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cursor.x, cursor.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
    }
  }, [background, backgroundColor, cursor, foregroundPoints, pan.x, pan.y, project.height, project.process, project.width, radius, tool, zoom]);
  drawRef.current = draw;

  useEffect(() => {
    const image = new Image();
    setLoading(true);
    image.onload = () => {
      imageRef.current = image;
      setLoading(false);
    };
    image.onerror = () => setLoading(false);
    image.src = localAssetUrl(project.preview_path, project.revision);
    return () => {
      image.onload = null;
      image.onerror = null;
    };
  }, [project.preview_path, project.revision]);

  useEffect(() => {
    draw();
  }, [draw, loading]);

  useEffect(() => {
    const observer = new ResizeObserver(() => drawRef.current());
    const host = hostRef.current;
    if (host) observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => {
    if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
  }, []);

  useEffect(() => {
    const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
    if (!isMac || !isTauriRuntime()) return;

    const unlisten = listen<MacosMagnifyPayload>("macos-preview-magnify", ({ payload }) => {
      const host = hostRef.current;
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const clientX = payload.x;
      const clientY = window.innerHeight - payload.y;
      const isInside = clientX >= rect.left
        && clientX <= rect.right
        && clientY >= rect.top
        && clientY <= rect.bottom;
      if (!isInside && !hoverRef.current) return;

      const factor = Math.max(0.1, 1 + payload.magnification * 0.72);
      zoomAt(clientX, clientY, factor);
    });
    return () => {
      void unlisten.then((dispose) => dispose());
    };
  }, [zoomAt]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
    const usesNativeMagnify = isMac && isTauriRuntime();

    const isOverPreview = (event: Event & { clientX?: number; clientY?: number }) => {
      if (event.target instanceof Node && host.contains(event.target)) return true;
      if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return false;
      const rect = host.getBoundingClientRect();
      return event.clientX! >= rect.left
        && event.clientX! <= rect.right
        && event.clientY! >= rect.top
        && event.clientY! <= rect.bottom;
    };

    const onWheel = (event: WheelEvent) => {
      if (!isOverPreview(event)) return;
      event.preventDefault();
      const delta = normalizeWheelDelta(
        event.deltaX,
        event.deltaY,
        event.deltaMode,
        host.clientHeight,
      );
      const wantsZoom = event.ctrlKey;

      if (wantsZoom) {
        // WebKit may emit both gesture and ctrl+wheel events for the same pinch.
        if (gestureRef.current.active) return;
        const limitedDelta = clamp(delta.y, -100, 100);
        const sensitivity = isMac ? 0.004 : 0.00135;
        zoomAt(event.clientX, event.clientY, Math.exp(-limitedDelta * sensitivity));
        return;
      }

      const current = viewRef.current;
      if (!isMac && event.shiftKey) {
        const horizontalDelta = delta.x !== 0 ? delta.x : delta.y;
        scheduleView({
          zoom: current.zoom,
          pan: { x: current.pan.x - horizontalDelta, y: current.pan.y },
        });
        return;
      }

      scheduleView({
        zoom: current.zoom,
        pan: { x: current.pan.x - delta.x, y: current.pan.y - delta.y },
      });
    };

    // WKWebView exposes native macOS pinch through WebKit gesture events.
    const onGestureStart = (rawEvent: Event) => {
      if (usesNativeMagnify) return;
      const event = rawEvent as WebKitGestureEvent;
      if (!isOverPreview(event)) return;
      event.preventDefault();
      gestureRef.current = { lastScale: event.scale || 1, active: true };
    };
    const onGestureChange = (rawEvent: Event) => {
      if (usesNativeMagnify) return;
      const event = rawEvent as WebKitGestureEvent;
      if (!gestureRef.current.active) return;
      event.preventDefault();
      const previous = gestureRef.current.lastScale || 1;
      const current = event.scale || previous;
      gestureRef.current.lastScale = current;
      // Dampen WebKit's cumulative scale slightly for a controlled native feel.
      zoomAt(event.clientX, event.clientY, Math.pow(current / previous, 0.72));
    };
    const onGestureEnd = (event: Event) => {
      if (usesNativeMagnify) return;
      if (!gestureRef.current.active) return;
      event.preventDefault();
      gestureRef.current = { lastScale: 1, active: false };
    };

    // Capture at window level: WKWebView can route macOS gestures to its scroll
    // view before they bubble through the element beneath the pointer.
    window.addEventListener("wheel", onWheel, { passive: false, capture: true });
    window.addEventListener("gesturestart", onGestureStart, { passive: false, capture: true });
    window.addEventListener("gesturechange", onGestureChange, { passive: false, capture: true });
    window.addEventListener("gestureend", onGestureEnd, { passive: false, capture: true });
    return () => {
      window.removeEventListener("wheel", onWheel, true);
      window.removeEventListener("gesturestart", onGestureStart, true);
      window.removeEventListener("gesturechange", onGestureChange, true);
      window.removeEventListener("gestureend", onGestureEnd, true);
    };
  }, [scheduleView, zoomAt]);

  const screenToCanonical = useCallback(
    (point: Point): Point | null => {
      const image = imageRef.current;
      if (!image) return null;
      const { scale, originX, originY } = transformRef.current;
      const previewX = (point.x - originX) / scale;
      const previewY = (point.y - originY) / scale;
      if (previewX < 0 || previewY < 0 || previewX >= image.naturalWidth || previewY >= image.naturalHeight) {
        return null;
      }
      return {
        x: Math.min(project.width - 0.5, Math.max(0.5, (previewX / image.naturalWidth) * project.width)),
        y: Math.min(project.height - 0.5, Math.max(0.5, (previewY / image.naturalHeight) * project.height)),
      };
    },
    [project.height, project.width],
  );

  const eventPoint = (event: React.PointerEvent): Point => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const onPointerDown = async (event: React.PointerEvent) => {
    if (disabled) return;
    const point = eventPoint(event);
    canvasRef.current?.setPointerCapture(event.pointerId);
    if (tool === "pan" || event.button === 1 || event.altKey || event.metaKey) {
      pointerRef.current = { id: event.pointerId, mode: "pan", last: point, points: [] };
      return;
    }
    const canonical = screenToCanonical(point);
    if (!canonical) return;
    if (tool === "wand-keep" || tool === "wand-remove") {
      await onWand(canonical);
      return;
    }
    if (tool === "subject") {
      await onSubject(canonical);
      return;
    }
    if (tool === "protect") {
      await onProtect(canonical, event.shiftKey);
      return;
    }
    if (tool === "keep" || tool === "remove") {
      pointerRef.current = { id: event.pointerId, mode: "brush", last: point, points: [canonical] };
    }
  };

  const onPointerMove = (event: React.PointerEvent) => {
    hoverRef.current = true;
    const point = eventPoint(event);
    setCursor(point);
    const active = pointerRef.current;
    if (!active || active.id !== event.pointerId) return;
    if (active.mode === "pan") {
      const dx = point.x - active.last.x;
      const dy = point.y - active.last.y;
      const current = viewRef.current;
      scheduleView({
        zoom: current.zoom,
        pan: { x: current.pan.x + dx, y: current.pan.y + dy },
      });
      active.last = point;
      return;
    }
    const canonical = screenToCanonical(point);
    if (canonical) active.points.push(canonical);
  };

  const finishPointer = async (event: React.PointerEvent) => {
    const active = pointerRef.current;
    pointerRef.current = null;
    if (active?.mode === "brush" && active.points.length > 0) {
      await onBrush(active.points);
    }
    canvasRef.current?.releasePointerCapture(event.pointerId);
  };

  return (
    <div className="canvas-host" ref={hostRef}>
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={finishPointer}
        onPointerCancel={finishPointer}
        onPointerEnter={() => { hoverRef.current = true; }}
        onPointerLeave={() => { hoverRef.current = false; setCursor(null); }}
      />
      <div className="canvas-hud">
        <button onClick={resetView}>Vừa khung</button>
        <span>{Math.round(zoom * 100)}%</span>
      </div>
      {loading && <div className="canvas-loading">Đang dựng preview…</div>}
    </div>
  );
}
