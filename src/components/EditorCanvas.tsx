import { useCallback, useEffect, useRef, useState } from "react";
import { localAssetUrl } from "../bridge";
import type { ProjectPayload, ToolMode } from "../types";

interface Point {
  x: number;
  y: number;
}

interface Props {
  project: ProjectPayload;
  tool: ToolMode;
  radius: number;
  background: "checker" | "white" | "black" | "garment";
  disabled?: boolean;
  onBrush: (points: Point[]) => Promise<void>;
  onWand: (point: Point) => Promise<void>;
}

interface ViewTransform {
  scale: number;
  originX: number;
  originY: number;
}

const checkerSize = 12;

export default function EditorCanvas({
  project,
  tool,
  radius,
  background,
  disabled,
  onBrush,
  onWand,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const transformRef = useRef<ViewTransform>({ scale: 1, originX: 0, originY: 0 });
  const pointerRef = useRef<{
    id: number;
    mode: "pan" | "brush";
    last: Point;
    points: Point[];
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [cursor, setCursor] = useState<Point | null>(null);
  const [loading, setLoading] = useState(true);

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
      ctx.fillStyle = background === "white" ? "#fff" : background === "black" ? "#080808" : "#263a58";
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
  }, [background, cursor, pan.x, pan.y, project.width, radius, tool, zoom]);

  useEffect(() => {
    const image = new Image();
    setLoading(true);
    image.onload = () => {
      imageRef.current = image;
      setLoading(false);
      draw();
    };
    image.onerror = () => setLoading(false);
    image.src = localAssetUrl(project.preview_path, project.revision);
    return () => {
      image.onload = null;
      image.onerror = null;
    };
  }, [project.preview_path, project.revision, draw]);

  useEffect(() => {
    const observer = new ResizeObserver(draw);
    if (hostRef.current) observer.observe(hostRef.current);
    draw();
    return () => observer.disconnect();
  }, [draw]);

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
    if (tool === "keep" || tool === "remove") {
      pointerRef.current = { id: event.pointerId, mode: "brush", last: point, points: [canonical] };
    }
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const point = eventPoint(event);
    setCursor(point);
    const active = pointerRef.current;
    if (!active || active.id !== event.pointerId) return;
    if (active.mode === "pan") {
      const dx = point.x - active.last.x;
      const dy = point.y - active.last.y;
      setPan((current) => ({ x: current.x + dx, y: current.y + dy }));
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

  const onWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.13 : 1 / 1.13;
    setZoom((current) => Math.min(32, Math.max(0.15, current * factor)));
  };

  return (
    <div className="canvas-host" ref={hostRef}>
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={finishPointer}
        onPointerCancel={finishPointer}
        onPointerLeave={() => setCursor(null)}
        onWheel={onWheel}
      />
      <div className="canvas-hud">
        <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Vừa khung</button>
        <span>{Math.round(zoom * 100)}%</span>
      </div>
      {loading && <div className="canvas-loading">Đang dựng preview…</div>}
    </div>
  );
}

