import { convertFileSrc, invoke } from "@tauri-apps/api/core";

interface ProtocolResponse<T> {
  id: string;
  ok: boolean;
  result?: T;
  error?: {
    code: string;
    message: string;
    details?: string;
  };
}

let requestSequence = 0;

export function isTauriRuntime(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

export async function coordinatorCall<T>(
  method: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  if (!isTauriRuntime()) {
    throw new Error("Editor cần chạy trong ứng dụng Tauri, không phải tab trình duyệt.");
  }

  const request = {
    id: `ui-${Date.now()}-${++requestSequence}`,
    method,
    params,
  };
  const response = await invoke<ProtocolResponse<T>>("coordinator_request", { request });
  if (!response.ok || response.result === undefined) {
    const suffix = response.error?.details ? `\n${response.error.details}` : "";
    throw new Error(`${response.error?.message ?? "Sidecar không phản hồi"}${suffix}`);
  }
  return response.result;
}

export function localAssetUrl(path: string, revision: string | number = 0): string {
  return `${convertFileSrc(path)}?revision=${revision}`;
}
