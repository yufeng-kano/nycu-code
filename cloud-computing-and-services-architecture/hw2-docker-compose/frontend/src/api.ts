export interface NameRecord {
  id: number;
  name: string;
}

interface ErrorBody {
  error: string;
}

// All requests go to the same origin. The proxy container routes /api to the backend.
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = (await response.json()) as T | ErrorBody;
  if (!response.ok) {
    throw new Error((body as ErrorBody).error ?? `HTTP ${response.status}`);
  }
  return body as T;
}

export function getNames(): Promise<{ names: NameRecord[] }> {
  return request("/api/get_names");
}

export function addName(name: string): Promise<NameRecord> {
  return request("/api/add_name", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function removeName(name: string): Promise<{ deleted: string }> {
  return request(`/api/remove_name?name=${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}
