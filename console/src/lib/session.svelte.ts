const KEY = "memorypilot.console";

export type Session = {
  baseUrl: string;
  apiKey: string;
  subject: string;
};

export function loadSession(): Session {
  if (typeof localStorage === "undefined") {
    return { baseUrl: "", apiKey: "", subject: "" };
  }
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? "null") as Session | null;
    return stored ?? { baseUrl: "", apiKey: "", subject: "" };
  } catch {
    return { baseUrl: "", apiKey: "", subject: "" };
  }
}

export function saveSession(session: Session): void {
  localStorage.setItem(KEY, JSON.stringify(session));
}

export function clearSession(): void {
  localStorage.removeItem(KEY);
}
