// Service worker for the Bridge AI PWA.
//
// Honest scope: this app's actual game state lives on the server (a
// single global GameSession per webapp/app.py's module docstring), not
// in the browser. There is nothing meaningful this worker could do to
// make gameplay itself work offline -- every call/card played is an
// API round-trip. So this worker only caches the static *shell* (the
// page, CSS, JS, manifest, icons): that's what makes the browser
// consider the app "installable" and lets the page repaint instantly
// on a slow connection, not a claim that you can play bridge with no
// network. Every /api/* request always goes straight to the network,
// never the cache -- serving a stale game state offline would be worse
// than just failing loudly.

const CACHE_NAME = "bridge-ai-shell-v1";
const SHELL_ASSETS = [
  "/",
  "/static/style.css",
  "/static/app.js",
  "/static/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept API calls -- always hit the live server. Caching
  // these would mean showing you an old board, an old auction, or a
  // stale save list, which is strictly worse than a network error.
  if (url.pathname.startsWith("/api/")) {
    return;
  }

  // Cache-first for the static shell, with a network fallback so a
  // newly added/changed asset still loads before the next install.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return (
        cached ||
        fetch(event.request).catch(() => {
          // Offline and not cached (e.g. first visit had no network) --
          // nothing sensible to serve instead.
          return new Response("Offline and not cached.", { status: 503 });
        })
      );
    })
  );
});
