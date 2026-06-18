"""
swagger_ui.py
=============
Serves the OpenAPI 3.0 spec and Swagger UI for AgniAI.

Register in app.py:
    from swagger_ui import swagger_bp
    app.register_blueprint(swagger_bp)

Endpoints:
    GET /docs          → Swagger UI HTML page
    GET /docs/spec     → Raw OpenAPI JSON spec
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, send_from_directory

swagger_bp = Blueprint("swagger", __name__)

_SPEC_PATH = Path(__file__).parent / "static" / "swagger.json"


def _load_spec() -> dict:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


@swagger_bp.route("/docs/spec")
def spec():
    """Serve the raw OpenAPI JSON spec."""
    return jsonify(_load_spec())


@swagger_bp.route("/docs")
@swagger_bp.route("/docs/")
def swagger_ui():
    """Serve the Swagger UI HTML page."""
    return Response(
        _SWAGGER_HTML,
        mimetype="text/html",
    )


# ---------------------------------------------------------------------------
# Inline HTML — no static files needed beyond the JSON spec.
# Uses the official Swagger UI CDN bundle.
# ---------------------------------------------------------------------------

_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AgniAI — API Docs</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui.min.css" />
  <style>
    /* ── Brand colours ── */
    :root {
      --agni-orange : #FF6B35;
      --agni-dark   : #1a1a2e;
      --agni-mid    : #16213e;
      --agni-card   : #0f3460;
      --agni-accent : #e94560;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--agni-dark);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Top navbar ── */
    .agni-nav {
      background: linear-gradient(135deg, var(--agni-dark) 0%, var(--agni-mid) 100%);
      border-bottom: 2px solid var(--agni-orange);
      padding: 0 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      height: 60px;
      position: sticky;
      top: 0;
      z-index: 9999;
      box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }

    .agni-nav .logo {
      font-size: 22px;
      font-weight: 800;
      color: var(--agni-orange);
      letter-spacing: 1px;
    }

    .agni-nav .tagline {
      font-size: 12px;
      color: #aaa;
      border-left: 1px solid #444;
      padding-left: 16px;
    }

    .agni-nav .badge {
      margin-left: auto;
      background: var(--agni-orange);
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      letter-spacing: 0.5px;
    }

    /* ── Server selector bar ── */
    .server-bar {
      background: var(--agni-mid);
      border-bottom: 1px solid #2a2a4a;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .server-bar label {
      color: #ccc;
      font-size: 13px;
      font-weight: 600;
    }

    .server-bar input {
      background: #1e1e3f;
      border: 1px solid #3a3a6a;
      border-radius: 6px;
      color: #e0e0e0;
      font-size: 13px;
      padding: 6px 12px;
      width: 280px;
      outline: none;
      transition: border-color 0.2s;
    }

    .server-bar input:focus {
      border-color: var(--agni-orange);
    }

    .server-bar button {
      background: var(--agni-orange);
      border: none;
      border-radius: 6px;
      color: #fff;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      padding: 7px 16px;
      transition: opacity 0.2s;
    }

    .server-bar button:hover { opacity: 0.85; }

    .server-bar .preset-btns {
      display: flex;
      gap: 8px;
    }

    .server-bar .preset-btn {
      background: var(--agni-card);
      border: 1px solid #3a3a6a;
      border-radius: 6px;
      color: #aaa;
      cursor: pointer;
      font-size: 12px;
      padding: 5px 12px;
      transition: all 0.2s;
    }

    .server-bar .preset-btn:hover {
      border-color: var(--agni-orange);
      color: var(--agni-orange);
    }

    /* ── Swagger UI wrapper ── */
    #swagger-ui-container {
      background: #fafafa;
      min-height: calc(100vh - 110px);
    }

    /* ── Swagger UI overrides ── */
    #swagger-ui-container .swagger-ui .topbar { display: none !important; }

    #swagger-ui-container .swagger-ui .info .title {
      color: var(--agni-dark) !important;
    }

    #swagger-ui-container .swagger-ui .opblock-tag {
      border-bottom: 1px solid #e0e0e0;
    }

    #swagger-ui-container .swagger-ui .opblock-tag-section h3 {
      font-size: 16px !important;
    }

    /* GET = blue, POST = green-ish, DELETE = red */
    #swagger-ui-container .swagger-ui .opblock.opblock-post {
      border-color: #49cc90;
    }
    #swagger-ui-container .swagger-ui .opblock.opblock-get {
      border-color: #61affe;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #3a3a6a; border-radius: 3px; }
  </style>
</head>
<body>

  <!-- ── Navbar ── -->
  <nav class="agni-nav">
    <span class="logo">🔥 AgniAI</span>
    <span class="tagline">Offline Agniveer Chatbot — API Documentation</span>
    <span class="badge">v1.0</span>
  </nav>

  <!-- ── Server selector ── -->
  <div class="server-bar">
    <label>Base URL:</label>
    <input id="server-url" type="text" value="http://localhost:5000" placeholder="http://localhost:5000" />
    <div class="preset-btns">
      <button class="preset-btn" onclick="setServer('http://localhost:7257')">:7257</button>
      <button class="preset-btn" onclick="setServer('http://localhost:5000')">:5000</button>
    </div>
    <button onclick="applyServer()">Apply &amp; Reload</button>
  </div>

  <!-- ── Swagger UI ── -->
  <div id="swagger-ui-container"></div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-standalone-preset.min.js"></script>
  <script>
    let _ui = null;

    function buildSpec(baseUrl) {
      return fetch('/docs/spec')
        .then(r => r.json())
        .then(spec => {
          // Override servers with the user-chosen base URL
          spec.servers = [{ url: baseUrl, description: 'Selected server' }];
          return spec;
        });
    }

    function loadUI(baseUrl) {
      buildSpec(baseUrl).then(spec => {
        if (_ui) {
          // Re-initialise with updated spec
          document.getElementById('swagger-ui-container').innerHTML = '';
        }
        _ui = SwaggerUIBundle({
          spec: spec,
          dom_id: '#swagger-ui-container',
          presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIStandalonePreset,
          ],
          layout: 'StandaloneLayout',
          deepLinking: true,
          displayRequestDuration: true,
          defaultModelsExpandDepth: 1,
          defaultModelExpandDepth: 2,
          docExpansion: 'list',
          filter: true,
          showExtensions: true,
          showCommonExtensions: true,
          tryItOutEnabled: true,
          requestInterceptor: (req) => {
            // Rewrite the URL to use the chosen base URL
            const chosen = document.getElementById('server-url').value.replace(/\\/$/, '');
            req.url = req.url.replace(/^https?:\\/\\/[^/]+/, chosen);
            return req;
          },
        });
      });
    }

    function setServer(url) {
      document.getElementById('server-url').value = url;
    }

    function applyServer() {
    const url = document.getElementById('server-url').value.replace(/\/$/, '') || 'http://localhost:5000';
      loadUI(url);
    }

    // Initial load
    loadUI(document.getElementById('server-url').value);
  </script>
</body>
</html>
"""
