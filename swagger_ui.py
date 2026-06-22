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

from flask import Blueprint, Response, jsonify, request, send_from_directory

swagger_bp = Blueprint("swagger", __name__)

_SPEC_PATH = Path(__file__).parent / "static" / "swagger.json"


def _load_spec() -> dict:
    spec = json.loads(
        _SPEC_PATH.read_text(encoding="utf-8")
    )
    host_url = request.host_url.rstrip("/")
    spec["servers"] = [
        {
            "url": host_url,
            "description": "Current server"
        }
    ]
    return spec


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
  <title>AgniAI — API Docs</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui.min.css" />
  <style>
    body {
      margin: 0;
      background: #fafafa;
    }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.min.js"></script>
  <script>
    window.onload = function() {
      window.ui = SwaggerUIBundle({
        url: window.location.origin + "/docs/spec",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis
        ],
        layout: "BaseLayout",
        defaultModelsExpandDepth: 1,
        defaultModelExpandDepth: 2,
        tryItOutEnabled: true,
        requestInterceptor: (req) => {
          const currentOrigin = window.location.origin;
          req.url = req.url
            .replace("http://localhost:5000", currentOrigin)
            .replace("http://127.0.0.1:5000", currentOrigin);
          return req;
        }
      });
    };
  </script>
</body>
</html>
"""
