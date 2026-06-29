import json

from app import app


def test_swagger_spec_dynamic_servers():
    client = app.test_client()

    # Request from localhost:5000
    resp_local = client.get("/docs/spec", headers={"Host": "localhost:5000"})
    assert resp_local.status_code == 200
    data_local = json.loads(resp_local.data.decode("utf-8"))
    assert data_local["servers"] == [
        {"url": "http://localhost:5000", "description": "Current server"}
    ]

    # Request from a LAN IP
    resp_lan = client.get("/docs/spec", headers={"Host": "192.168.1.5:5000"})
    assert resp_lan.status_code == 200
    data_lan = json.loads(resp_lan.data.decode("utf-8"))
    assert data_lan["servers"] == [
        {"url": "http://192.168.1.5:5000", "description": "Current server"}
    ]

    # Request from a reverse proxy / custom hostname
    resp_proxy = client.get("/docs/spec", headers={"Host": "example.com"})
    assert resp_proxy.status_code == 200
    data_proxy = json.loads(resp_proxy.data.decode("utf-8"))
    assert data_proxy["servers"] == [
        {"url": "http://example.com", "description": "Current server"}
    ]


def test_swagger_ui_page_loads():
    client = app.test_client()
    resp = client.get("/docs")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "SwaggerUIBundle" in html
    assert 'window.location.origin + "/docs/spec"' in html
    assert "requestInterceptor" in html
