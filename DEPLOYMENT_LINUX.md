# 🐧 AgniAI Linux Server Deployment Guide

This document provides instructions for deploying the pre-built **AgniAI** Linux executable package (`agniai-linux-x86_64.zip`) onto a target Linux server.

---

## 📦 What is Included in the Build Package

The distribution archive (`agniai-linux-x86_64.zip`) contains all necessary binaries, configuration templates, launcher scripts, and system dependency installers:

| File / Folder | Purpose & Description |
|---|---|
| `agniai` | **Main Standalone Executable** — 64-bit Linux ELF binary application. |
| `_internal/` | **Bundled Dependencies** — Embedded Python runtime, shared C libraries, PyODBC binaries, and framework dependencies. |
| `start.sh` | **1-Click Launcher** — Automatically verifies Ollama installation, extracts offline model archives, checks `.env`, starts Ollama daemon, and launches `./agniai`. |
| `install_deps.sh` | **System Dependency Installer** — Shell script to install required Linux OS packages (unixODBC, Microsoft SQL Server ODBC driver `msodbcsql18`, and Ollama). |
| `agniai.service` | **Systemd Unit File** — Unit configuration for running AgniAI as a persistent system daemon service. |
| `.env.example` | **Environment Template** — Pre-configured template for Linux database credentials, Ollama endpoints, and server settings. |
| `models.zip` / `models--*.zip` | *(Optional)* Offline LLM / embedding model archives automatically unpacked into `~/.ollama` and `~/.cache/huggingface/`. |

---

## 🚀 Server Deployment Steps

Follow these steps to deploy and run AgniAI on a target Linux server (Ubuntu / Debian / RHEL / CentOS).

### Step 1: Copy and Extract the Build Package
Upload `agniai-linux-x86_64.zip` to your target Linux server and extract it:

```bash
# Unzip the deployment package
unzip agniai-linux-x86_64.zip
cd agniai
```

---

### Step 2: Install System Dependencies (First-Time Setup)
Run the automated dependency installer script with root privileges to install required system packages (such as `unixodbc`, `msodbcsql18` for SQL Server connectivity, and Ollama):

```bash
sudo bash install_deps.sh
```

---

### Step 3: Configure Environment Variables (`.env`)
Create your runtime `.env` configuration file from the provided template:

```bash
cp .env.example .env
nano .env
```

Edit the required configuration parameters:
- **`DB_SERVER`**: SQL Server hostname or IP address
- **`DB_NAME`**: Target database name
- **`DB_USER`** & **`DB_PASSWORD`**: Database authentication credentials
- **`OLLAMA_BASE_URL`**: `http://localhost:11434` (or external Ollama service URL)
- **`PORT`**: Application HTTP port (default: `5000`)

---

### Step 4: Make Launcher Executable & Start Application
Grant execution permissions to `start.sh` and launch the application:

```bash
chmod +x start.sh
./start.sh
```

#### What `start.sh` Automatically Does:
1. **Checks Ollama**: Verifies if `ollama` is installed on the host.
2. **Extracts Offline Models**: If offline model archives (`models.zip` / `models--*.zip`) are present in the folder, extracts them to `~/.ollama` and `~/.cache/huggingface/hub`.
3. **Starts Ollama Daemon**: Ensures the `ollama serve` process is running in the background.
4. **Launches AgniAI**: Executes the `./agniai` executable binary using the specified `.env` settings.

---

### Step 5: Verify Deployment
Confirm that the application is running by testing the health check endpoint:

```bash
curl http://localhost:5000/api/health
```

Expected JSON response:
```json
{"status": "healthy"}
```

---

## ⚙️ Setting Up AgniAI as a Production Daemon (Systemd)

To run AgniAI as a background system service that automatically starts on server reboot:

### 1. Copy Application Directory to `/opt/agniai`
```bash
sudo cp -r . /opt/agniai
```

### 2. Install the Systemd Unit File
```bash
sudo cp /opt/agniai/agniai.service /etc/systemd/system/agniai.service
```

### 3. Enable and Start the Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable agniai
sudo systemctl start agniai
```

### 4. Service Management & Logging
```bash
# Check service status
sudo systemctl status agniai

# View live logs
sudo journalctl -u agniai -f -n 100

# Restart or Stop service
sudo systemctl restart agniai
sudo systemctl stop agniai
```

---

## 🛠️ Quick Reference & Troubleshooting

| Action | Command |
|---|---|
| Check Ollama status | `curl http://localhost:11434/api/tags` |
| View active application logs | `sudo journalctl -u agniai -f` |
| Verify installed ODBC drivers | `odbcinst -q -d` |
| Check port usage (port 5000) | `sudo netstat -tulpn \| grep 5000` |

