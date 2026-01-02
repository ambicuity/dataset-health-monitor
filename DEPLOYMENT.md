# Dataset Health Monitor - GitHub App Deployment Guide

This guide covers deploying the Dataset Health Monitor as a GitHub App.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Creating a GitHub App](#creating-a-github-app)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [Render](#render)
  - [Fly.io](#flyio)
  - [AWS](#aws)
- [Configuration](#configuration)
- [Webhook Setup](#webhook-setup)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- A GitHub account with admin access to create GitHub Apps

## Creating a GitHub App

1. Go to **GitHub Settings** → **Developer settings** → **GitHub Apps**
2. Click **New GitHub App**
3. Fill in the required fields:

### Basic Information

| Field | Value |
|-------|-------|
| GitHub App name | `Dataset Health Monitor` (or your preferred name) |
| Homepage URL | Your deployment URL or repository URL |
| Webhook URL | `https://your-domain.com/webhook` |
| Webhook secret | Generate a secure random string |

### Permissions

Set the following repository permissions:

| Permission | Access |
|------------|--------|
| Contents | Read |
| Issues | Read & Write |
| Pull requests | Read & Write |
| Metadata | Read |

### Events

Subscribe to these events:

- ✅ Installation
- ✅ Push
- ✅ Repository
- ✅ Workflow run (optional)

4. Click **Create GitHub App**
5. After creation, generate a **Private Key** and save it securely
6. Note down the **App ID**

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/ambicuity/dataset-health-monitor.git
cd dataset-health-monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt -r requirements-app.txt
```

### 3. Set environment variables

```bash
export GITHUB_APP_ID="your-app-id"
export GITHUB_PRIVATE_KEY="$(cat path/to/private-key.pem)"
export GITHUB_WEBHOOK_SECRET="your-webhook-secret"
export PORT=8000
```

Or create a `.env` file:

```env
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY_PATH=/path/to/private-key.pem
GITHUB_WEBHOOK_SECRET=your-webhook-secret
PORT=8000
DEBUG=true
```

### 4. Run the application

```bash
python -m app.app
```

### 5. Expose for webhooks (using ngrok)

```bash
ngrok http 8000
```

Update your GitHub App's webhook URL to the ngrok URL.

## Docker Deployment

### Build the image

```bash
docker build -t dataset-health-monitor .
```

### Run with Docker

```bash
docker run -d \
  --name dataset-health-monitor \
  -p 8000:8000 \
  -e GITHUB_APP_ID="your-app-id" \
  -e GITHUB_PRIVATE_KEY="$(cat private-key.pem)" \
  -e GITHUB_WEBHOOK_SECRET="your-webhook-secret" \
  -v $(pwd)/state:/app/state \
  dataset-health-monitor
```

### Run with Docker Compose

```bash
# Set environment variables
export GITHUB_APP_ID="your-app-id"
export GITHUB_PRIVATE_KEY="$(cat private-key.pem)"
export GITHUB_WEBHOOK_SECRET="your-webhook-secret"

# Start services
docker-compose up -d
```

## Cloud Deployment

### Render

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure the service:

| Setting | Value |
|---------|-------|
| Environment | Docker |
| Instance Type | Free (or higher for production) |
| Health Check Path | `/health` |

4. Add environment variables:
   - `GITHUB_APP_ID`
   - `GITHUB_PRIVATE_KEY`
   - `GITHUB_WEBHOOK_SECRET`

5. Deploy and update your GitHub App's webhook URL

### Fly.io

1. Install the [Fly CLI](https://fly.io/docs/hands-on/install-flyctl/)

2. Create `fly.toml`:

```toml
app = "dataset-health-monitor"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[services]]
  http_checks = []
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443

  [[services.http_checks]]
    interval = 10000
    grace_period = "5s"
    method = "get"
    path = "/health"
    protocol = "http"
    timeout = 2000
```

3. Deploy:

```bash
fly launch
fly secrets set GITHUB_APP_ID="your-app-id"
fly secrets set GITHUB_PRIVATE_KEY="$(cat private-key.pem)"
fly secrets set GITHUB_WEBHOOK_SECRET="your-webhook-secret"
fly deploy
```

### AWS

#### Option 1: AWS App Runner

1. Push your image to ECR:

```bash
aws ecr create-repository --repository-name dataset-health-monitor
docker tag dataset-health-monitor:latest <account>.dkr.ecr.<region>.amazonaws.com/dataset-health-monitor:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/dataset-health-monitor:latest
```

2. Create an App Runner service in the AWS Console
3. Configure environment variables in the service settings

#### Option 2: AWS ECS with Fargate

1. Create an ECS cluster
2. Create a task definition with the container image
3. Configure environment variables as secrets in AWS Secrets Manager
4. Create a service with an Application Load Balancer

#### Option 3: AWS Lambda with API Gateway

For serverless deployment, you'll need to adapt the FastAPI app using [Mangum](https://mangum.io/):

```python
from mangum import Mangum
from app.app import app

handler = Mangum(app)
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_APP_ID` | Yes | Your GitHub App ID |
| `GITHUB_PRIVATE_KEY` | Yes* | PEM-encoded private key content |
| `GITHUB_PRIVATE_KEY_PATH` | Yes* | Path to private key file |
| `GITHUB_WEBHOOK_SECRET` | Recommended | Webhook signature verification secret |
| `HOST` | No | Server host (default: `0.0.0.0`) |
| `PORT` | No | Server port (default: `8000`) |
| `DEBUG` | No | Enable debug mode (default: `false`) |

*Either `GITHUB_PRIVATE_KEY` or `GITHUB_PRIVATE_KEY_PATH` is required.

### Storing Secrets

For production deployments, use a secrets manager:

- **Render**: Environment variables are encrypted
- **Fly.io**: Use `fly secrets set`
- **AWS**: Use AWS Secrets Manager or Parameter Store
- **Docker**: Use Docker secrets in Swarm mode

## Webhook Setup

### Verifying Webhooks

The app automatically verifies webhook signatures using the `GITHUB_WEBHOOK_SECRET`. Ensure this matches the secret configured in your GitHub App settings.

### Testing Webhooks

1. Use GitHub's webhook delivery feature to redeliver events
2. Use the [GitHub Webhooks CLI](https://github.com/cli/cli) for local testing:

```bash
gh webhook forward --repo=owner/repo --events=push --url=http://localhost:8000/webhook
```

### Webhook Events

| Event | Description |
|-------|-------------|
| `installation` | App installed/uninstalled |
| `installation_repositories` | Repos added/removed from installation |
| `push` | Code pushed (triggers dataset check if datasets.yaml changed) |
| `workflow_run` | Workflow completed |
| `ping` | Webhook verification |

## Troubleshooting

### Common Issues

#### "Private key not valid"

- Ensure the private key is in PEM format
- Check for proper newline characters (use `\n` in environment variables)
- Verify the key matches your GitHub App

#### "Webhook signature verification failed"

- Verify `GITHUB_WEBHOOK_SECRET` matches GitHub App settings
- Ensure the secret doesn't have trailing whitespace

#### "Installation token generation failed"

- Verify the App ID is correct
- Check that the app is installed on the target repository
- Ensure the private key is valid and not expired

### Logs

View application logs:

```bash
# Docker
docker logs dataset-health-monitor

# Docker Compose
docker-compose logs -f

# Fly.io
fly logs

# Render
# Check the Render dashboard
```

### Health Check

Verify the app is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "app_name": "Dataset Health Monitor"
}
```

## Security Considerations

1. **Never commit private keys** - Use environment variables or secrets management
2. **Use HTTPS** - All production deployments should use TLS
3. **Verify webhooks** - Always configure `GITHUB_WEBHOOK_SECRET`
4. **Minimal permissions** - Only request permissions you need
5. **Rotate keys** - Periodically regenerate private keys

## Support

- 📖 [Documentation](README.md)
- 🐛 [Issues](https://github.com/ambicuity/dataset-health-monitor/issues)
- 💬 [Discussions](https://github.com/ambicuity/dataset-health-monitor/discussions)
