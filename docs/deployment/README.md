# Deployment Guide

## Overview

This guide covers deploying the LLM Evaluation Platform in various environments, from local development to production-scale cloud deployments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployments](#cloud-deployments)
5. [On-Premises Deployment](#on-premises-deployment)
6. [Security Configuration](#security-configuration)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 20GB SSD
- **Network**: 100 Mbps
- **OS**: Ubuntu 20.04+, CentOS 8+, or macOS 10.15+

### Recommended Production Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **Network**: 1 Gbps
- **Database**: PostgreSQL 13+ (separate server)
- **Cache**: Redis 6+ (separate server)

### Software Dependencies
- Python 3.9+
- Node.js 18+ (for frontend)
- PostgreSQL 13+
- Redis 6+
- Docker 20.10+ (optional)
- Nginx (recommended for production)

## Local Development Setup

### Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/llm-eval-platform.git
cd llm-eval-platform
```

2. **Set Up Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set Up Database**
```bash
# Install PostgreSQL locally or use Docker
docker run --name postgres-eval \
  -e POSTGRES_DB=llm_eval_platform \
  -e POSTGRES_USER=eval_user \
  -e POSTGRES_PASSWORD=eval_password \
  -p 5432:5432 \
  -d postgres:13
```

4. **Set Up Redis**
```bash
# Install Redis locally or use Docker
docker run --name redis-eval \
  -p 6379:6379 \
  -d redis:6-alpine
```

5. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your database and Redis settings
```

6. **Initialize Database**
```bash
cd backend
python -c "
from database.connection import create_tables
import asyncio
asyncio.run(create_tables())
"
```

7. **Start Backend**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

8. **Start Frontend** (in a new terminal)
```bash
cd frontend
npm install
npm run dev
```

### Development Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://eval_user:eval_password@localhost:5432/llm_eval_platform
DATABASE_ECHO=false

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# API Keys (for testing)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Create docker-compose.yml**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: llm_eval_platform
      POSTGRES_USER: eval_user
      POSTGRES_PASSWORD: eval_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U eval_user"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://eval_user:eval_password@postgres:5432/llm_eval_platform
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend

volumes:
  postgres_data:
  redis_data:
```

2. **Create Backend Dockerfile**
```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. **Create Frontend Dockerfile**
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
```

4. **Deploy with Docker Compose**
```bash
docker-compose up -d
```

### Production Docker Configuration

For production, use multi-stage builds and optimized images:

```dockerfile
# backend/Dockerfile.prod
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

## Cloud Deployments

### AWS Deployment

#### Using AWS ECS with Fargate

1. **Create ECS Task Definition**
```json
{
  "family": "llm-eval-platform",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/llm-eval-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql+asyncpg://user:pass@rds-endpoint:5432/db"
        },
        {
          "name": "REDIS_HOST",
          "value": "elasticache-endpoint"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-eval-platform",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

2. **Create RDS Instance**
```bash
aws rds create-db-instance \
  --db-instance-identifier llm-eval-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 13.7 \
  --master-username eval_user \
  --master-user-password your_password \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name your-subnet-group
```

3. **Create ElastiCache Redis Cluster**
```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id llm-eval-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-group-name your-cache-subnet-group
```

#### Using AWS Lambda (Serverless)

For lighter workloads, deploy as serverless functions:

```python
# lambda_handler.py
import json
from mangum import Mangum
from main import app

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

**SAM Template (template.yaml)**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  LLMEvalAPI:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: backend/
      Handler: lambda_handler.lambda_handler
      Runtime: python3.9
      Timeout: 30
      MemorySize: 1024
      Environment:
        Variables:
          DATABASE_URL: !Ref DatabaseURL
          REDIS_HOST: !Ref RedisHost
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
```

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Build and Push Container**
```bash
# Build the container
docker build -t gcr.io/your-project/llm-eval-backend ./backend

# Push to Container Registry
docker push gcr.io/your-project/llm-eval-backend
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy llm-eval-platform \
  --image gcr.io/your-project/llm-eval-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="postgresql+asyncpg://user:pass@/db?host=/cloudsql/project:region:instance" \
  --set-env-vars REDIS_HOST="redis-ip" \
  --add-cloudsql-instances project:region:instance
```

3. **Set Up Cloud SQL (PostgreSQL)**
```bash
gcloud sql instances create llm-eval-db \
  --database-version POSTGRES_13 \
  --tier db-f1-micro \
  --region us-central1

gcloud sql databases create llm_eval_platform \
  --instance llm-eval-db

gcloud sql users create eval_user \
  --instance llm-eval-db \
  --password your_password
```

### Microsoft Azure

#### Using Azure Container Instances

1. **Create Resource Group**
```bash
az group create --name llm-eval-rg --location eastus
```

2. **Deploy Container**
```bash
az container create \
  --resource-group llm-eval-rg \
  --name llm-eval-platform \
  --image your-registry.azurecr.io/llm-eval-backend:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL="postgresql+asyncpg://user:pass@server.postgres.database.azure.com:5432/db" \
    REDIS_HOST="cache.redis.cache.windows.net" \
  --secure-environment-variables \
    REDIS_PASSWORD="redis_password"
```

## On-Premises Deployment

### Kubernetes Deployment

1. **Create Namespace**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-eval-platform
```

2. **PostgreSQL Deployment**
```yaml
# postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: llm-eval-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: llm_eval_platform
        - name: POSTGRES_USER
          value: eval_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: llm-eval-platform
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

3. **Backend Deployment**
```yaml
# backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: llm-eval-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: your-registry/llm-eval-backend:latest
        env:
        - name: DATABASE_URL
          value: postgresql+asyncpg://eval_user:password@postgres-service:5432/llm_eval_platform
        - name: REDIS_HOST
          value: redis-service
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: llm-eval-platform
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

4. **Ingress Configuration**
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-eval-ingress
  namespace: llm-eval-platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - eval.yourdomain.com
    secretName: llm-eval-tls
  rules:
  - host: eval.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
```

### Traditional Server Deployment

For traditional server deployment without containers:

1. **Install Dependencies**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv postgresql-13 redis-server nginx

# CentOS/RHEL
sudo yum install python39 postgresql13-server redis nginx
```

2. **Set Up Application**
```bash
# Create application user
sudo useradd -m -s /bin/bash llmeval

# Set up application directory
sudo mkdir -p /opt/llm-eval-platform
sudo chown llmeval:llmeval /opt/llm-eval-platform

# Switch to application user
sudo -u llmeval -i

# Clone and set up application
cd /opt/llm-eval-platform
git clone https://github.com/your-org/llm-eval-platform.git .
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Create Systemd Service**
```ini
# /etc/systemd/system/llm-eval-backend.service
[Unit]
Description=LLM Evaluation Platform Backend
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=llmeval
Group=llmeval
WorkingDirectory=/opt/llm-eval-platform/backend
Environment=PATH=/opt/llm-eval-platform/venv/bin
ExecStart=/opt/llm-eval-platform/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

4. **Configure Nginx**
```nginx
# /etc/nginx/sites-available/llm-eval-platform
server {
    listen 80;
    server_name eval.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name eval.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/eval.yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/eval.yourdomain.com.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static/ {
        alias /opt/llm-eval-platform/frontend/public/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Security Configuration

### SSL/TLS Setup

1. **Using Let's Encrypt (Recommended)**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d eval.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

2. **Using Custom Certificates**
```bash
# Generate private key
openssl genrsa -out eval.yourdomain.com.key 2048

# Generate certificate signing request
openssl req -new -key eval.yourdomain.com.key -out eval.yourdomain.com.csr

# Install certificate files
sudo cp eval.yourdomain.com.crt /etc/ssl/certs/
sudo cp eval.yourdomain.com.key /etc/ssl/private/
sudo chmod 600 /etc/ssl/private/eval.yourdomain.com.key
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

### Database Security

1. **PostgreSQL Configuration**
```sql
-- Create dedicated user with limited privileges
CREATE USER eval_user WITH PASSWORD 'strong_password';
CREATE DATABASE llm_eval_platform OWNER eval_user;

-- Grant only necessary privileges
GRANT CONNECT ON DATABASE llm_eval_platform TO eval_user;
GRANT USAGE ON SCHEMA public TO eval_user;
GRANT CREATE ON SCHEMA public TO eval_user;
```

2. **Connection Security**
```bash
# Edit postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'

# Edit pg_hba.conf
hostssl all eval_user 0.0.0.0/0 md5
```

### Environment Variables Security

Use a secrets management system in production:

```bash
# Using HashiCorp Vault
vault kv put secret/llm-eval-platform \
  database_password="secure_password" \
  redis_password="redis_password" \
  secret_key="jwt_secret_key"

# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name llm-eval-platform/database \
  --secret-string '{"password":"secure_password"}'
```

## Performance Optimization

### Database Optimization

1. **PostgreSQL Configuration**
```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

2. **Connection Pooling**
```python
# Use PgBouncer for connection pooling
# /etc/pgbouncer/pgbouncer.ini
[databases]
llm_eval_platform = host=localhost port=5432 dbname=llm_eval_platform

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

### Redis Optimization

```bash
# redis.conf optimizations
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Performance

1. **Gunicorn Configuration**
```python
# gunicorn.conf.py
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
```

2. **Nginx Optimization**
```nginx
# nginx.conf optimizations
worker_processes auto;
worker_connections 1024;

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

## Monitoring and Maintenance

### Health Checks

1. **Application Health Check**
```python
# Add to main.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    # Check database connection
    db_healthy = await check_database_health()
    
    # Check Redis connection
    redis_healthy = cache_service.health_check()
    
    return {
        "status": "healthy" if db_healthy and redis_healthy else "unhealthy",
        "database": db_healthy,
        "redis": redis_healthy,
        "timestamp": datetime.utcnow().isoformat()
    }
```

2. **External Monitoring**
```bash
# Uptime monitoring with curl
*/5 * * * * curl -f https://eval.yourdomain.com/health || echo "Health check failed" | mail -s "LLM Eval Platform Down" admin@yourdomain.com
```

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/llm-eval-platform/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": "INFO",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Backup Strategy

1. **Database Backup**
```bash
#!/bin/bash
# backup_database.sh
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/llm_eval_platform_$DATE.sql"

mkdir -p $BACKUP_DIR

pg_dump -h localhost -U eval_user -d llm_eval_platform > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_FILE.gz s3://your-backup-bucket/database/
```

2. **Redis Backup**
```bash
#!/bin/bash
# backup_redis.sh
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Create Redis backup
redis-cli BGSAVE
sleep 10  # Wait for backup to complete

# Copy RDB file
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Compress and upload
gzip $BACKUP_DIR/redis_$DATE.rdb
aws s3 cp $BACKUP_DIR/redis_$DATE.rdb.gz s3://your-backup-bucket/redis/
```

### Update and Maintenance

1. **Rolling Updates (Kubernetes)**
```bash
# Update deployment
kubectl set image deployment/backend backend=your-registry/llm-eval-backend:v2.0.0 -n llm-eval-platform

# Monitor rollout
kubectl rollout status deployment/backend -n llm-eval-platform

# Rollback if needed
kubectl rollout undo deployment/backend -n llm-eval-platform
```

2. **Blue-Green Deployment**
```bash
# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Test green environment
curl -f https://green.eval.yourdomain.com/health

# Switch traffic (update load balancer)
# Terminate blue environment after verification
```

### Troubleshooting

Common issues and solutions:

1. **High Memory Usage**
```bash
# Check memory usage
free -h
docker stats

# Optimize PostgreSQL
# Reduce shared_buffers if needed
# Check for memory leaks in application
```

2. **Slow Database Queries**
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Analyze slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

3. **High CPU Usage**
```bash
# Check CPU usage
top
htop

# Profile application
py-spy top --pid $(pgrep -f "uvicorn main:app")
```

For additional deployment support, consult the [troubleshooting guide](../troubleshooting/) or contact support. 