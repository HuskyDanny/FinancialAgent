# Getting Started - Financial Agent Development

## Quick Start

### Prerequisites
- **kubectl** configured with AKS cluster access
- **Azure CLI** (az) authenticated
- **Node.js 18+** (for local frontend development)
- **Python 3.12+** (for local backend development)
- **Git**

### 1. Clone Repository

```bash
git clone <repository>
cd financial_agent
```

### 2. Access Deployed Services

The application is deployed on Kubernetes. Access it via:

**Production URLs:**
- **Frontend**: https://klinematrix.com
- **Backend API**: https://klinematrix.com/api/health
- **API Docs**: https://klinematrix.com/api/docs

**Verify deployment:**
```bash
# Check pod status
kubectl get pods -n klinematrix-test

# Check services
kubectl get svc -n klinematrix-test

# View backend logs
kubectl logs -f deployment/backend -n klinematrix-test

# View frontend logs
kubectl logs -f deployment/frontend -n klinematrix-test
```

### 3. Local Development (Optional)

For local code development with hot reload:

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:5173
```

**Note**: Local development requires port-forwarding to access Kubernetes services:
```bash
# Forward MongoDB
kubectl port-forward -n klinematrix-test svc/mongodb-service 27017:27017

# Forward Redis
kubectl port-forward -n klinematrix-test svc/redis-service 6379:6379
```

### 4. Verify Walking Skeleton

**Via production deployment:**
```bash
curl https://klinematrix.com/api/health | python3 -m json.tool
```

Expected response:
```json
{
  "status": "ok",
  "environment": "test",
  "dependencies": {
    "mongodb": {"connected": true},
    "redis": {"connected": true}
  }
}
```

## Development Commands

```bash
# Code Quality (REQUIRED before commits)
make fmt          # Format all code
make lint         # Lint all code
make test         # Run all tests

# Backend-specific
cd backend
make test-backend # Run Python tests
make lint-backend # Lint Python code

# Frontend-specific
cd frontend
npm test          # Run React tests
npm run lint      # Lint TypeScript code

# Version Management
./scripts/bump-version.sh backend patch   # Increment backend version
./scripts/bump-version.sh frontend minor  # Increment frontend version
```

**Note**: Every commit must increment at least one component version. Pre-commit hooks enforce this.

## Project Structure

```
financial_agent/
├── backend/                 # FastAPI backend
│   ├── src/
│   │   ├── api/            # REST API endpoints
│   │   ├── core/           # Configuration and utilities
│   │   ├── database/       # MongoDB and Redis connections
│   │   └── main.py         # Application entry point
│   ├── pyproject.toml      # Python dependencies and tools
│   └── Dockerfile          # Backend container
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API client services
│   │   ├── types/          # TypeScript type definitions
│   │   └── App.tsx         # Main application
│   ├── package.json        # Node.js dependencies
│   └── Dockerfile          # Frontend container
├── docs/                   # Documentation
├── .pipeline/              # Kubernetes and CI/CD
│   └── k8s/               # Kubernetes manifests
├── scripts/               # Utility scripts
├── Makefile              # Development commands
└── README.md             # Project overview
```

## Tech Stack Overview

### Backend
- **Python 3.12** with modern syntax
- **FastAPI** for async API framework
- **MongoDB** for document storage
- **Redis** for caching
- **LangChain + LangGraph** for AI agent orchestration
- **LangSmith** for observability

### Frontend
- **React 18** with TypeScript 5.x
- **Vite** for fast build tooling
- **TailwindCSS** for styling
- **React Query** for state management
- Modern chat interface

### Infrastructure
- **Docker Compose** for local development
- **Kubernetes** for production deployment
- **GitHub Actions** for CI/CD
- **Azure** for cloud hosting

## Local Development Workflow

### Backend Development

```bash
# Option 1: Direct Python execution (fastest hot reload)
cd backend
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend Hot Reload:**
- Function/method changes: Auto-reload ✅
- New routes/endpoints: Auto-reload ✅
- New dependencies: Restart required ❌
- Global/module-level changes: Restart required ❌

**Testing backend directly:**
```bash
curl http://localhost:8000/api/health
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:5173
```

**Frontend Hot Reload:**
- Component changes: Auto-reload ✅
- CSS/styling changes: Auto-reload ✅
- New dependencies: Restart required ❌

**Testing frontend:**
Visit http://localhost:5173 in your browser.

### Database Access

**MongoDB:**
```bash
# Via port-forward to Kubernetes
kubectl port-forward -n klinematrix-test svc/mongodb-service 27017:27017

# Then connect locally
mongosh mongodb://localhost:27017/klinematrix_test
```

**Redis:**
```bash
# Via port-forward to Kubernetes
kubectl port-forward -n klinematrix-test svc/redis-service 6379:6379

# Then connect locally
redis-cli -h localhost -p 6379
```

**Or access directly in pods:**
```bash
# MongoDB shell in pod
kubectl exec -it deployment/mongodb -n klinematrix-test -- mongosh

# Redis CLI in pod
kubectl exec -it deployment/redis -n klinematrix-test -- redis-cli
```

## Code Standards

### Python
- Use modern syntax: `|` unions, `match/case`, f-strings, `@dataclass`
- Type hints required
- Black for formatting
- Ruff for linting
- MyPy for type checking
- Maximum 500 lines per file

### TypeScript
- ES modules
- Optional chaining
- `satisfies` operator
- ESLint for linting
- Prettier for formatting
- Maximum 500 lines per file

### Documentation
- Descriptive docstrings at top of every file
- Rich comments for key business logic (explain "why", not "what")
- No code duplication (DRY principle)

## Quality Gates

Before committing, always run:

```bash
make fmt && make test && make lint
```

This ensures:
- ✅ Code is properly formatted
- ✅ All tests pass
- ✅ No linting violations

## Common Development Tasks

### Adding a New API Endpoint

1. Create endpoint in `backend/src/api/`
2. Add tests in `backend/tests/api/`
3. Update API client in `frontend/src/services/`
4. Run quality gates: `make fmt && make test && make lint`

### Adding a New React Component

1. Create component in `frontend/src/components/`
2. Add TypeScript types in `frontend/src/types/`
3. Write tests
4. Run quality gates

### Adding a New Database Model

1. Create Pydantic model in `backend/src/models/`
2. Add database operations in `backend/src/database/`
3. Write tests
4. Ensure TypeScript types match in frontend

## Environment Variables

### Backend (Kubernetes Deployment)
Environment variables are configured in Kubernetes deployments and ConfigMaps:

**Key Variables:**
```yaml
ENVIRONMENT: test
MONGODB_URL: mongodb://mongodb-service:27017/klinematrix_test
REDIS_URL: redis://redis-service:6379
CORS_ORIGINS: '["https://klinematrix.com"]'
EMAIL_BYPASS_MODE: true  # Test environment only
```

See `.pipeline/k8s/base/backend/deployment.yaml` for complete configuration.

### Frontend (Local Development)
Create `frontend/.env.local` for local development:
```env
VITE_API_BASE_URL=http://localhost:8000
```

For production builds, API URL is auto-detected from window.location.

## Troubleshooting

### Backend Not Responding

```bash
# Check pod status
kubectl get pods -n klinematrix-test -l app=backend

# Check pod logs
kubectl logs -f deployment/backend -n klinematrix-test

# Restart backend pods
kubectl rollout restart deployment/backend -n klinematrix-test

# Check service endpoints
kubectl get endpoints backend-service -n klinematrix-test
```

### Frontend Shows 503 Error

```bash
# Check frontend pod status
kubectl get pods -n klinematrix-test -l app=frontend

# Check frontend logs
kubectl logs -f deployment/frontend -n klinematrix-test

# Restart frontend pods
kubectl rollout restart deployment/frontend -n klinematrix-test
```

### Database Connection Errors

```bash
# Check database pod status
kubectl get pods -n klinematrix-test -l app=mongodb
kubectl get pods -n klinematrix-test -l app=redis

# Check database logs
kubectl logs deployment/mongodb -n klinematrix-test
kubectl logs deployment/redis -n klinematrix-test

# Verify service endpoints
kubectl get endpoints mongodb-service -n klinematrix-test
kubectl get endpoints redis-service -n klinematrix-test

# Restart backend after database restart
kubectl rollout restart deployment/backend -n klinematrix-test
```

### Local Development Issues

**Port conflicts:**
```bash
# Check for processes using ports
lsof -i :5173  # Frontend dev server
lsof -i :8000  # Backend
lsof -i :27017 # MongoDB port-forward
lsof -i :6379  # Redis port-forward
```

**Port-forward not working:**
```bash
# Kill existing port-forwards
pkill -f "port-forward"

# Restart port-forwards
kubectl port-forward -n klinematrix-test svc/mongodb-service 27017:27017 &
kubectl port-forward -n klinematrix-test svc/redis-service 6379:6379 &
```

### Hot Reload Not Working

**If code changes aren't reflected:**

**For local development:**
1. Check if change requires restart (dependencies, module-level code)
2. Stop and restart the dev server:
   ```bash
   # Backend: Ctrl+C, then restart uvicorn
   # Frontend: Ctrl+C, then npm run dev
   ```

**For Kubernetes deployment:**
```bash
# Rebuild image with new version
./scripts/bump-version.sh backend patch
az acr build --registry financialagent --image klinematrix/backend:test-v{VERSION} --file backend/Dockerfile backend/

# Update deployment
kubectl set image deployment/backend backend=financialagent-gxftdbbre4gtegea.azurecr.io/klinematrix/backend:test-v{VERSION} -n klinematrix-test

# Or force pod restart
kubectl rollout restart deployment/backend -n klinematrix-test
```

## Next Steps

Once your development environment is running:

1. **Explore the Interface** - Visit https://klinematrix.com
2. **Review the Code** - Check out backend and frontend source code
3. **Read the Architecture** - See [System Design](../architecture/system-design.md)
4. **Review Coding Standards** - See [Coding Standards](coding-standards.md)
5. **Review Deployment** - See [Deployment Workflow](../deployment/workflow.md)
6. **Start Building** - Pick a task and start developing!

## Resources

- **API Documentation**: https://klinematrix.com/api/docs
- **Complete Documentation**: [docs/README.md](../README.md)
- **Project Specifications**: [docs/project/specifications.md](../project/specifications.md)
- **12-Factor Agent Guide**: [docs/architecture/agent-12-factors.md](../architecture/agent-12-factors.md)
- **Troubleshooting**: [docs/troubleshooting/README.md](../troubleshooting/README.md)
- **Deployment Guide**: [docs/deployment/workflow.md](/Users/allenpan/Desktop/repos/projects/financial_agent/docs/deployment/workflow.md)
