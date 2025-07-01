# scripts/start_dev.sh
#!/bin/bash
# Start dev environment (backend + frontend)
echo "Starting Backend..."
cd backend && uvicorn main:app --reload &
cd ../frontend && echo "Starting Frontend..." && npm run dev
