# scripts/build_all.sh
#!/bin/bash
cd backend && echo "Installing backend deps..." && pip install -r requirements.txt && cd ..
cd frontend && echo "Installing frontend deps..." && npm install && npm run build && cd ..
echo "✅ Full build complete"
