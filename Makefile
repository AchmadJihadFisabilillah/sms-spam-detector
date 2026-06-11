.PHONY: backend frontend docker train clean

backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

docker:
	docker compose up --build

train:
	curl -X POST http://localhost:8000/api/train \
	  -H "Content-Type: application/json" \
	  -d '{"test_size":0.2,"random_state":42,"max_features":8000}'

clean:
	rm -f backend/artifacts/*.joblib backend/artifacts/*.json backend/artifacts/*.csv
