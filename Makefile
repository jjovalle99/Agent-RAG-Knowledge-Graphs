format:
	isort app.py
	black app.py
	isort src/*.py
	black src/*.py

pycache:
	find ./ -type d -name '__pycache__' -exec rm -rf {} +