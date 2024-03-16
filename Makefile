format:
	isort serve.py
	black serve.py
	isort src/*.py
	black src/*.py

pycache:
	find ./ -type d -name '__pycache__' -exec rm -rf {} +