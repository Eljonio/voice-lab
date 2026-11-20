# заглушка: GigaAM импортирует pyannote только для transcribe_longform (мы его не зовём); transformers проверяет импорты заранее
# установка в venv обработки: mkdir -p <venv>/lib/python3.12/site-packages/pyannote && cp dataprep/pyannote_stub.py <тот же путь>/pyannote/__init__.py
