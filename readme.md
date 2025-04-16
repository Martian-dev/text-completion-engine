## Setup instructions

- first create a `.env` file in the project root directory and add `DOCS_PATH` and `GEMINI_API_KEY`.
- if you want the documents folder to be inside the project itself then create a `documents` folder in the project root directory and add the absolute path to that under `DOCS_PATH` in the `.env` file.
- make sure to create a virtual environment, activate it and install all the required packages given in the `requirements.txt` file.
- now you can execute the following command to run the program

```bash
python src/main.py
```

to get the completion we hit the endpoint `localhost:5000/complete`

```bash
curl -X POST http://localhost:5000/complete -H "Content-Type: application/json" -d '{"text": "<the text you want completion for>"}'
```

memory and cpu usage metrics:

```
=== System Metrics ===
Total RAM (rss): 442.56 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Total RAM (rss): 435.82 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Total RAM (rss): 435.88 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Total RAM (rss): 442.61 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Total RAM (rss): 437.91 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Total RAM (rss): 441.18 MB
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================
```
