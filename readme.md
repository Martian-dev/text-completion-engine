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
Timestamp: 2025-04-16 21:46:16
Total RAM (rss): 442.49 MB
CPU Usage: 0.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Timestamp: 2025-04-16 21:46:33
Total RAM (rss): 441.74 MB
CPU Usage: 0.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Timestamp: 2025-04-16 21:55:21
Total RAM (rss): 442.47 MB
CPU Usage: 0.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Timestamp: 2025-04-16 21:56:18
Total RAM (rss): 498.39 MB
CPU Usage: 7.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Timestamp: 2025-04-16 21:56:30
Total RAM (rss): 501.70 MB
CPU Usage: 13.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================

=== System Metrics ===
Timestamp: 2025-04-16 21:56:42
Total RAM (rss): 501.70 MB
CPU Usage: 7.00%
FAISS Index Size: 0.08 MB
Metadata Size: 0.03 MB
Embedding Model Size: 0.00 MB
=======================
```
