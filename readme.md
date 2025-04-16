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
