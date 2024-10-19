from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pypdf import PdfReader 
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from time import sleep
import os
import chromadb
import re
import ollama

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
print(api_key)

app = FastAPI()
UPLOAD_DIRECTORY = "uploaded_pdfs"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer
groq = Groq(api_key=api_key)

def QnAextract(client, doc):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
    You are given a passage of text. Your task is to extract question-answer pairs from this text. Each pair should consist of a question that can be logically derived from the text, and a corresponding answer that directly addresses the question based on the content provided. Follow these guidelines:

    1. **Identify Key Information:** Look for main points, facts, and statements in the text that can be transformed into questions.
    2. **Formulate Clear Questions:** Create questions that are clear and specific, targeting the key information.
    3. **Provide Accurate Answers:** Ensure that the answers are precise and directly taken from the text. Keep the answers verbose and detailed.
    4. **Format in JSON:** Return the question-answer pairs in JSON format. Your output must be in valid JSON. Do not output anything other than the JSON. Surround your JSON output with <result> </result> tags.


    **Example:**

    "The Eiffel Tower, located in Paris, France, was completed in 1889. It was designed by the engineer Gustave Eiffel and has become a global cultural icon of France and one of the most recognizable structures in the world. The tower stands 324 meters tall and was the tallest man-made structure in the world until the completion of the Chrysler Building in New York in 1930."

    **Extracted Question-Answer Pairs: (JSON format)**
    {json.dumps({
    "question_answer_pairs": [
        {
        "question": "Where is the Eiffel Tower located?",
        "answer": "The Eiffel Tower is located in Paris, France."
        },
        {
        "question": "When was the Eiffel Tower completed?",
        "answer": "The Eiffel Tower was completed in 1889."
        },
        {
        "question": "Who designed the Eiffel Tower?",
        "answer": "The Eiffel Tower was designed by the engineer Gustave Eiffel."
        },
        {
        "question": "How tall is the Eiffel Tower?",
        "answer": "The Eiffel Tower stands 324 meters tall."
        },
        {
        "question": "What structure surpassed the Eiffel Tower in height in 1930?",
        "answer": "The Chrysler Building in New York surpassed the Eiffel Tower in height in 1930."
        }
    ]
    })}

    **Text:** {doc}""",
            }
        ],
        model="llama3-8b-8192",
    )
    return(chat_completion.choices[0].message.content)


def extractJSON(res):
    json_object_match = re.search(r'\{.*\}', res, re.DOTALL)

    if json_object_match:
        json_object = json_object_match.group()
        try:
            # Parse JSON to ensure it is valid
            parsed_json = json.loads(json_object)
            # Print the JSON object
            print(json.dumps(parsed_json, indent=2))
            return parsed_json
        except json.JSONDecodeError:
            print('This is the response')
            print(res)
            raise ValueError("The extracted text is not a valid JSON object.")
    else:
        print(res)
        raise ValueError("No JSON object found in the text.")


class ChatRequest(BaseModel):
    message: str

client = chromadb.PersistentClient(os.getcwd())
collections = client.list_collections()
print(collections)
collection = client.get_or_create_collection(name="docs")
print(collection)


@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message

    
    # # Decode the response
    # response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    if collection.count() == 0:
        chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Respond to this prompt: {user_message}"
            }
            ],
            model="llama3-8b-8192",
        )
    else:
        response = ollama.embeddings(
            prompt=user_message,
            model="mxbai-embed-large"
        )
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=3
        )
        data = results['documents'][0][0]
        chat_completion = groq.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Using this data: {data}. Respond to this prompt: {user_message}"
                }
                ],
                model="llama3-8b-8192",
        )

    print(chat_completion.choices[0].message.content)

    return {"response": chat_completion.choices[0].message.content}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        reader = PdfReader(os.path.join(UPLOAD_DIRECTORY, file.filename)) 
        page = reader.pages[1] 
        text = page.extract_text()
        print((text))
        if collection.count() == 0:
            index = 0
            for j in range(2, len(reader.pages)):
                print("page: ", j)
                page = reader.pages[j] 
                text = page.extract_text()
                text = text.replace('\n', ' ')
                # print(text)
                for attempt in range(3):
                    try:
                        QnA = QnAextract(groq, text)
                        print(QnA)
                        clean = extractJSON(QnA)
                        break
                    except:
                        print(f"Attempt {attempt+1} failed with error")
                        if attempt == 2:  # If this was the last attempt, re-raise the exception
                            raise
                key = []
                for k in clean.keys():
                    key.append(k)
                # print(key[0])
                for i, pair in enumerate(clean[key[0]]):
                    index = index + i        
                    text = pair["question"] + " " + pair["answer"]
                    # print(text)
                    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
                    embedding = response["embedding"]
                    collection.add(
                        ids=[str(index)],
                        embeddings=[embedding],
                        documents=[text]
                    )
                index = index + 1
                sleep(15)
        return JSONResponse(content={"message": "PDF uploaded successfully", "file_path": file_location})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)