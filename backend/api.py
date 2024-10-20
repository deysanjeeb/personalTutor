from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pypdf import PdfReader 
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq
from dotenv import load_dotenv
from time import sleep
import os
import chromadb
import re
import ollama
import requests

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
vectara_api = os.getenv('VECTARA_API_KEY')
vectara_corpus_id = os.getenv('VECTARA_CORPUS_ID')
vectara_customer_id = os.getenv('VECTARA_CUSTOMER_ID')
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


def query_response(query_str, customer_id, corpus_id, api_key, n=3):
    """
    Sends a query to the Vectara API and retrieves results.

    Parameters:
    - query_str (str): The query string to search.
    - customer_id (str): The ID of the customer associated with the query.
    - corpus_id (str): The ID of the corpus to query.
    - api_key (str): The API key for authentication.
    - n (int, optional): The number of top results to retrieve (default is 3).

    Returns:
    - dict: The response from the API containing the search results.
    """

    url = "https://api.vectara.io/v1/query"

    payload = json.dumps({
    "query": [
        {
          "query": query_str,
          "start": 0,
          "numResults": 10,
          "contextConfig": {
            "sentencesBefore": 0,
            "sentencesAfter": 0
          },
          "corpusKey": [
            {
              "customerId": customer_id,
              "corpusId": corpus_id
            }
          ],
        }
      ]
    })

    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'x-api-key': api_key
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()

    return res


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

@app.post("/upload")
async def upload_pdf_file(file: UploadFile = File(...)):
    """
    Uploads a PDF file to the Vectara API.

    Parameters:
    - customer_id (str): The ID of the customer.
    - corpus_id (str): The ID of the corpus to which the file belongs.
    - file_path (str): The path to the PDF file to upload.
    - api_key (str): The API key for authentication.
    # """
    # vectara_customer_id
    # vectara_corpus_id
    # vectara_api
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    file_dir = os.path.join(UPLOAD_DIRECTORY, file.filename)
    url = f"https://api.vectara.io/v1/upload?c={vectara_customer_id}&o={vectara_corpus_id}"

    post_headers = { 
        "x-api-key": vectara_api,
        "customer-id": str(vectara_customer_id)
    }
    files = {
        "file": (file_dir, open(file_dir, 'rb')),
    }  
    response = requests.post(url, files=files, verify=True, headers=post_headers)

    if response.status_code == 200:
        print("File uploaded successfully")
        return JSONResponse(content={"message": "PDF uploaded successfully", "file_path": file_location})

    else:
        print(f"Error uploading file: {response.text}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


    # return response


# @app.post("/vidGen")
# async def chat(request: ChatRequest):


#LAVANYA - PHASE_2

from bs4 import BeautifulSoup
import requests
import os

'''
Getting references
'''


# Define your search query
query = 'python'
search_engine = 'https://www.google.com/search'

proxies = {
    'http': 'http://brd-customer-hl_e62aa94d-zone-datacenter_proxy1:0did46f1tu65@brd.superproxy.io:22225',
    'https': 'http://brd-customer-hl_e62aa94d-zone-datacenter_proxy1:0did46f1tu65@brd.superproxy.io:22225'
}

def get_references(query, search_engine, proxies):
    

    # Headers to avoid being blocked
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
    }
    
    # Search query parameters
    parameters = {'q': query}
    
    # Send the request and parse content
    content = requests.get(search_engine, headers=headers, params=parameters).text
    soup = BeautifulSoup(content, 'html.parser')
    
    # Attempt to find search results
    search_results = soup.find_all('div', class_='tF2Cxc')  # Common class for search results
    
    
    # Check if any results are found
    if search_results:
        # Get up to 5 results
        for index, result in enumerate(search_results[:5]):
            # Find the link in each result
            link = result.find('a')
            
            # Print the result number and the link
            if link and link['href']:
                print(f"Result {index + 1}: {link['href']}")
            else:
                print(f"Result {index + 1}: No valid link found.")
                
            response = requests.get(link['href'], headers=headers, params=parameters, proxies=proxies)
    
            content = response.text
            
            soup = BeautifulSoup(content, 'html.parser')
    
    # Extract relevant text from paragraphs, headings, or divs with content
            important_text = []
            
    # Extracting <p> (paragraph) tags
            for paragraph in soup.find_all('p'):
                important_text.append(paragraph.get_text())
    
    # Optionally extracting headings like h1-h6
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                important_text.append(heading.get_text())
            
            # Get the first 10 important points and convert them to string form
            first_10_points = important_text[:10]  # Get the first 10 points
            formatted_string = '\n'.join([f"{i + 1}. {point}" for i, point in enumerate(first_10_points)])
    
            # Print the formatted string of first 10 important points
            print(f"{formatted_string}\n")
    
            # Save the scraped content to a text file
            filename = f"important_text_{idx+1}.txt"
            save_text_to_file(formatted_string, filename)
        
    else:
        print("No search results found.")


import requests
import json

'''
generate audio
'''

# Function to get the list of available voices
def get_voices(api_key, text):
    url = "https://api.elevenlabs.io/v1/voices"
    # text = formatted_string
    headers = {
        'xi-api-key': 'sk_9abac15db603c850879e3a568abf4543fcd32140f62ed8c6',
        'accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    print(response)
    if response.status_code == 200:
        voices = response.json()
        return voices['voices']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to convert text to speech using Eleven Labs API
def text_to_speech(text, api_key, voice_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    #text = text[:1000]
    headers = {
        'xi-api-key': 'sk_9abac15db603c850879e3a568abf4543fcd32140f62ed8c6',
        'Content-Type': 'application/json'
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",  # Optional, default model
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Save the audio file locally
        with open("/Users/lavanyadalinannappa/Desktop/output2.mp3", "wb") as f:
            f.write(response.content)
        print("Audio has been saved successfully!")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Your Eleven Labs API key
api_key = "sk_9abac15db603c850879e3a568abf4543fcd32140f62ed8c6"

def generate_audio(text, api_key, voice_id=0):

    # Get available voices
    voices = get_voices(api_key, text)
    
    if voices:
        for voice in voices:
            print(f"Voice: {voice['name']}, ID: {voice['voice_id']}")
    
        # Use the first voice for the demo
        selected_voice_id = voices[voice_id]['voice_id']
        
        # Convert text to audio
        text_to_speech(text, api_key, selected_voice_id)

    else:
        print('could not generate audio')


# if voices:
#     for voice in voices:
#         print(f"Voice: {voice['name']}, ID: {voice['voice_id']}")

#     # Use the first voice for the demo
#     selected_voice_id = voices[4]['voice_id']

#     # Scraped text (example)
#     #scraped_text = "Your scraped content here that you want to convert into speech."

#     # Convert text to audio
#     text_to_speech(formatted_string, api_key, selected_voice_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
