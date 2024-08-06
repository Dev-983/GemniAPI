from flask import Flask, request, redirect, url_for, render_template,jsonify
from pypdf import PdfReader
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import textwrap
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

load_dotenv()
print(os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ALLOWED_EXTENSIONS = {'pdf'}
model = 'models/embedding-001'

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

prompt_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="Your name is Angel. Your role is to find the best and most relevant answer with step by step to the user's question.",
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_file():
    return render_template('index.html')

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

def find_best_passage(query, dataframe):
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Data']

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    data_str = request.json.get('data')
    data = json.loads(data_str)
    print("Type",type(data))
    model = 'models/embedding-001'
    embeddings_list = genai.embed_content(model=model,content=user_message,task_type="retrieval_query")
    df = pd.DataFrame(data)
    passage = find_best_passage(user_message, df)

    # prompt= f"{passage} \n Question - {user_message}."
    prompt = make_prompt(user_message, passage)
    response = prompt_model.generate_content(prompt)

    return jsonify({'response': response.text})

@app.route('/chat_2', methods=['POST'])
def chat_2():
    user_message = request.json.get('message')
    # prompt= f"{passage} \n Question - {user_message}."
    prompt = "Please find the best answer to my question.\nQUESTION -"+user_message
    response = prompt_model.generate_content(prompt)

    return jsonify({'response': response.text})



def embed_fn(title):
  return genai.embed_content(model=model,
                             content=title,
                             task_type="retrieval_document"
                             )["embedding"]

@app.route('/upload', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Read PDF content directly from memory
            pdf_chunks = read_pdf(file)
            #print(pdf_chunks)

            df = pd.DataFrame(pdf_chunks)
            df.columns = ['Data']
            df['Embeddings'] = df.apply(lambda row: embed_fn(row['Data']), axis=1)
            embeddings_list = df[['Data', 'Embeddings']].to_dict(orient='records')
            
            #chunk_display = ''.join([f'<p>Chunk {i+1}: {chunk}</p>' for i, chunk in enumerate(pdf_chunks)])
            #return f'<h1>Uploaded File: {file.filename}</h1>{chunk_display}'
            #print(embeddings_list)
            return jsonify(embeddings_list)
    return redirect(url_for('upload_file'))

def read_pdf(file, chunk_size=500):
    reader = PdfReader(file)
    num_pages = len(reader.pages)
    pdf_content = ''
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        page_text = page.extract_text()
        pdf_content += page_text

    # Split content into chunks of `chunk_size` characters
    chunks = [pdf_content[i:i+chunk_size] for i in range(0, len(pdf_content), chunk_size)]
    
    return chunks

##### URL CHECK 
@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get('url')
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print('valid')
            return jsonify({'valid': True, 'message': 'ok'})
        else:
            return jsonify({'valid': False, 'message': 'URL is not accessible (status code: {})'.format(response.status_code)})
    except requests.exceptions.RequestException as e:
        return jsonify({'valid': False, 'message': 'URL is not valid or accessible: {}'.format(e)})

## SCRAP CONTENT
@app.route('/scrapehtml', methods=['POST'])
def scrapehtml():
    data = request.json
    url = data.get('url')
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)})
    
    chunk_size=500
    urlchunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    df = pd.DataFrame(urlchunks)
    df.columns = ['Data']
    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Data']), axis=1)
    embeddings_list = df[['Data', 'Embeddings']].to_dict(orient='records') 
    return jsonify(embeddings_list)




if __name__ == '__main__':
    app.run(debug=True)
