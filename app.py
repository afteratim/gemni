import os
import time
import base64
import requests
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

PROMPT = """
Extract data from this handwritten table image following these strict rules:
1. Find the Date at the top right and format it as DD-MM-YYYY.
2. Ignore subheading text rows (like city or company names written across the middle of the table).
3. For each numbered data row in the table:
   - Take the G.R. number (ONLY the digits BEFORE the slash '/').
   - Take the CTNS number from the first column (ignore the number after '/' in G.R., ignore the circled total at the bottom, and ignore any '-' or dots).
4. Format each row exactly like this:
G.R.,"Date - CTNS CRTNS"

Example output:
17195,"04-11-2026 - 15 CRTNS"
54739,"04-11-2026 - 10 CRTNS"

Return ONLY the formatted lines. Do not include any extra text, markdown, or explanations.
"""

def clean_output(text):
    # Clean any markdown blocks or extra spaces
    cleaned = text.replace("```csv", "").replace("```text", "").replace("```", "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip() and ',"' in line]
    return "\n".join(lines) if lines else cleaned

def call_gemini(image_bytes, mime_type):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in Render Environment")

    client = genai.Client(api_key=api_key)
    last_error = None

    # Try up to 2 times before switching to Groq
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    PROMPT
                ]
            )
            if response.text:
                return clean_output(response.text)
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                time.sleep(2)
            else:
                break
    raise Exception(f"Gemini Error: {str(last_error)}")

def call_groq(image_bytes, mime_type):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in Render Environment")

    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Newest active Groq Llama 4 Vision models
    models = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]

    last_error = None
    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "temperature": 0.1
        }
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        if res.status_code == 200:
            content = res.json()['choices'][0]['message']['content']
            return clean_output(content)
        else:
            last_error = f"{res.status_code} - {res.text}"

    raise Exception(f"Groq Error: {last_error}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    provider = request.form.get('provider', 'auto')
    files = request.files.getlist('images')
    all_results = []

    for file in files:
        if file.filename == '':
            continue
        
        image_bytes = file.read()
        mime_type = file.mimetype or 'image/jpeg'

        try:
            if provider == 'gemini':
                result = call_gemini(image_bytes, mime_type)
            elif provider == 'groq':
                result = call_groq(image_bytes, mime_type)
            else:
                # AUTO MODE: Try Gemini first, fallback to Groq if Gemini is busy/down
                try:
                    result = call_gemini(image_bytes, mime_type)
                except Exception as gemini_err:
                    result = call_groq(image_bytes, mime_type)

            all_results.append(result)

        except Exception as e:
            all_results.append(f"Error processing {file.filename}: {str(e)}")

    return jsonify({'result': "\n".join(all_results)})

if __name__ == '__main__':
    app.run(debug=True)
