import os
import base64
import requests
from flask import Flask, render_template, request, jsonify

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
    cleaned = text.replace("```csv", "").replace("```text", "").replace("```", "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip() and ',"' in line]
    return "\n".join(lines) if lines else cleaned

def get_gemini_models(api_key):
    """Automatically fetch all active Gemini models for this API key"""
    models = ["gemini-3.8-flash"]
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            skip_words = ['2.5', '2.0', '1.5', '1.0', 'embedding', 'imagen', 'veo', 'tts', 'audio', 'aqa', 'learnlm', 'gemma', 'robotics']
            for m in res.json().get('models', []):
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    name = m['name'].replace('models/', '')
                    if not any(w in name.lower() for w in skip_words) and name not in models:
                        models.append(name)
    except Exception:
        pass
    return models

def call_gemini(image_bytes, mime_type):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in Render Environment")

    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    models = get_gemini_models(api_key)
    
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": b64_image}}
            ]
        }]
    }

    last_error = None
    # If one Gemini model is busy (503), automatically try the next active Gemini model
    for model_name in models[:4]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                return clean_output(text)
            else:
                last_error = f"{model_name} ({res.status_code})"
        except Exception as e:
            last_error = f"{model_name} ({str(e)})"

    raise Exception(f"Gemini busy/failed: {last_error}")

def get_groq_models(api_key):
    """Automatically fetch active models from Groq account"""
    headers = {"Authorization": f"Bearer {api_key}"}
    vision_priority = []
    other_models = []
    
    res = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
    if res.status_code == 200:
        skip = ['whisper', 'tts', 'guard', 'orpheus', 'playai', 'embed', 'allam', 'safeguard']
        vision_keywords = ['vision', 'vl', 'scout', 'maverick', 'pixtral', 'llava', 'omni', 'llama-4']
        for m in res.json().get('data', []):
            mid = m.get('id', '')
            if any(s in mid.lower() for s in skip):
                continue
            if any(vk in mid.lower() for vk in vision_keywords):
                vision_priority.append(mid)
            else:
                other_models.append(mid)
    return vision_priority + other_models[:3]

def call_groq(image_bytes, mime_type):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in Render Environment")

    models = get_groq_models(api_key)
    if not models:
        raise Exception("No active vision models found on Groq account")

    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

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
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12
            )
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                return clean_output(content)
            else:
                last_error = f"{model_name}: {res.status_code}"
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"

    raise Exception(f"Groq Error (No vision model available, tried {models}): {last_error}")

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
                # AUTO MODE: Try all active Gemini models first, then Groq models
                try:
                    result = call_gemini(image_bytes, mime_type)
                except Exception as gem_err:
                    try:
                        result = call_groq(image_bytes, mime_type)
                    except Exception as groq_err:
                        raise Exception(f"{gem_err} | {groq_err}")

            all_results.append(result)

        except Exception as e:
            all_results.append(f"Error processing {file.filename}: {str(e)}")

    return jsonify({'result': "\n".join(all_results)})

if __name__ == '__main__':
    app.run(debug=True)
