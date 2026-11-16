import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Initialize Gemini API
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """
Extract data from this handwritten table image following these strict rules:
1. Find the Date at the top right and format it as DD-MM-YYYY.
2. For each row in the table:
   - Take the G.R. number (ONLY the digits BEFORE the slash '/').
   - Take the CTNS number from the first column (ignore the circled total at the bottom, and ignore any '-' or dots before the number).
3. Format each row exactly like this:
G.R.,"Date - CTNS CRTNS"

Example output:
17195,"04-11-2026 - 15 CRTNS"
54739,"04-11-2026 - 10 CRTNS"

Return ONLY the formatted lines. Do not include any extra text, markdown, or explanations.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    all_results = []

    for file in files:
        if file.filename == '':
            continue
        try:
            # Read raw image bytes directly in memory
            image_bytes = file.read()
            mime_type = file.mimetype or 'image/jpeg'
            
            # Send directly to Gemini
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    PROMPT
                ]
            )
            
            if response.text:
                all_results.append(response.text.strip())
                
        except Exception as e:
            all_results.append(f"Error processing {file.filename}: {str(e)}")

    final_output = "\n".join(all_results)
    return jsonify({'result': final_output})

if __name__ == '__main__':
    app.run(debug=True)
