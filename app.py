import os
import base64
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# Initialize Groq API
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
            # Read image and convert to Base64
            image_bytes = file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Send directly to Groq's Vision Model
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                model="llama-3.2-11b-vision-preview", 
            )
            
            text = chat_completion.choices[0].message.content
            if text:
                all_results.append(text.strip())
                
        except Exception as e:
            all_results.append(f"Error processing {file.filename}: {str(e)}")

    final_output = "\n".join(all_results)
    return jsonify({'result': final_output})

if __name__ == '__main__':
    app.run(debug=True)
