from flask import Flask, request, jsonify
import openai
import json
from dotenv import load_dotenv
import os
import re
import requests
import fitz  # PyMuPDF
import io

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Detect file ID from Google Drive URL
def extract_drive_file_id(url):
    patterns = [r"/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Convert Google Drive URL to direct downloadable URL
def make_direct_download_url(url):
    file_id = extract_drive_file_id(url)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Check content type (PDF or Image)
def get_content_type(download_url):
    try:
        head = requests.head(download_url, allow_redirects=True)
        return head.headers.get("Content-Type", "")
    except:
        return ""
    

def get_file_type(url):
    try:
        response = requests.get(url, stream=True)
        content_type = response.headers.get("Content-Type", "").lower()

        # Check content-type first
        if "pdf" in content_type:
            return "pdf"
        if "image" in content_type:
            return "image"

        # Fallback: Read first bytes to check PDF signature
        first_bytes = response.raw.read(4)
        if first_bytes == b"%PDF":
            return "pdf"

        return "image"  # assume image if unsure

    except Exception as e:
        print(f"Error detecting file type: {e}")
        return "unknown"


# Send image URL to GPT-4o
def analyze_image_url(image_url):
    messages = [
        {
            "role": "system",
            "content": "You are a bilingual medical assistant who analyzes medical lab report images and summarizes them for patients and doctors."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
You are analyzing an image of a medical lab report.

Your task:
- Focus on clearly stating **abnormal test results** (values that are too high or too low).
- Mention **normal results only briefly**, grouped together (e.g., "Other tests are within normal range").
- Keep the tone professional, clear, and friendly for patients.

Return a **short bilingual summary (3–4 lines)** using this exact JSON format:
{
  "summary": {
    "english": "Your English summary here...",
    "arabic": "الملخص باللغة العربية هنا..."
  }
}

Rules:
1. Do NOT wrap the output in code blocks or markdown.
2. Keep it simple and understandable for patients.
"""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    return json.loads(response["choices"][0]["message"]["content"])



# Extract text from PDF and send to GPT-4o
def analyze_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        pdf_data = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])

        prompt = f"""
        You are a bilingual medical assistant analyzing the following medical lab report.

        Your task:
        - Focus on highlighting **abnormal test results** (values that are too high or too low).
        - Mention **normal results only generally**, grouped together (e.g., "Other values are within normal range").
        - Use a **calm, professional, patient-friendly tone**.
        - Return a short summary (3–4 lines maximum) in **both English and Arabic**.

        Output Format:
        {{
        "summary": {{
            "english": "Your English summary here...",
            "arabic": "الملخص باللغة العربية هنا..."
        }}
        }}

        Instructions:
        - Do NOT wrap the output in code blocks or markdown.
        - Keep it structured, easy to understand, and concise.

        Lab Report:
        {text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical lab report summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return json.loads(response["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": str(e)}




# ----------------- NEW Weight Report Analyzer -----------------
# ----------------- NEW Weight Report Analyzer -----------------
def analyze_weight_report(image_url):
    """
    Sends weight/body composition report image to GPT and extracts structured JSON.
    Works with any report format by dynamically extracting all available fields.
    """

    messages = [
        {
            "role": "system",
            "content": "You are an assistant that extracts structured health metrics from body composition/weight analysis images and returns clean JSON."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
You are analyzing an image of a weight/body composition report.

Rules:
1. Extract **all available fields** exactly as they appear in the report (e.g., weight, BMI, fat %, fat mass, bone density, water %, visceral fat, muscle %, muscle mass, metabolic rate, etc.).
2. Store them under `patient_info` as key-value pairs in **snake_case**. 
   Example: "Body Fat %" → "body_fat_percentage".
3. Values should be numbers when possible. Keep units out of values, just numeric.
4. If a field is missing, simply do not include it.
5. Add a `diagnoses` section with booleans if you can infer:
   - obesity (true if BMI >= 30 or body_fat_percentage >= 35)
   - high_visceral_fat (true if visceral_fat > 10)
   - low_water_percentage (true if water_percentage < 50)
   - high_metabolism (true if metabolism_kcal_per_day > 1500)
   If the required field is not available, omit that diagnosis.

Return ONLY valid JSON in this format:
{
  "data": {
    "patient_info": {
      "field1": value,
      "field2": value,
      ...
    },
    "diagnoses": {
      "obesity": true/false,
      "high_visceral_fat": true/false,
      "low_water_percentage": true/false,
      "high_metabolism": true/false
    }
  }
}
Do NOT add explanations or extra text.
"""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    # Ensure response is JSON
    try:
        return json.loads(response["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}

# ----------------- New API endpoint -----------------
# ----------------- New API endpoint -----------------
@app.route("/analyze-weight", methods=["POST"])
def analyze_weight():
    data = request.json
    input_url = data.get("url") or data.get("image_url")  # allow both keys

    if not input_url:
        return jsonify({"status": "error", "message": "Missing 'url' or 'image_url'"}), 400

    # Convert Drive link to direct download
    download_url = make_direct_download_url(input_url)
    if not download_url:
        return jsonify({"status": "error", "message": "Invalid Google Drive link"}), 400

    # Detect type
    file_type = get_file_type(download_url)
    if file_type != "image":
        return jsonify({"status": "error", "message": "Weight analyzer works only with images"}), 400

    # Analyze weight/body composition
    result = analyze_weight_report(download_url)
    return jsonify({"status": "success", "type": "image", "result": result})


# Main endpoint
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    input_url = data.get("url")

    if not input_url:
        return jsonify({"status": "error", "message": "Missing 'url'"}), 400

    download_url = make_direct_download_url(input_url)
    if not download_url:
        return jsonify({"status": "error", "message": "Invalid Google Drive link"}), 400

    file_type = get_file_type(download_url)

    if file_type == "pdf":
        result = analyze_pdf_text(download_url)
        return jsonify({"status": "success", "type": "pdf", "result": result})

    elif file_type == "image":
        result = analyze_image_url(download_url)
        return jsonify({"status": "success", "type": "image", "result": result})

    else:
        return jsonify({"status": "error", "message": "Unsupported or undetectable file type"}), 400
@app.route("/")
def home():
    return "Flask app is running — send a POST to /webhook with a Google Drive 'url'."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run("0.0.0.0", port=port)
