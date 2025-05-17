import os
import json
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Union
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import uuid

app = Flask(__name__)
#pytesseract.pytesseract.tesseract_cmd = r"D:\tes\tesseract.exe"


# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class QuestionPaperProcessor:
    
    def __init__(self, api_key: str = None):
   
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def preprocess_image(self, image_path: str) -> Image.Image:
   
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        pil_img = Image.fromarray(denoised)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(2.0)
        
        sharpened_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        return sharpened_img
    
    def extract_text(self, image: Image.Image, lang: str = 'eng') -> str:
  
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        
        return text
    
    def parse_questions(self, text: str) -> Dict[str, Any]:
    
        lines = text.strip().split('\n')
        questions = []
        current_q_num = None
        current_q_text = []
        current_q_marks = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('Q.', 'Q ', 'Question')) or \
               (line[0].isdigit() and '.' in line[:3]):
                
                if current_q_num is not None:
                    questions.append({
                        "number": current_q_num,
                        "text": ' '.join(current_q_text).strip(),
                        "marks": current_q_marks
                    })
                
                parts = line.split('.', 1)
                if len(parts) > 1:
                    current_q_num = parts[0].replace('Q', '').replace('Question', '').strip()
                    current_q_text = [parts[1].strip()]
                else:
                    current_q_num = "unknown"
                    current_q_text = [line.strip()]
                
                marks_match = None
                if current_q_text and '[' in current_q_text[0] and ']' in current_q_text[0]:
                    marks_text = current_q_text[0].split('[')[-1].split(']')[0]
                    if 'marks' in marks_text.lower() or 'mark' in marks_text.lower():
                        try:
                            current_q_marks = int(''.join(filter(str.isdigit, marks_text)))
                        except:
                            current_q_marks = None
            
            elif current_q_num is not None:
                current_q_text.append(line)
                
                if current_q_marks is None and '[' in line and ']' in line:
                    marks_text = line.split('[')[-1].split(']')[0]
                    if 'marks' in marks_text.lower() or 'mark' in marks_text.lower():
                        try:
                            current_q_marks = int(''.join(filter(str.isdigit, marks_text)))
                        except:
                            current_q_marks = None
        
        if current_q_num is not None and current_q_text:
            questions.append({
                "number": current_q_num,
                "text": ' '.join(current_q_text).strip(),
                "marks": current_q_marks
            })
        
        return questions
    
    def structure_with_ai(self, text: str) -> Dict[str, Any]:
       
        if not self.api_key:
            raise ValueError("API key required for AI structuring")
        
        prompt = f"""
        This text was extracted from a question paper using OCR. The text may have errors 
        or formatting issues. Please organize this into a clean JSON structure with the format:
        {{
          "questions": [
            {{
              "number": "1",
              "text": "The question text",
              "marks": 5  // if available, otherwise null
            }},
            ...
          ]
        }}
        
        Raw OCR text:
        {text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from AI response", "questions": []}
            
        except Exception as e:
            return {"error": f"AI processing failed: {str(e)}", "raw_text": text, "questions": []}
    
    def process_single_image(self, image_path: str, use_ai: bool = False) -> Dict[str, Any]:
      
        try:
            # Preprocess and extract text
            preprocessed_img = self.preprocess_image(image_path)
            text = self.extract_text(preprocessed_img)
            
            # Parse the text
            if use_ai and self.api_key:
                result = self.structure_with_ai(text)
            else:
                questions = self.parse_questions(text)
                result = {"questions": questions}
                
            # Add the original image path for reference
            result["image_path"] = image_path.split('/')[-1]
            return result
        except Exception as e:
            return {"error": str(e), "image_path": image_path.split('/')[-1], "questions": []}

# Initialize the processor with your API key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyApZJTfRrbA69b60wmLjejSUrv3eiXzspI")
processor = QuestionPaperProcessor(api_key=GOOGLE_API_KEY)

@app.route('/process_images', methods=['POST'])
def process_images():
    """API endpoint to process multiple question paper images."""
    
    # Check if files are in the request
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')  # Get multiple files
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Get parameters
    use_ai = request.form.get('use_ai', 'false').lower() == 'true'
    
    results = []
    temp_files = []
    
    try:
        # Process each file
        for file in files:
            if file and allowed_file(file.filename):
                # Create unique filename to avoid collisions
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                temp_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                # Save the file temporarily
                file.save(temp_path)
                temp_files.append(temp_path)
                
                # Process the image
                result = processor.process_single_image(temp_path, use_ai=use_ai)
                results.append(result)
            else:
                results.append({
                    "error": f"Invalid file or file type: {file.filename if file.filename else 'unknown'}",
                    "image_path": file.filename if file.filename else 'unknown',
                    "questions": []
                })
        
        # Prepare the full response
        response = {
            "processed_files": len(results),
            "results": results
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "service": "question-paper-processor"}), 200

if __name__ == "__main__":
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
