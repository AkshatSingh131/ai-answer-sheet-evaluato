import os
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from flask import Flask, request, render_template, jsonify, redirect, url_for
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import re
from datetime import datetime
from PIL import Image as PILImage
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Loading TrOCR model...")
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print("TrOCR model loaded.")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ── In-memory store (replaces MySQL for prototype) ──
results_store = []

# ── Preloaded answer key (teacher sets this) ──
ANSWER_KEY = {
    "Q1": {
        "question": "What is photosynthesis?",
        "model_answer": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose.",
        "max_marks": 5,
        "keywords": ["sunlight", "water", "carbon dioxide", "glucose", "oxygen", "plants"]
    },
    "Q2": {
        "question": "Explain Newton's First Law of Motion.",
        "model_answer": "Newton's First Law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced external force. This is also called the law of inertia.",
        "max_marks": 5,
        "keywords": ["rest", "motion", "force", "inertia", "external", "speed"]
    },
    "Q3": {
        "question": "What is the Pythagorean theorem?",
        "model_answer": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. It is expressed as a² + b² = c², where c is the hypotenuse.",
        "max_marks": 5,
        "keywords": ["right", "triangle", "hypotenuse", "square", "sum", "a²", "b²", "c²"]
    }
}


# ──────────────────────────────────────────────────
# PIPELINE STAGE 1: Image Preprocessing (OpenCV)
# ──────────────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to improve OCR — Tesseract works better on larger images
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE — improves contrast locally, handles uneven lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu thresholding — automatically finds best threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save debug
    cv2.imwrite('debug_mask.png', thresh)

    preprocessed_path = image_path.rsplit('.', 1)[0] + '_preprocessed.png'
    cv2.imwrite(preprocessed_path, thresh)
    return preprocessed_path, thresh

def extract_text(preprocessed_path):
    img = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert — black text on white background for Tesseract
    inverted = cv2.bitwise_not(img)
    
    # Scale up 2x — Tesseract works better on larger images
    inverted = cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Save inverted debug
    cv2.imwrite('debug_inverted.png', inverted)
    
    pil_img = PILImage.fromarray(inverted)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_blacklist=|'
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    return text.strip()
def semantic_score(student_text, model_answer, keywords, max_marks):
    if not student_text.strip():
        return 0, 0.0, [], "No text could be extracted from the answer."

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([student_text.lower(), model_answer.lower()])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except Exception:
        sim = 0.0

    found_keywords = [kw for kw in keywords if kw.lower() in student_text.lower()]
    keyword_ratio = len(found_keywords) / len(keywords) if keywords else 0

    blended = 0.6 * sim + 0.4 * keyword_ratio
    marks = round(blended * max_marks, 1)
    marks = min(marks, max_marks)

    confidence = "HIGH" if blended > 0.5 else "LOW"

    return marks, blended, found_keywords, confidence

# ──────────────────────────────────────────────────
# PIPELINE STAGE 5: Feedback Generation
# ──────────────────────────────────────────────────
def generate_feedback(question_id, student_text, marks, max_marks,
                      found_keywords, all_keywords, confidence, model_answer):
    missing_keywords = [kw for kw in all_keywords if kw not in found_keywords]
    ratio = marks / max_marks if max_marks > 0 else 0

    if ratio >= 0.85:
        feedback = f"Excellent answer! You covered the key concepts well."
    elif ratio >= 0.6:
        feedback = f"Good attempt. You got {marks}/{max_marks} marks."
        if missing_keywords:
            feedback += f" Consider including: {', '.join(missing_keywords[:3])}."
    elif ratio >= 0.35:
        feedback = f"Partial credit awarded ({marks}/{max_marks}). "
        if missing_keywords:
            feedback += f"Key concepts missing: {', '.join(missing_keywords)}."
        feedback += " Review the topic more carefully."
    else:
        feedback = f"Answer needs significant improvement ({marks}/{max_marks}). "
        feedback += f"Important concepts to include: {', '.join(all_keywords[:4])}."

    if confidence == "LOW":
        feedback += " ⚠️ Flagged for teacher review (low confidence match)."

    return feedback


# ──────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', answer_key=ANSWER_KEY)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'answer_sheet' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['answer_sheet']
    student_name = request.form.get('student_name', 'Unknown')
    question_id = request.form.get('question_id', 'Q1')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Stage 1: Preprocess
        preprocessed_path, _ = preprocess_image(filepath)

        # Stage 2: OCR
        extracted_text = extract_text(preprocessed_path)

        if not extracted_text:
            extracted_text = "[No text could be extracted — check image quality]"

        # Stage 3 & 4: Score
        qa = ANSWER_KEY.get(question_id, ANSWER_KEY["Q1"])
        marks, similarity, found_kw, confidence = semantic_score(
            extracted_text,
            qa["model_answer"],
            qa["keywords"],
            qa["max_marks"]
        )

        # Stage 5: Feedback
        feedback = generate_feedback(
            question_id, extracted_text, marks,
            qa["max_marks"], found_kw, qa["keywords"],
            confidence, qa["model_answer"]
        )

        result = {
            "student_name": student_name,
            "question_id": question_id,
            "question": qa["question"],
            "extracted_text": extracted_text,
            "marks": marks,
            "max_marks": qa["max_marks"],
            "similarity": round(float(similarity) * 100, 1),
            "found_keywords": found_kw,
            "missing_keywords": [k for k in qa["keywords"] if k not in found_kw],
            "confidence": confidence,
            "feedback": feedback,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "needs_review": confidence == "LOW"
        }

        results_store.append(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    return render_template('results.html', results=results_store)


@app.route('/api/results')
def api_results():
    return jsonify(results_store)


@app.route('/teacher')
def teacher_dashboard():
    flagged = [r for r in results_store if r.get('needs_review')]
    return render_template('teacher.html', flagged=flagged, all_results=results_store)


@app.route('/teacher/approve/<int:idx>', methods=['POST'])
def approve(idx):
    if 0 <= idx < len(results_store):
        new_marks = request.form.get('marks')
        if new_marks:
            results_store[idx]['marks'] = float(new_marks)
        results_store[idx]['needs_review'] = False
        results_store[idx]['teacher_reviewed'] = True
    return redirect(url_for('teacher_dashboard'))

@app.route('/score_text', methods=['POST'])
def score_text():
    data = request.get_json()
    student_text = data.get('text', '')
    question_id = data.get('question_id', 'Q1')
    student_name = data.get('student_name', 'Unknown')

    qa = ANSWER_KEY.get(question_id, ANSWER_KEY["Q1"])
    marks, similarity, found_kw, confidence = semantic_score(
        student_text, qa["model_answer"], qa["keywords"], qa["max_marks"]
    )
    feedback = generate_feedback(
        question_id, student_text, marks, qa["max_marks"],
        found_kw, qa["keywords"], confidence, qa["model_answer"]
    )

    result = {
        "student_name": student_name,
        "question_id": question_id,
        "question": qa["question"],
        "extracted_text": student_text,
        "marks": marks,
        "max_marks": qa["max_marks"],
        "similarity": round(float(similarity) * 100, 1),
        "found_keywords": found_kw,
        "missing_keywords": [k for k in qa["keywords"] if k not in found_kw],
        "confidence": confidence,
        "feedback": feedback,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "needs_review": confidence == "LOW"
    }

    results_store.append(result)
    return jsonify(result)
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)

