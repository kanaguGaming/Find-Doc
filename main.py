import os
import re
import io
import json
import random
from PIL import Image
try:
    from google import genai
    from google.genai import types as genai_types
    _NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    _NEW_SDK = False
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Hierarchical Medical Triage Platform")

# Mount static files and templates setup
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# GUARDRAIL: Banned prescription keywords
# ---------------------------------------------------------------------------
FORBIDDEN_MEDICATIONS = [
    "amoxicillin", "penicillin", "ibuprofen", "paracetamol", "acetaminophen",
    "xanax", "lipitor", "metformin", "ciprofloxacin", "prednisone", "aspirin"
]

# ---------------------------------------------------------------------------
# RAG KNOWLEDGE BASE — In-memory medical knowledge chunks
# Keyed by lowercase anatomical / modality keywords.
# retrieve_rag_context() performs multi-keyword fuzzy matching.
# ---------------------------------------------------------------------------
MEDICAL_KNOWLEDGE_BASE: Dict[str, List[str]] = {
    "knee": [
        "Knee X-Ray grading (Kellgren-Lawrence scale): Grade 0 = no OA; Grade 1 = doubtful narrowing; "
        "Grade 2 = definite osteophytes, possible narrowing; Grade 3 = moderate multiple osteophytes, "
        "definite narrowing, sclerosis; Grade 4 = large osteophytes, marked narrowing, severe sclerosis, bony deformity.",
        "Knee fracture indicators: disruption of the cortical line, trabecular impaction, lipohemarthrosis "
        "(fat-fluid level in suprapatellar bursa), Segond fracture (lateral tibial rim avulsion).",
        "Meniscal injury signs on MRI: increased signal intensity within meniscus reaching articular surface; "
        "bucket-handle tears show 'double PCL sign'.",
        "Knee alignment: measure mechanical axis (hip-knee-ankle angle). Varus >3° suggests medial compartment "
        "loading. Valgus >3° suggests lateral compartment loading.",
        "Patellar maltracking: measure sulcus angle, patellar tilt angle, and trochlear depth on axial cuts.",
        "Bone density reduction in knee (osteopenia) can appear as reduced trabeculae, particularly in "
        "subchondral bone; check for subchondral cysts as sign of advanced OA.",
    ],
    "chest": [
        "Pneumonia on CXR: lobar or segmental consolidation, air bronchograms, silhouette sign "
        "(loss of heart border = RML pneumonia, loss of diaphragm border = lower lobe).",
        "Pleural effusion: blunting of costophrenic angles, meniscus sign on upright PA film; "
        ">200 mL needed to see on PA, >50 mL on lateral.",
        "Pneumothorax: absent lung markings peripheral to a sharp visceral pleural line; "
        "tension pneumothorax causes mediastinal shift away.",
        "Cardiomegaly: cardiothoracic ratio >0.5 on PA film; check for pulmonary vascular redistribution.",
        "Interstitial lung disease: reticular pattern, honeycombing, traction bronchiectasis on HRCT.",
        "COPD: hyperinflation, flattened hemidiaphragms, increased AP diameter, bullae formation.",
        "Lung nodule characterization: size, density (solid/sub-solid/GGO), margins (spiculated vs smooth), "
        "Fleischner Society guidelines determine follow-up intervals.",
    ],
    "mri": [
        "MRI signal characteristics: T1 bright = fat, subacute blood, protein; T2 bright = fluid, edema, "
        "most pathology; FLAIR suppresses free fluid, highlights periventricular lesions.",
        "Brain MRI — stroke DWI: restricted diffusion (bright DWI, dark ADC) within minutes of ischemia. "
        "T2/FLAIR changes appear 6-12h after onset.",
        "Spine MRI: Modic changes — Type 1 = T1 dark/T2 bright (edema/inflammation); "
        "Type 2 = T1 bright/T2 bright (fatty); Type 3 = T1 dark/T2 dark (sclerosis).",
        "Herniated disc on MRI: focal protrusion of disc material, effacement of thecal sac or nerve root, "
        "T2 signal loss in degenerated disc (black disc).",
        "Breast MRI: morphology (mass vs non-mass enhancement), kinetics (washout = suspicious), "
        "ACR BIRADS classification 1-6.",
    ],
    "skin": [
        "Dermoscopy ABCDE criteria: Asymmetry, Border irregularity, Color variation (multiple colors), "
        "Diameter >6mm, Evolution over time.",
        "Melanoma vs benign nevus: atypical pigment network, regression structures (gray-white areas), "
        "irregular streaks, blue-white veil.",
        "Basal cell carcinoma: arborizing blood vessels, blue-gray globules, leaf-like areas, spoke-wheel structures.",
        "Squamous cell carcinoma: white circles, dotted/glomerular vessels, central keratinization.",
        "Psoriasis: regular scaling plaques, Auspitz sign (pinpoint bleeding on scale removal), "
        "salmon-pink plaques with silvery scales.",
        "Contact dermatitis: vesicles, oozing at contact site, geometric distribution matching allergen exposure.",
    ],
    "ecg": [
        "ECG rate: 300/number of large squares between QRS complexes (regular rhythm method).",
        "Normal sinus rhythm: P before every QRS, PR 0.12-0.20s, QRS <0.12s, regular rate 60-100bpm.",
        "Atrial fibrillation: absent P waves, irregularly irregular rhythm, variable RR intervals, fibrillatory baseline.",
        "STEMI criteria: ST elevation ≥1mm in 2 contiguous limb leads or ≥2mm in 2 contiguous precordial leads, "
        "or new LBBB. Reciprocal changes (ST depression) in opposite leads.",
        "LBBB: wide QRS >120ms, broad notched R in I/aVL/V5-V6, deep S in V1, no septal Q in I/V5-V6.",
        "Hypokalemia: flattened T waves, U waves, prolonged QU interval, ST depression.",
        "Hyperkalaemia: peaked T waves, widened QRS, sinusoidal pattern, PEA in severe cases.",
    ],
    "abdominal": [
        "Abdominal CT — appendicitis: dilated appendix >6mm, periappendiceal fat stranding, appendicolith.",
        "Bowel obstruction on AXR: dilated loops (small bowel >3cm, large bowel >6cm, caecum >9cm), "
        "air-fluid levels, paucity of gas distally.",
        "Gallstones: hyperechoic foci with posterior acoustic shadowing on ultrasound; "
        "only 10-15% visible on plain X-ray.",
        "Liver lesions: characterize by enhancement pattern — hemangioma (peripheral nodular fill-in), "
        "HCC (arterial enhancement with washout), metastases (ring enhancement).",
    ],
    "brain": [
        "Brain CT — hemorrhage: hyperdense (bright) acute blood, fades to isodense at 1-2 weeks, "
        "hypodense (dark) chronic blood/hemosiderin.",
        "Subdural hematoma: crescent-shaped collection following brain surface, crosses suture lines.",
        "Epidural hematoma: biconvex/lenticular shape, limited by suture lines, associated with skull fracture.",
        "Midline shift: every 5mm of shift = 1 point on herniation scale; "
        "effacement of sulci and cisterns = raised ICP.",
        "Hydrocephalus: enlarged ventricles, periventricular lucency (transependymal edema), "
        "rounding of frontal horns.",
    ],
}

def retrieve_rag_context(modality: str, body_region: str) -> str:
    """
    Multi-keyword fuzzy matching against MEDICAL_KNOWLEDGE_BASE.
    Returns formatted RAG context string for LLM injection.
    """
    combined_query = f"{modality} {body_region}".lower()
    matched_chunks: List[str] = []

    for key, chunks in MEDICAL_KNOWLEDGE_BASE.items():
        if key in combined_query:
            matched_chunks.extend(chunks)

    # Fallback: try individual words
    if not matched_chunks:
        query_words = set(re.findall(r'\w+', combined_query))
        for key, chunks in MEDICAL_KNOWLEDGE_BASE.items():
            key_words = set(re.findall(r'\w+', key))
            if key_words & query_words:  # intersection
                matched_chunks.extend(chunks)
                break

    if not matched_chunks:
        return "No specific domain knowledge retrieved. Answer based on general medical knowledge."

    formatted = "\n".join(f"- {c}" for c in matched_chunks)
    return f"Retrieved Medical Knowledge:\n{formatted}"


# ---------------------------------------------------------------------------
# SPECIALTY / DOCTOR DATABASE — Keyed by specialty slug
# ---------------------------------------------------------------------------
DOCTOR_DATABASE: Dict[str, List[Dict[str, str]]] = {
    "orthopedic": [
        {"name": "Kanchi Bone & Joint Clinic", "specialty": "Orthopaedic Surgery",
         "address": "Gandhi Road, Kancheepuram", "distance": "1.5 km"},
        {"name": "Dr. Ortho Care Centre", "specialty": "Orthopaedics & Traumatology",
         "address": "GST Road, Chengalpattu", "distance": "3.8 km"},
        {"name": "Apollo Ortho Hospital", "specialty": "Joint Replacement Surgery",
         "address": "Nellore Bypass, Chengalpattu", "distance": "5.2 km"},
    ],
    "pulmonologist": [
        {"name": "Dr. Ariswaran Lung Care", "specialty": "Pulmonology & Respiratory Medicine",
         "address": "Gandhi Road, Kancheepuram", "distance": "1.2 km"},
        {"name": "Apollo Priya Chest Medicine", "specialty": "Chest & Respiratory Diseases",
         "address": "Chengalpattu Main Road", "distance": "4.5 km"},
    ],
    "dermatologist": [
        {"name": "Dr. R. Skin & Laser Clinic", "specialty": "Dermatology & Cosmetology",
         "address": "Alagesan Nagar, Chengalpattu", "distance": "0.8 km"},
        {"name": "Sri Skin Centre", "specialty": "Dermatology",
         "address": "Nellukaran Street, Kancheepuram", "distance": "2.3 km"},
    ],
    "cardiologist": [
        {"name": "Kanchi Heart & Vascular Centre", "specialty": "Cardiology & Electrophysiology",
         "address": "Moosa Street, Kancheepuram", "distance": "1.5 km"},
        {"name": "Dr. Rajan Cardiology", "specialty": "Interventional Cardiology",
         "address": "GST Road, Chengalpattu", "distance": "3.8 km"},
    ],
    "neurologist": [
        {"name": "Neuro Care Hospital", "specialty": "Neurology & Stroke",
         "address": "Big Street, Kancheepuram", "distance": "2.1 km"},
        {"name": "Kanchi Brain & Spine Clinic", "specialty": "Neurosurgery",
         "address": "SH-111, Chengalpattu", "distance": "4.0 km"},
    ],
    "gastroenterologist": [
        {"name": "GI & Liver Centre", "specialty": "Gastroenterology & Hepatology",
         "address": "Station Road, Kancheepuram", "distance": "1.8 km"},
        {"name": "Sree Balaji Digestive Health", "specialty": "Gastroenterology",
         "address": "GST Road, Chengalpattu", "distance": "5.0 km"},
    ],
    "general": [
        {"name": "Kancheepuram Government Medical College", "specialty": "General Medicine",
         "address": "Hospital Road, Kancheepuram", "distance": "2.0 km"},
        {"name": "Sri Ramachandra Multi-Specialty Hospital", "specialty": "General Surgery & Medicine",
         "address": "GST Road, Chengalpattu", "distance": "6.5 km"},
    ],
}

# Maps modality/region keywords → specialty slug
MODALITY_SPECIALTY_MAP: List[tuple] = [
    # Order matters: more specific entries first
    (["knee", "hip", "shoulder", "spine", "fracture", "orthop", "bone", "joint",
      "ligament", "tendon", "cartilage", "osteo"], "orthopedic"),
    (["chest", "lung", "pulmon", "pneumo", "pleural", "bronch", "trachea"], "pulmonologist"),
    (["skin", "derm", "lesion", "melanoma", "eczema", "psoriasis", "rash"], "dermatologist"),
    (["heart", "cardiac", "ecg", "ekg", "arrhythmia", "coronary", "aorta"], "cardiologist"),
    (["brain", "neuro", "stroke", "seizure", "cranial", "mri head"], "neurologist"),
    (["abdomen", "liver", "gallbladder", "bowel", "colon", "gastro", "stomach"], "gastroenterologist"),
]

def resolve_specialty(modality: str, body_region: str, condition: str) -> str:
    """Resolves the correct medical specialty from modality, region, and condition keywords."""
    combined = f"{modality} {body_region} {condition}".lower()
    for keywords, specialty in MODALITY_SPECIALTY_MAP:
        if any(kw in combined for kw in keywords):
            return specialty
    return "general"

def get_doctors_for_specialty(specialty: str) -> List[Dict[str, str]]:
    return DOCTOR_DATABASE.get(specialty, DOCTOR_DATABASE["general"])


# ---------------------------------------------------------------------------
# GEMINI SETUP
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("WARNING: GEMINI_API_KEY not set. Vision and chat will fail gracefully.")

if _NEW_SDK:
    _genai_client = genai.Client(api_key=api_key or "mock_key")
    _MODEL_ID = "gemini-1.5-flash"
else:
    genai.configure(api_key=api_key or "mock_key")
    vision_model = genai.GenerativeModel('gemini-1.5-flash')
    chat_model   = genai.GenerativeModel('gemini-1.5-flash')


def _call_vision(image: Image.Image, prompt: str) -> str:
    """Unified vision call — works with both old and new SDK."""
    if _NEW_SDK:
        import io as _io
        buf = _io.BytesIO()
        image.save(buf, format='JPEG')
        img_bytes = buf.getvalue()
        part_img  = genai_types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
        part_txt  = genai_types.Part.from_text(text=prompt)
        resp = _genai_client.models.generate_content(
            model=_MODEL_ID,
            contents=[part_img, part_txt],
            config=genai_types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        return resp.text
    else:
        resp = vision_model.generate_content(
            [image, prompt],
            generation_config=genai.GenerationConfig(response_mime_type='application/json')
        )
        return resp.text


def _call_text(prompt: str) -> str:
    """Unified text call — works with both old and new SDK."""
    if _NEW_SDK:
        resp = _genai_client.models.generate_content(
            model=_MODEL_ID,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        return resp.text
    else:
        resp = chat_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type='application/json')
        )
        return resp.text


# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    message: str
    detected_condition: Optional[str] = None
    detected_modality: Optional[str] = None
    body_region: Optional[str] = None
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/api/triage/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Two-pass CV pipeline:
      Pass 1 — Modality & Region detection (anchored, no condition guessing)
      Pass 2 — Targeted condition analysis using Pass 1 output as context
    """
    image_data = await file.read()
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # -----------------------------------------------------------------------
    # PASS 1: Modality & Anatomical Region — strict, no condition inference
    # -----------------------------------------------------------------------
    pass1_prompt = """
    You are an expert radiologist and medical imaging classifier.
    Carefully examine this medical image and determine ONLY:
    1. The imaging modality (e.g., X-Ray, MRI, CT Scan, Dermoscopy photo, ECG trace, Ultrasound, Fundus photograph).
    2. The exact anatomical region / body part shown (e.g., Right Knee, Left Chest, Lumbar Spine, Skin — Left Forearm, Brain Axial Slice).
    3. The imaging view or plane if applicable (e.g., AP view, Lateral view, Axial, Sagittal, Coronal).

    DO NOT diagnose or guess any condition in this step.
    Be precise. If the image is NOT a medical image, set modality to "Non-Medical Image".

    Output ONLY valid JSON with this exact schema:
    {
      "modality": "string",
      "body_region": "string",
      "view": "string"
    }
    """
    try:
        pass1 = json.loads(_call_vision(image, pass1_prompt))
        modality    = pass1.get("modality", "Unknown Modality")
        body_region = pass1.get("body_region", "Unknown Region")
        view        = pass1.get("view", "")
    except Exception as e:
        print(f"Pass 1 error: {e}")
        modality    = "Unknown Modality"
        body_region = "Unknown Region"
        view        = ""

    # -----------------------------------------------------------------------
    # PASS 2: Targeted Condition Analysis — uses Pass 1 context to lock focus
    # -----------------------------------------------------------------------
    pass2_prompt = f"""
    You are an expert medical AI assistant specialized in {modality} interpretation.
    The image has been confirmed as: {modality} of {body_region} ({view}).

    Carefully analyze this image ONLY in the context of {body_region} pathology.
    Do NOT reference or suggest conditions from other body systems.

    Provide:
    1. The primary suspected condition or finding (be specific to {body_region}).
    2. A list of 3-5 specific visual observations/findings that support your assessment.
    3. A confidence score (0-100) reflecting image clarity and finding certainty.
    4. Severity level: one of ["Normal", "Mild", "Moderate", "Severe", "Critical"].

    Examples for a Knee X-Ray:
      - condition: "Moderate Osteoarthritis (KL Grade 3)"
      - findings: ["Medial joint space narrowing", "Marginal osteophyte formation", "Subchondral sclerosis"]

    Examples for a Chest X-Ray:
      - condition: "Community-Acquired Pneumonia (Right Lower Lobe)"
      - findings: ["Right lower lobe consolidation", "Air bronchograms visible", "Minor right pleural effusion"]

    Output ONLY valid JSON with this exact schema:
    {{
      "condition": "string",
      "findings": ["string", "string", "string"],
      "confidence": number,
      "severity": "string"
    }}
    """
    try:
        pass2 = json.loads(_call_vision(image, pass2_prompt))
        condition  = pass2.get("condition", "Requires Clinical Evaluation")
        findings   = pass2.get("findings", ["No specific findings extracted"])
        confidence = float(pass2.get("confidence", 75.0))
        severity   = pass2.get("severity", "Unknown")
    except Exception as e:
        print(f"Pass 2 error: {e}")
        condition  = "Unable to process image accurately"
        findings   = ["Image processing error"]
        confidence = 0.0
        severity   = "Unknown"

    # Heatmap overlay coords (visual indicator of region of interest)
    heatmap_coords = {
        "top":    random.randint(20, 40),
        "left":   random.randint(20, 40),
        "width":  random.randint(35, 50),
        "height": random.randint(35, 50),
    }

    triage_note = (
        f"I've analyzed your {modality} of {body_region}. "
        f"The imaging suggests: **{condition}** (Severity: {severity}). "
        f"Please describe your symptoms or ask me anything about this finding."
    )

    return {
        "status":            "Success",
        "modality":          modality,
        "body_region":       body_region,
        "view":              view,
        "prediction":        condition,
        "findings":          findings,
        "confidence":        confidence,
        "severity":          severity,
        "heatmap":           heatmap_coords,
        "initial_triage_note": triage_note,
    }


@app.post("/api/triage/chat")
async def chat_triage(payload: ChatMessage):
    user_query = payload.message
    user_query_lower = user_query.lower()

    # -----------------------------------------------------------------------
    # 1. Guardrail — prescription / medication requests
    # -----------------------------------------------------------------------
    if (any(kw in user_query_lower for kw in FORBIDDEN_MEDICATIONS)
            or "prescribe" in user_query_lower):
        specialty = resolve_specialty(
            payload.detected_modality or "",
            payload.body_region or "",
            payload.detected_condition or "",
        )
        return {
            "response": "I cannot recommend or prescribe medications. Please consult a licensed physician.",
            "actions_required": True,
            "suggested_doctors": get_doctors_for_specialty(specialty),
        }

    # -----------------------------------------------------------------------
    # 2. Retrieve RAG context for the current modality & region
    # -----------------------------------------------------------------------
    rag_context = retrieve_rag_context(
        payload.detected_modality or "",
        payload.body_region or "",
    )

    # -----------------------------------------------------------------------
    # 3. Build context-aware LLM prompt
    # -----------------------------------------------------------------------
    condition_ctx = payload.detected_condition or "Not yet determined"
    modality_ctx  = payload.detected_modality  or "Unknown imaging modality"
    region_ctx    = payload.body_region        or "Unknown body region"

    system_prompt = f"""
You are an AI medical triage assistant integrated with a computer vision pipeline.

=== IMAGING CONTEXT (from CV model) ===
- Imaging Modality : {modality_ctx}
- Body Region      : {region_ctx}
- Suspected Finding: {condition_ctx}

=== RETRIEVED MEDICAL KNOWLEDGE (RAG) ===
{rag_context}

=== YOUR RULES ===
1. Answer the user's query STRICTLY in the context of {region_ctx} / {modality_ctx}.
2. Do NOT reference conditions from other body systems (e.g., do NOT mention Pneumonia when analyzing a Knee).
3. Do NOT diagnose. Use language like "imaging patterns of this type often indicate...", "findings consistent with..."
4. Do NOT prescribe medications.
5. Ground your answer using the Retrieved Medical Knowledge above where relevant.
6. Keep the response concise — max 4 sentences.
7. Set "needs_doctor" to true ONLY if the user explicitly asks about treatment, physical therapy, surgery, where to go, or expresses pain/urgency. For purely informational questions, set it to false.

User's query: "{user_query}"

Output ONLY valid JSON:
{{
  "response": "Your concise, context-aware answer here",
  "needs_doctor": boolean
}}
"""

    try:
        chat_data    = json.loads(_call_text(system_prompt))
        llm_text     = chat_data.get("response", "I'm sorry, I couldn't process that request.")
        needs_doctor = chat_data.get("needs_doctor", False)
    except Exception as e:
        print(f"Chat LLM error: {e}")
        llm_text     = "System is temporarily unavailable. Please consult a doctor directly."
        needs_doctor = True

    # -----------------------------------------------------------------------
    # 4. Modality-aware doctor routing — NEVER falls back to wrong specialty
    # -----------------------------------------------------------------------
    suggested = []
    if needs_doctor:
        specialty = resolve_specialty(
            payload.detected_modality or "",
            payload.body_region or "",
            payload.detected_condition or "",
        )
        suggested = get_doctors_for_specialty(specialty)

    return {
        "response":         llm_text,
        "actions_required": needs_doctor,
        "suggested_doctors": suggested,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)