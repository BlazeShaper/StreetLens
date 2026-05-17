# StreetLens 📊

> **AI-powered pedestrian demographic analysis from street camera footage.**

StreetLens analyzes video files from street cameras to detect people and estimate their demographic profiles (age, gender). It then leverages Google Gemini to generate data-driven business location recommendations based on the foot traffic observed.

---

## ✨ Features

- **Person Detection** — YOLOv8 with ByteTrack multi-object tracking to accurately count unique pedestrians
- **Demographic Estimation** — DeepFace (age & gender) with automatic Gemini Vision fallback when DeepFace fails
- **Business Recommendations** — Google Gemini analyzes the demographic summary and suggests the most suitable businesses for that location
- **Time-aware Analysis** — Supports known video start times or brightness-based time estimation for unclocked footage
- **Interactive Charts** — Gender pie chart, age group bar chart, and hourly density chart powered by Plotly
- **Bilingual UI** — Full English and Turkish interface via Streamlit sidebar toggle
- **Privacy-first** — Face crops are processed in memory only; no images are saved to disk

---

## 🖥️ Tech Stack

| Component | Library |
|---|---|
| Web UI | [Streamlit](https://streamlit.io) |
| Person Detection | [Ultralytics YOLOv8](https://docs.ultralytics.com) + ByteTrack |
| Demographic Analysis | [DeepFace](https://github.com/serengil/deepface) |
| AI Fallback & Recommendations | [Google Gemini API](https://ai.google.dev) (`gemini-2.5-flash`) |
| Video Processing | [OpenCV](https://opencv.org) |
| Charts | [Plotly](https://plotly.com) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BlazeShaper/StreetLens.git
cd StreetLens
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow and DeepFace installation may take several minutes. Python 3.10 is recommended.

### 4. Configure your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
# Optional: override the default model
# GEMINI_MODEL=gemini-2.5-flash
```

You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How It Works

```
Video File
    │
    ▼
┌───────────────┐
│  Frame Sampler │  ← 1 frame per second (configurable)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  YOLOv8 + ByteTrack │  ← Detects & tracks people across frames
└───────┬───────┘
        │  (unique person crop)
        ▼
┌───────────────────────┐
│  DeepFace             │  ← Estimates age & gender
│  (Gemini fallback)    │  ← Used when DeepFace fails
└───────┬───────────────┘
        │
        ▼
┌───────────────┐
│  Statistics   │  ← Age groups, gender split, hourly density
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Gemini LLM   │  ← Generates business recommendations as JSON
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Streamlit UI │  ← Interactive charts & recommendation cards
└───────────────┘
```

### Detection Pipeline Details

1. **Frame sampling** — The video is sampled once per second to balance speed and accuracy.
2. **Person detection** — YOLOv8 (`yolov8s.pt`) detects people with a confidence threshold of 0.40 and minimum bounding box area filter to remove distant or partial detections.
3. **Track caching** — ByteTrack assigns persistent IDs to each person. Demographic analysis runs only once per unique track ID; subsequent frames reuse the cached result.
4. **Demographic analysis** — The upper 55% of the detected bounding box (roughly head + torso) is cropped and sent to DeepFace. If DeepFace cannot find a face, the crop is sent to Gemini Vision as a fallback.
5. **Short video handling** — For videos ≤30 seconds, the single most populated frame is used as the demographic sample instead of track-based deduplication.
6. **Gemini recommendation** — Aggregated statistics (age buckets, gender split, hourly density) are sent to Gemini with a structured system prompt. The response is a strict JSON object with business suggestions, avoidances, and a location profile score (0–100).

---

## 🗂️ Project Structure

```
StreetLens/
├── app.py              # Main application (Streamlit + all logic)
├── requirements.txt    # Python dependencies
├── yolov8n.pt          # YOLOv8 nano weights (lightweight)
├── yolov8s.pt          # YOLOv8 small weights (used by default)
├── .env                # API keys (not committed to version control)
└── README.md
```

---

## ⚙️ Configuration

Key constants at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `FRAME_INTERVAL_SECONDS` | `1` | How often (in seconds) to sample a video frame |
| `DEFAULT_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for analysis |
| `YOLO_MODEL_NAME` | `yolov8s.pt` | YOLO weights file |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.40` | Minimum YOLO detection confidence |
| `YOLO_IMAGE_SIZE` | `960` | Image size fed to YOLO (higher = more accurate, slower) |
| `MIN_PERSON_BOX_AREA_RATIO` | `0.0025` | Minimum bounding box area as fraction of frame (filters out tiny detections) |

You can also set `GEMINI_MODEL` in your `.env` file to switch models without editing source code.

---

## 🔒 Privacy

- **No face images are stored.** Person crops are processed in RAM and immediately discarded.
- The `.env` file containing your API key is excluded from version control via `.gitignore`.
- Video files are saved to a system temporary directory during processing and automatically deleted after analysis completes.

---

## 📋 Requirements

- Python **3.10** (recommended; TensorFlow 2.19 requires ≥3.9, ≤3.12)
- A valid **Google Gemini API key**
- Supported video formats: `.mp4`, `.mov`, `.avi`, `.mkv`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
