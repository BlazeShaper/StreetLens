from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time as time_module
from html import escape
from collections import Counter
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)


APP_TITLE = "StreetLens"
FRAME_INTERVAL_SECONDS = 1
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
YOLO_MODEL_NAME = "yolov8s.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_IMAGE_SIZE = 960
MIN_PERSON_BOX_AREA_RATIO = 0.0025

UI_TEXT = {
    "tr": {
        "language": "Dil",
        "title": "StreetLens Demografik Analiz",
        "caption": "YOLO ile insan tespiti, DeepFace ile demografik tahmin, Gemini ile is yeri onerisi.",
        "video_section": "Video ve zaman araligi",
        "privacy": "Gizlilik odakli analiz - yuz gorselleri saklanmaz.",
        "video_file": "Video dosyasi",
        "unknown_time": "Video saati bilinmiyor",
        "video_start": "Video baslangic saati",
        "start": "Baslangic",
        "end": "Bitis",
        "unknown_time_help": "Saat bilinmiyorsa analiz tum videoya uygulanir; Gemini'ye parlaklik seviyesi ve video suresi gonderilir.",
        "analyze": "Analiz Et",
        "current_detection": "Anlik tespit",
        "people_detected": "{count} kisi tespit edildi",
        "running": "Analiz arka planda suruyor. Sayfa otomatik yenilenir.",
        "stats_title": "Demografik istatistikler",
        "gemini_title": "Gemini analizi",
        "empty": "Bir video secip analiz baslattiginizda istatistikler ve Gemini onerileri burada gorunecek.",
        "total_people": "Toplam kisi",
        "reliability": "Veri guvenilirligi",
        "active_hours": "Aktif saat sayisi",
        "duration": "Video suresi",
        "unknown_time_info": "Video saati bilinmiyor. Aydinlik analizine gore tahmin: {estimate}.",
        "gender_chart": "Cinsiyet dagilimi",
        "age_chart": "Yas gruplari",
        "age_group": "Yas grubu",
        "person_count": "Kisi sayisi",
        "gender_empty": "Cinsiyet grafigi icin yeterli veri yok.",
        "age_empty": "Yas gruplari grafigi icin yeterli veri yok.",
        "hour_chart": "Saate gore yogunluk",
        "hour": "Saat",
    },
    "en": {
        "language": "Language",
        "title": "StreetLens Demographic Analysis",
        "caption": "YOLO person detection, DeepFace demographic estimates, Gemini business recommendations.",
        "video_section": "Video and time range",
        "privacy": "Privacy-focused analysis - face images are not stored.",
        "video_file": "Video file",
        "unknown_time": "Video time is unknown",
        "video_start": "Video start time",
        "start": "Start",
        "end": "End",
        "unknown_time_help": "When the time is unknown, the full video is analyzed; brightness and duration are sent to Gemini.",
        "analyze": "Analyze",
        "current_detection": "Live detection",
        "people_detected": "{count} people detected",
        "running": "Analysis is running in the background. The page refreshes automatically.",
        "stats_title": "Demographic statistics",
        "gemini_title": "Gemini analysis",
        "empty": "Choose a video and start analysis to see statistics and Gemini recommendations here.",
        "total_people": "Total people",
        "reliability": "Data reliability",
        "active_hours": "Active hours",
        "duration": "Video duration",
        "unknown_time_info": "Video time is unknown. Brightness-based estimate: {estimate}.",
        "gender_chart": "Gender distribution",
        "age_chart": "Age groups",
        "age_group": "Age group",
        "person_count": "People count",
        "gender_empty": "Not enough data for the gender chart.",
        "age_empty": "Not enough data for the age chart.",
        "hour_chart": "Hourly density",
        "hour": "Hour",
    },
}

SYSTEM_PROMPT = """
Sen bir lokasyon analisti ve iÅŸ geliÅŸtirme danÄ±ÅŸmanÄ±sÄ±n. Sana bir sokak kamerasÄ±ndan elde edilmiÅŸ demografik veri seti verilecek. Bu veriyi analiz ederek o lokasyona en uygun iÅŸ yeri Ã¶nerilerini sunacaksÄ±n.

KurallarÄ±n:
- Sadece veriye dayan, tahmin veya varsayÄ±m ekleme
- EÄŸer veri yetersizse (toplam kişi sayısı 50'nin altÄ±ndaysa) bunu aÃ§Ä±kÃ§a belirt ve Ã¶nerilerin gÃ¼ven skorunu dÃ¼ÅŸÃ¼r
- Ã–nerilerini verinin gÃ¼Ã§lÃ¼ yÃ¶nlerine gÃ¶re Ã¶nceliklendir
- TÃ¼rkÃ§e yanÄ±t ver
- YanÄ±tÄ±nÄ± kesinlikle dÃ¼z metin veya aÃ§Ä±klama olmadan, sadece geÃ§erli JSON formatÄ±nda ver
""".strip()


@dataclass(frozen=True)
class Detection:
    timestamp_seconds: float
    clock_minutes: int
    age: int | None
    gender: str


@dataclass(frozen=True)
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int | None

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class TrackState:
    timestamp_seconds: float
    clock_minutes: int
    age: int | None
    gender: str
    confidence: float
    area: int


@dataclass(frozen=True)
class VideoContext:
    duration_seconds: float
    time_known: bool
    estimated_time_label: str
    average_brightness: float | None


def get_gemini_env_error() -> str | None:
    if not ENV_PATH.exists():
        return ".env dosyasÄ± bulunamadÄ±. Proje klasÃ¶rÃ¼nde .env dosyasÄ± oluÅŸturup GEMINI_API_KEY deÄŸerini ekleyin."
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return ".env dosyasÄ±nda GEMINI_API_KEY boÅŸ veya tanÄ±mlÄ± deÄŸil. LÃ¼tfen geÃ§erli Gemini API anahtarÄ±nÄ± ekleyin."
    return None


def time_to_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def seconds_to_clock_minutes(video_start: time, elapsed_seconds: float) -> int:
    return (time_to_minutes(video_start) + int(elapsed_seconds // 60)) % (24 * 60)


def format_minutes(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:00"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} saniye"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes} dakika {remainder} saniye"


def estimate_frame_brightness(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def estimate_time_label_from_brightness(average_brightness: float | None) -> str:
    if average_brightness is None:
        return "Unknown video time; brightness could not be analyzed"
    if average_brightness < 55:
        return "Night or very low-light environment"
    if average_brightness < 100:
        return "Evening/night-like or weakly lit environment"
    if average_brightness < 150:
        return "Early morning, late afternoon, or cloudy daylight"
    return "Daylight or strongly lit environment"


def is_between_clock_range(value: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def normalize_gender(value: Any) -> str | None:
    if isinstance(value, dict) and value:
        man_score = float(value.get("Man", value.get("man", 0)) or 0)
        woman_score = float(value.get("Woman", value.get("woman", 0)) or 0)
        return "Erkek" if man_score >= woman_score else "Kadın"

    text = str(value or "").strip().lower()
    if text in {"man", "male", "erkek"}:
        return "Erkek"
    if text in {"woman", "female", "kadın", "kadin"}:
        return "Kadın"
    return None


def parse_age(value: Any) -> int | None:
    if value is None:
        return None
    try:
        age = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(age, 0)


def age_bucket(age: int) -> str:
    if age <= 17:
        return "0-17"
    if age <= 25:
        return "18-25"
    if age <= 40:
        return "26-40"
    if age <= 60:
        return "41-60"
    return "60+"


def percentage(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100 / total, 1)


def reliability_label(total: int) -> str:
    if total < 50:
        return "low"
    if total < 200:
        return "medium"
    return "high"


def detect_people_boxes(model: Any, frame: Any) -> list[PersonBox]:
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[YOLO_PERSON_CLASS_ID],
        conf=YOLO_CONFIDENCE_THRESHOLD,
        iou=0.5,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )
    boxes: list[PersonBox] = []
    height, width = frame.shape[:2]
    min_area = int(width * height * MIN_PERSON_BOX_AREA_RATIO)

    for result in results:
        track_ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        for box, track_id in zip(result.boxes, track_ids):
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            person_box = PersonBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                track_id=int(track_id) if track_id is not None else None,
            )
            if person_box.area < min_area:
                continue
            boxes.append(person_box)

    return boxes


def analyze_person_crop(crop: Any) -> tuple[int | None, str]:
    from deepface import DeepFace

    if crop.size == 0:
        return None, "Bilinmiyor"

    upper_body = crop[: max(1, int(crop.shape[0] * 0.55)), :]
    rgb_crop = cv2.cvtColor(upper_body, cv2.COLOR_BGR2RGB)
    try:
        result = DeepFace.analyze(
            img_path=rgb_crop,
            actions=["age", "gender"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
    except Exception:
        print("[DEBUG] DeepFace başarısız → Gemini'ye düşüldü", flush=True)
        import base64
        import google.generativeai as genai

        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return None, "Bilinmiyor"

        genai.configure(api_key=os.environ["GEMINI_API_KEY"].strip())
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
            system_instruction='Bu görseldeki kişinin tahmini yaşını ve cinsiyetini belirle. Sadece JSON döndür, başka hiçbir şey yazma, markdown kullanma: {"age": 25, "gender": "Kadın"} — gender değeri yalnızca \'Erkek\' veya \'Kadın\' olabilir, yaş 1-100 arası integer olmalı.',
        )
        image_base64 = base64.b64encode(encoded).decode("utf-8")
        try:
            response = model.generate_content(
                [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64,
                        }
                    }
                ],
                generation_config={"response_mime_type": "application/json"},
            )
        except Exception as exc:
            print(f"[DEBUG] Gemini başarısız: {exc}", flush=True)
            return None, "Bilinmiyor"
        try:
            parsed = parse_gemini_json(response.text or "")
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, "Bilinmiyor"

        age = parse_age(parsed.get("age"))
        gender = normalize_gender(parsed.get("gender")) or "Bilinmiyor"
        if age is not None and not 1 <= age <= 100:
            age = None
        return age, gender

    print("[DEBUG] DeepFace başarılı", flush=True)

    faces = result if isinstance(result, list) else [result]
    for face in faces:
        if not isinstance(face, dict):
            continue
        age_value = face.get("age")
        gender_value = face.get("dominant_gender") or face.get("gender")
        age = parse_age(age_value)
        gender = normalize_gender(gender_value) or "Bilinmiyor"
        return age, gender

    return None, "Bilinmiyor"


def process_video(
    video_path: str,
    video_start: time,
    progress_events: queue.Queue,
    time_unknown: bool = False,
    interval_seconds: int = FRAME_INTERVAL_SECONDS,
) -> tuple[list[Detection], VideoContext]:
    from ultralytics import YOLO

    yolo_model = YOLO(YOLO_MODEL_NAME)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Video dosyasÄ± aÃ§Ä±lamadÄ±. Dosya biÃ§imini veya dosyanÄ±n bozuk olup olmadÄ±ÄŸÄ±nÄ± kontrol edin.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = frame_count / fps if frame_count else 0
    sample_count = max(1, int(duration_seconds // interval_seconds) + 1) if duration_seconds else 1
    print(
        f"[DEBUG] Video analizi baÅŸladÄ±: fps={fps:.2f}, frame_count={frame_count}, "
        f"duration={duration_seconds:.2f}s, interval={interval_seconds}s, samples~={sample_count}",
        flush=True,
    )

    track_states: dict[int, TrackState] = {}
    fallback_detections: list[Detection] = []
    frame_snapshots: list[list[Detection]] = []
    brightness_values: list[float] = []
    sample_index = 0
    current_second = 0

    while True:
        if duration_seconds and current_second > duration_seconds:
            break

        capture.set(cv2.CAP_PROP_POS_MSEC, current_second * 1000)
        ok, frame = capture.read()
        if not ok:
            print(f"[DEBUG] Kare okunamadÄ±: index={sample_index + 1}, time={current_second:.1f}s", flush=True)
            break

        brightness = estimate_frame_brightness(frame)
        brightness_values.append(brightness)
        clock_minutes = 0 if time_unknown else seconds_to_clock_minutes(video_start, current_second)
        clock_label = "bilinmiyor" if time_unknown else f"{clock_minutes // 60:02d}:{clock_minutes % 60:02d}"
        boxes = detect_people_boxes(yolo_model, frame)
        people: list[tuple[int | None, str]] = []
        frame_detections: list[Detection] = []
        for person_index, person_box in enumerate(boxes, start=1):
            existing = track_states.get(person_box.track_id) if person_box.track_id is not None else None
            if person_box.track_id is None:
                age, gender = None, "Bilinmiyor"
                print("[DEBUG] track_id yok → Gemini çağrılmadı", flush=True)
            elif existing is not None:
                age, gender = existing.age, existing.gender
                print(f"[DEBUG] track cache kullanıldı: track={person_box.track_id}", flush=True)
            else:
                crop = frame[person_box.y1 : person_box.y2, person_box.x1 : person_box.x2].copy()
                age, gender = analyze_person_crop(crop)
                del crop
            print(
                f"[DEBUG] YOLO person {person_index}/{len(boxes)} | track={person_box.track_id} | "
                f"bbox=({person_box.x1},{person_box.y1},{person_box.x2},{person_box.y2}) | "
                f"conf={person_box.confidence:.2f} | age={age} | gender={gender}",
                flush=True,
            )
            people.append((age, gender))
            detection = Detection(
                timestamp_seconds=current_second,
                clock_minutes=clock_minutes,
                age=age,
                gender=gender,
            )
            frame_detections.append(detection)
            if person_box.track_id is None:
                fallback_detections.append(detection)
                continue

            existing = track_states.get(person_box.track_id)
            should_update = (
                existing is None
                or (existing.age is None and age is not None)
                or (
                    age is not None
                    and gender != "Bilinmiyor"
                    and person_box.area > existing.area
                )
            )
            if should_update:
                track_states[person_box.track_id] = TrackState(
                    timestamp_seconds=current_second if existing is None else existing.timestamp_seconds,
                    clock_minutes=clock_minutes if existing is None else existing.clock_minutes,
                    age=age,
                    gender=gender,
                    confidence=person_box.confidence,
                    area=person_box.area,
                )
        frame_snapshots.append(frame_detections)
        people_summary = ", ".join(f"{age if age is not None else '?'}/{gender}" for age, gender in people) or "kişi bulunamadÄ±"
        unique_count = len(track_states) if track_states else len(fallback_detections)
        print(
            f"[DEBUG] Kare {sample_index + 1}/{sample_count} | video_saniyesi={current_second:.1f} | "
            f"hesaplanan_saat={clock_label} | parlaklik={brightness:.1f} | "
            f"yolo_kisi={len(boxes)} | unique_kisi={unique_count} | demografi={len(people)} | {people_summary}",
            flush=True,
        )

        sample_index += 1
        progress_events.put(
            {
                "type": "progress",
                "value": min(sample_index / sample_count, 1.0),
                "message": f"{current_second:.0f}s sampled, {len(boxes)} people in frame, {unique_count} unique people so far.",
                "current_people_count": len(boxes),
            }
        )
        current_second += interval_seconds

    capture.release()
    track_detections = [
        Detection(
            timestamp_seconds=state.timestamp_seconds,
            clock_minutes=state.clock_minutes,
            age=state.age,
            gender=state.gender,
        )
        for state in track_states.values()
    ]
    if duration_seconds and duration_seconds <= 30 and frame_snapshots:
        detections = max(frame_snapshots, key=len)
    else:
        detections = track_detections or fallback_detections
    average_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else None
    video_context = VideoContext(
        duration_seconds=duration_seconds,
        time_known=not time_unknown,
        estimated_time_label=(
            estimate_time_label_from_brightness(average_brightness)
            if time_unknown
            else "Video start time was provided by the user"
        ),
        average_brightness=average_brightness,
    )
    print(
        f"[DEBUG] Video analysis finished: processed_frames={sample_index}, total_detections={len(detections)}",
        flush=True,
    )
    return detections, video_context


def filter_detections(detections: list[Detection], start: time, end: time) -> list[Detection]:
    start_minutes = time_to_minutes(start)
    end_minutes = time_to_minutes(end)
    return [item for item in detections if is_between_clock_range(item.clock_minutes, start_minutes, end_minutes)]


def should_analyze_full_video(video_start: time) -> bool:
    return time_to_minutes(video_start) == 0


def analysis_time_label(start: time, end: time, full_video: bool = False) -> str:
    if full_video:
        return "TÃ¼m video"
    return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"


def calculate_statistics(detections: list[Detection], video_context: VideoContext | None = None) -> dict[str, Any]:
    analyzed_detections = detections
    short_video_mode = bool(video_context and video_context.duration_seconds <= 30)

    total = len(analyzed_detections)
    age_counts = Counter(age_bucket(item.age) for item in analyzed_detections if item.age is not None)
    gender_counts = Counter(item.gender for item in analyzed_detections)
    time_known = video_context.time_known if video_context else True
    hour_counts = Counter(format_minutes(item.clock_minutes) for item in analyzed_detections) if time_known else Counter()

    age_order = ["0-17", "18-25", "26-40", "41-60", "60+"]
    gender_order = ["Erkek", "Kadın", "Bilinmiyor"]
    hour_order = [f"{hour:02d}:00" for hour in range(24)]

    return {
        "total": total,
        "reliability": reliability_label(total),
        "age_counts": {bucket: age_counts.get(bucket, 0) for bucket in age_order},
        "age_percentages": {bucket: percentage(age_counts.get(bucket, 0), total) for bucket in age_order},
        "gender_counts": {gender: gender_counts.get(gender, 0) for gender in gender_order},
        "gender_percentages": {gender: percentage(gender_counts.get(gender, 0), total) for gender in gender_order},
        "hour_counts": {hour: hour_counts.get(hour, 0) for hour in hour_order if hour_counts.get(hour, 0) > 0},
        "time_known": time_known,
        "video_duration_seconds": video_context.duration_seconds if video_context else None,
        "estimated_time_label": video_context.estimated_time_label if video_context else None,
        "average_brightness": video_context.average_brightness if video_context else None,
        "observation_count": len(detections),
        "short_video_mode": short_video_mode,
    }


def build_user_prompt(stats: dict[str, Any], start: time, end: time, full_video: bool = False) -> str:
    age = stats["age_percentages"]
    gender = stats["gender_percentages"]
    time_known = bool(stats.get("time_known", True))
    if time_known:
        time_context = f"Zaman dilimi: {analysis_time_label(start, end, full_video)}"
        hourly_lines = "\n".join(f"- {hour}: {count} kişi" for hour, count in stats["hour_counts"].items()) or "- Veri yok"
    else:
        brightness = stats.get("average_brightness")
        brightness_text = f"{brightness:.1f}" if isinstance(brightness, (int, float)) else "hesaplanamadÄ±"
        time_context = (
            "Video baÅŸlangÄ±Ã§ saati: bilinmiyor\n"
            f"AydÄ±nlÄ±k seviyesine gÃ¶re tahmini zaman baÄŸlamÄ±: {stats.get('estimated_time_label') or 'belirsiz'}\n"
            f"Ortalama parlaklÄ±k skoru: {brightness_text}\n"
            f"Video sÃ¼resi: {format_duration(float(stats.get('video_duration_seconds') or 0))}\n"
            "Saat bazlÄ± yoÄŸunluk bilinmiyor; 00:00 deÄŸerini gece yarÄ±sÄ± olarak yorumlama."
        )
        hourly_lines = "- GerÃ§ek saat bilinmediÄŸi iÃ§in saat bazlÄ± yoÄŸunluk hesaplanmadÄ±"

    sample_context = ""
    if stats.get("short_video_mode"):
        sample_context = (
            f"\nKÄ±sa video notu: Video 30 saniyeden kÄ±sa olduÄŸu iÃ§in aynÄ± kişiler kareler arasÄ±nda tekrar sayÄ±lmasÄ±n diye "
            f"demografik toplam en kalabalÄ±k temsilci kareden hesaplandÄ±. Ham YOLO gÃ¶zlem sayısı: {stats.get('observation_count')}."
        )

    return f"""
AÅŸaÄŸÄ±daki demografik veriye gÃ¶re analiz yap:

{time_context}
Toplam geÃ§en kişi: {stats["total"]}
Veri gÃ¼venilirliÄŸi: {stats["reliability"]}
{sample_context}

Yaş dağılımÄ±:
- 0-17 yaÅŸ: %{age["0-17"]}
- 18-25 yaÅŸ: %{age["18-25"]}
- 26-40 yaÅŸ: %{age["26-40"]}
- 41-60 yaÅŸ: %{age["41-60"]}
- 60+ yaÅŸ: %{age["60+"]}

Cinsiyet dağılımÄ±:
- Erkek: %{gender["Erkek"]}
- Kadın: %{gender["Kadın"]}
- Bilinmiyor: %{gender["Bilinmiyor"]}

Saate gÃ¶re yoÄŸunluk:
{hourly_lines}

AÅŸaÄŸÄ±daki JSON formatÄ±nda yanÄ±t ver, baÅŸka hiÃ§bir ÅŸey yazma:

{{
  "zaman_dilimi": "string",
  "demografik_ozet": "string - 3-4 cÃ¼mle, bu sokaktan geÃ§en kitleyi tanÄ±mla",
  "profil_skoru": integer (0-100, lokasyonun iÅŸ potansiyelini gÃ¶sterir),
  "oneriler": [
    {{
      "isyeri_turu": "string",
      "neden": "string - veriye dayalÄ± gerekÃ§e",
      "hedef_kitle": "string",
      "en_iyi_saat": "string",
      "tahmini_potansiyel": "yÃ¼ksek | orta | dÃ¼ÅŸÃ¼k",
      "guven_skoru": integer (0-100)
    }}
  ],
  "kacininmasi_gerekenler": [
    {{
      "isyeri_turu": "string",
      "neden": "string - neden bu lokasyona uygun deÄŸil"
    }}
  ],
  "veri_uyarisi": "string veya null - veri yetersizse uyarÄ± mesajÄ±"
}}

Ã–neri sayısı: minimum 2, maksimum 6 (veriye gÃ¶re karar ver)
KaÃ§Ä±nÄ±lmasÄ± gereken sayısı: minimum 1, maksimum 3
""".strip()


def parse_gemini_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def ask_gemini(stats: dict[str, Any], start: time, end: time, full_video: bool = False) -> dict[str, Any]:
    import google.generativeai as genai

    env_error = get_gemini_env_error()
    if env_error:
        raise RuntimeError(env_error)
    api_key = os.environ["GEMINI_API_KEY"].strip()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    fallback = {
        "gemini_unavailable": True,
        "zaman_dilimi": analysis_time_label(start, end, full_video),
        "demografik_ozet": "Gemini kotası dolduğu için otomatik iş yeri yorumu üretilemedi. Demografik grafikler ve temel istatistikler başarıyla hesaplandı.",
        "profil_skoru": 0,
        "oneriler": [],
        "kacininmasi_gerekenler": [],
        "veri_uyarisi": "Gemini API kotası/rate limit nedeniyle öneri üretilemedi. Bir süre sonra tekrar deneyin veya GEMINI_MODEL / API planını güncelleyin.",
    }
    try:
        response = model.generate_content(
            build_user_prompt(stats, start, end, full_video),
            generation_config={"response_mime_type": "application/json"},
        )
    except Exception as exc:
        print(f"[DEBUG] Gemini öneri analizi başarısız: {exc}", flush=True)
        return fallback

    raw_text = response.text or ""
    try:
        return parse_gemini_json(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[DEBUG] Gemini öneri JSON parse başarısız: {exc.msg}. Ham yanıt: {raw_text[:800]}", flush=True)
        return fallback


def worker(
    video_path: str,
    video_start: time,
    analysis_start: time,
    analysis_end: time,
    time_unknown: bool,
    events: queue.Queue,
) -> None:
    try:
        detections, video_context = process_video(video_path, video_start, events, time_unknown=time_unknown)
        events.put({"type": "progress", "value": 1.0, "message": "Video processed, preparing statistics."})
        full_video = time_unknown or should_analyze_full_video(video_start)
        filtered = detections if full_video else filter_detections(detections, analysis_start, analysis_end)
        stats = calculate_statistics(filtered, video_context)
        gemini = ask_gemini(stats, analysis_start, analysis_end, full_video)
        events.put({"type": "done", "detections": len(detections), "stats": stats, "gemini": gemini})
    except Exception as exc:
        events.put({"type": "error", "message": str(exc)})
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            pass


def ensure_state() -> None:
    defaults = {
        "events": None,
        "thread": None,
        "running": False,
        "progress": 0.0,
        "status_message": "Waiting for analysis.",
        "stats": None,
        "gemini": None,
        "error": None,
        "raw_detection_count": 0,
        "current_people_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def drain_events() -> None:
    events = st.session_state.events
    if events is None:
        return

    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break

        if event["type"] == "progress":
            st.session_state.progress = event["value"]
            st.session_state.status_message = event["message"]
            if "current_people_count" in event:
                st.session_state.current_people_count = event["current_people_count"]
        elif event["type"] == "done":
            st.session_state.running = False
            st.session_state.progress = 1.0
            st.session_state.status_message = "Analysis complete."
            st.session_state.current_people_count = 0
            st.session_state.stats = event["stats"]
            st.session_state.gemini = event["gemini"]
            st.session_state.raw_detection_count = event["detections"]
        elif event["type"] == "error":
            st.session_state.running = False
            st.session_state.error = event["message"]
            st.session_state.status_message = "Analysis stopped."
            st.session_state.current_people_count = 0


def save_upload_to_temp(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def start_background_analysis(
    uploaded_file: Any,
    video_start: time,
    analysis_start: time,
    analysis_end: time,
    time_unknown: bool,
) -> None:
    video_path = save_upload_to_temp(uploaded_file)
    events: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=worker,
        args=(video_path, video_start, analysis_start, analysis_end, time_unknown, events),
        daemon=True,
    )
    st.session_state.events = events
    st.session_state.thread = thread
    st.session_state.running = True
    st.session_state.progress = 0.0
    st.session_state.status_message = "Video processing started."
    st.session_state.stats = None
    st.session_state.gemini = None
    st.session_state.error = None
    st.session_state.current_people_count = 0
    thread.start()


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def text_for(lang: str, key: str, **kwargs: Any) -> str:
    template = UI_TEXT.get(lang, UI_TEXT["tr"]).get(key, key)
    return template.format(**kwargs)


def render_stats(stats: dict[str, Any], lang: str) -> None:
    top_cols = st.columns(3)
    with top_cols[0]:
        metric_card(text_for(lang, "total_people"), str(stats["total"]))
    with top_cols[1]:
        metric_card(text_for(lang, "reliability"), str(stats["reliability"]).upper())
    with top_cols[2]:
        if stats.get("time_known", True):
            metric_card(text_for(lang, "active_hours"), str(len(stats["hour_counts"])))
        else:
            metric_card(text_for(lang, "duration"), format_duration(float(stats.get("video_duration_seconds") or 0)))

    if not stats.get("time_known", True):
        st.info(
            text_for(lang, "unknown_time_info", estimate=stats.get("estimated_time_label") or "unknown")
        )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        gender_counts = {
            gender: int(count)
            for gender, count in stats["gender_counts"].items()
            if gender != "Bilinmiyor" and int(count) > 0
        }
        if gender_counts:
            gender_labels = {
                "en": {"Erkek": "Male", "Kadın": "Female"},
                "tr": {"Erkek": "Erkek", "Kadın": "Kadin"},
            }.get(lang, {})
            fig = px.pie(
                names=[gender_labels.get(gender, gender) for gender in gender_counts.keys()],
                values=list(gender_counts.values()),
                title=text_for(lang, "gender_chart"),
                hole=0.45,
                color_discrete_sequence=["#3867d6", "#eb3b5a"],
            )
            fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True, key="gender_chart")
        else:
            st.info(text_for(lang, "gender_empty"))

    with chart_cols[1]:
        age_counts = {group: int(count) for group, count in stats["age_counts"].items()}
        if any(age_counts.values()):
            y_values = list(age_counts.values())
            y_max = max(y_values)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(age_counts.keys()),
                        y=y_values,
                        marker_color=["#20bf6b", "#45aaf2", "#fed330", "#fa8231", "#8854d0"],
                    )
                ]
            )
            fig.update_layout(
                title=text_for(lang, "age_chart"),
                xaxis_title=text_for(lang, "age_group"),
                yaxis_title=text_for(lang, "person_count"),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig.update_yaxes(range=[0, max(1, y_max) * 1.15], rangemode="tozero", dtick=1)
            st.plotly_chart(fig, use_container_width=True, key="age_chart")
        else:
            st.info(text_for(lang, "age_empty"))

    hour_counts = stats["hour_counts"]
    if hour_counts:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(hour_counts.keys()),
                    y=[int(value) for value in hour_counts.values()],
                    marker_color="#2d98da",
                )
            ]
        )
        fig.update_layout(
            title=text_for(lang, "hour_chart"),
            xaxis_title=text_for(lang, "hour"),
            yaxis_title=text_for(lang, "person_count"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_yaxes(rangemode="tozero", dtick=1)
        st.plotly_chart(fig, use_container_width=True, key="hour_chart")

def render_gemini_result(result: dict[str, Any]) -> None:
    if result.get("gemini_unavailable"):
        warning = result.get("veri_uyarisi")
        if warning:
            st.warning(warning)
        summary = result.get("demografik_ozet")
        if summary:
            st.write(summary)
        return

    score = escape(str(result.get("profil_skoru", 0)))
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">Profile score</div>
            <div class="score-value">{score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    warning = result.get("veri_uyarisi")
    if warning:
        st.warning(warning)

    summary = result.get("demografik_ozet")
    if summary:
        st.subheader("Demographic summary")
        st.write(summary)

    st.subheader("Business recommendations")
    recommendations = result.get("oneriler", [])
    if recommendations:
        for item in recommendations:
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div class="card-title">{escape(str(item.get("isyeri_turu", "Recommendation")))}</div>
                    <div class="card-body">{escape(str(item.get("neden", "")))}</div>
                    <div class="card-grid">
                        <span>Target audience</span><strong>{escape(str(item.get("hedef_kitle", "-")))}</strong>
                        <span>Best time</span><strong>{escape(str(item.get("en_iyi_saat", "-")))}</strong>
                        <span>Potential</span><strong>{escape(str(item.get("tahmini_potansiyel", "-")))}</strong>
                        <span>Confidence</span><strong>{escape(str(item.get("guven_skoru", "-")))}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    avoid_items = result.get("kacininmasi_gerekenler", [])
    if avoid_items:
        st.subheader("Avoid")
    for item in avoid_items:
        st.markdown(
            f"""
            <div class="avoid-card">
                <div class="card-title">{escape(str(item.get("isyeri_turu", "Not suitable")))}</div>
                <div class="card-body">{escape(str(item.get("neden", "")))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            max-width: 1280px;
        }
        .score-card {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
            margin-bottom: 16px;
        }
        .score-label {
            color: #52616b;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .score-value {
            color: #102a43;
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1;
            margin-top: 4px;
        }
        .recommendation-card,
        .avoid-card {
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
            border: 1px solid #d9e2ec;
            background: #ffffff;
        }
        .recommendation-card {
            border-left: 5px solid #20bf6b;
        }
        .avoid-card {
            border-left: 5px solid #eb3b5a;
            background: #fff8f8;
        }
        .card-title {
            font-weight: 800;
            font-size: 1.05rem;
            color: #102a43;
            margin-bottom: 6px;
        }
        .card-body {
            color: #334e68;
            margin-bottom: 10px;
        }
        .card-grid {
            display: grid;
            grid-template-columns: minmax(90px, 0.45fr) 1fr;
            gap: 5px 12px;
            color: #52616b;
            font-size: 0.92rem;
        }
        .card-grid strong {
            color: #102a43;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="StreetLens", page_icon="📊", layout="wide")
    ensure_state()
    drain_events()
    render_styles()

    with st.sidebar:
        st.header("StreetLens")
        language_label = st.selectbox("Language / Dil", ["English", "Turkce"], index=0)
        lang = "tr" if language_label == "Turkce" else "en"
        st.divider()

        env_error = get_gemini_env_error()

        st.subheader(text_for(lang, "video_section"))
        st.caption(text_for(lang, "privacy"))
        uploaded = st.file_uploader(text_for(lang, "video_file"), type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            st.video(uploaded)

        st.divider()
        time_unknown = st.checkbox(text_for(lang, "unknown_time"), value=True)
        video_start = st.time_input(text_for(lang, "video_start"), value=time(0, 0), step=60, disabled=time_unknown)
        interval_cols = st.columns(2)
        with interval_cols[0]:
            analysis_start = st.time_input(text_for(lang, "start"), value=time(13, 0), step=60, disabled=time_unknown)
        with interval_cols[1]:
            analysis_end = st.time_input(text_for(lang, "end"), value=time(17, 0), step=60, disabled=time_unknown)
        if time_unknown:
            st.caption(text_for(lang, "unknown_time_help"))

        st.divider()
        disabled = st.session_state.running or uploaded is None or env_error is not None
        if st.button(text_for(lang, "analyze"), type="primary", use_container_width=True, disabled=disabled):
            if env_error:
                st.session_state.error = env_error
            else:
                start_background_analysis(uploaded, video_start, analysis_start, analysis_end, time_unknown)
                st.rerun()

        st.progress(st.session_state.progress, text=st.session_state.status_message)
        st.metric(
            text_for(lang, "current_detection"),
            text_for(lang, "people_detected", count=st.session_state.current_people_count),
        )
        if st.session_state.running:
            st.info(text_for(lang, "running"))
            time_module.sleep(1)
            st.rerun()

        if st.session_state.error:
            st.error(st.session_state.error)

        if st.session_state.stats:
            st.divider()
            st.metric(text_for(lang, "total_people"), str(st.session_state.stats["total"]))
            st.metric(text_for(lang, "reliability"), str(st.session_state.stats["reliability"]).upper())

    st.title(text_for(lang, "title"))
    st.caption(text_for(lang, "caption"))

    if env_error:
        st.error(env_error)

    if st.session_state.stats:
        st.subheader(text_for(lang, "stats_title"))
        render_stats(st.session_state.stats, lang)
        st.divider()
        st.subheader(text_for(lang, "gemini_title"))
        render_gemini_result(st.session_state.gemini or {})
    else:
        st.info(text_for(lang, "empty"))


if __name__ == "__main__":
    main()
