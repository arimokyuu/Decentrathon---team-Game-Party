import streamlit as st
import requests
from PIL import Image, ImageDraw
from openai import OpenAI
import base64

# 🔑 API-ключи
API_KEY_ROBOFLOW = "EFy7lQ764ESqzbgxlhug"
MODEL_URL = "https://detect.roboflow.com/car-damages-detection-lnsbx/1"
client = OpenAI(api_key=" ")  # <-- вставь сюда ключ 

# --- Функция для GPT описания ---
def classify_with_gpt(image_bytes):
    """Отправка фото + инструкции в GPT для классификации"""
    try:
        # Кодируем картинку в base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": """Посмотри на фото автомобиля.
Классифицируй строго по двум критериям:
1. Целостность: битый или небитый.
2. Чистота: чистый, слегка грязный или сильно грязный.

Ответ дай строго в формате:
Автомобиль [целостность], [чистота]."""
                         },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка GPT: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Car Inspector AI", layout="wide")
st.title("🚗 AI-Осмотр автомобиля")
st.write("Загрузите фото, и AI найдёт повреждения и определит состояние авто.")

uploaded_file = st.file_uploader("Выберите фото машины", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное фото", width=600)

    # Делаем копию файла в байтах
    image_bytes = uploaded_file.getvalue()

    st.info("⌛ Анализ изображения в Roboflow...")

    # --- Roboflow для сегментации ---
    response = requests.post(
        f"{MODEL_URL}?api_key={API_KEY_ROBOFLOW}",
        files={"file": image_bytes},
        data={"confidence": 20, "overlap": 30}
    )

    result = response.json()
    damage_detected = False

    if "predictions" in result and len(result["predictions"]) > 0:
        damage_detected = True
        draw_img = image.convert("RGBA")
        overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for pred in result["predictions"]:
            label = pred["class"]
            conf = pred["confidence"]

            if "points" in pred:
                points = [(p["x"], p["y"]) for p in pred["points"]]
                draw.polygon(points, outline="red", fill=(255, 0, 0, 80))
                draw.text(points[0], f"{label} {conf:.2f}", fill="red")
            else:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                x0, y0 = x - w / 2, y - h / 2
                x1, y1 = x + w / 2, y + h / 2
                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                draw.text((x0, y0 - 10), f"{label} {conf:.2f}", fill="red")

        final_img = Image.alpha_composite(draw_img, overlay).convert("RGB")
        st.image(final_img, caption="🚨 Найденные повреждения", width=600)
    else:
        st.warning("Повреждений не найдено ✅")

    # --- GPT классификация по фото ---
    st.subheader("📊 Классификация автомобиля (GPT Анализ)")
    classification = classify_with_gpt(image_bytes)
    st.success(classification)
