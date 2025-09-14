# 🚗 Car Inspector AI

Веб-приложение на **Streamlit**, которое:
- определяет повреждения автомобиля с помощью модели Roboflow,  
- классифицирует его состояние (битый/небитый, чистота) с помощью **OpenAI GPT**.  

---

## 📦 Установка

1. Склонируйте или скопируйте проект:
   ```command panel
   git clone https://github.com/arimokyuu/Decentrathon---team-Game-Party.git
   cd Decentrathon---team-Game-Party
   ```

2. Установите зависимости:
   ```command panel
   pip install -r requirements.txt
   ```

   Если `requirements.txt` нет, установите вручную:
   ```command panel
   pip install streamlit requests pillow openai
   ```

---

## 🔑 Настройка API-ключей

В файле `app.py` есть два ключа, которые нужно указать:

1. **Roboflow API Key**  
   ```python
   API_KEY_ROBOFLOW = "ВАШ_КЛЮЧ"
   MODEL_URL = "https://detect.roboflow.com/car-damages-detection-lnsbx/1"
   ```
   👉 Получить можно на [Roboflow](https://roboflow.com).

2. **OpenAI API Key**  
   ```python
   client = OpenAI(api_key="sk-...")
   ```
   👉 Сгенерируйте на [OpenAI Platform](https://platform.openai.com/).

---

## ▶️ Запуск

После настройки ключей выполните:
```command panel
streamlit run app.py
```

Приложение откроется в браузере по адресу:  
👉 [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Использование

1. Загрузите фото автомобиля (`.jpg`, `.jpeg`, `.png`).
2. Приложение:
   - Отправит изображение в **Roboflow** для поиска повреждений.  
   - Отобразит найденные дефекты на картинке.  
   - Отправит фото в **GPT**, чтобы классифицировать:  
     - целостность (битый/небитый),  
     - чистоту (чистый/слегка грязный/сильно грязный).  
3. Результат выводится на экране.

---

## 📁 Структура проекта

```
app.py            # основной код приложения
requirements.txt  # список зависимостей (добавьте при необходимости)
README.md         # инструкция
```

---

## 🚀 Пример результата

- Загруженное фото  
- Картинка с подсвеченными повреждениями  
- Текстовый вывод:
  ```
  Автомобиль битый, слегка грязный.
  ```
