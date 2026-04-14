# mlops-dga-classifier

Курсовой проект по MLOps: сервис бинарной классификации доменных имен (`domain -> label`) с использованием ClearML.

DGA (Domain Generation Algorithm) - это алгоритм, который автоматически генерирует большое количество доменных имен, обычно используемых вредоносным ПО для связи с командным сервером. Такие домены трудно заранее заблокировать вручную, потому что они постоянно меняются и часто выглядят как случайный набор символов.

Данный сервис нужен для автоматической классификации доменных имен, чтобы ускорить первичную фильтрацию подозрительного сетевого трафика, помочь системам мониторинга безопасности и сократить время реакции на угрозы.

## Что реализовано

Проект покрывает минимальный жизненный цикл ML-модели:

1. Подготовка подвыборки датасета (в рамках курсового проекта используем не весь датасет).
2. Регистрация датасета в ClearML Dataset.
3. Обучение модели через ClearML Agent.
4. Логирование гиперпараметров, метрик и артефактов.
5. Публикация лучшей модели в ClearML Model Registry.
6. Деплой модели через ClearML Serving.
7. Streamlit UI, который работает через HTTP endpoint и не загружает модель локально.

---

## Технологии

- Python
- ClearML Server
- ClearML Agent
- ClearML Dataset
- ClearML Model Registry
- ClearML Serving
- scikit-learn
- Streamlit
- Docker / Docker Compose

---

## Структура проекта

```text
.
├── data
│   ├── dga_sample_100k.csv
│   └── raw
│       └── train.csv
├── serving
│   └── preprocess.py
├── src
│   ├── create_dataset.py
│   ├── prepare_sample.py
│   └── train.py
├── ui
│   └── app.py
├── requirements.txt
└── README.md
```

### Назначение файлов

- `src/prepare_sample.py` — создание подвыборки из исходного датасета;
- `src/create_dataset.py` — регистрация датасета в ClearML;
- `src/train.py` — обучение модели, логирование метрик и сохранение модели;
- `serving/preprocess.py` — pre/postprocessing для ClearML Serving;
- `ui/app.py` — Streamlit UI.

---

## Модель

Используется простой baseline для текстовой классификации:

- `TfidfVectorizer(analyzer="char")`;
- `LogisticRegression`;
- `Pipeline` из `scikit-learn`.

Модель сохраняется как единый inference pipeline.

---

## Требования

Перед началом должны быть установлены:

- Git
- Python 3.14
- Docker
- Docker Compose

---

## 1. Клонирование репозитория

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd mlops-dga-classifier
```

---

## 2. Виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Разворачивание ClearML Server

### 3.1. Клонировать официальный репозиторий сервера

```bash
cd ..
git clone https://github.com/clearml/clearml-server.git
cd clearml-server
```

### 3.2. Поднять сервер

```bash
docker compose pull
docker compose up -d
docker compose ps
```

После запуска должны быть доступны:

- Web UI: `http://<HOST_IP>:8080`
- API: `http://<HOST_IP>:8008`
- Fileserver: `http://<HOST_IP>:8081`


### 3.3. Первая настройка
Открыть в браузере:

```text
http://<HOST_IP>:8080
```

Создать пользователя и credentials в UI.

---

## 4. Настройка ClearML SDK

Вернуться в проект:

```bash
cd ../mlops-dga-classifier
source .venv/bin/activate
clearml-init
```

При `clearml-init` нужно вставить конфигурацию, выданную ClearML UI.

### Проверка подключения

```bash
python - <<'PY'
from clearml import Task
task = Task.init(project_name="MLOPS DGA", task_name="sdk_connection_test")
print("Connected OK")
task.close()
PY
```

---

## 5. Подготовка подвыборки датасета

Исходный файл должен лежать в:

```text
data/raw/train.csv
```

Создание подвыборки:

```bash
python src/prepare_sample.py
```

Результат:
- `data/dga_sample_100k.csv`

---

## 6. Регистрация датасета в ClearML

```bash
python src/create_dataset.py
```

После этого датасет должен появиться в разделе `Datasets` в ClearML UI.

Сохраните `dataset_id`, он используется дальше при обучении.

---

## 7. Запуск ClearML Agent

В ClearML UI нужно создать очередь:

```text
students
```

Потом в терминале:

```bash
clearml-agent daemon --queue students
```

Агент должен появиться в UI и слушать очередь `students`.

---

## 8. Обучение модели

### Запуск обучения

```bash
python src/train.py
```

Что делает `train.py`:

- создает ClearML Task;
- отправляет задачу в очередь `students`;
- агент забирает задачу и запускает обучение удаленно;
- загружает датасет из ClearML по `dataset_id`;
- обучает модель;
- логирует:
  - `accuracy`
  - `f1`
  - confusion matrix;
- сохраняет pipeline как модель.

### После обучения при нескольких вариантах гиперпараметров:
В ClearML UI должны быть видны:

- варианты training experiments;
- различия в гиперпараметрах;
- различия в метриках;
- output model.

---

## 9. Публикация лучшей модели в Registry

После сравнения экспериментов нужно:

1. Выбрать лучшую модель.
2. Открыть ее в разделе `Models`.
3. Нажать `Publish` для опубликования в Registry.

Проверка:
- модель видна в `Models`;
- статус `Published`;
- есть теги;
- есть `MODEL URL`.

---

## 10. Разворачивание ClearML Serving

### 10.1. Клонировать официальный репозиторий serving

```bash
cd ..
git clone https://github.com/clearml/clearml-serving.git
cd clearml-serving/docker
```

### 10.2. Создать serving service

```bash
clearml-serving create --name "dga-serving-clean"
```

Сохраните выданный `Serving Service ID`.

### 10.3. Настроить `local.env`

Создайте файл `local.env` в директории `clearml-serving/docker` со значениями:

```env
CLEARML_WEB_HOST=http://<HOST_IP>:8080
CLEARML_API_HOST=http://<HOST_IP>:8008
CLEARML_FILES_HOST=http://<HOST_IP>:8081
CLEARML_API_ACCESS_KEY=<YOUR_ACCESS_KEY>
CLEARML_API_SECRET_KEY=<YOUR_SECRET_KEY>
CLEARML_SERVING_TASK_ID=<SERVING_SERVICE_ID>
CLEARML_EXTRA_PYTHON_PACKAGES=scikit-learn==1.8.0 numpy==2.4.4 joblib==1.5.3 pandas==3.0.2 scipy==1.17.1
```

### 10.4. Поднять serving stack

```bash
docker compose --env-file local.env -f docker-compose.yml up -d --force-recreate
docker compose --env-file local.env -f docker-compose.yml ps
```

Serving поднимается на:

```text
http://<HOST_IP>:8085
```

---

## 11. Добавление endpoint в ClearML Serving

После публикации рабочей модели в Registry нужно взять ее `model_id`.

Далее добавить endpoint:

```bash
clearml-serving --id <SERVING_SERVICE_ID> model add \
  --engine sklearn \
  --endpoint dga-classifier \
  --version 1 \
  --model-id <MODEL_ID> \
  --preprocess "/absolute/path/to/mlops-dga-classifier/serving/preprocess.py"
```

### Проверка конфигурации endpoint

```bash
clearml-serving --id <SERVING_SERVICE_ID> model list
```

---

## 12. Проверка HTTP endpoint

Рабочий endpoint:

```text
http://<HOST_IP>:8085/serve/dga-classifier/1
```

Примеры запросов:

```bash
curl -X POST "http://localhost:8085/serve/dga-classifier/1" \
  -H "Content-Type: application/json" \
  -d '{"domain": "google.com"}'
```

```bash
curl -X POST "http://localhost:8085/serve/dga-classifier/1" \
  -H "Content-Type: application/json" \
  -d '{"domain": "facebook.com"}'
```

```bash
curl -X POST "http://localhost:8085/serve/dga-classifier/1" \
  -H "Content-Type: application/json" \
  -d '{"domain": "xj39qkzpwla.biz"}'
```

Пример ответа:

```json
{"label":0}
```

```json
{"label":1}
```

---

## 13. Запуск Streamlit UI

Запуск:

```bash
cd ../mlops-dga-classifier
source .venv/bin/activate
streamlit run ui/app.py
```

Открыть:

```text
http://localhost:8501
```

### Что умеет UI
- поле ввода домена;
- кнопка `Predict`;
- отображение `label`;
- отображение latency;
- обработка ошибки, если endpoint недоступен.

---

## 14. Полный порядок запуска с нуля

### 14.1. Поднять ClearML Server
```bash
cd ~/.../clearml-server
docker compose up -d
```

### 14.2. Поднять ClearML Serving
```bash
cd ~/.../clearml-serving/docker
docker compose --env-file local.env -f docker-compose.yml up -d
```

### 14.3. Активировать окружение проекта
```bash
cd ~/.../mlops-dga-classifier
source .venv/bin/activate
```

### 14.4. Запустить Agent
```bash
clearml-agent daemon --queue students
```

### 14.5. Подготовить sample
```bash
python src/prepare_sample.py
```

### 14.6. Зарегистрировать датасет
```bash
python src/create_dataset.py
```

### 14.7. Запустить обучение
```bash
python src/train.py
```

### 14.8. Опубликовать лучшую модель
Сделать в ClearML UI вручную через `Models -> Publish`.

### 14.9. Добавить endpoint в serving
```bash
clearml-serving --id <SERVING_SERVICE_ID> model add \
  --engine sklearn \
  --endpoint dga-classifier \
  --version 1 \
  --model-id <MODEL_ID> \
  --preprocess "/absolute/path/to/mlops-dga-classifier/serving/preprocess.py"
```

### 14.10. Проверить endpoint
```bash
curl -X POST "http://localhost:8085/serve/dga-classifier/1" \
  -H "Content-Type: application/json" \
  -d '{"domain": "google.com"}'
```

### 14.11. Запустить UI
```bash
streamlit run ui/app.py
```

---

## 15. Остановка сервисов

### Остановить Agent
Во вкладке с агентом:

```text
Ctrl + C
```

### Остановить Streamlit
Во вкладке со Streamlit:

```text
Ctrl + C
```

### Остановить ClearML Serving
```bash
cd ~/.../clearml-serving/docker
docker compose --env-file local.env -f docker-compose.yml down
```

### Остановить ClearML Server
```bash
cd ~/.../clearml-server
docker compose down
```

---


## 16. Полезные замечания

- `clearml.conf` хранится в домашней директории пользователя и не коммитится;
- для self-hosted запуска важно использовать корректный `<HOST_IP>`;
- если IP машины меняется, нужно обновить:
  - `clearml.conf`
  - `local.env`
  - рекомендуется использовать переменные окружения проекта для `<HOST_IP>`.

---
