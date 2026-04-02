# mlops-dga-classifier

Курсовой проект по MLOps: сервис классификации доменных имен с использованием ClearML.

## Задача
Бинарная классификация строк `domain` -> `label`.

## Стек
- ClearML Server
- ClearML Agent
- ClearML Dataset
- ClearML Model Registry
- ClearML Serving
- Streamlit
- scikit-learn

## План пайплайна
1. Подготовка подвыборки датасета
2. Регистрация датасета в ClearML
3. Обучение модели через ClearML Agent
4. Логирование метрик и артефактов
5. Публикация лучшей модели в Model Registry
6. Деплой через ClearML Serving
7. UI на Streamlit, работающий через HTTP endpoint

## Структура проекта
- `src/create_dataset.py` — регистрация датасета в ClearML
- `src/train.py` — обучение и логирование
- `ui/app.py` — пользовательский интерфейс
- `screenshots/` — скриншоты для сдачи
