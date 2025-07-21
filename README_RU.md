# A Data-Centric Framework for Addressing Phonetic and Prosodic Challenges in Russian Speech Generative Models

Русский синтез речи сталкивается c рядом особенностей: редукция гласных, оглушение согласных, подвижное ударение, омонимия. В данной работе представлен датасет Balalaika — более 2 000 часов студийной русской речи с полными текстовыми аннотациями (включая пунктуацию и ударения). Модели, обученные на Balalaika, заметно превосходят аналоги по задачам синтеза и улучшения речи. 

## Быстрый старт 👟
```bash
git clone https://github.com/mtuciru/balalaika && cd balalaika
bash create_user_env.sh       # cоздаёт виртуальное окружение и устанавливает зависимости
bash use_meta_500h.sh         # можно выбрать 100h / 500h / 1000h / 2000h
```

## Содержание

1. [Предварительные требования](#предварительные-требования)  
2. [Установка](#установка)  
3. [Подготовка данных](#подготовка-данных)  
   - [Быстрая настройка](#быстрая-настройка)  
   - [Загрузка датасета по мете](#загрузка-датасета-по-мете)  
4. [Запуск пайплайна разметки](#запуск-пайплайна-разметки)  
   - [Базовый сценарий (локально)](#базовый-сценарий-локально)  
5. [Конфигурация](#конфигурация)  
6. [Переменные окружения](#переменные-окружения)  
7. [Модели](#модели)  
8. [Ссылка на цитирование](#ссылка-на-цитирование)  
<!-- 9. [Благодарности](#благодарности)   -->

## Предварительные требования

```bash
sudo apt update && sudo apt install -y \
  ffmpeg                 # инструменты для аудио/видео
  python3                # Python
  python3-pip            # менеджер пакетов Pip
  python3-venv           # виртуальные окружения
  python3-dev            # заголовки для сборки wheels
  python-is-python3
wget -qO- https://astral.sh/uv/install.sh | sh
```

## Установка

Склонируйте репозиторий и создайте окружение

```bash
git clone https://github.com/mtuciru/balalaika
cd balalaika

# Используется для скриптов, создающих новую аннотацию или модифицирующих датасет
bash create_dev_env.sh     

# Используется, если надо загрузить готовый датасет
bash create_user_env.sh    
```

## Подготовка данных

### Быстрая настройка

Выберите один из заранее подготовленных объёмов:

* **100 часов**
  ```bash
  bash use_meta_100h.sh
  ```

* **500 часов**
  ```bash
  bash use_meta_500h.sh
  ```

* **1 000 часов**
  ```bash
  bash use_meta_1000h.sh
  ```

* **2 000 часов**
  ```bash
  bash use_meta_2000h.sh
  ```

Метаданные также доступны на [Hugging Face – MTUCI](https://huggingface.co/MTUCI).

### Загрузка датасета по мете

Если у вас уже есть `balalaika.parquet` и `balalaika.pkl`, скопируйте их в корень проекта и запустите:

```bash
bash use_meta.sh
```

## Запуск пайплайна разметки

### Базовый сценарий (локально)

Пайплайн:

1. Скачивает датасеты  
2. Режет аудио на семантические фрагменты  
3. Транскрибирует сегменты  
4. Делает сегментацию по спикерам  
5. Применяет фонемизацию  


```bash
bash base.sh configs/config.yaml
```

Результат сохраняется в `podcasts/result.csv`.

## Конфигурация

Главный файл — `configs/config.yaml`. Ниже кратко описаны ключевые параметры.

### Глобальные

* `podcasts_path` — абсолютный путь к каталогу с подкастами и выводом всех стадий.

### `download`

* `episodes_limit` — максимум эпизодов на плейлист  
* `num_workers` — количество параллельных загрузок  
* `podcasts_urls_file` — путь к `.pkl` со списком ссылок

### `preprocess`

* `duration` — максимальная длина сегмента, сек.  
* `whisper_model` — модель Faster-Whisper  
* `compute_type` — тип вычислений  
* `beam_size` — размер beam-поиска  
* `num_workers` — параллельные процессы

### `separation`

* `nisqa_config` — конфиг NISQA  
* `one_speaker` — загружать только одноголосые записи  
* `num_workers` — процессы

### `transcription`

* `model_name` — CTC или RNN-T  
* `with_timestamps` — добавлять тайм-коды (только CTC)  
* `lm_path` — путь к языковой модели  
* `num_workers` — процессы

### `punctuation`

* `model_name` — RUPunct  
* `num_workers` — процессы

### `accent`

* `model_name` — ruAccent  
* `num_workers` — процессы

### `phonemizer`

* `num_workers` — процессы

### `classification`

* `threshold` — порог уверенности (0.6–0.9)  
* `model_path` — `.pt` файл модели  
* `num_workers` — процессы

### Скрипты запуска

* `*_yaml.sh` — берут все параметры из `config.yaml`  
* `*_args.sh` — жёстко прописанные аргументы внутри скрипта

## Переменные окружения

Создайте `.env`:

```ini
HF_TOKEN=
YANDEX_KEY=
```

* `HF_TOKEN` — нужен для оценки числа спикеров  
* `YANDEX_KEY` — нужен для загрузки подкастов  

## Важные замечания

- Запускайте скрипты **из корня проекта**.  
- Пути в конфиге должны быть **абсолютными**.  
- Стадии пунктуация → акценты  выполняются **поочерёдно**.  
- Необходимы ключи Yandex Music и Hugging Face.

## Модели

```
models/
├── voxblink_resnet/ 
│   └── ...
└── nisqa_s.tar     
```

Поддерживаются:

- [NISQA](https://github.com/deepvk/NISQA-s)  – Оценка качества аудио.
- [GigaAM](https://github.com/salute-developers/GigaAM)  – ASR.
- [ruAccent](https://github.com/Den4ikAI/ruaccent)  – Расстановка ударений.
- [RUPunct](https://huggingface.co/RUPunct/RUPunct_big)  – Пунктуация.
- [VoxBlink ResNet](https://github.com/wenet-e2e/wespeaker)  – Получение эмбеддингов спикеров для кластеризации.
- [TryIPaG2P](https://github.com/NikiPshg/TryIPaG2P)  – Фонемизация.
- [Speaker Diarization](https://github.com/pyannote/pyannote-audio)  – Диаризация.
- [Whisper](https://github.com/SYSTRAN/faster-whisper)  – ASR + сегментация
 

## Ссылка на цитирование

Если вы используете датасет в своей работе, пожалуйста процитируйте нас
```
@misc{borodin2025datacentricframeworkaddressingphonetic,
      title={A Data-Centric Framework for Addressing Phonetic and Prosodic Challenges in Russian Speech Generative Models}, 
      author={Kirill Borodin and Nikita Vasiliev and Vasiliy Kudryavtsev and Maxim Maslov and Mikhail Gorodnichev and Oleg Rogov and Grach Mkrtchian},
      year={2025},
      eprint={2507.13563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.13563}, 
}
```

<!-- ## Благодарности

Спасибо всем разработчикам и контрибьюторам, сделавшим этот проект возможным. -->

## Лицензия

### Датасет Balalaika  
- **CC BY-NC-ND 4.0** – некоммерческое использование, без производных работ, только для научных исследований.  
- Обязательно цитируйте корпус и **не** распространяйте файлы без письменного разрешения.

### Код  
- **CC BY-NC-SA 3.0** – допускается использовать, изменять и распространять материал лишь в академических, некоммерческих целях.  
- Сохраняйте уведомления об авторских правах и лицензии; для коммерческого использования свяжитесь с авторами.

### Сторонние модели и библиотеки  
Помимо вышесказанного, необходимо соблюдать лицензии каждого компонента:

| Компонент          | Лицензия        |
|--------------------|-----------------|
| NISQA-s            | Apache 2.0      |
| GigaAM             | MIT             |
| ruAccent           | CC BY-NC-ND 4.0 |
| RUPunct            | CC BY-NC-ND 4.0 |
| VoxBlink ResNet    | Apache 2.0      |
| TryIPaG2P          | MIT             |
| pyannote-audio     | MIT             |
| Faster-Whisper     | MIT             |