# Пилотная Docker-версия Balalaika

В этом каталоге находится контейнерный слой для запуска на одной ноде в рамках
мультинодовой архитектуры, описанной в
[`docs/multinode_architecture.md`](../docs/multinode_architecture.md). Необязательный
SSH/rsync controller для 4-5 нод находится в [`cluster_admin`](../cluster_admin/)
и описан в [`docs/cluster_admin.md`](../docs/cluster_admin.md). Ручной запуск
этого Docker-слоя и обычный `base.sh` остаются самостоятельными сценариями.

## Требования к хосту

На хосте необходимы только:

- совместимый драйвер NVIDIA;
- Docker Engine;
- NVIDIA Container Toolkit, настроенный для Docker;
- достаточно места на локальном диске для слоёв образа, моделей, кэшей и
  партиции датасета.

Python, пользовательские wheel-пакеты CUDA, ONNX Runtime, TensorRT и FFmpeg
устанавливаются внутри образа. Не запускайте `create_dev_env.sh` в образе.

## Сборка

### CUDA 12.8 (текущий стабильный образ)

```bash
bash docker/build.sh
```

При сборке используется `requirements_dev.cuda128.txt`, после чего последним
переустанавливается `onnxruntime-gpu==1.26.0`. Благодаря этому CPU-зависимость
`onnxruntime`, объявленная пакетом ruaccent, не может перезаписать общие файлы
Python-модуля.

### CUDA 13.0

CUDA 13 собирается отдельной командой и не заменяет образ CUDA 12.8:

```bash
bash docker/build_cuda13.sh
```

В образ копируется текущий checkout репозитория в `/opt/balalaika/app`, а
Python-окружение создаётся в `/opt/balalaika/.venv`. Поэтому после изменения
кода образ нужно пересобрать. CUDA 13 lock использует Python 3.12,
PyTorch 2.11, ONNX Runtime GPU 1.28 и TensorRT 10.16.

Для запуска CUDA 13 нужны NVIDIA Container Toolkit, GPU Turing (compute
capability 7.5) или новее и Linux-драйвер NVIDIA не ниже `580.95.05`.

Проверка на одной карте:

```bash
BALALAIKA_GPU_DEVICES=0 bash docker/run_cuda13.sh smoke
BALALAIKA_GPU_DEVICES=0 bash docker/run_cuda13.sh smoke --tensorrt
```

Запуск пайплайна на нескольких выбранных картах:

```bash
BALALAIKA_GPU_DEVICES=0,1 \
BALALAIKA_HOST_DATA=/absolute/path/to/dataset \
BALALAIKA_HOST_MODELS=/absolute/path/to/models \
BALALAIKA_HOST_CONFIG=/absolute/path/to/config.yaml \
BALALAIKA_HOST_CACHE=/absolute/path/to/cache \
BALALAIKA_HOST_OUTPUT=/absolute/path/to/output \
  bash docker/run_cuda13.sh \
  pipeline --stage 1 --stop_stage 15 --strict
```

`BALALAIKA_GPU_DEVICES` содержит индексы физических GPU хоста. Контейнер видит
только выбранные карты, перенумерованные в `cuda:0..N-1`; существующий код
Balalaika сам поднимает локальные процессы по `torch.cuda.device_count()`.
Датасет, модели, config, cache и output не копируются в image и передаются как
bind mounts. Исходный config остаётся read-only, а entrypoint создаёт его
рабочую контейнерную копию.

Модели и данные времени выполнения исключены через `.dockerignore`. Локальный
каталог `models/` размером 5,3 ГБ монтируется при запуске, а не копируется в слой
образа.

Образ по умолчанию создаёт пользователя с UID/GID `10001`, чтобы не
конфликтовать со стандартным пользователем Ubuntu с UID `1000`. Локальная
обёртка запуска использует UID/GID вызывающего пользователя для доступа к bind
mounts. В Nomad-контуре каталоги ноды будут заранее принадлежать фиксированному
UID/GID образа.

## Проверка работоспособности на GPU 0

```bash
bash docker/run_gpu0.sh smoke
```

Обёртка всегда одновременно использует:

```text
docker run --gpus device=0
CUDA_VISIBLE_DEVICES=0
```

Для проверки должен быть виден ровно один GPU. Тест выполняет матричное
умножение через Torch CUDA, запускает реальную CUDA-сессию ONNX Runtime и
проверяет декодирование и кодирование через TorchCodec/FFmpeg. Профиль ONNX
Runtime подтверждает, что вычислительный узел действительно выполнился на CUDA,
а не только зарегистрировал provider. Необязательная проверка TensorRT таким же
образом подтверждает выполнение небольшого графа через TensorRT:

```bash
bash docker/run_gpu0.sh smoke --tensorrt
```

Эти проверки не запускают ни одной стадии пайплайна.

## Запуск одной стадии пайплайна на GPU 0

```bash
BALALAIKA_HOST_DATA=/absolute/path/to/isolated-partition \
  bash docker/run_gpu0.sh \
  pipeline --stage 4 --stop_stage 4 --strict
```

Исходный YAML монтируется только для чтения. `prepare_config.py` создаёт внутри
контейнера рабочую копию и изменяет только пути, специфичные для контейнера:

- `runtime.venv_path` -> `/opt/balalaika/.venv`;
- `runtime.log_dir` -> `/logs`;
- `runtime.trt_cache_path` -> `/cache/trt`;
- каждый существующий `podcasts_path` -> `/data`, если задана переменная
  `BALALAIKA_HOST_DATA`;
- известные пути к ONNX-моделям -> `/models/<filename>`;
- `export.output_path` -> `/output`;
- пути к кэшу G2P и общему кэшу -> `/cache/balalaika`.

Конфигурация на хосте никогда не изменяется.

Полезные переопределения:

```text
BALALAIKA_IMAGE             тег образа, по умолчанию balalaika:cuda12.8
BALALAIKA_HOST_CONFIG       исходный YAML на хосте
BALALAIKA_HOST_MODELS       набор моделей на хосте
BALALAIKA_HOST_CACHE        корень постоянного локального кэша ноды
BALALAIKA_HOST_DATA         корень изолированной партиции
BALALAIKA_HOST_OUTPUT       каталог результатов партиции
BALALAIKA_HOST_LOGS         постоянные логи, по умолчанию <output>/logs
BALALAIKA_MODEL_MOUNT_MODE  режим models: ro по умолчанию, rw только для prefetch
BALALAIKA_SHM_SIZE          разделяемая память Docker, по умолчанию 8g
BALALAIKA_RUN_UID/GID       UID/GID локального процесса контейнера
BALALAIKA_UID/GID           UID/GID пользователя при сборке, по умолчанию 10001
```

Если `HF_TOKEN` и `YANDEX_KEY` заданы на хосте, они передаются при запуске и
никогда не добавляются в образ. Рабочие процессы промышленного контура должны
использовать заранее загруженный набор моделей с проверенными контрольными
суммами и офлайн-режим Hugging Face.

Для одноразовой предварительной загрузки отсутствующих моделей разрешите запись
явно, затем верните read-only mount для рабочих запусков:

```bash
BALALAIKA_MODEL_MOUNT_MODE=rw BALALAIKA_HOST_DATA=/absolute/path/to/partition \
  bash docker/run_gpu0.sh warmup
```

## Текущие ограничения

- Это образ и обёртка запуска для одной ноды, а не распределённый контур
  управления.
- В образе используется полный lock-файл окружения разработки, включая пакеты
  для тестирования и форматирования. Уменьшенный lock-файл зависимостей времени
  выполнения можно подготовить после стабилизации пилотной версии.
- Базовые образы закреплены по OCI digest, а Python-пакеты — по точной версии.
  Python lock пока не содержит хэши wheel/sdist.
- Upstream RUAccent частично игнорирует свой параметр `workdir` и ожидает
  ресурсы `koziev` внутри установленного пакета. Совместимый shim Balalaika
  перенаправляет все его runtime assets в постоянный `/cache/ruaccent`. Первый
  prefetch требует запись и доступ к Hugging Face; рабочие ноды должны получать
  уже проверенный cache bundle.
- Для стадии 12 ONNX-файл должен присутствовать в смонтированном наборе моделей;
  текущая конфигурация не содержит полных метаданных для его автоматической
  загрузки.
- Для работы в офлайн-режиме ASR и восстановление пунктуации всё ещё зависят от
  постоянной структуры кэшей Hugging Face/onnx-asr.
- Пайплайн должен получать корень изолированной партиции. Не монтируйте один и
  тот же доступный для записи датасет в контейнеры на нескольких нодах.
