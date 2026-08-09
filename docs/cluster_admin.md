# Управление небольшим Balalaika-кластером

## Статус и границы

`cluster_admin` - дополнительный слой управления для 4-5 worker-нод. Нода может
запускать отдельный Docker-контейнер (`runtime: docker`) или выполнять pipeline
внутри уже поднятого worker-контейнера (`runtime: direct`). Обычный
односерверный запуск остаётся без изменений:

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 15
```

Новый код не импортируется из `base.sh` и не меняет реализацию стадий. На
worker-ноде controller запускает тот же образ с командой:

```text
pipeline --stage <start> --stop_stage <stop> --strict
```

Docker entrypoint этого образа, как и раньше, вызывает `base.sh`. Поэтому
мультинодовый режим можно проверять отдельно и просто не использовать при
любых проблемах.

Это первый рабочий MVP. Он умеет разбить dataset, безопасно доставить части,
использовать заданный набор GPU каждой ноды, пережить разрыв SSH, остановить
задачу из CLI или панели, собрать результаты и показать статус. После успешного
run отдельный finalizer публикует общий Parquet, aggregate audit и manifest
глобально уникальных WebDataset shards.

## Как это устроено

```text
                         управляющая нода
  исходный dataset  ->  manifest + SQLite + scheduler  ->  панель
          |                         |
          | один SSH transfer      | SSH RPC (JSON)
          v                         v
   +--------------+        +--------------+        +--------------+
   | node-01 SSD  |        | node-02 SSD  |  ...   | node-05 SSD  |
   | partition A  |        | partition B  |        | partition E  |
   | GPUs 0,1     |        | GPUs 2,3     |        | GPUs 0,1,2,3 |
   +--------------+        +--------------+        +--------------+
```

Controller хранит только управляющее состояние в SQLite. Он передаёт файлы
напрямую из исходного каталога на локальный диск выбранной ноды по SSH: через
`rsync`, если он есть с обеих сторон, либо через встроенный проверяемый stream
protocol. После одного копирования все стадии многократно читают локальный
SSD/NVMe. Общий HDFS, NFS или CephFS в hot path не нужен.

Управляющая машина и worker - независимые роли. Наличие controller не даёт ему
GPU-работу автоматически. Если admin-машина также должна считать, её явно
добавляют в `nodes` с собственным SSH endpoint, GPU и worker paths; scheduler
обращается к ней по тому же протоколу, что и к остальным. Если её не добавить,
она только планирует, передаёт partitions, собирает результаты и обслуживает
панель.

В текущем варианте полный исходный dataset должен быть доступен на диске
управляющей ноды в момент планирования и первичной передачи. Это соответствует
сценарию с текущим `/mnt/hdd_6tb_1/youtube_data_incoming`. Позже источник можно
заменить на immutable object storage или peer-to-peer staging, не меняя
контейнерный worker protocol.

### Единица работы

Для текущего layout `<date>/<recording-id>/...` используется `group_depth: 2`.
Все audio и sidecar-файлы одной записи всегда попадают в одну партицию. Размер
работы считается по `total_duration` из `balalaika.parquet`, а при отсутствии
duration - по размеру audio. Детерминированный LPT-алгоритм сначала раскладывает
самые тяжёлые записи в наименее загруженные партиции.

Обычно создаётся 4 партиции на ноду. Ноды не получают заранее жёстко заданную
четверть dataset: освободившаяся нода берёт следующую самую тяжёлую партицию.
Это лучше выравнивает записи разной длительности.

Если в корне уже есть `balalaika.parquet`, команда `run plan` потоково делит его
по тем же партициям и переписывает `filepath` в переносимый вид `/data/...`.
Snapshot pipeline config также сохраняется рядом с manifest и больше не зависит
от последующего редактирования исходного YAML.

При планировании controller также сохраняет SHA256 execution-конфигурации:
topology нод, SSH endpoints, runtime, GPU, remote paths, image overrides, stages
и Docker `shm_size`. Resume с изменившейся execution-конфигурацией
останавливается до любого RPC или transfer; нужно вернуть исходный config либо
создать новый run.

## Что хранится где

На controller:

```text
.balalaika-cluster/
  controller.sqlite3
  controller.lock
  manifests/<run-id>/
    run.manifest.json
    pipeline.config.yaml
    part-0000.manifest.json
    part-0000.files0
    part-0000.balalaika.parquet
  results/<run-id>/partitions/<partition-id>/
```

На каждой worker-ноде:

```text
/var/lib/balalaika/
  bin/balalaika-node-runner.py
  models/                         # weights, один раз на ноду
  cache/                          # HF/ruAccent/TRT cache, локальный
  locks/gpu0.*                    # legacy filename; reserves configured GPU set
  runs/<run-id>/partitions/<partition-id>/attempt-.../
    control/{job.json,manifest.json,config.yaml}
    data/                         # отдельная writable копия партиции
    logs/
    output/
    result.json
    _SUCCESS
```

Исходный dataset controller не изменяется. Деструктивные filter stages и
denoising работают только с attempt-local копией. Результат возвращается в
`.balalaika-cluster/results`, а не накладывается поверх исходника.

## Требования

На controller:

- Python 3.10+;
- PyYAML и PyArrow из окружения Balalaika;
- OpenSSH client;
- `rsync` 3.1+;
- исходный dataset и достаточно места для metadata/принятых результатов.

На каждой ноде независимо от runtime:

- Linux, OpenSSH server, Python 3 и `rsync` 3.1+;
- NVIDIA runtime, видящий все GPU из `gpu_devices`;
- локальный каталог models и постоянный cache;
- место под как минимум одну партицию, её рабочую копию и результат.

Для `runtime: docker` дополнительно нужны Docker Engine, NVIDIA Container
Toolkit и уже собранный или загруженный Balalaika image. Для `runtime: direct`
SSH server, node runner, pipeline, venv, GPU и рабочие каталоги должны
находиться **в одном уже поднятом контейнере**, то есть в одном
filesystem/PID/CUDA namespace. Direct-режим не требует и не использует Docker
socket. Текущий базовый `Dockerfile` не устанавливает `sshd`, поэтому для direct
используется внешний worker image/контейнер, где SSH уже настроен, либо
производный образ с `sshd` и `rsync`. SSH должен приводить непосредственно в этот
контейнер, а не на хост с недоступным runner.

Установка host-зависимостей на Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server rsync docker.io
```

Первый пилот использует только `transfer: rsync`. Встроенный stream transport
остаётся запасным режимом для будущих окружений без `rsync`; он не участвует в
штатном запуске.

NVIDIA Container Toolkit устанавливается по официальной инструкции NVIDIA, а
затем проверяется обычным Docker smoke test из этого репозитория.

## Подготовка нод

### 1. Отдельный пользователь и каталоги

На каждой ноде выполните с нужным именем пользователя:

```bash
sudo useradd --create-home --shell /bin/bash balalaika 2>/dev/null || true
sudo usermod -aG docker balalaika
sudo install -d -o balalaika -g balalaika -m 0700 \
  /var/lib/balalaika \
  /var/lib/balalaika/models \
  /var/lib/balalaika/cache
```

Членство в группе `docker` практически эквивалентно root-доступу. Не используйте
для cluster key общий пользовательский аккаунт и не принимайте недоверенные
pipeline config/image.

### 2. Runtime, image, models и cache

На каждой Docker-ноде должен разрешаться один и тот же image reference:

```bash
docker image inspect balalaika:cuda12.8 --format '{{.Id}}'
```

Для воспроизводимого запуска лучше один раз собрать image, передать его как OCI
archive или запушить в registry и использовать digest `repo@sha256:...`. Равный
tag сам по себе не доказывает, что слои одинаковы.

Weights не копируются с каждой партицией. Их один раз помещают в
`/var/lib/balalaika/models`, cache - в `/var/lib/balalaika/cache`; каждый
контейнер читает их локально. Модель всё равно отдельно загружается процессом
стадии в RAM/VRAM, как в обычном `base.sh`.

Проверьте image на каждой ноде до подключения controller:

```bash
BALALAIKA_HOST_MODELS=/var/lib/balalaika/models \
BALALAIKA_HOST_CACHE=/var/lib/balalaika/cache \
  bash docker/run_gpu0.sh smoke
```

### 3. SSH key и pinned host keys

Создайте отдельный ключ на controller и добавьте public key пользователю
`balalaika` на нодах:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/balalaika_cluster -C balalaika-controller
ssh-copy-id -i ~/.ssh/balalaika_cluster.pub balalaika@10.10.0.11
```

Host key нельзя принимать через `StrictHostKeyChecking=no`. Получите fingerprint
через консоль ноды или доверенный канал, сравните его, и только затем добавьте
ключ в отдельный файл:

```bash
ssh-keyscan -H 10.10.0.11 > ~/.ssh/balalaika_known_hosts.candidate
ssh-keygen -lf ~/.ssh/balalaika_known_hosts.candidate
# Сравнить fingerprint, затем:
cat ~/.ssh/balalaika_known_hosts.candidate >> ~/.ssh/balalaika_known_hosts
chmod 600 ~/.ssh/balalaika_known_hosts ~/.ssh/balalaika_cluster
```

Controller запускает OpenSSH с `BatchMode=yes`, `StrictHostKeyChecking=yes` и
`-F /dev/null`. Значения run/partition/stage не вставляются в remote command:
runner получает JSON через stdin и поддерживает только фиксированный список RPC.

## Конфигурация controller

Создайте шаблон:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml init
```

Отредактируйте созданный `cluster_admin/config.yaml`. Полный пример находится в
[`cluster_admin/config.example.yaml`](../cluster_admin/config.example.yaml).
Минимальные важные поля:

```yaml
controller:
  source_root: /mnt/hdd_6tb_1/youtube_data_incoming
  state_dir: ../.balalaika-cluster

pipeline:
  config: ../configs/config.yaml
  image: balalaika:cuda12.8
  stage_start: "1"
  stage_stop: "15"

partitioning:
  partitions_per_node: 4
  group_depth: 2

ssh:
  identity_file: ~/.ssh/balalaika_cluster
  known_hosts: ~/.ssh/balalaika_known_hosts

nodes:
  - id: node-01
    host: 10.10.0.11
    user: balalaika
    runtime: docker
    transfer: rsync
    gpu_devices: [0, 1]
    max_gpu_memory_used_mib: 1024
    max_gpu_utilization_percent: 20
    work_root: /var/lib/balalaika
    pipeline_root: /opt/balalaika/app
    venv_path: /opt/balalaika/.venv
    models_root: /var/lib/balalaika/models
    cache_root: /var/lib/balalaika/cache
```

Если SSH приводит сразу внутрь заранее созданного worker-контейнера, нода
описывается так:

```yaml
  - id: node-02
    host: 10.10.0.12
    user: balalaika
    runtime: direct
    transfer: rsync
    gpu_devices: [0, 1]
    max_gpu_memory_used_mib: 1024
    max_gpu_utilization_percent: 20
    work_root: /worker/balalaika
    pipeline_root: /workspace/balalaika
    venv_path: /workspace/.venv
    models_root: /worker/models
    cache_root: /worker/cache
```

`gpu_devices` - индексы, видимые `nvidia-smi` и PyTorch именно в namespace
node runner. Внешний контейнер может получить физические карты 4 и 7, но внутри
увидеть их как 0 и 1; тогда здесь указывается `[0, 1]`. Probe разрешает индексы
в стабильные GPU UUID и фиксирует их в attempt. Дочерний pipeline получает
компактный набор `cuda:0..N-1`, поэтому существующий multi-GPU код Balalaika сам
поднимает процессы на всех выбранных картах.

`transfer` принимает `rsync`, `auto` или `stream`; значение по умолчанию и для
первого пилота - `rsync`. Запасной stream принимает только файлы из immutable
manifest и отклоняет traversal, symlink, special files, дубликаты и несовпадение
size/SHA256.

Перед claim и ещё раз непосредственно перед новым start controller проверяет
`memory.used` и `utilization.gpu` через `nvidia-smi`. Если любая выбранная карта
превышает `max_gpu_memory_used_mib` или `max_gpu_utilization_percent`, нода
остаётся `DEGRADED` и данные на неё не копируются. Direct probe импортирует Torch
из указанной venv для проверки CUDA build, но не вызывает CUDA device API и не
создаёт вычислительный context.

Один attempt пока резервирует всю ноду и весь перечисленный набор GPU. Это
предотвращает случайный запуск двух партиций на одних картах. Разбиение одной
ноды на несколько независимых GPU slots будет отдельным расширением.

Референс для уже существующего контейнера вида `lab`:

```yaml
  - id: lab
    host: 100.64.0.3
    port: 23
    user: nikita
    runtime: direct
    transfer: rsync
    # Указывать только фактически зарезервированные и исправные карты.
    gpu_devices: [0]
    max_gpu_memory_used_mib: 1024
    max_gpu_utilization_percent: 20
    pipeline_root: /home/nikita/balalaika_claude/balalaika
    venv_path: /home/nikita/balalaika/.dev_venv
    models_root: /home/nikita/balalaika/models
    # Не размещать work/cache на почти полном /home mount.
    work_root: /mnt/disk_1tb/balalaika-cluster
    cache_root: /mnt/disk_1tb/balalaika-cache
```

Это именно container-local пути: controller не пытается найти их на физическом
хосте. `nodes bootstrap` также не делает `git pull` и не изменяет dirty checkout.
Standalone runner и renderer config устанавливаются в `<work_root>/bin`, а из
`pipeline_root` требуется `base.sh`. Поэтому управляющий слой обновляется
отдельно от пользовательских изменений pipeline.

Read-only проверка этого примера подтвердила ожидаемую границу: SSH попадает в
Docker container, Docker socket отсутствует, `/dev/shm` равен 24 ГБ, venv
использует Python 3.12/Torch CUDA 12.8, models доступны локально. При этом
`/home/nikita` почти заполнен, поэтому attempt data/cache должны идти на отдельный
writable mount. Одна из проброшенных карт в момент проверки возвращала NVML
`Unknown Error`; `nodes probe` обязан отклонить такую конфигурацию до копирования
данных. Никакие файлы, процессы или GPU-задачи на референсной ноде проверкой не
изменялись.

В direct-режиме controller не запускает исходный YAML напрямую. Node runner
сначала создаёт attempt-local `control/runtime.config.yaml` через
`docker/prepare_config.py`: data/log/output указывают в текущий attempt, а
models/cache/venv - в пути этой ноды. Затем supervisor запускает:

```text
<venv>/bin/python .../prepare_config.py --input config.yaml --output runtime.config.yaml
bash <pipeline_root>/base.sh --config_path runtime.config.yaml \
  --stage <start> --stop_stage <stop> --strict
```

Supervisor фиксирует PID, process group, boot ID, `/proc` start time, GPU UUID и
fencing token, пишет exit marker атомарно и продолжает работу после разрыва SSH.
Остановка посылает всей process group `SIGINT`, затем при зависании `SIGTERM` и
`SIGKILL`. Поэтому кнопка панели останавливает и `base.sh`, и созданные им
локальные GPU workers, а не только родительский shell.

`exclude_top_level` по умолчанию исключает текущие `.balalaika_work`,
`balalaika_analysis`, `filter_report.md` и `filter_summary.csv`. В партиции
попадают только группы, содержащие расширение из `audio_extensions`, но вместе
с audio передаются все sidecar-файлы этой группы.

## Первый безопасный тест

Не начинайте проверку с 490 ГБ. Создайте отдельный небольшой source root с
несколькими целыми каталогами записей, укажите его в cluster config и ограничьте
диапазон одной недеструктивной стадией. Для dataset без подготовленного state
начинайте со стадии 1 и используйте `--no-split-state`.

Проверка controller:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml doctor
```

Установка/обновление standalone runner и probe всех нод:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml nodes bootstrap
python3 -m cluster_admin --config cluster_admin/config.yaml nodes probe
python3 -m cluster_admin --config cluster_admin/config.yaml nodes list
```

План на две партиции:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml \
  run plan pilot-001 --partitions 2 --no-split-state
```

Перед запуском проверьте manifests и свободный диск, затем запустите scheduler в
`tmux` или отдельном systemd scope. Команда остаётся foreground, пока run не
завершён, и последовательно выдаёт новые партиции освободившимся нодам:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml run start pilot-001
```

Один цикл reconcile без постоянного процесса:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml \
  run start pilot-001 --once
```

Если controller был перезапущен, снова запустите обычный `run start` с тем же
run ID. Именованный detached container продолжает работать после разрыва SSH;
controller прочитает его фактическое состояние перед новой выдачей.

## CLI и панель

Текстовый статус:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml status pilot-001
python3 -m cluster_admin --config cluster_admin/config.yaml status pilot-001 --watch
python3 -m cluster_admin --config cluster_admin/config.yaml status pilot-001 --json
python3 -m cluster_admin --config cluster_admin/config.yaml partitions pilot-001
python3 -m cluster_admin --config cluster_admin/config.yaml run cancel pilot-001 part-0003
# Запретить новые назначения на ноду / вернуть её в scheduler:
python3 -m cluster_admin --config cluster_admin/config.yaml nodes drain node-01
python3 -m cluster_admin --config cluster_admin/config.yaml nodes resume node-01
# Без partition ID команда отменяет queued/running партиции всего run:
python3 -m cluster_admin --config cluster_admin/config.yaml run cancel pilot-001
```

Локальная панель:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml \
  serve --bind 127.0.0.1 --port 8765
```

Открыть её с рабочего компьютера безопаснее через tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 controller-host
```

URL: `http://127.0.0.1:8765`. Панель показывает доступность нод, выбранные GPU,
свободный диск, текущую партицию, stage, retries и ошибки. Progress внутри
worker пока stage-level: существующий pipeline не публикует единый точный
счётчик файлов для каждой стадии.

У занятой ноды есть кнопка `Остановить`. После подтверждения controller сначала
durable-переводит ноду в `drained`, затем отменяет показанную на карточке
партицию. Поэтому scheduler не выдаст этой ноде следующую работу сразу после
остановки. Контракт запроса привязан к run, partition и node ID, поэтому
устаревшая карточка не остановит уже выданную следующую работу. Намерения drain
и cancel сохраняются в SQLite даже при потере SSH-ответа. Вернуть ноду в очередь
можно только явно через `nodes resume <node-id>` или соответствующий API.
Изменяющие POST защищены одноразовым для процесса CSRF-токеном и проверкой
same-origin; оставляйте panel на `127.0.0.1` и открывайте её через SSH tunnel.
Server программно отклоняет `--bind 0.0.0.0` и любой другой non-loopback адрес;
публичный bind потребует отдельной аутентификации и в MVP не поддерживается.

## Надёжность

- SQLite работает в WAL + `synchronous=FULL`; scheduler защищён singleton lock.
- На одной ноде может быть только один активный attempt; он получает весь
  настроенный ordered-набор `gpu_devices`.
- `drained` хранится в SQLite отдельно от YAML `enabled`: scheduler продолжает
  наблюдать текущий attempt, но атомарно запрещает любые новые назначения до
  явного `nodes resume`.
- Каждый retry получает новый `attempt_id` и случайный fencing token.
- Поздний результат старого attempt не может обновить текущую партицию.
- Отмена run или отдельной партиции сначала сохраняется как desired state в
  SQLite, поэтому потерянный SSH-ответ не может вернуть её в очередь. Для
  аварийной отмены сохраняется исходный SSH endpoint attempt: изменение других
  параметров YAML не мешает остановке, а при смене host/user/port/work_root
  controller требует сначала вернуть прежний endpoint.
- Scheduler reconciles активные attempts всех run перед новой выдачей: job,
  завершившийся во время рестарта controller, не блокирует следующую очередь.
- До claim нода обязана подтвердить совместимую версию runner, все выбранные
  GPU, runtime-specific зависимости, models и достаточное свободное место.
  Неисправная нода не
  получает даже входную партицию.
- Source manifest проверяется по path/size/mtime непосредственно до и после
  transfer; symlink, special file, traversal и изменение source останавливают job.
- Worker принимает данные в `data.partial`, проверяет manifest и только потом
  делает atomic rename в `data`.
- Container имеет детерминированное имя и запускается detached без `--rm`.
- `_SUCCESS` создаётся последним. Controller принимает result только при
  совпадении attempt, token, manifest SHA256, config SHA256 и exit code.
- Один входной transfer одновременно защищает исходный HDD от пяти параллельных
  полных чтений. Уже запущенные ноды в это время считают локально.
- При исчерпании retries одной партиции остальные jobs не прерываются.

## Текущие ограничения MVP

1. Scheduler пока foreground-процесс. Для постоянной службы
   нужен отдельный systemd unit, но worker containers уже переживают его restart.
2. Stage 0 лучше выполнять до partitioning на controller. Cluster launcher
   рассчитан на уже имеющиеся immutable input files.
3. Stage 13-15 выполняются отдельно внутри каждой партиции. После успешного run
   команда `run finalize <run-id>` атомарно публикует объединённый Parquet,
   aggregate audit и WebDataset shards с partition/global-rank namespace.
   Формат описан в [`cluster_finalizer.md`](cluster_finalizer.md).
4. Results сохраняются по partition и не накладываются на source. Автоматическая
   очистка remote attempts пока не включена, чтобы не удалить данные до ручной
   проверки.
5. Автоматические retries покрывают завершение container с ошибкой. Недоступная
   нода остаётся `UNKNOWN/OFFLINE`, пока связь не восстановится; controller не
   запускает дубликат вслепую.
6. Для полностью воспроизводимых retries нужно отдельно детерминировать случайный
   crop antispoofing. Это ограничение существующей стадии, а не launcher.
7. Forced-command/`rrsync` ключи и object-storage staging оставлены как hardening
   следующей версии. Текущий MVP рассчитан на доверенный закрытый кластер 4-5 нод.
8. `runtime: direct` не может изнутри изменить `/dev/shm`, Linux capabilities
   или readonly mounts. Внешний worker-контейнер должен быть создан non-root,
   без Docker socket и host PID namespace, с readonly pipeline/models,
   persistent work/cache и достаточным `--shm-size`. Direct job переживает
   разрыв SSH, но не уничтожение самого внешнего контейнера.
9. Встроенный stream transport публикует каждый файл атомарно, но не продолжает
   оборванный файл с середины. Для очень больших исходников используйте `rsync`
   либо учитывайте, что retry повторно отправит текущую партицию.

## Проверки разработчика

```bash
python3 -m pytest -q \
  tests/test_cluster_partitioner.py \
  tests/test_cluster_admin_core.py \
  tests/test_cluster_finalizer.py \
  tests/test_cluster_scheduler.py \
  tests/test_cluster_server.py \
  tests/test_cluster_stream_transfer.py
python3 -m py_compile cluster_admin/*.py
flake8 --extend-ignore=E501 cluster_admin tests/test_cluster_admin_core.py \
  tests/test_cluster_partitioner.py tests/test_cluster_scheduler.py
```

Чтобы полностью отключить мультинодовый слой, остановите scheduler и продолжайте
запускать `base.sh` напрямую. Никакого rollback основного pipeline для этого не
требуется.
