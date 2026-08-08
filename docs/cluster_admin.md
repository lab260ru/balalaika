# Управление небольшим Balalaika-кластером

## Статус и границы

`cluster_admin` - дополнительный слой управления для 4-5 Docker-нод. Обычный
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
запустить контейнеры на GPU 0, пережить разрыв SSH, собрать результаты, показать
CLI-статус и read-only панель. Глобальное объединение результатов всех партиций
в один WebDataset/Parquet пока намеренно не выполняется.

## Как это устроено

```text
                         управляющая нода
  исходный dataset  ->  manifest + SQLite + scheduler  ->  панель
          |                         |
          | один rsync одновременно| SSH RPC (JSON)
          v                         v
   +--------------+        +--------------+        +--------------+
   | node-01 SSD  |        | node-02 SSD  |  ...   | node-05 SSD  |
   | partition A  |        | partition B  |        | partition E  |
   | Docker GPU 0 |        | Docker GPU 0 |        | Docker GPU 0 |
   +--------------+        +--------------+        +--------------+
```

Controller хранит только управляющее состояние в SQLite. Аудио через него не
проксируется: `rsync` идёт напрямую с исходного каталога на локальный диск
выбранной ноды. После одного копирования все стадии многократно читают локальный
SSD/NVMe. Общий HDFS, NFS или CephFS в hot path не нужен.

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
topology нод, SSH endpoints, remote paths, image overrides, stages и Docker
`shm_size`. Resume с изменившейся execution-конфигурацией останавливается до
любого RPC или `rsync`; нужно вернуть исходный config либо создать новый run.

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
  locks/gpu0.*
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

На каждой ноде:

- Linux, OpenSSH server и `rsync`;
- Docker Engine и NVIDIA Container Toolkit;
- NVIDIA driver, видящий GPU с локальным индексом 0;
- уже собранный или загруженный одинаковый Balalaika image;
- локальный каталог models и постоянный cache;
- место под как минимум одну партицию, её рабочую копию и результат.

Установка host-зависимостей на Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server rsync docker.io
```

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

### 2. Image, models и cache

На каждой ноде должен разрешаться один и тот же image reference:

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
    work_root: /var/lib/balalaika
    models_root: /var/lib/balalaika/models
    cache_root: /var/lib/balalaika/cache
```

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
# Без partition ID команда отменяет queued/running партиции всего run:
python3 -m cluster_admin --config cluster_admin/config.yaml run cancel pilot-001
```

Read-only панель:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml \
  serve --bind 127.0.0.1 --port 8765
```

Открыть её с рабочего компьютера безопаснее через tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 controller-host
```

URL: `http://127.0.0.1:8765`. Панель показывает доступность нод, GPU 0, свободный
диск, текущую партицию, stage, retries и ошибки. Progress внутри контейнера пока
stage-level: существующий pipeline не публикует единый точный счётчик файлов для
каждой стадии.

## Надёжность

- SQLite работает в WAL + `synchronous=FULL`; scheduler защищён singleton lock.
- На одной ноде может быть только один активный attempt, всегда на GPU 0.
- Каждый retry получает новый `attempt_id` и случайный fencing token.
- Поздний результат старого attempt не может обновить текущую партицию.
- Отмена run или отдельной партиции сначала сохраняется как desired state в
  SQLite, поэтому потерянный SSH-ответ не может вернуть её в очередь. Для
  аварийной отмены сохраняется исходный SSH endpoint attempt: изменение других
  параметров YAML не мешает остановке, а при смене host/user/port/work_root
  controller требует сначала вернуть прежний endpoint.
- Scheduler reconciles активные attempts всех run перед новой выдачей: job,
  завершившийся во время рестарта controller, не блокирует следующую очередь.
- До claim нода обязана подтвердить совместимую версию runner, Docker daemon,
  GPU 0, image, models и достаточное свободное место. Неисправная нода не
  получает даже входную партицию.
- Source manifest проверяется по path/size/mtime непосредственно до и после
  `rsync`; symlink, special file, traversal и изменение source останавливают job.
- Worker принимает данные в `data.partial`, проверяет manifest и только потом
  делает atomic rename в `data`.
- Container имеет детерминированное имя и запускается detached без `--rm`.
- `_SUCCESS` создаётся последним. Controller принимает result только при
  совпадении attempt, token, manifest SHA256, config SHA256 и exit code.
- Один входной transfer одновременно защищает исходный HDD от пяти параллельных
  полных чтений. Уже запущенные ноды в это время считают локально.
- При исчерпании retries одной партиции остальные jobs не прерываются.

## Текущие ограничения MVP

1. Scheduler пока foreground-процесс, а панель read-only. Для постоянной службы
   нужен отдельный systemd unit, но worker containers уже переживают его restart.
2. Stage 0 лучше выполнять до partitioning на controller. Cluster launcher
   рассчитан на уже имеющиеся immutable input files.
3. Stage 13-15 выполняются отдельно внутри каждой партиции. Dataset-level
   finalizer для объединённого Parquet, audit и списка WebDataset shards будет
   следующим слоем после базового пилота.
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

## Проверки разработчика

```bash
python3 -m pytest -q \
  tests/test_cluster_partitioner.py \
  tests/test_cluster_admin_core.py \
  tests/test_cluster_scheduler.py
python3 -m py_compile cluster_admin/*.py
flake8 --extend-ignore=E501 cluster_admin tests/test_cluster_admin_core.py \
  tests/test_cluster_partitioner.py tests/test_cluster_scheduler.py
```

Чтобы полностью отключить мультинодовый слой, остановите scheduler и продолжайте
запускать `base.sh` напрямую. Никакого rollback основного pipeline для этого не
требуется.
