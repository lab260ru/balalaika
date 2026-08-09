# Balalaika Cluster Admin

Необязательный управляющий слой для запуска существующего Balalaika pipeline на
4-5 нодах через SSH. Штатный transport - `rsync`; встроенный SSH-stream оставлен
как запасной режим.
Поддерживаются отдельные Docker-контейнеры и запуск в уже поднятом
worker-контейнере, заданный набор GPU каждой ноды и durable
остановка из CLI/панели. Он не изменяет контракт `base.sh`: каждая партиция
по-прежнему запускает обычный pipeline с тем же диапазоном стадий.

Полная схема, подготовка нод, первый тест и эксплуатационные ограничения
описаны в [`docs/cluster_admin.md`](../docs/cluster_admin.md).

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml init
python3 -m cluster_admin --config cluster_admin/config.yaml doctor
python3 -m cluster_admin --config cluster_admin/config.yaml nodes bootstrap
python3 -m cluster_admin --config cluster_admin/config.yaml nodes probe
python3 -m cluster_admin --config cluster_admin/config.yaml run plan pilot-001 --partitions 2
python3 -m cluster_admin --config cluster_admin/config.yaml run start pilot-001
python3 -m cluster_admin --config cluster_admin/config.yaml run finalize pilot-001
# Панельная остановка оставляет ноду в drain; вернуть её в scheduler:
python3 -m cluster_admin --config cluster_admin/config.yaml nodes resume node-01
```

Формат итогового Parquet, WebDataset shard manifest и атомарная публикация
описаны в [`docs/cluster_finalizer.md`](../docs/cluster_finalizer.md).

Панель запускается отдельной командой и по умолчанию доступна только локально:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml serve
```
