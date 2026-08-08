# Balalaika Cluster Admin

Необязательный управляющий слой для запуска существующего Balalaika pipeline на
4-5 Docker-нодах через SSH и rsync. Он не изменяет `base.sh`: внутри каждой
партиции контейнер по-прежнему вызывает `pipeline`, а entrypoint запускает
обычный `base.sh`.

Полная схема, подготовка нод, первый тест и эксплуатационные ограничения
описаны в [`docs/cluster_admin.md`](../docs/cluster_admin.md).

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml init
python3 -m cluster_admin --config cluster_admin/config.yaml doctor
python3 -m cluster_admin --config cluster_admin/config.yaml nodes bootstrap
python3 -m cluster_admin --config cluster_admin/config.yaml nodes probe
python3 -m cluster_admin --config cluster_admin/config.yaml run plan pilot-001 --partitions 2
python3 -m cluster_admin --config cluster_admin/config.yaml run start pilot-001
```

Панель запускается отдельной командой и по умолчанию доступна только локально:

```bash
python3 -m cluster_admin --config cluster_admin/config.yaml serve
```
