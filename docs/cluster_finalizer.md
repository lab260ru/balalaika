# Финализация multinode run

После того как все партиции run перешли в `SUCCEEDED`, controller может
опубликовать единый результат:

```bash
python -m cluster_admin --config cluster_admin/config.yaml \
  run finalize <run-id>
```

Команда работает только на admin-ноде с уже собранными результатами. Она не
подключается по SSH и не запускает код на worker-нодах.

Перед публикацией finalizer проверяет состояние run, полный набор партиций,
текущие `COMPLETED` attempts, fencing token, config/manifest digest и
partition-level `_SUCCESS`. Любая незавершённая или устаревшая партиция
останавливает финализацию.

Результат находится в:

```text
.balalaika-cluster/results/<run-id>/
  partitions/<partition-id>/...       # неизменённые результаты worker
  dataset/
    balalaika.parquet
    filter_summary.csv                # только если audit был хотя бы в одной партиции
    webdataset/
      shards.jsonl
      part-0000-r000000-s000000.tar
      part-0001-r000001-s000000.tar
    manifest.json
    _SUCCESS
```

`balalaika.parquet` объединяется потоково. Его `filepath` приводится к
переносимому пути относительно корня run:
`partitions/<partition-id>/data/<path>`. Поэтому переносить или архивировать
нужно весь каталог `<run-id>`, а не только `dataset/`.

WebDataset audio не распаковывается и не перепаковывается. Finalizer создаёт
hard link на каждый partition shard и назначает уникальное имя, содержащее
partition и global rank. `shards.jsonl` содержит переносимые пути, размер,
partition, rank и исходный путь. Hard links требуют, чтобы `dataset/` и
partition results находились на одной файловой системе, что гарантируется
стандартной структурой `results/<run-id>`.

Для `filter_summary.csv` берётся последняя запись каждой стадии в каждой
партиции, после чего числовые показатели суммируются. При разных параметрах
стадии параметры сохраняются по partition, а не теряются.

Публикация выполняется через временный каталог и один atomic rename.
`_SUCCESS` записывается последним и содержит SHA-256 `manifest.json`.
Повторный `run finalize` проверяет готовый artifact и возвращает его, но никогда
не перезаписывает. Изменённый manifest, Parquet, audit или shard manifest
считается ошибкой и требует ручного разбора.
