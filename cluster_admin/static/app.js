(function () {
  "use strict";

  const API_URL = "/api/v1/overview";
  const DEFAULT_POLL_INTERVAL_MS = 5000;
  const TERMINAL_PARTITION_STATES = new Set([
    "completed",
    "failed",
    "cancelled",
  ]);

  const STATUS = {
    planned: { label: "Запланировано", tone: "planned", order: 70 },
    queued: { label: "В очереди", tone: "queued", order: 60 },
    ready: { label: "Готово", tone: "ready", order: 50 },
    transferring: { label: "Передача", tone: "transferring", order: 30 },
    running: { label: "В работе", tone: "running", order: 10 },
    verifying: { label: "Проверка", tone: "verifying", order: 20 },
    collecting: { label: "Сбор", tone: "collecting", order: 25 },
    completed: { label: "Завершено", tone: "completed", order: 90 },
    failed: { label: "Ошибка", tone: "failed", order: 0 },
    cancelled: { label: "Отменено", tone: "cancelled", order: 80 },
    online: { label: "Доступна", tone: "online", order: 10 },
    busy: { label: "Занята", tone: "busy", order: 10 },
    draining: { label: "Завершает работу", tone: "draining", order: 30 },
    drained: { label: "Приостановлена", tone: "drained", order: 35 },
    offline: { label: "Недоступна", tone: "offline", order: 0 },
    success: { label: "Завершено", tone: "success", order: 90 },
    warning: { label: "Предупреждение", tone: "warning", order: 20 },
    error: { label: "Ошибка", tone: "error", order: 0 },
    unknown: { label: "Неизвестно", tone: "unknown", order: 100 },
  };

  const STATE_ALIASES = {
    pending: "queued",
    waiting: "queued",
    assigned: "ready",
    staged: "ready",
    staging: "transferring",
    starting: "running",
    uploading: "collecting",
    committed: "completed",
    complete: "completed",
    succeeded: "completed",
    done: "completed",
    canceled: "cancelled",
    failure: "failed",
    unhealthy: "error",
    active: "running",
    idle: "online",
  };

  const state = {
    overview: null,
    lastSuccessAt: null,
    loading: false,
    request: null,
    pollTimer: null,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    csrfToken: "",
    stoppingPartitions: new Set(),
  };

  const elements = {};

  document.addEventListener("DOMContentLoaded", initialize);

  function initialize() {
    [
      "alert",
      "connectionDot",
      "connectionLabel",
      "updatedAt",
      "refreshButton",
      "runTitle",
      "runState",
      "configDigest",
      "imageDigest",
      "runProgress",
      "partitionProgress",
      "audioProgress",
      "audioTotal",
      "onlineNodes",
      "runningJobs",
      "failedJobs",
      "elapsedTime",
      "startedAt",
      "overallProgressBar",
      "nodeCount",
      "nodesGrid",
      "visiblePartitionCount",
      "searchInput",
      "stateFilter",
      "nodeFilter",
      "partitionsTableBody",
    ].forEach((id) => {
      elements[id] = document.getElementById(id);
    });

    elements.refreshButton.addEventListener("click", () =>
      refreshOverview(true),
    );
    elements.searchInput.addEventListener("input", renderPartitions);
    elements.stateFilter.addEventListener("change", renderPartitions);
    elements.nodeFilter.addEventListener("change", renderPartitions);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    refreshOverview(false);
    state.pollTimer = window.setInterval(() => {
      if (!document.hidden) {
        refreshOverview(false);
      }
    }, state.pollIntervalMs);
  }

  function handleVisibilityChange() {
    if (!document.hidden && state.lastSuccessAt) {
      const ageMs = Date.now() - state.lastSuccessAt.getTime();
      if (ageMs >= state.pollIntervalMs) {
        refreshOverview(false);
      }
    }
  }

  async function refreshOverview(manual) {
    if (state.loading) {
      if (!manual) return;
      state.request?.abort();
    }

    state.loading = true;
    state.request = new AbortController();
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
        credentials: "same-origin",
        signal: state.request.signal,
      });

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status} ${response.statusText}`.trim(),
        );
      }

      const payload = await response.json();
      state.csrfToken = stringValue(asObject(payload.controller).csrf_token);
      state.overview = normalizeOverview(payload);
      state.lastSuccessAt = new Date();
      hideAlert();
      renderOverview();
      setConnection("online", controllerLabel(state.overview.controller));
    } catch (error) {
      if (error.name === "AbortError") return;
      setConnection("offline", "Нет связи");
      showAlert(
        state.overview
          ? `Не удалось обновить данные: ${error.message}. Показан последний полученный снимок.`
          : `Не удалось получить состояние кластера: ${error.message}.`,
      );
    } finally {
      state.loading = false;
      state.request = null;
      setLoading(false);
      renderUpdatedAt();
    }
  }

  function normalizeOverview(payload) {
    const raw = asObject(payload);
    const nodesRaw = arrayFrom(raw.nodes || raw.workers);
    const partitionsRaw = arrayFrom(raw.partitions || raw.jobs || raw.tasks);
    const runs = arrayFrom(raw.runs);
    const activeRunRaw = asObject(
      raw.active_run ||
        raw.run ||
        runs.find((run) => isActiveRun(run)) ||
        runs[0],
    );

    const nodes = nodesRaw.map(normalizeNode);
    const partitions = partitionsRaw.map(normalizePartition);
    const activeRun = normalizeRun(activeRunRaw, partitions);

    return {
      generatedAt: dateOrNull(raw.generated_at || raw.timestamp),
      controller: asObject(raw.controller),
      activeRun,
      nodes,
      partitions,
    };
  }

  function normalizeRun(raw, partitions) {
    const summary = asObject(raw.summary || raw.stats || raw.progress);
    const relevant = raw.id
      ? partitions.filter(
          (partition) => !partition.runId || partition.runId === raw.id,
        )
      : partitions;
    const completed = relevant.filter(
      (partition) => partition.state === "completed",
    ).length;
    const failed = relevant.filter(
      (partition) => partition.state === "failed",
    ).length;
    const running = relevant.filter((partition) =>
      ["transferring", "running", "verifying", "collecting"].includes(
        partition.state,
      ),
    ).length;
    const totalAudio = sum(relevant.map((partition) => partition.audioSeconds));
    const processedAudio = sum(
      relevant.map((partition) => partition.processedAudioSeconds),
    );
    const total = numberOrNull(
      firstDefined(
        summary.partitions_total,
        summary.total_partitions,
        raw.partitions_total,
      ),
    );
    const progress = normalizePercent(
      firstDefined(
        summary.progress_percent,
        raw.progress_percent,
        raw.progress,
      ),
    );

    return {
      id: stringValue(raw.id || raw.run_id || raw.name),
      name: stringValue(raw.name || raw.label || raw.id || raw.run_id),
      state: canonicalState(raw.state || raw.status),
      startedAt: dateOrNull(raw.started_at || raw.created_at),
      finishedAt: dateOrNull(raw.finished_at || raw.completed_at),
      configDigest: stringValue(raw.config_digest || raw.config_sha256),
      imageDigest: stringValue(raw.image_digest || raw.image),
      partitionsTotal: total ?? relevant.length,
      partitionsCompleted:
        numberOrNull(
          firstDefined(
            summary.partitions_completed,
            summary.completed_partitions,
            raw.partitions_completed,
          ),
        ) ?? completed,
      partitionsRunning:
        numberOrNull(
          firstDefined(summary.partitions_running, raw.partitions_running),
        ) ?? running,
      partitionsFailed:
        numberOrNull(
          firstDefined(summary.partitions_failed, raw.partitions_failed),
        ) ?? failed,
      audioSeconds:
        numberOrNull(
          firstDefined(
            summary.audio_seconds,
            summary.total_audio_seconds,
            raw.audio_seconds,
          ),
        ) ?? totalAudio,
      processedAudioSeconds:
        numberOrNull(
          firstDefined(
            summary.processed_audio_seconds,
            summary.audio_processed_seconds,
            raw.processed_audio_seconds,
          ),
        ) ?? processedAudio,
      progressPercent:
        progress ??
        calculateProgress(
          relevant,
          completed,
          total ?? relevant.length,
          processedAudio,
          totalAudio,
        ),
    };
  }

  function normalizeNode(rawValue, index) {
    const raw = asObject(rawValue);
    const details = asObject(raw.details);
    let gpuValues = arrayFrom(raw.gpus).filter(
      (value) => Object.keys(asObject(value)).length,
    );
    if (!gpuValues.length) {
      gpuValues = arrayFrom(details.gpus).filter(
        (value) => Object.keys(asObject(value)).length,
      );
    }
    if (!gpuValues.length) {
      const legacyGpu = raw.gpu || details.gpu;
      if (Object.keys(asObject(legacyGpu)).length) gpuValues = [legacyGpu];
    }
    let gpus = gpuValues.map(normalizeGpu);
    let configuredGpuValues = arrayFrom(raw.gpu_devices);
    if (!configuredGpuValues.length) {
      configuredGpuValues = arrayFrom(details.gpu_devices);
    }
    const configuredGpuIndexes = [
      ...new Set(
        configuredGpuValues
          .map(numberOrNull)
          .filter((value) => value !== null),
      ),
    ];
    if (configuredGpuIndexes.length) {
      const configuredSet = new Set(configuredGpuIndexes);
      const byIndex = new Map(
        gpus
          .filter((gpu) => gpu.index !== null)
          .map((gpu) => [gpu.index, gpu]),
      );
      gpus = [
        ...configuredGpuIndexes.map(
          (gpuIndex) =>
            byIndex.get(gpuIndex) || normalizeGpu({ index: gpuIndex }),
        ),
        ...gpus.filter(
          (gpu) => gpu.index === null || !configuredSet.has(gpu.index),
        ),
      ];
    }
    const disk = asObject(
      raw.disk || raw.storage || details.disk || details.storage,
    );
    const stateValue = canonicalState(raw.state || raw.status || "unknown");
    const drained =
      stateValue === "drained" ||
      truthyFlag(firstDefined(raw.drained, details.drained));

    return {
      id: stringValue(raw.id || raw.node_id || raw.name || `node-${index + 1}`),
      name: stringValue(
        raw.name || raw.id || raw.node_id || `node-${index + 1}`,
      ),
      host: stringValue(raw.host || raw.hostname || raw.address),
      state: stateValue,
      lastSeenAt: dateOrNull(
        raw.last_seen_at || raw.heartbeat_at || raw.updated_at,
      ),
      currentPartitionId: stringValue(
        raw.current_partition_id || raw.partition_id || raw.current_job_id,
      ),
      error: stringValue(raw.error || raw.last_error),
      drained,
      gpus,
      disk: {
        freeBytes: bytesValue(disk.free_bytes, disk.free_mb, disk.free),
        totalBytes: bytesValue(disk.total_bytes, disk.total_mb, disk.total),
      },
    };
  }

  function normalizeGpu(rawValue, index) {
    const gpu = asObject(rawValue);
    return {
      name: stringValue(gpu.name || gpu.model),
      index: numberOrNull(gpu.index ?? gpu.device_index ?? index),
      utilizationPercent: normalizePercent(
        gpu.utilization_percent ?? gpu.utilization ?? gpu.util,
      ),
      memoryUsedBytes: bytesValue(
        gpu.memory_used_bytes,
        firstDefined(gpu.memory_used_mib, gpu.memory_used_mb),
        gpu.memory_used,
      ),
      memoryTotalBytes: bytesValue(
        gpu.memory_total_bytes,
        firstDefined(gpu.memory_total_mib, gpu.memory_total_mb),
        gpu.memory_total,
      ),
    };
  }

  function normalizePartition(rawValue, index) {
    const raw = asObject(rawValue);
    const progressRaw = asObject(raw.progress);
    const filesTotal = numberOrNull(
      firstDefined(raw.files_total, raw.total_files, progressRaw.files_total),
    );
    const filesProcessed = numberOrNull(
      firstDefined(
        raw.files_processed,
        raw.processed_files,
        progressRaw.files_processed,
      ),
    );
    const audioSeconds = numberOrNull(
      firstDefined(
        raw.audio_seconds,
        raw.total_audio_seconds,
        progressRaw.audio_seconds,
      ),
    );
    let processedAudioSeconds = numberOrNull(
      firstDefined(
        raw.processed_audio_seconds,
        raw.audio_processed_seconds,
        progressRaw.processed_audio_seconds,
      ),
    );
    let progressPercent = normalizePercent(
      firstDefined(raw.progress_percent, progressRaw.percent, raw.progress),
    );

    if (progressPercent === null && filesTotal > 0 && filesProcessed !== null) {
      progressPercent = (filesProcessed / filesTotal) * 100;
    }
    if (
      progressPercent === null &&
      audioSeconds > 0 &&
      processedAudioSeconds !== null
    ) {
      progressPercent = (processedAudioSeconds / audioSeconds) * 100;
    }

    const stateValue = canonicalState(raw.state || raw.status);
    if (progressPercent === null && stateValue === "completed") {
      progressPercent = 100;
    }
    if (
      processedAudioSeconds === null &&
      audioSeconds !== null &&
      progressPercent !== null
    ) {
      processedAudioSeconds = (audioSeconds * progressPercent) / 100;
    }

    const stage = asObject(raw.stage);
    return {
      id: stringValue(
        raw.id || raw.partition_id || raw.job_id || `part-${index + 1}`,
      ),
      runId: stringValue(raw.run_id),
      nodeId: stringValue(raw.node_id || raw.worker_id || raw.assigned_node),
      state: stateValue,
      desiredState: canonicalState(raw.desired_state),
      stage: stringValue(
        stage.name ||
          stage.id ||
          raw.current_stage ||
          raw.stage_name ||
          (typeof raw.stage === "string" || typeof raw.stage === "number"
            ? raw.stage
            : ""),
      ),
      progressPercent: clamp(progressPercent ?? 0, 0, 100),
      attempt: numberOrNull(raw.attempt ?? raw.attempt_number),
      audioSeconds,
      processedAudioSeconds,
      filesTotal,
      filesProcessed,
      startedAt: dateOrNull(raw.started_at),
      updatedAt: dateOrNull(raw.updated_at || raw.heartbeat_at),
      error: stringValue(raw.error || raw.last_error || raw.message),
    };
  }

  function renderOverview() {
    const overview = state.overview;
    renderRun(overview.activeRun, overview.nodes);
    renderNodes(overview.nodes, overview.partitions);
    refreshFilterOptions(overview.partitions, overview.nodes);
    renderPartitions();
    renderUpdatedAt();
  }

  function renderRun(run, nodes) {
    const hasRun = Boolean(run.id || run.name || run.partitionsTotal);
    const onlineCount = nodes.filter(
      (node) =>
        !node.drained &&
        ["online", "busy", "running", "draining"].includes(node.state),
    ).length;

    setText(
      elements.runTitle,
      hasRun ? run.name || run.id : "Нет активного запуска",
    );
    setBadge(elements.runState, hasRun ? run.state : "unknown");
    setText(elements.configDigest, compactDigest(run.configDigest));
    elements.configDigest.title = run.configDigest || "";
    setText(elements.imageDigest, compactDigest(run.imageDigest));
    elements.imageDigest.title = run.imageDigest || "";
    setText(elements.runProgress, `${formatNumber(run.progressPercent, 1)}%`);
    setText(
      elements.partitionProgress,
      `${formatInteger(run.partitionsCompleted)} из ${formatInteger(run.partitionsTotal)} частей`,
    );
    setText(elements.audioProgress, formatDuration(run.processedAudioSeconds));
    setText(elements.audioTotal, `из ${formatDuration(run.audioSeconds)}`);
    setText(elements.onlineNodes, `${onlineCount} / ${nodes.length}`);
    setText(elements.runningJobs, formatInteger(run.partitionsRunning));
    setText(elements.failedJobs, formatInteger(run.partitionsFailed));
    elements.failedJobs.classList.toggle(
      "metric__value--danger",
      run.partitionsFailed > 0,
    );
    setText(elements.elapsedTime, formatElapsed(run.startedAt, run.finishedAt));
    setText(
      elements.startedAt,
      run.startedAt ? `с ${formatDateTime(run.startedAt)}` : "Не запущено",
    );
    elements.overallProgressBar.style.width = `${clamp(run.progressPercent, 0, 100)}%`;
  }

  function renderNodes(nodes, partitions) {
    elements.nodesGrid.replaceChildren();
    setText(elements.nodeCount, String(nodes.length));

    if (!nodes.length) {
      elements.nodesGrid.append(
        createElement("p", "empty-state", "Ноды не зарегистрированы."),
      );
      return;
    }

    const sorted = [...nodes].sort((a, b) => {
      const orderA = STATUS[a.state]?.order ?? 100;
      const orderB = STATUS[b.state]?.order ?? 100;
      return orderA - orderB || a.name.localeCompare(b.name, "ru");
    });
    const fragment = document.createDocumentFragment();
    sorted.forEach((node) =>
      fragment.append(createNodeCard(node, partitions)),
    );
    elements.nodesGrid.append(fragment);
  }

  function createNodeCard(node, partitions) {
    const card = createElement("article", "node-card");
    if (node.state === "offline") card.classList.add("node-card--offline");
    if (node.drained) card.classList.add("node-card--drained");
    if (node.error || node.state === "error")
      card.classList.add("node-card--error");

    const header = createElement("div", "node-card__header");
    const identity = createElement("div", "node-card__identity");
    const name = createElement("strong", "node-card__name", node.name);
    name.title = node.name;
    const host = createElement(
      "span",
      "node-card__host",
      node.host || "Адрес не указан",
    );
    host.title = node.host || "";
    identity.append(name, host);
    const statuses = createElement("div", "node-card__statuses");
    statuses.append(createBadge(node.state));
    if (node.drained && node.state !== "drained") {
      statuses.append(createBadge("drained"));
    }
    header.append(identity, statuses);

    const resources = createElement("div", "node-card__resources");
    resources.append(
      createGpuList(node.gpus),
      createResourceRow(
        "Локальный диск",
        formatFreeTotal(node.disk.freeBytes, node.disk.totalBytes),
        diskUsedPercent(node.disk.freeBytes, node.disk.totalBytes),
      ),
    );

    const footer = createElement("div", "node-card__footer");
    const current = createElement(
      "span",
      "node-card__detail",
      node.currentPartitionId
        ? `Задача: ${node.currentPartitionId}`
        : "Свободна",
    );
    current.title = node.currentPartitionId || "";
    const seen = createElement(
      "span",
      "node-card__detail",
      node.lastSeenAt ? relativeTime(node.lastSeenAt) : "Нет heartbeat",
    );
    seen.title = node.lastSeenAt ? formatDateTime(node.lastSeenAt) : "";
    const details = createElement("div", "node-card__details");
    details.append(current, seen);
    footer.append(details);

    const partition = partitions.find(
      (item) =>
        item.id === node.currentPartitionId && item.nodeId === node.id,
    );
    if (partition && !TERMINAL_PARTITION_STATES.has(partition.state)) {
      footer.append(createStopButton(node, partition));
    }

    card.append(header, resources, footer);
    if (node.error) {
      const error = createElement("p", "node-card__error", node.error);
      error.title = node.error;
      card.append(error);
    }
    return card;
  }

  function createGpuList(gpus) {
    const list = createElement("div", "gpu-list");
    list.setAttribute("role", "table");
    list.setAttribute("aria-label", "GPU");

    const heading = createElement("div", "gpu-list__heading");
    heading.setAttribute("role", "row");
    const deviceHeading = createElement("span", "", "GPU");
    const utilizationHeading = createElement("span", "", "Загрузка");
    const memoryHeading = createElement("span", "", "Память");
    [deviceHeading, utilizationHeading, memoryHeading].forEach((cell) =>
      cell.setAttribute("role", "columnheader"),
    );
    heading.append(
      deviceHeading,
      utilizationHeading,
      memoryHeading,
    );
    list.append(heading);

    if (!gpus.length) {
      const emptyRow = createElement("div", "gpu-list__empty");
      const emptyCell = createElement("span", "", "Данные GPU недоступны");
      emptyRow.setAttribute("role", "row");
      emptyCell.setAttribute("role", "cell");
      emptyCell.setAttribute("aria-colspan", "3");
      emptyRow.append(emptyCell);
      list.append(emptyRow);
      return list;
    }

    gpus.forEach((gpu) => list.append(createGpuRow(gpu)));
    return list;
  }

  function createGpuRow(gpu) {
    const row = createElement("div", "gpu-row");
    row.setAttribute("role", "row");
    const device = createElement("div", "gpu-row__device");
    device.setAttribute("role", "cell");
    const index = createElement(
      "strong",
      "gpu-row__index",
      `GPU ${gpu.index ?? "-"}`,
    );
    const name = createElement(
      "span",
      "gpu-row__name",
      gpu.name || "Модель не указана",
    );
    device.title = gpu.name
      ? `GPU ${gpu.index ?? "-"}: ${gpu.name}`
      : `GPU ${gpu.index ?? "-"}`;
    device.append(index, name);

    const utilization = createElement(
      "span",
      "gpu-row__utilization",
      gpu.utilizationPercent === null
        ? "-"
        : `${formatNumber(gpu.utilizationPercent, 0)}%`,
    );
    utilization.title =
      gpu.utilizationPercent === null
        ? "Загрузка недоступна"
        : `Загрузка: ${formatNumber(gpu.utilizationPercent, 0)}%`;
    utilization.setAttribute("role", "cell");

    const memory = createElement(
      "span",
      "gpu-row__memory",
      formatGpuMemory(gpu.memoryUsedBytes, gpu.memoryTotalBytes),
    );
    memory.title = `Память: ${formatUsedTotal(
      gpu.memoryUsedBytes,
      gpu.memoryTotalBytes,
    )}`;
    memory.setAttribute("role", "cell");

    row.append(device, utilization, memory);
    return row;
  }

  function createStopButton(node, partition) {
    const key = `${partition.runId}:${partition.id}`;
    const stopping =
      state.stoppingPartitions.has(key) || partition.desiredState === "cancelled";
    const button = createElement(
      "button",
      "button button--danger node-card__stop",
    );
    button.type = "button";
    button.disabled = stopping || !partition.runId || !state.csrfToken;
    button.classList.toggle("is-loading", stopping);
    button.title = stopping
      ? "Остановка уже запрошена"
      : `Остановить ${partition.id} на ${node.name}`;
    button.setAttribute(
      "aria-label",
      stopping
        ? `Остановка ${partition.id} запрошена`
        : `Остановить ${partition.id} на ноде ${node.name}`,
    );
    button.append(
      createElement("span", "stop-symbol", "■"),
      createElement(
        "span",
        "node-card__stop-label",
        stopping ? "Остановка" : "Остановить",
      ),
    );
    if (!stopping) {
      button.addEventListener("click", () => requestStop(node, partition));
    }
    return button;
  }

  async function requestStop(node, partition) {
    const key = `${partition.runId}:${partition.id}`;
    const confirmed = window.confirm(
      [
        `Остановить партицию ${partition.id} на ноде ${node.name}?`,
        "Нода будет приостановлена и не получит новую задачу.",
      ].join(" "),
    );
    if (!confirmed) return;

    state.stoppingPartitions.add(key);
    renderNodes(state.overview.nodes, state.overview.partitions);
    try {
      const runId = encodeURIComponent(partition.runId);
      const partitionId = encodeURIComponent(partition.id);
      const url = `/api/v1/runs/${runId}/partitions/${partitionId}/cancel`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-CSRF-Token": state.csrfToken,
        },
        credentials: "same-origin",
        body: JSON.stringify({ node_id: node.id }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      await refreshOverview(true);
      const drainedNodeId = stringValue(payload.drained_node_id || node.id);
      const resumeHint =
        `Нода ${drainedNodeId} приостановлена до ` +
        `nodes resume ${drainedNodeId}.`;
      showAlert(
        payload.warning
          ? `Остановка ${partition.id} сохранена, но нода пока не подтвердила её: ${payload.warning}. ${resumeHint}`
          : `Партиция ${partition.id} остановлена. ${resumeHint}`,
        payload.warning ? "warning" : "success",
      );
    } catch (error) {
      showAlert(`Не удалось остановить ${partition.id}: ${error.message}.`);
    } finally {
      state.stoppingPartitions.delete(key);
      if (state.overview) {
        renderNodes(state.overview.nodes, state.overview.partitions);
      }
    }
  }

  function createResourceRow(label, value, percent) {
    const row = createElement("div", "resource-row");
    const heading = createElement("div", "resource-row__heading");
    heading.append(
      createElement("span", "resource-row__label", label),
      createElement("span", "resource-row__value", value),
    );
    const track = createElement("div", "resource-track");
    const normalized = clamp(percent ?? 0, 0, 100);
    if (normalized >= 92) track.classList.add("resource-track--danger");
    else if (normalized >= 80) track.classList.add("resource-track--warning");
    const bar = document.createElement("span");
    bar.style.width = `${normalized}%`;
    track.append(bar);
    row.append(heading, track);
    return row;
  }

  function refreshFilterOptions(partitions, nodes) {
    const currentState = elements.stateFilter.value;
    const currentNode = elements.nodeFilter.value;
    const states = [
      ...new Set(partitions.map((partition) => partition.state)),
    ].sort((a, b) => (STATUS[a]?.order ?? 100) - (STATUS[b]?.order ?? 100));
    const nodeIds = [
      ...new Set(
        [
          ...nodes.map((node) => node.id),
          ...partitions.map((partition) => partition.nodeId),
        ].filter(Boolean),
      ),
    ].sort((a, b) => a.localeCompare(b, "ru"));

    replaceSelectOptions(
      elements.stateFilter,
      "Все состояния",
      states.map((value) => ({ value, label: statusInfo(value).label })),
      currentState,
    );
    replaceSelectOptions(
      elements.nodeFilter,
      "Все ноды",
      nodeIds.map((value) => ({ value, label: value })),
      currentNode,
    );
  }

  function renderPartitions() {
    if (!state.overview) return;
    const query = elements.searchInput.value.trim().toLocaleLowerCase("ru");
    const stateFilter = elements.stateFilter.value;
    const nodeFilter = elements.nodeFilter.value;
    const filtered = state.overview.partitions
      .filter((partition) => !stateFilter || partition.state === stateFilter)
      .filter((partition) => !nodeFilter || partition.nodeId === nodeFilter)
      .filter((partition) => {
        if (!query) return true;
        return [
          partition.id,
          partition.runId,
          partition.nodeId,
          partition.stage,
          partition.error,
        ]
          .filter(Boolean)
          .some((value) => value.toLocaleLowerCase("ru").includes(query));
      })
      .sort(comparePartitions);

    setText(
      elements.visiblePartitionCount,
      filtered.length === state.overview.partitions.length
        ? String(filtered.length)
        : `${filtered.length} / ${state.overview.partitions.length}`,
    );
    elements.partitionsTableBody.replaceChildren();

    if (!filtered.length) {
      const row = createElement("tr", "table-message-row");
      const cell = createElement(
        "td",
        "",
        state.overview.partitions.length
          ? "Партиции по выбранным фильтрам не найдены."
          : "Партиции ещё не созданы.",
      );
      cell.colSpan = 9;
      row.append(cell);
      elements.partitionsTableBody.append(row);
      return;
    }

    const fragment = document.createDocumentFragment();
    filtered.forEach((partition) => {
      fragment.append(createPartitionRow(partition));
      if (partition.error) fragment.append(createPartitionErrorRow(partition));
    });
    elements.partitionsTableBody.append(fragment);
  }

  function createPartitionRow(partition) {
    const row = createElement("tr", "partition-row");
    row.append(
      createTableCell("Партиция", partition.id, "cell-id", partition.id),
      createTableCellWithNode("Состояние", createBadge(partition.state)),
      createTableCell(
        "Нода",
        partition.nodeId || "-",
        "cell-id",
        partition.nodeId,
      ),
      createTableCell("Этап", formatStage(partition.stage), "cell-muted"),
      createTableCellWithNode(
        "Прогресс",
        createTableProgress(partition.progressPercent),
      ),
      createTableCell(
        "Файлы",
        formatCountPair(partition.filesProcessed, partition.filesTotal),
        "cell-number",
      ),
      createTableCell(
        "Аудио",
        formatAudioPair(
          partition.processedAudioSeconds,
          partition.audioSeconds,
        ),
        "cell-number",
      ),
      createTableCell(
        "Попытка",
        partition.attempt === null ? "-" : String(partition.attempt),
        "cell-number",
      ),
      createTableCell(
        "Обновлено",
        partition.updatedAt ? relativeTime(partition.updatedAt) : "-",
        "cell-muted cell-number",
        partition.updatedAt ? formatDateTime(partition.updatedAt) : "",
      ),
    );
    return row;
  }

  function createPartitionErrorRow(partition) {
    const row = createElement("tr", "partition-error-row");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    const error = createElement("div", "partition-error", partition.error);
    error.setAttribute("aria-label", `Ошибка ${partition.id}`);
    cell.append(error);
    row.append(cell);
    return row;
  }

  function createTableCell(label, value, className, title) {
    const cell = createElement("td", className, value);
    cell.dataset.label = label;
    if (title) cell.title = title;
    return cell;
  }

  function createTableCellWithNode(label, child) {
    const cell = document.createElement("td");
    cell.dataset.label = label;
    cell.append(child);
    return cell;
  }

  function createTableProgress(percent) {
    const wrapper = createElement("div", "table-progress");
    const track = createElement("div", "progress-track");
    const bar = document.createElement("span");
    const normalized = clamp(percent ?? 0, 0, 100);
    bar.style.width = `${normalized}%`;
    track.append(bar);
    wrapper.append(
      track,
      createElement(
        "span",
        "progress-value",
        `${formatNumber(normalized, 0)}%`,
      ),
    );
    return wrapper;
  }

  function replaceSelectOptions(select, allLabel, options, selected) {
    const fragment = document.createDocumentFragment();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = allLabel;
    fragment.append(all);
    options.forEach((option) => {
      const element = document.createElement("option");
      element.value = option.value;
      element.textContent = option.label;
      fragment.append(element);
    });
    select.replaceChildren(fragment);
    select.value = options.some((option) => option.value === selected)
      ? selected
      : "";
  }

  function comparePartitions(a, b) {
    const orderA = STATUS[a.state]?.order ?? 100;
    const orderB = STATUS[b.state]?.order ?? 100;
    const updateA = a.updatedAt?.getTime() ?? 0;
    const updateB = b.updatedAt?.getTime() ?? 0;
    return (
      orderA - orderB || updateB - updateA || a.id.localeCompare(b.id, "ru")
    );
  }

  function calculateProgress(
    partitions,
    completed,
    total,
    processedAudio,
    totalAudio,
  ) {
    if (totalAudio > 0 && processedAudio >= 0) {
      return clamp((processedAudio / totalAudio) * 100, 0, 100);
    }
    if (partitions.length) {
      const progressSum = sum(
        partitions.map((partition) => partition.progressPercent),
      );
      return progressSum / partitions.length;
    }
    if (total > 0) return (completed / total) * 100;
    return 0;
  }

  function setLoading(loading) {
    elements.refreshButton.disabled = loading;
    elements.refreshButton.classList.toggle("is-loading", loading);
    if (loading && !state.lastSuccessAt)
      setConnection("loading", "Подключение...");
  }

  function setConnection(kind, label) {
    elements.connectionDot.className = `connection-dot connection-dot--${kind}`;
    setText(elements.connectionLabel, label);
  }

  function controllerLabel(controller) {
    const name = stringValue(controller.name || controller.id);
    const version = stringValue(controller.version);
    if (name && version) return `${name} ${version}`;
    return name || (version ? `Controller ${version}` : "Подключено");
  }

  function renderUpdatedAt() {
    if (!state.lastSuccessAt) {
      setText(elements.updatedAt, "Нет данных");
      return;
    }
    setText(elements.updatedAt, `Обновлено ${formatTime(state.lastSuccessAt)}`);
    elements.updatedAt.title = formatDateTime(state.lastSuccessAt);
  }

  function showAlert(message, tone = "error") {
    setText(elements.alert, message);
    elements.alert.className = `alert alert--${tone}`;
    elements.alert.hidden = false;
  }

  function hideAlert() {
    elements.alert.hidden = true;
    elements.alert.className = "alert";
    setText(elements.alert, "");
  }

  function createBadge(status) {
    const badge = createElement("span", "status-badge");
    setBadge(badge, status);
    return badge;
  }

  function setBadge(element, status) {
    const info = statusInfo(status);
    element.className = `status-badge status-badge--${info.tone}`;
    setText(element, info.label);
  }

  function statusInfo(status) {
    const normalized = canonicalState(status);
    return (
      STATUS[normalized] || {
        label: humanize(normalized),
        tone: "unknown",
        order: 100,
      }
    );
  }

  function canonicalState(value) {
    const raw = stringValue(value).trim().toLowerCase().replaceAll("-", "_");
    if (!raw) return "unknown";
    return STATE_ALIASES[raw] || raw;
  }

  function isActiveRun(run) {
    return ![
      "completed",
      "committed",
      "failed",
      "cancelled",
      "canceled",
    ].includes(canonicalState(asObject(run).state || asObject(run).status));
  }

  function formatStage(stage) {
    if (!stage) return "-";
    const raw = String(stage);
    return /^\d+(\.\d+)?$/.test(raw) ? `Этап ${raw}` : raw;
  }

  function formatCountPair(processed, total) {
    if (processed === null && total === null) return "-";
    return `${formatInteger(processed ?? 0)} / ${formatInteger(total ?? 0)}`;
  }

  function formatAudioPair(processed, total) {
    if (processed === null && total === null) return "-";
    return `${formatDurationShort(processed ?? 0)} / ${formatDurationShort(total ?? 0)}`;
  }

  function formatDuration(seconds) {
    const value = finiteNumber(seconds) ?? 0;
    if (value >= 3600) return `${formatNumber(value / 3600, 1)} ч`;
    if (value >= 60) return `${formatNumber(value / 60, 0)} мин`;
    return `${formatNumber(value, 0)} с`;
  }

  function formatDurationShort(seconds) {
    const value = finiteNumber(seconds) ?? 0;
    if (value >= 3600) return `${formatNumber(value / 3600, 1)} ч`;
    if (value >= 60) return `${formatNumber(value / 60, 0)} м`;
    return `${formatNumber(value, 0)} с`;
  }

  function formatElapsed(start, end) {
    if (!start) return "-";
    const seconds = Math.max(
      0,
      ((end || new Date()).getTime() - start.getTime()) / 1000,
    );
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days) return `${days} д ${hours} ч`;
    if (hours) return `${hours} ч ${minutes} мин`;
    return `${minutes} мин`;
  }

  function relativeTime(date) {
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const absolute = Math.abs(seconds);
    const formatter = new Intl.RelativeTimeFormat("ru", { numeric: "auto" });
    if (absolute < 60) return formatter.format(seconds, "second");
    if (absolute < 3600)
      return formatter.format(Math.round(seconds / 60), "minute");
    if (absolute < 86400)
      return formatter.format(Math.round(seconds / 3600), "hour");
    return formatter.format(Math.round(seconds / 86400), "day");
  }

  function formatDateTime(date) {
    return new Intl.DateTimeFormat("ru-RU", {
      dateStyle: "medium",
      timeStyle: "medium",
    }).format(date);
  }

  function formatTime(date) {
    return new Intl.DateTimeFormat("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  }

  function formatNumber(value, maximumFractionDigits) {
    const number = finiteNumber(value) ?? 0;
    return new Intl.NumberFormat("ru-RU", {
      maximumFractionDigits,
    }).format(number);
  }

  function formatInteger(value) {
    return formatNumber(value, 0);
  }

  function formatBytes(value) {
    const bytes = finiteNumber(value);
    if (bytes === null) return "-";
    const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"];
    let amount = Math.max(0, bytes);
    let index = 0;
    while (amount >= 1024 && index < units.length - 1) {
      amount /= 1024;
      index += 1;
    }
    return `${formatNumber(amount, amount >= 10 ? 0 : 1)} ${units[index]}`;
  }

  function formatUsedTotal(used, total) {
    if (used === null && total === null) return "Нет данных";
    return `${formatBytes(used ?? 0)} / ${formatBytes(total ?? 0)}`;
  }

  function formatGpuMemory(used, total) {
    if (used === null && total === null) return "-";
    const largest = Math.max(used ?? 0, total ?? 0);
    const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"];
    let divisor = 1;
    let unitIndex = 0;
    while (largest / divisor >= 1024 && unitIndex < units.length - 1) {
      divisor *= 1024;
      unitIndex += 1;
    }
    const value = (bytes) =>
      bytes === null
        ? "-"
        : formatNumber(bytes / divisor, bytes / divisor >= 10 ? 0 : 1);
    return `${value(used)} / ${value(total)} ${units[unitIndex]}`;
  }

  function formatFreeTotal(free, total) {
    if (free === null && total === null) return "Нет данных";
    return `${formatBytes(free ?? 0)} свободно`;
  }

  function compactDigest(value) {
    if (!value) return "-";
    const raw = String(value);
    if (raw.length <= 24) return raw;
    return `${raw.slice(0, 12)}...${raw.slice(-8)}`;
  }

  function ratioPercent(used, total) {
    if (!Number.isFinite(used) || !Number.isFinite(total) || total <= 0)
      return 0;
    return clamp((used / total) * 100, 0, 100);
  }

  function diskUsedPercent(free, total) {
    if (!Number.isFinite(free) || !Number.isFinite(total) || total <= 0)
      return 0;
    return clamp(((total - free) / total) * 100, 0, 100);
  }

  function bytesValue(bytes, megabytes, fallback) {
    const exact = numberOrNull(bytes);
    if (exact !== null) return exact;
    const mb = numberOrNull(megabytes);
    if (mb !== null) return mb * 1024 * 1024;
    return numberOrNull(fallback);
  }

  function normalizePercent(value) {
    const number = finiteNumber(value);
    if (number === null) return null;
    return number;
  }

  function dateOrNull(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function numberOrNull(value) {
    return finiteNumber(value);
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function truthyFlag(value) {
    if (value === true || value === 1) return true;
    if (typeof value !== "string") return false;
    return ["1", "true"].includes(value.trim().toLowerCase());
  }

  function stringValue(value) {
    return value === null || value === undefined ? "" : String(value);
  }

  function firstDefined(...values) {
    return values.find((value) => value !== undefined && value !== null);
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : {};
  }

  function arrayFrom(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.values(value);
    return [];
  }

  function sum(values) {
    return values.reduce(
      (total, value) => total + (finiteNumber(value) ?? 0),
      0,
    );
  }

  function clamp(value, min, max) {
    const number = finiteNumber(value) ?? min;
    return Math.min(max, Math.max(min, number));
  }

  function humanize(value) {
    const text = stringValue(value).replaceAll("_", " ");
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Неизвестно";
  }

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function setText(element, value) {
    element.textContent = value;
  }
})();
