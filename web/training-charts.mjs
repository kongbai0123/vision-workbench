// Training plots consume recorded metrics only. Pure model helpers also serve the
// Epoch table and keep plotting independent of desktop/browser rendering.
const DEFINITIONS = {
  'train/loss': ['訓練損失 · Train Loss', false, 'lower'],
  'train/learning_rate': ['實際學習率 · LR（每輪最後一步）', false, null],
  'lr/group_0': ['學習率 · 參數組 0', false, null],
  'lr/group_1': ['學習率 · 參數組 1', false, null],
  'lr/group_2': ['學習率 · 參數組 2', false, null],
  'val/loss': ['驗證損失 · Validation Loss', false, 'lower'],
  'val/mean_iou': ['Validation · mIoU', true, 'higher'],
  'val/box_mean_iou': ['Validation · Box IoU', true, 'higher'],
  'val/recall_50': ['Validation · Recall@0.5', true, 'higher'],
  'val/mean_dice': ['Validation · Dice', true, 'higher'],
  'val/accuracy': ['Validation · Accuracy', true, 'higher'],
  'val/macro_f1': ['Validation · Macro F1', true, 'higher'],
  'val/macro_recall': ['Validation · Macro Recall', true, 'higher'],
  'val/box_map50_95': ['Validation · Box mAP50–95', true, 'higher'],
  'val/box_map50': ['Validation · Box mAP50', true, 'higher'],
  'val/box_precision': ['Validation · Box Precision', true, 'higher'],
  'val/box_recall': ['Validation · Box Recall', true, 'higher'],
  'val/mask_map50_95': ['Validation · Mask mAP50–95', true, 'higher'],
  'val/mask_map50': ['Validation · Mask mAP50', true, 'higher'],
  'val/mask_precision': ['Validation · Mask Precision', true, 'higher'],
  'val/mask_recall': ['Validation · Mask Recall', true, 'higher'],
  'train/box_loss': ['Train · Box Loss', false, 'lower'],
  'train/seg_loss': ['Train · Segmentation Loss', false, 'lower'],
  'train/cls_loss': ['Train · Classification Loss', false, 'lower'],
  'train/dfl_loss': ['Train · DFL Loss', false, 'lower'],
  'train/giou_loss': ['Train · GIoU Loss', false, 'lower'],
  'train/l1_loss': ['Train · L1 Loss', false, 'lower'],
  'train/sem_loss': ['Train · Semantic Loss', false, 'lower'],
  'val/box_loss': ['Validation · Box Loss', false, 'lower'],
  'val/seg_loss': ['Validation · Segmentation Loss', false, 'lower'],
  'val/cls_loss': ['Validation · Classification Loss', false, 'lower'],
  'val/dfl_loss': ['Validation · DFL Loss', false, 'lower'],
  'val/giou_loss': ['Validation · GIoU Loss', false, 'lower'],
  'val/l1_loss': ['Validation · L1 Loss', false, 'lower'],
  'val/sem_loss': ['Validation · Semantic Loss', false, 'lower'],
  threshold: ['分割閾值 · Threshold', false, null],
};
const PRIORITY = ['train/loss', 'train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss',
  'train/giou_loss', 'train/l1_loss', 'train/sem_loss', 'val/loss', 'val/box_loss', 'val/seg_loss',
  'val/cls_loss', 'val/dfl_loss', 'val/giou_loss', 'val/l1_loss', 'val/sem_loss',
  'val/box_precision', 'val/box_recall', 'val/box_map50_95', 'val/box_map50',
  'val/mask_precision', 'val/mask_recall', 'val/mask_map50_95', 'val/mask_map50',
  'val/mean_iou', 'val/box_mean_iou', 'val/accuracy'];
const META_KEYS = new Set(['epoch', 'step', 'timestamp', 'time', 'created_at', 'progress',
  'optimizer_steps', 'optimizer_steps_total', 'effective_batch_size', 'gradient_accumulation',
  'train/optimizer_steps', 'train/optimizer_steps_epoch', 'train/optimizer_attempts_epoch',
  'train/optimizer_skipped_epoch']);
const PALETTE = ['#55d4bd', '#8ebdff', '#efb35b', '#c795f5', '#f28fab', '#c4d46e'];
const DASHES = ['', '7 4', '2 3', '10 3 2 3', '9 5', '3 2 3 6'];

export function finiteMetric(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

export function normalizedMetricRows(rows = []) {
  const byEpoch = new Map();
  for (const row of rows) {
    if (!row || !Number.isInteger(row.epoch) || row.epoch < 1) continue;
    byEpoch.set(row.epoch, {...byEpoch.get(row.epoch), ...row});
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

function reportRows(reports, run) {
  const report = reports?.get(run.run_id);
  if (report?.run?.run_id && report.run.run_id !== run.run_id) return [];
  return normalizedMetricRows(report?.metrics || []);
}

function definition(key) {
  const [label, unit, direction] = DEFINITIONS[key] || [key, false, null];
  return {key, label, unit, direction};
}

function displayDefinition(key, runs) {
  const descriptor = definition(key);
  if (!key.startsWith('val/')) return descriptor;
  const splits = new Set(runs.map(run => run.metricSplit || 'val'));
  if (splits.size === 1 && splits.has('test')) {
    descriptor.label = descriptor.label.replace('Validation', 'Test（訓練期間評估）').replace('驗證損失', '評估損失');
  } else if (splits.has('unknown')) {
    descriptor.label = descriptor.label.replace('Validation', '評估（分割未知）');
  } else if (splits.size > 1) {
    descriptor.label = descriptor.label.replace('Validation', '評估（Val／Test）');
  }
  return descriptor;
}

export function metricDescriptors(runs = [], reports = new Map()) {
  const available = new Map();
  for (const run of runs) {
    for (const row of reportRows(reports, run)) {
      for (const [key, value] of Object.entries(row)) {
        if (META_KEYS.has(key) || !finiteMetric(value)) continue;
        if (!available.has(key)) available.set(key, new Set());
        available.get(key).add(run.run_id);
      }
    }
  }
  const keys = [...available.keys()].sort((a, b) => {
    const ia = PRIORITY.indexOf(a), ib = PRIORITY.indexOf(b);
    return (ia < 0 ? 100 : ia) - (ib < 0 ? 100 : ib) || a.localeCompare(b);
  });
  return keys.map(key => ({...displayDefinition(key, runs.filter(run => available.get(key).has(run.run_id))),
    availableRunIds: [...available.get(key)]}));
}

export function defaultMetricKeys(descriptors = []) {
  const keys = descriptors.map(item => item.key);
  const loss = keys.includes('train/loss') ? 'train/loss' : keys.find(key => key.endsWith('/loss'));
  const primary = keys.find(key => key.startsWith('val/') && !key.endsWith('/loss'));
  return [...new Set([loss, primary].filter(Boolean))];
}

export function runAppearance(run, index = 0) {
  const appearance = Number.isInteger(run?.appearanceIndex) && run.appearanceIndex >= 0 ? run.appearanceIndex : index;
  const slot = Number.isInteger(appearance) && appearance >= 0 ? appearance : 0;
  return {color: slot < PALETTE.length ? PALETTE[slot] : `hsl(${Math.round(slot * 137.508) % 360} 65% 68%)`,
    dash: DASHES[slot % DASHES.length]};
}

function engineFamily(run) {
  const engine = run.engine || run.config?.engine || '';
  if (engine.startsWith('maskrcnn_')) return 'maskrcnn';
  if (engine.startsWith('fasterrcnn_')) return 'fasterrcnn';
  if (engine.startsWith('deeplabv3_')) return 'deeplab';
  if (engine.endsWith('_classification')) return 'classification';
  if (engine.startsWith('rt_detr_')) return 'rtdetr';
  if (engine.startsWith('yolo26')) return 'yolo26';
  return engine || `unknown:${run.run_id}`;
}

function metricMeaning(run, key) {
  const family = engineFamily(run);
  const split = key.startsWith('val/') ? `:${run.metricSplit || 'val'}` : '';
  const size = run.config?.image_size ?? 'default';
  if (key === 'val/mean_iou') {
    // These engines write the same field name for different evaluation procedures.
    if (family === 'maskrcnn') return `merged-class-mask-iou${split}`;
    if (family === 'deeplab') return `semantic-class-iou:${size}${split}`;
    if (family === 'pixel_prototype_v1') return `pixel-prototype-iou${split}`;
    return family + split;
  }
  if (key.endsWith('/loss')) return `${run.engine || run.config?.engine || family}:${size}${split}`;
  return key + split;
}

export function comparisonWarnings(runs = [], metricKeys = []) {
  if (runs.length < 2) return [];
  const warnings = [];
  if (new Set(runs.map(run => run.dataset_version_id || `unknown:${run.run_id}`)).size > 1) {
    warnings.push('Run 使用不同資料版本；請選擇相同資料版本與評估分割，才能比較模型效果。');
  }
  if (new Set(runs.map(run => run.metricSplit || 'val')).size > 1) {
    warnings.push('Run 的 Epoch 指標使用不同評估分割（Validation／Test），已停用這些指標的疊圖。');
  }
  for (const key of metricKeys) {
    if (new Set(runs.map(run => metricMeaning(run, key))).size > 1) {
      warnings.push(`${definition(key).label} 的計算定義、模型設定或評估分割不同，已停用此指標的疊圖；請選擇相容的 Run 或改看共同的評估指標。`);
    }
  }
  return warnings;
}

function roundedBound(value) {
  if (value === 0) return 0;
  const step = 10 ** Math.floor(Math.log10(Math.abs(value)));
  const rounded = Math.ceil(Math.abs(value) / step * 2) / 2 * step;
  return finiteMetric(rounded) ? rounded : Math.abs(value);
}

function pointSegments(points) {
  const segments = [];
  for (const point of points) {
    const last = segments.at(-1);
    if (!last || last.at(-1).epoch + 1 !== point.epoch) segments.push([point]);
    else last.push(point);
  }
  return segments;
}

function epochRanges(values) {
  const ranges=[];
  for(const value of values){const last=ranges.at(-1);if(last&&last[1]+1===value)last[1]=value;else ranges.push([value,value])}
  return ranges.map(([start,end])=>start===end?String(start):`${start}–${end}`).join('、');
}

/** Retain `domains` across refreshes, clearing it when changing projects. */
export function buildChartModel(options, key) {
  const {runs = [], reports = new Map(), visibleRunIds = new Set(runs.map(run => run.run_id)),
    comparison = false, domains = new Map()} = options;
  const descriptor = displayDefinition(key, runs);
  const collectSeries = (run, index) => {
    const rows = reportRows(reports, run);
    const points = rows.filter(row => finiteMetric(row[key])).map(row => ({epoch: row.epoch, value: row[key]}));
    return {run, ...runAppearance(run, index), rows, points, segments: pointSegments(points)};
  };
  const allSeries = runs.map(collectSeries);
  // Comparison candidates hold the coordinate range while the checked Runs alone
  // determine plotted data and whether their metric definitions are compatible.
  const domainRuns = options.domainRuns || runs;
  const domainSeries = domainRuns === runs ? allSeries : domainRuns.map(collectSeries);
  const metricRuns = allSeries.filter(series => series.points.length).map(series => series.run);
  const incompatible = comparison && new Set(metricRuns.map(run => metricMeaning(run, key))).size > 1;
  const domainKey = JSON.stringify([domainRuns.map(run => run.run_id).sort(), key]);
  const prior = domains.get(domainKey);
  const observedEpoch = domainSeries.reduce((maximum, series) => Math.max(maximum, series.rows.at(-1)?.epoch || 1), 1);
  const configuredEpoch = domainRuns.reduce((maximum, run) => Math.max(maximum,
    finiteMetric(run.config?.epochs) ? run.config.epochs : 1), 1);
  const xMax = Math.max(1, Math.ceil(configuredEpoch), observedEpoch, prior?.xMax || 1);
  const values = domainSeries.flatMap(series => series.points.map(point => point.value));
  const isRate = key==='train/learning_rate'||key.startsWith('lr/');
  const rateCeiling = isRate ? domainRuns.reduce((max,run)=>Math.max(max,run.config?.learning_rate||0),1e-12) : 1;
  const maximum = values.reduce((max, value) => Math.max(max, value), isRate ? rateCeiling : 0);
  const minimum = values.reduce((min, value) => Math.min(min, value), 0);
  const zoomedUnit = descriptor.unit && maximum > 0 && maximum < .1;
  const unitMaximum = zoomedUnit ? Math.max(.001, roundedBound(maximum)) : 1;
  const positiveUnitValue = descriptor.unit && values.some(value => value > 0 && value <= 1);
  const priorUnitMaximum = prior?.hadPositiveUnitValue ? prior.yMax : 0;
  const yMax = descriptor.unit ? Math.max(unitMaximum, priorUnitMaximum) : Math.max(roundedBound(maximum), prior?.yMax || rateCeiling);
  const yMin = descriptor.unit ? 0 : Math.min(-roundedBound(Math.min(0, minimum)), prior?.yMin || 0);
  const expanded = Boolean(prior?.expanded || (prior && (yMax > prior.yMax || yMin < prior.yMin || xMax > prior.xMax)));
  domains.set(domainKey, {xMax, yMax, yMin, expanded,
    hadPositiveUnitValue: Boolean(prior?.hadPositiveUnitValue || positiveUnitValue)});
  const outOfRange = descriptor.unit && allSeries.some(series => series.points.some(point => point.value < 0 || point.value > 1));
  const visible = allSeries.filter(series => visibleRunIds.has(series.run.run_id));
  const series = (incompatible ? [] : visible).map(item => {
    const points = descriptor.unit ? item.points.filter(point => point.value >= 0 && point.value <= 1) : item.points;
    return {...item, points, segments: pointSegments(points)};
  });
  const warnings = [];
  if (incompatible) warnings.push(`${descriptor.label} 的計算定義、模型設定或評估分割不同，無法疊圖比較。請選擇相容的 Run，或改看共同的評估指標。`);
  if (outOfRange) warnings.push('部分指標超出 0–1 範圍，未繪製異常值；請查看數值表與訓練日誌。');
  if (expanded) warnings.push('新資料超出原座標範圍，已擴大座標以完整顯示；之後不會自動縮小。');
  for(const item of series){
    if(item.points.length<2)continue;
    const present=new Set(item.points.map(point=>point.epoch)),first=item.points[0].epoch,last=item.points.at(-1).epoch,missing=[];
    for(let epoch=first;epoch<=last;epoch++)if(!present.has(epoch))missing.push(epoch);
    const noUpdates=isRate?missing.filter(epoch=>item.rows.find(row=>row.epoch===epoch)?.['train/optimizer_steps_epoch']===0):[];
    const unexplained=missing.filter(epoch=>!noUpdates.includes(epoch));
    if(noUpdates.length){
      const diagnosed=noUpdates.filter(epoch=>finiteMetric(item.rows.find(row=>row.epoch===epoch)?.['train/optimizer_attempts_epoch']));
      const diagnosis=diagnosed.map(epoch=>{const row=item.rows.find(entry=>entry.epoch===epoch),attempts=row['train/optimizer_attempts_epoch'],skipped=row['train/optimizer_skipped_epoch'];return `Epoch ${epoch}：嘗試 ${attempts} 次、AMP 跳過 ${finiteMetric(skipped)?skipped:'—'} 次`}).join('；');
      const allAmp=diagnosed.length&&diagnosed.every(epoch=>{const row=item.rows.find(entry=>entry.epoch===epoch);return row['train/optimizer_attempts_epoch']>0&&row['train/optimizer_attempts_epoch']===row['train/optimizer_skipped_epoch']});
      warnings.push(`${item.run.run_id} 的 Epoch ${epochRanges(noUpdates)} 成功權重更新為 0 次，所以沒有「實際更新時的學習率」，並非學習率等於 0。${diagnosed.length?`${diagnosis}。${allAmp?'AMP 因非有限梯度拒絕了全部更新。':'請依嘗試／跳過次數判讀。'}`:'此舊 Run 未記錄更新嘗試與 AMP 跳過原因，無法事後確定；常見原因是 AMP 偵測到非有限梯度。'}`);
    }
    if(unexplained.length)warnings.push(`${item.run.run_id} 的 ${descriptor.label} 缺少 Epoch ${epochRanges(unexplained)}；折線保留缺口，表示該輪沒有有效指標，不代表訓練程序中斷。請查看 Epoch 明細或技術日誌。`);
  }
  const notes=zoomedUnit?[`數值低於 0.1，Y 軸已自動放大至 ${formatMetric(yMax)}；指標原始範圍仍為 0–1。`]:[];
  return {...descriptor, xMax, yMin, yMax, expanded, incompatible, outOfRange, series, allSeries, warnings, notes};
}

export function valuesAtEpoch(model, epoch) {
  return model.series.map(series => ({runId: series.run.run_id, color: series.color,
    label: series.label || series.run.run_id,
    value: series.points.find(point => point.epoch === epoch)?.value ?? null}));
}

const isLearningRate = key => key === 'train/learning_rate' || key.startsWith('lr/');
const rateGroupLabel = key => key === 'train/learning_rate' ? '主學習率' : key.replace(/^lr\/group_/, '參數組 ');

/** One LR panel; merge aliases only when every selected Run has identical points. */
export function buildLearningRateChartModel(options = {}) {
  const runs = options.runs || [], reports = options.reports || new Map();
  const domainRuns = options.domainRuns || runs, domains = options.domains || new Map();
  const keys = metricDescriptors(runs, reports).map(item => item.key).filter(isLearningRate)
    .sort((a, b) => a === 'train/learning_rate' ? -1 : b === 'train/learning_rate' ? 1 : a.localeCompare(b, undefined, {numeric: true}));
  const domainKeys = metricDescriptors(domainRuns, reports).map(item => item.key).filter(isLearningRate);
  const models = new Map([...new Set([...keys, ...domainKeys])].map(key => [key, buildChartModel({...options, domains}, key)]));
  const groups = [];
  for (const key of keys) {
    const model = models.get(key);
    const signature = JSON.stringify(model.allSeries.map(series => [series.run.run_id, series.points]));
    const same = groups.find(group => group.signature === signature);
    if (same) same.keys.push(key);
    else groups.push({signature, keys: [key], model});
  }
  const domainKey = JSON.stringify([domainRuns.map(run => run.run_id).sort(), 'learning-rate-panel']);
  const prior = domains.get(domainKey);
  const bounds = [...models.values()];
  const xMax = Math.max(1, prior?.xMax || 1, ...bounds.map(model => model.xMax));
  const yMax = Math.max(1e-12, prior?.yMax || 0, ...bounds.map(model => model.yMax));
  const yMin = Math.min(0, prior?.yMin || 0, ...bounds.map(model => model.yMin));
  const expanded = Boolean(prior?.expanded || (prior && (xMax > prior.xMax || yMax > prior.yMax || yMin < prior.yMin)));
  domains.set(domainKey, {xMax, yMax, yMin, expanded});
  const series = groups.flatMap((group, index) => group.model.series.filter(item => item.points.length).map(item => ({
    ...item, metricKeys: group.keys, label: `${item.run.run_id} · ${group.keys.map(rateGroupLabel).join('／')}`,
    // Run identity stays in its color; distinguish parameter groups by line style.
    dash: groups.length === 1 ? item.dash : DASHES[index % DASHES.length],
  })));
  const warnings = [...new Set(groups.flatMap(group => group.model.warnings))];
  if (expanded && !warnings.some(message => message.includes('已擴大座標'))) warnings.push('新資料超出原座標範圍，已擴大座標以完整顯示；之後不會自動縮小。');
  const notes = ['每輪最後一步的實測學習率；顏色代表 Run，線型代表不同參數組。'];
  if (groups.some(group => group.keys.length > 1)) notes.push('所選 Run 中完全相同的學習率序列已合併，圖例列出其參數組；Epoch 明細保留各組數值。');
  return {...definition('train/learning_rate'), xMax, yMax, yMin, expanded, incompatible: false,
    outOfRange: false, series, allSeries: groups.flatMap(group => group.model.allSeries), warnings, notes,
    metricGroups: groups.map(group => group.keys)};
}

export function buildChartModels(options, keys) {
  const models = [];
  let hasRate = false;
  for (const key of new Set(keys)) {
    if (isLearningRate(key)) {
      if (!hasRate) models.push(buildLearningRateChartModel(options));
      hasRate = true;
    } else models.push(buildChartModel(options, key));
  }
  return models;
}

function html(document, tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function svgNode(document, tag, attributes = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export function formatMetric(value) {
  if (!finiteMetric(value)) return '—';
  return Math.abs(value) >= 100000 || (value !== 0 && Math.abs(value) < .0001) ? value.toExponential(3) : value.toFixed(4);
}
const format = formatMetric;

/**
 * Owns container contents. Destroy before replacing the view to disconnect its
 * ResizeObserver. `interaction` optionally keeps the inspected Epoch on refresh.
 */
export function createTrainingCharts(container, options = {}) {
  const document = container.ownerDocument, view = document.defaultView;
  const interaction = options.interaction || {};
  const descriptors = metricDescriptors(options.runs, options.reports);
  const keys = options.metricKeys === undefined ? defaultMetricKeys(descriptors) : options.metricKeys;
  const grid = html(document, 'div', 'training-chart-grid');
  const models = buildChartModels(options, keys);
  const charts = [];
  let destroyed = false;
  function showEpoch(epoch) {
    interaction.epoch = epoch;
    for (const chart of charts) chart.showEpoch?.(epoch);
  }
  function draw(chart) {
    if (destroyed) return;
    const {host, model, hover} = chart;
    const width = Math.max(180, Math.floor(host.getBoundingClientRect().width || container.clientWidth || 360));
    const height = 260, left = 53, right = 14, top = 18, bottom = 43;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const x = epoch => left + (model.xMax === 1 ? 0 : (epoch - 1) / (model.xMax - 1) * plotWidth);
    const y = value => top + (model.yMax - value) / (model.yMax - model.yMin) * plotHeight;
    const svg = svgNode(document, 'svg', {viewBox: `0 0 ${width} ${height}`, width: '100%', height,
      role: 'img', 'aria-label': `${model.label}，Epoch 1 到 ${model.xMax}，可使用左右方向鍵檢視數值`,
      'data-x-max': model.xMax, 'data-y-max': model.yMax, 'data-y-min': model.yMin});
    for (let index = 0; index <= 4; index++) {
      const value = model.yMin + (model.yMax - model.yMin) * index / 4;
      const label = Math.abs(value) > 999 || (Math.abs(value) > 0 && Math.abs(value) < .01) ? value.toExponential(1) : value.toFixed(model.unit ? 2 : 1);
      svg.append(svgNode(document, 'line', {x1: left, y1: y(value), x2: width - right, y2: y(value), stroke: '#304550'}),
        svgNode(document, 'text', {x: left - 7, y: y(value) + 4, 'text-anchor': 'end', fill: '#9aafba', 'font-size': 11}, label));
    }
    const tickCount = Math.min(width < 360 ? 3 : 5, model.xMax);
    const ticks = new Set(Array.from({length: tickCount}, (_, index) => Math.round(1 + (model.xMax - 1) * index / Math.max(1, tickCount - 1))));
    for (const tick of ticks) svg.append(svgNode(document, 'text', {x: x(tick), y: height - 23,
      'text-anchor': tick === 1 ? 'start' : tick === model.xMax ? 'end' : 'middle', fill: '#9aafba', 'font-size': 11}, tick));
    svg.append(svgNode(document, 'text', {x: left + plotWidth / 2, y: height - 4, 'text-anchor': 'middle', fill: '#9aafba', 'font-size': 11}, 'Epoch'));
    for (const series of model.series) {
      for (const segment of series.segments) {
        if (segment.length > 1) svg.append(svgNode(document, 'polyline', {
          points: segment.map(point => `${x(point.epoch)},${y(point.value)}`).join(' '), fill: 'none',
          stroke: series.color, 'stroke-width': 2, 'stroke-dasharray': series.dash, 'data-run-id': series.run.run_id}));
      }
      for (const point of series.points) {
        const dot = svgNode(document, 'circle', {cx: x(point.epoch), cy: y(point.value), r: 3,
          fill: series.color, 'data-run-id': series.run.run_id, 'data-epoch': point.epoch});
        dot.append(svgNode(document, 'title', {}, `${series.label || series.run.run_id} · Epoch ${point.epoch} · ${format(point.value)}`));
        svg.append(dot);
      }
    }
    const guide = svgNode(document, 'line', {y1: top, y2: height - bottom, stroke: '#9aafba', 'stroke-dasharray': '3 3', visibility: 'hidden'});
    svg.append(guide);
    const hit = svgNode(document, 'rect', {x: left, y: top, width: plotWidth, height: plotHeight, fill: 'transparent',
      'data-chart-hit': '', tabindex: 0, role: 'slider', 'aria-label': `${model.label}，檢視 Epoch 數值`,
      'aria-valuemin': 1, 'aria-valuemax': model.xMax, 'aria-valuenow': 1});
    chart.showEpoch = epoch => {
      const bounded = Math.min(model.xMax, Math.max(1, Math.round(epoch)));
      guide.setAttribute('x1', x(bounded)); guide.setAttribute('x2', x(bounded));
      guide.setAttribute('visibility', model.series.length ? 'visible' : 'hidden');
      const values = valuesAtEpoch(model, bounded);
      hover.textContent = values.length ? `Epoch ${bounded} · ${values.map(item => `${item.label} ${item.value === null ? '無資料' : format(item.value)}`).join('　｜　')}` : model.incompatible ? '此指標無法疊圖比較。' : '請開啟至少一個 Run 以顯示曲線。';
      hit.setAttribute('aria-valuenow', bounded);
      hit.setAttribute('aria-valuetext', hover.textContent);
    };
    const inspect = event => {
      const rect = svg.getBoundingClientRect();
      const position = (event.clientX - rect.left) * width / Math.max(1, rect.width);
      showEpoch(Math.min(model.xMax, Math.max(1, Math.round(1 + (position - left) / plotWidth * (model.xMax - 1)))));
    };
    hit.addEventListener('pointermove', inspect);
    hit.addEventListener('click', inspect);
    hit.addEventListener('focus', () => chart.showEpoch(interaction.epoch || 1));
    hit.addEventListener('keydown', event => {
      let epoch = Number(hit.getAttribute('aria-valuenow'));
      if (event.key === 'ArrowLeft') epoch--;
      else if (event.key === 'ArrowRight') epoch++;
      else if (event.key === 'Home') epoch = 1;
      else if (event.key === 'End') epoch = model.xMax;
      else return;
      event.preventDefault(); showEpoch(Math.min(model.xMax, Math.max(1, epoch)));
    });
    svg.append(hit);
    host.replaceChildren(svg);
    if (Number.isInteger(interaction.epoch)) chart.showEpoch(interaction.epoch);
  }
  for (const model of models) {
    const card = html(document, 'section', 'training-chart-card'); card.dataset.metricKey = model.key;
    const heading = html(document, 'div', 'training-chart-heading');
    heading.append(html(document, 'h3', '', model.label));
    const latest = html(document, 'div', 'training-chart-latest');
    for (const series of model.series) {
      const point = series.points.at(-1), chip = html(document, series.metricKeys ? 'div' : 'span', '', `${series.label || series.run.run_id} 最新 ${point ? `${format(point.value)} · E${point.epoch}` : '無資料'}`);
      chip.style.setProperty('--series-color', series.color);
      chip.dataset.metricKeys = (series.metricKeys || [model.key]).join(',');
      chip.title = series.dash ? '虛線：' + (series.metricKeys || [model.key]).map(rateGroupLabel).join('／') : '實線';
      if (series.metricKeys) {
        chip.style.cssText += ';display:flex;align-items:center;gap:5px;color:#c4d9e0;min-width:0';
        const swatch = svgNode(document, 'svg', {width: 28, height: 10, 'aria-hidden': 'true'});
        swatch.style.flexShrink = '0';
        swatch.append(svgNode(document, 'line', {x1: 0, y1: 5, x2: 28, y2: 5, stroke: series.color, 'stroke-width': 2, 'stroke-dasharray': series.dash}));
        chip.insertBefore(swatch, chip.firstChild);
      }
      latest.append(chip);
    }
    heading.append(latest); card.append(heading);
    const host = html(document, 'div', 'training-chart-host'), hover = html(document, 'p', 'training-chart-hover', '移到圖表或點選 Epoch，可同時查看各 Run 的數值。');
    if (model.incompatible || !model.series.length) hover.textContent = model.incompatible ? '此指標無法疊圖比較。' : '請開啟至少一個 Run 以顯示曲線。';
    card.append(host, hover);
    for (const note of model.notes || []) card.append(html(document, 'p', 'metric-chart-note', note));
    if (model.direction) card.append(html(document, 'p', 'training-chart-direction', model.direction === 'lower' ? '越低越好' : '越高越好'));
    for (const warning of model.warnings) card.append(html(document, 'p', 'training-chart-warning', warning));
    grid.append(card); charts.push({host, model, hover});
  }
  if (!models.length) grid.append(html(document, 'p', 'metric-empty', descriptors.length ? '請選擇要顯示的指標。' : '尚無可繪製的 Epoch 指標；收到訓練資料後會顯示曲線。'));
  container.replaceChildren(grid);
  charts.forEach(draw);
  const observer = view.ResizeObserver ? new view.ResizeObserver(entries => {
    for (const entry of entries) {
      const chart = charts.find(item => item.host === entry.target);
      if (chart) draw(chart);
    }
  }) : null;
  charts.forEach(chart => observer?.observe(chart.host));
  const resize = () => charts.forEach(draw);
  if (!observer) view.addEventListener('resize', resize);
  return {destroy() { destroyed = true; observer?.disconnect(); if (!observer) view.removeEventListener('resize', resize); }};
}
