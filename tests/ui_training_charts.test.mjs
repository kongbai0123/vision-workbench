import test from 'node:test';
import assert from 'node:assert/strict';
import {finiteMetric, normalizedMetricRows, metricDescriptors, defaultMetricKeys,
  runAppearance, comparisonWarnings, buildChartModel, valuesAtEpoch, formatMetric} from '../web/training-charts.mjs';

const run = (id, options = {}) => ({run_id: id, engine: 'maskrcnn_resnet50_fpn',
  dataset_version_id: 'D001', config: {epochs: 10, image_size: 640}, ...options});
const reportsFor = (...entries) => new Map(entries.map(([item, metrics]) => [item.run_id, {run: item, metrics}]));

test('small model thresholds remain nonzero in chart and Epoch value labels', () => {
  assert.equal(formatMetric(1e-8), '1.000e-8');
  assert.equal(formatMetric(2e-5), '2.000e-5');
  assert.equal(formatMetric(0), '0.0000');
  assert.equal(formatMetric(null), '—');
});

test('metric values reject coercion of missing, boolean and string input', () => {
  for (const value of [null, undefined, false, true, '', '0', [], {}, Infinity, NaN]) assert.equal(finiteMetric(value), false);
  assert.equal(finiteMetric(0), true);
  assert.equal(finiteMetric(-.12), true);
  const item = run('R1');
  const reports = reportsFor([item, [{epoch: 1, 'train/loss': null, 'val/accuracy': false},
    {epoch: 2, 'train/loss': '0', 'val/accuracy': ''}, {epoch: 3, 'val/accuracy': 0}]]);
  assert.deepEqual(metricDescriptors([item], reports).map(item => item.key), ['val/accuracy']);
  assert.deepEqual(buildChartModel({runs: [item], reports}, 'train/loss').series[0].points, []);
  assert.deepEqual(buildChartModel({runs: [item], reports}, 'val/accuracy').series[0].points, [{epoch: 3, value: 0}]);
});

test('invalid Epochs are excluded and duplicate observations merge in file order', () => {
  const rows = normalizedMetricRows([{epoch: 3, 'train/loss': .4}, {epoch: 1, 'train/loss': .7},
    {epoch: 1, 'val/mean_iou': .8}, {epoch: 3, 'train/loss': null},
    {epoch: '2', 'train/loss': 99}, {epoch: 2.5}, {epoch: 0}, {epoch: -1}]);
  assert.deepEqual(rows, [{epoch: 1, 'train/loss': .7, 'val/mean_iou': .8}, {epoch: 3, 'train/loss': null}]);
});

test('descriptors discover actual model metrics and default to loss plus a primary score', () => {
  const item = run('R1'), reports = reportsFor([item, [{epoch: 1, timestamp: 999, 'val/macro_f1': .6,
    'val/accuracy': .8, 'train/loss': .4, 'val/macro_recall': .5, 'train/learning_rate': .001}]]);
  const descriptors = metricDescriptors([item], reports);
  assert.deepEqual(defaultMetricKeys(descriptors), ['train/loss', 'val/accuracy']);
  assert.equal(descriptors.some(item => item.key === 'timestamp'), false);
  assert.equal(descriptors.some(item => item.key === 'train/learning_rate'), true);
  assert.equal(descriptors.find(item => item.key === 'val/accuracy').unit, true);
  assert.deepEqual(descriptors.find(item => item.key === 'val/accuracy').availableRunIds, ['R1']);
  const baseline = run('R2', {engine: 'pixel_prototype_v1'});
  assert.deepEqual(defaultMetricKeys(metricDescriptors([baseline], reportsFor([baseline,
    [{epoch: 1, threshold: .2, 'val/mean_iou': .7}]]))), ['val/mean_iou']);
});

test('mAP50–95 takes precedence over mAP50 as primary evaluation metric', () => {
  const item = run('R1', {engine: 'yolo26n_seg'});
  const descriptors = metricDescriptors([item], reportsFor([item,
    [{epoch: 1, 'train/loss': .4, 'val/mask_map50': .8, 'val/mask_map50_95': .6}]]));
  assert.deepEqual(defaultMetricKeys(descriptors), ['train/loss', 'val/mask_map50_95']);
});

test('a stale report for another Run cannot be plotted as the selected Run', () => {
  const item = run('R1'), other = run('R2');
  const reports = new Map([['R1', {run: other, metrics: [{epoch: 1, 'train/loss': 1}]}]]);
  assert.deepEqual(metricDescriptors([item], reports), []);
  assert.deepEqual(buildChartModel({runs: [item], reports}, 'train/loss').series[0].points, []);
});

test('Run visibility changes neither axis limits nor stable series appearance', () => {
  const first = run('R1', {config: {epochs: 10}}), second = run('R2', {config: {epochs: 20}});
  const runs = [first, second], reports = reportsFor([first, [{epoch: 1, 'train/loss': .4}]],
    [second, [{epoch: 25, 'train/loss': 4.4}]]), domains = new Map();
  const both = buildChartModel({runs, reports, comparison: true, domains}, 'train/loss');
  const one = buildChartModel({runs, reports, comparison: true, domains, visibleRunIds: new Set(['R1'])}, 'train/loss');
  const none = buildChartModel({runs, reports, comparison: true, domains, visibleRunIds: new Set()}, 'train/loss');
  assert.equal(both.xMax, 25); assert.equal(one.xMax, 25); assert.equal(none.xMax, 25);
  assert.equal(both.yMax, one.yMax); assert.equal(one.yMax, none.yMax);
  assert.equal(one.series[0].color, both.series[0].color);
  assert.equal(none.series.length, 0);
  const secondOnly = buildChartModel({runs, reports, comparison: true, domains, visibleRunIds: new Set(['R2'])}, 'train/loss');
  assert.equal(secondOnly.series[0].color, both.series[1].color);
  assert.equal(secondOnly.series[0].dash, both.series[1].dash);
  assert.notEqual(runAppearance(first, 0).color, runAppearance(second, 1).color);
});

test('stable appearance allocation keeps a Run color and line style when another Run is removed or reordered', () => {
  const first = run('R1', {appearanceIndex: 0}), second = run('R2', {appearanceIndex: 1});
  const reports = reportsFor(...[first, second].map(item => [item, [{epoch: 1, 'train/loss': 1}]]));
  const before = buildChartModel({runs: [first, second], reports, comparison: true}, 'train/loss');
  const removed = buildChartModel({runs: [second], reports, comparison: true}, 'train/loss');
  const reordered = buildChartModel({runs: [second, first], reports, comparison: true}, 'train/loss');
  assert.equal(before.series[1].color, removed.series[0].color);
  assert.equal(before.series[1].color, reordered.series[0].color);
  assert.equal(before.series[1].dash, reordered.series[0].dash);
  assert.deepEqual(runAppearance(run('R3', {appearanceIndex: '2'}), 1), runAppearance(second, 0));
  assert.deepEqual(runAppearance(run('R3', {appearanceIndex: Infinity}), 1), runAppearance(second, 0));
});

test('comparison appearance supports more than six saved models without undefined or repeated palette slots', () => {
  const appearances=Array.from({length:12},(_,index)=>runAppearance(run(`R${index}`,{appearanceIndex:index}),0));
  assert.equal(new Set(appearances.map(item=>item.color)).size,12);
  assert.ok(appearances.every(item=>typeof item.color==='string'&&typeof item.dash==='string'));
});

test('loss limits only expand to fit new outliers and never follow falling losses', () => {
  const item = run('R1'), domains = new Map();
  const create = value => buildChartModel({runs: [item], domains,
    reports: reportsFor([item, [{epoch: 1, 'train/loss': value}]])}, 'train/loss');
  const initial = create(1.7), decreased = create(.2), outlier = create(4.1), after = create(.1);
  assert.equal(initial.yMax, decreased.yMax);
  assert.equal(initial.expanded, false); assert.equal(decreased.expanded, false);
  assert.ok(outlier.yMax >= 4.1); assert.ok(outlier.yMax > initial.yMax);
  assert.equal(after.yMax, outlier.yMax); assert.equal(after.expanded, true);
  assert.match(outlier.warnings.join(' '), /已擴大座標/);
});

test('domain cache is separate between comparison selections and between metrics', () => {
  const first = run('R1'), second = run('R2'), domains = new Map();
  const reports = reportsFor([first, [{epoch: 1, 'train/loss': .4, 'val/mean_iou': .8}]],
    [second, [{epoch: 1, 'train/loss': 20}]]);
  assert.ok(buildChartModel({runs: [first, second], reports, domains}, 'train/loss').yMax >= 20);
  assert.equal(buildChartModel({runs: [first], reports, domains}, 'train/loss').yMax, 1);
  assert.equal(buildChartModel({runs: [first], reports, domains}, 'val/mean_iou').yMax, 1);
});

test('unchecked model candidates preserve comparison coordinates without blocking checked compatible Runs', () => {
  const first = run('R1', {appearanceIndex: 0}), second = run('R2', {appearanceIndex: 1});
  const other = run('R3', {engine: 'deeplabv3_resnet50', config: {epochs: 30, image_size: 640}, appearanceIndex: 2});
  const domainRuns = [first, second, other], domains = new Map();
  const reports = reportsFor([first, [{epoch: 1, 'train/loss': .4, 'val/mean_iou': .8}]],
    [second, [{epoch: 1, 'train/loss': 1.7, 'val/mean_iou': .7}]],
    [other, [{epoch: 1, 'train/loss': 3.4, 'val/mean_iou': .9}]]);
  const checked = buildChartModel({runs: [first, second], domainRuns, reports, domains, comparison: true}, 'train/loss');
  const unchecked = buildChartModel({runs: [first], domainRuns, reports, domains, comparison: true}, 'train/loss');
  assert.equal(checked.incompatible, false);
  assert.equal(unchecked.incompatible, false);
  assert.equal(checked.xMax, 30); assert.equal(checked.xMax, unchecked.xMax);
  assert.equal(checked.yMax, unchecked.yMax);
  assert.deepEqual(unchecked.series.map(series => series.run.run_id), ['R1']);
  assert.equal(checked.series[0].color, unchecked.series[0].color);
  assert.equal(buildChartModel({runs: [first, second], domainRuns, reports, domains, comparison: true}, 'val/mean_iou').incompatible, false);
  assert.equal(buildChartModel({runs: [first, other], domainRuns, reports, domains, comparison: true}, 'val/mean_iou').incompatible, true);
});

test('missing Epochs split line segments and exact Epoch comparison never substitutes neighboring values', () => {
  const first = run('R1'), second = run('R2');
  const reports = reportsFor([first, [{epoch: 1, 'train/loss': 1}, {epoch: 2, 'train/loss': .9},
    {epoch: 3, 'train/loss': null}, {epoch: 4, 'train/loss': .6}, {epoch: 5, 'train/loss': .5}, {epoch: 7, 'train/loss': .4}]],
    [second, [{epoch: 1, 'train/loss': 1.2}, {epoch: 3, 'train/loss': .7}]]);
  const model = buildChartModel({runs: [first, second], reports, comparison: true}, 'train/loss');
  assert.deepEqual(model.series[0].segments.map(segment => segment.map(point => point.epoch)), [[1, 2], [4, 5], [7]]);
  assert.deepEqual(valuesAtEpoch(model, 3).map(item => [item.runId, item.value]), [['R1', null], ['R2', .7]]);
  assert.deepEqual(valuesAtEpoch(model, 8).map(item => item.value), [null, null]);
});

test('unit metrics retain 0–1 axes and reject out-of-range points instead of clamping them', () => {
  const item = run('R1'), reports = reportsFor([item, [{epoch: 1, 'val/accuracy': -.2},
    {epoch: 2, 'val/accuracy': .6}, {epoch: 3, 'val/accuracy': 1.2}]]);
  const model = buildChartModel({runs: [item], reports}, 'val/accuracy');
  assert.equal(model.yMin, 0); assert.equal(model.yMax, 1); assert.equal(model.outOfRange, true);
  assert.deepEqual(model.series[0].points, [{epoch: 2, value: .6}]);
  assert.match(model.warnings.join(' '), /未繪製異常值/);
});

test('different IoU procedures cannot be overlaid just because they share a field name', () => {
  const first = run('R1'), second = run('R2', {engine: 'deeplabv3_resnet50'});
  const reports = reportsFor([first, [{epoch: 1, 'val/mean_iou': .8}]], [second, [{epoch: 1, 'val/mean_iou': .9}]]);
  const model = buildChartModel({runs: [first, second], reports, comparison: true}, 'val/mean_iou');
  assert.equal(model.incompatible, true); assert.deepEqual(model.series, []);
  assert.match(model.warnings.join(' '), /計算定義/);
  assert.equal(comparisonWarnings([first, second], ['val/mean_iou']).length, 1);
});

test('semantic IoU with different image size is blocked but same-size model variants share the metric', () => {
  const first = run('R1', {engine: 'deeplabv3_resnet50'});
  const second = run('R2', {engine: 'deeplabv3_mobilenet_v3_large', config: {epochs: 10, image_size: 320}});
  const reports = reportsFor([first, [{epoch: 1, 'val/mean_iou': .8}]], [second, [{epoch: 1, 'val/mean_iou': .9}]]);
  assert.equal(buildChartModel({runs: [first, second], reports, comparison: true}, 'val/mean_iou').incompatible, true);
  second.config.image_size = 640;
  assert.equal(buildChartModel({runs: [first, second], reports, comparison: true}, 'val/mean_iou').incompatible, false);
});

test('different engine loss is blocked while a common evaluation metric remains available', () => {
  const first = run('R1', {engine: 'fasterrcnn_mobilenet_v3_large_fpn'});
  const second = run('R2', {engine: 'fasterrcnn_resnet50_fpn_v2'});
  const reports = reportsFor(...[first, second].map(item => [item, [{epoch: 1, 'train/loss': 1, 'val/box_mean_iou': .8}]]));
  assert.equal(buildChartModel({runs: [first, second], reports, comparison: true}, 'train/loss').incompatible, true);
  assert.equal(buildChartModel({runs: [first, second], reports, comparison: true}, 'val/box_mean_iou').incompatible, false);
});

test('Test fallback cannot be compared as the same split as Validation', () => {
  const first = run('R1', {metricSplit: 'val'}), second = run('R2', {metricSplit: 'test'});
  const reports = reportsFor(...[first, second].map(item => [item, [{epoch: 1, 'val/accuracy': .8}]]));
  assert.equal(buildChartModel({runs: [first, second], reports, comparison: true}, 'val/accuracy').incompatible, true);
  assert.match(comparisonWarnings([first, second], ['val/accuracy']).join(' '), /不同評估分割/);
  assert.match(comparisonWarnings([first, {...second, dataset_version_id: 'D002'}], []).join(' '), /不同資料版本/);
  assert.match(buildChartModel({runs: [second], reports}, 'val/accuracy').label, /Test（訓練期間評估）/);
  assert.doesNotMatch(metricDescriptors([second], reports)[0].label, /Validation/);
});
