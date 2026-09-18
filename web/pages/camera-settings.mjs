export function controlValues(controls = {}) {
  return Object.fromEntries(Object.entries(controls).map(([key, value]) => [key, {value: value.value, auto: value.auto}]));
}

export function modesForFormat(modes, format) {
  return format === 'auto' ? modes : modes.filter(mode => mode.pixel_format === format);
}

export function createCameraSettings({$, state, api, toast, cameraCommand, cameraConfiguration, updateCameraMode, updateCameraControls}) {
  let busy = false, profiles = {}, profileDevice = '', stagedControls = {}, signature = '';
  const deviceKey = () => `${$('cameraDevice').selectedOptions[0]?.textContent || '相機'} [${$('cameraDevice').value}]`;
  const engaged = () => state.camera || ['starting', 'stopping'].includes(state.cameraDetails?.state);
  const locked = () => busy || state.busy || state.recording || state.autoCapture?.active;
  async function run(action) {
    if (busy) return;
    busy = true; updateCameraControls();
    try { await action(); }
    catch (error) { toast(error.message, true); }
    finally { busy = false; updateCameraControls(); }
  }
  function showProfiles() {
    const previous = $('cameraProfile').value;
    $('cameraProfile').replaceChildren(new Option('選擇設定檔', ''), ...Object.keys(profiles).map(name => new Option(name, name)));
    if (profiles[previous]) $('cameraProfile').value = previous;
  }
  async function loadProfiles() {
    const device = deviceKey();
    profileDevice = device; stagedControls = {}; profiles = {}; showProfiles();
    try {
      const result = await api(`/api/camera/profiles?device=${encodeURIComponent(device)}`);
      if (device !== deviceKey()) return;
      profiles = result.profiles || {}; showProfiles();
    } catch (error) { $('cameraProfileStatus').textContent = error.message; }
    updateCameraControls();
  }
  function addControl(key, spec) {
    const row = document.createElement('div'); row.className = 'camera-parameter'; row.dataset.parameter = key;
    const heading = document.createElement('div'); heading.className = 'camera-parameter-heading';
    const label = document.createElement('label'); label.htmlFor = `cameraParam-${key}`; label.textContent = spec.label;
    heading.append(label);
    const autoLabel = document.createElement('label'); autoLabel.className = 'check-label';
    const auto = document.createElement('input'); auto.type = 'checkbox'; auto.dataset.part = 'auto';
    autoLabel.append(auto, document.createTextNode('自動')); autoLabel.hidden = !spec.supports_auto; heading.append(autoLabel);
    const fields = document.createElement('div'); fields.className = 'camera-parameter-fields';
    const slider = document.createElement('input'); slider.type = 'range'; slider.setAttribute('aria-label', spec.label);
    const number = document.createElement('input'); number.type = 'number'; number.id = label.htmlFor;
    for (const input of [slider, number]) { input.min = spec.min; input.max = spec.max; input.step = spec.step; }
    slider.oninput = () => { number.value = slider.value; };
    const apply = input => {
      const valid = input.checkValidity() && input.value !== '';
      const value = {value: Number(input.value), auto: auto.checked};
      return run(async () => {
        if (!valid) throw Error('請輸入有效範圍與步進值。');
        await cameraCommand('/api/camera/controls', {values: {[key]: value}});
      });
    };
    slider.onchange = () => apply(slider); number.onchange = () => apply(number);
    auto.onchange = () => apply(number);
    fields.append(slider, number); row.append(heading, fields);
    $(spec.advanced ? 'cameraAdvancedControls' : 'cameraImageControls').append(row);
  }
  function render() {
    const details = state.cameraDetails || {}, controls = details.controls || {};
    const next = JSON.stringify(Object.entries(controls).map(([key, s]) => [key, s.min, s.max, s.step, s.supports_auto, s.supports_manual]));
    if (signature !== next) {
      signature = next; $('cameraImageControls').replaceChildren(); $('cameraAdvancedControls').replaceChildren();
      for (const [key, spec] of Object.entries(controls)) addControl(key, spec);
    }
    for (const row of document.querySelectorAll('[data-parameter]')) {
      const spec = controls[row.dataset.parameter], auto = row.querySelector('[data-part="auto"]');
      auto.checked = spec.auto;
      auto.disabled = locked() || !state.camera || !spec.supports_auto || !spec.supports_manual;
      for (const input of row.querySelectorAll('input[type="range"],input[type="number"]')) {
        if (document.activeElement !== input) input.value = spec.value;
        input.disabled = locked() || !state.camera || spec.auto || !spec.supports_manual;
      }
    }
    $('cameraControlsStatus').textContent = !state.camera ? '啟動相機後讀取可用參數。' : details.controls_error || (Object.keys(controls).length ? '調整後即時套用並回讀確認；自動模式下手動值鎖定。' : '此裝置或目前擷取後端未提供可調整參數。');
    $('cameraControlsStatus').classList.toggle('camera-warning', !!details.controls_error);
    $('resetCameraControls').disabled = locked() || !state.camera || !Object.keys(controls).length;
    $('applyCameraSettings').disabled = locked() || !state.camera || details.state === 'stopping';
    $('loadCameraProfile').disabled = locked() || engaged() || !profiles[$('cameraProfile').value];
    $('saveCameraProfile').disabled = locked() || ['starting','stopping'].includes(details.state);
    $('cameraProfile').disabled = locked() || engaged();
    for (const id of ['cameraResolution','cameraFPS','cameraWidth','cameraHeight','cameraPixelFormat']) {
      $(id).disabled ||= locked() || ['starting','stopping'].includes(details.state);
    }
  }
  $('cameraPixelFormat').onchange = updateCameraMode;
  $('cameraProfile').onchange = updateCameraControls;
  $('applyCameraSettings').onclick = () => run(async () => {
    const config = cameraConfiguration();
    config.controls = controlValues(state.cameraDetails?.controls);
    await cameraCommand('/api/camera/stop', {});
    if (state.cameraDetails?.state !== 'stopped') throw Error('相機尚未停止，請等待停止後再啟動。');
    await cameraCommand('/api/camera/start', config);
  });
  $('resetCameraControls').onclick = () => run(() => cameraCommand('/api/camera/controls', {reset: true}));
  $('saveCameraProfile').onclick = () => run(async () => {
    const name = $('cameraProfileName').value.trim();
    if (!name) throw Error('請輸入設定檔名稱。');
    // Save actual device output while running, rather than unapplied form edits.
    const details = state.cameraDetails;
    if (state.camera && (details.fps_reported === false || !details.pixel_format || details.pixel_format === 'unknown')) throw Error('驅動未回報完整輸出模式，請停止相機後儲存要求設定。');
    const settings = state.camera ? {width: details.width, height: details.height, fps: details.fps, pixel_format: details.pixel_format, controls: controlValues(details.controls)} : cameraConfiguration();
    settings.preview_fps = Number($('cameraPreviewFPS').value);
    const result = await api('/api/camera/profiles', 'POST', {device: deviceKey(), name, settings});
    profiles = result.profiles; showProfiles(); $('cameraProfile').value = name;
    $('cameraProfileStatus').textContent = `已儲存「${name}」${state.camera ? '：使用實際輸出與目前影像參數。' : '。'}`;
  });
  $('loadCameraProfile').onclick = () => run(async () => {
    const profile = profiles[$('cameraProfile').value]; if (!profile || engaged()) return;
    const resolution = `${profile.width}x${profile.height}`;
    $('cameraResolution').value = [...$('cameraResolution').options].some(o => o.value === resolution) ? resolution : 'custom';
    $('cameraWidth').value = profile.width; $('cameraHeight').value = profile.height;
    // Rebuild choices for the selected resolution before choosing the saved format.
    updateCameraMode();
    if (![...$('cameraPixelFormat').options].some(o => o.value === profile.pixel_format)) throw Error('目前裝置未提供設定檔的輸出格式，請重新偵測。');
    $('cameraPixelFormat').value = profile.pixel_format; updateCameraMode(); $('cameraFPS').value = profile.fps;
    $('cameraPreviewFPS').value = profile.preview_fps || 15;
    stagedControls = profile.controls || {}; $('cameraProfileName').value = $('cameraProfile').value;
    $('cameraProfileStatus').textContent = '設定已載入；啟動相機後驗證輸出與影像參數。';
  });
  return {render, loadProfiles, get controls() { return profileDevice === deviceKey() ? structuredClone(stagedControls) : {}; }, get busy() { return busy; }};
}
