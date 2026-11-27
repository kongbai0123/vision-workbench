const kindLabels = {image:'圖片', video:'影片', annotation:'標註原檔'};
const batchSize = 40;
const secondsFormat = new Intl.NumberFormat('zh-TW', {maximumFractionDigits:3});

export function createProjectMaterialsDialog({$, api, element, number}) {
  const dialog = $('projectMaterialsDialog');
  const viewport = $('projectMaterialViewport');
  const list = $('projectMaterialList');
  const status = $('projectMaterialStatus');
  const search = $('projectMaterialSearch');
  const filters = [...$('projectMaterialFilters').querySelectorAll('[data-material-kind]')];
  let currentKind = 'all';
  let materials = [], names = new Map(), matches = [];
  let rendered = 0, request = 0;

  function appendRelations(row, relations, describe, label) {
    const visible = 3;
    row.append(...relations.slice(0, visible).map(relation => element('small', describe(relation))));
    if (relations.length <= visible) return;
    const details = element('details', undefined, 'project-material-relations');
    details.append(element('summary', `其餘 ${number(relations.length - visible)} 筆${label}`));
    details.ontoggle = () => {
      if (!details.open) return;
      details.append(...relations.slice(visible).map(relation => element('small', describe(relation))));
      details.ontoggle = null;
    };
    row.append(details);
  }

  function materialRow(material) {
    const row = element('li', undefined, 'project-material-row');
    row.dataset.materialId = material.id;
    const heading = element('div', undefined, 'project-material-row-heading');
    heading.append(element('b', material.name), element('span', kindLabels[material.kind]));
    if (material.state === 'trashed') heading.append(element('span', '已移至垃圾桶'));
    row.append(heading, element('code', material.relative_path, 'project-material-path'));

    const {video_sources = [], linked_images = []} = material.metadata || {};
    appendRelations(row, video_sources, source => {
      const video = names.get(source.video_id) || source.relative_path;
      const seconds = secondsFormat.format(source.timestamp_seconds);
      const method = source.timestamp_method === 'frame_fps' ? '（依 FPS 估算）' : '';
      return `${video} · 第 ${source.frame_index} 幀 · ${seconds} 秒${method}`;
    }, '取樣來源');
    appendRelations(row, linked_images, link => {
      const image = names.get(link.image_id) || link.image_id;
      return `${image} · 修訂 ${link.annotation_revision}`;
    }, '圖片關聯');
    return row;
  }

  function appendRows() {
    const next = matches.slice(rendered, rendered + batchSize);
    list.append(...next.map(materialRow));
    rendered += next.length;
  }

  function renderMatches() {
    const query = search.value.trim().toLowerCase();
    matches = materials.filter(material =>
      (currentKind === 'all' || material.kind === currentKind) &&
      (!query || `${material.name} ${material.relative_path}`.toLowerCase().includes(query)));
    list.replaceChildren();
    rendered = 0;
    viewport.scrollTop = 0;
    appendRows();
    status.textContent = matches.length
      ? `${number(matches.length)} 筆素材`
      : materials.length ? '沒有符合條件的素材。' : '目前專案尚無素材。';
  }

  function selectKind(kind) {
    currentKind = kind;
    for (const filter of filters) filter.setAttribute('aria-pressed', String(filter.dataset.materialKind === kind));
  }

  function setControls(enabled) {
    search.disabled = !enabled;
    for (const filter of filters) filter.disabled = !enabled;
  }

  for (const filter of filters) filter.onclick = () => {
    selectKind(filter.dataset.materialKind);
    renderMatches();
  };
  search.oninput = renderMatches;
  viewport.onscroll = () => {
    if (rendered < matches.length && viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 120) {
      appendRows();
    }
  };
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    request++;
    materials = matches = [];
    names.clear();
    list.replaceChildren();
  });

  return async function openProjectMaterials(project) {
    if (!project || dialog.open) return;
    const currentRequest = ++request;
    $('projectMaterialProject').textContent = project.name;
    search.value = '';
    selectKind('all');
    list.replaceChildren();
    setControls(false);
    status.textContent = '正在讀取素材清單…';
    dialog.showModal();
    try {
      const result = await api(`/api/projects/${encodeURIComponent(project.id)}/materials`);
      if (currentRequest !== request || !dialog.open) return;
      materials = result.materials;
      names = new Map(materials.map(material => [material.id, material.name]));
      renderMatches();
      setControls(true);
    } catch (error) {
      if (currentRequest === request && dialog.open) status.textContent = `素材清單讀取失敗。 ${error.message}`;
    }
  };
}
