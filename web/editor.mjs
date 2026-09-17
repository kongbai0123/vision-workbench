import {kind, bounds, vector, validateShape, decodeMask, encodeMask, brush, moved, rotated, resizeOBB, history} from './shapes.mjs';

export const shapeNames = {rectangle:'矩形', obb:'旋轉框', polygon:'多邊形', linestrip:'折線', point:'關鍵點', mask:'遮罩'};
export function colorFor(label) {
  let n = 0;
  for (const c of label) n = (n * 31 + c.charCodeAt(0)) >>> 0;
  return `hsl(${n % 360} 70% 65%)`;
}
export function maskMoved(shape, dx, dy, width, height) {
  dx = Math.round(dx); dy = Math.round(dy);
  const original = decodeMask(shape.counts, width, height), next = new Uint8Array(width * height);
  let minX=width,minY=height,maxX=-1,maxY=-1;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(original[y*width+x]){
    minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
  }
  if(maxX<0)return {...shape,counts:[width*height]};
  // Moving an object must retain its pixels at image boundaries, including holes.
  dx=Math.max(-minX,Math.min(width-1-maxX,dx));dy=Math.max(-minY,Math.min(height-1-maxY,dy));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx >= 0 && sy >= 0 && sx < width && sy < height) next[y * width + x] = original[sy * width + sx];
  }
  return {...shape, counts: encodeMask(next, width, height)};
}
function svgElement(name, attrs = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  return element;
}
function closestEdge(points, point, closed) {
  let result = {distance: Infinity, index: 0};
  for (let i = 0; i < points.length - (closed ? 0 : 1); i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((point.x - a[0]) * dx + (point.y - a[1]) * dy) / (dx*dx + dy*dy || 1)));
    const d = Math.hypot(point.x - a[0] - t * dx, point.y - a[1] - t * dy);
    if (d < result.distance) result = {distance: d, index: i + 1};
  }
  return result;
}

export class AnnotationEditor {
  constructor({onChange, onSelection, onCreated, onNotice, onAssetNavigation}) {
    this.$ = id => document.getElementById(id);
    this.onChange = onChange;
    this.onSelection = onSelection;
    this.onCreated = onCreated;
    this.notice = onNotice;
    this.onAssetNavigation = onAssetNavigation;
    this.asset = null;
    this.tool = 'select';
    this.selection = new Set();
    this.draft = [];
    this.gesture = null;
    this.prompts = {positive: [], negative: []};
    this.candidate = null;this.proposals=[];
    this.history = history(40);
    this.maskCache = new WeakMap();
    this.zoom = 1;
    this.pan = {x: 0, y: 0};
    this.scale = 1;
    this.locked = false;
    this.active = false;
    const properties=document.querySelector('.properties-scroll');
    properties.insertBefore(document.querySelector('.object-section'),properties.children[1]);
    this.installEvents();
    this.resizeObserver = new ResizeObserver(() => this.fit(false));
    this.resizeObserver.observe(this.$('canvasArea'));
  }
  load(asset) {
    this.crosshair.hidden = true;
    this.asset = asset;
    this.asset.shapes ||= [];
    this.selection.clear();
    this.draft = [];
    this.gesture = null;
    this.prompts = {positive: [], negative: []};
    this.candidate = null;this.proposals=[];
    this.history = history(40);
    this.maskCache = new WeakMap();
    this.zoom = 1; this.pan = {x:0, y:0};
    const img = this.$('assetImage');
    img.onload = () => this.fit(false);
    img.onerror = () => this.notice('原圖讀取失敗，請重新開啟圖片。', true);
    img.src = asset.url;
    this.$('canvasEmpty').hidden = true;
    this.$('imageStage').hidden = false;
    this.$('overlay').setAttribute('viewBox', `0 0 ${asset.width} ${asset.height}`);
    this.$('imageDimensions').textContent = `${asset.width} × ${asset.height} px`;
    this.fit();
    this.render();
    this.updateCandidate();
  }
  clear() {
    this.crosshair.hidden = true;
    this.asset = null; this.selection.clear(); this.draft = []; this.candidate = null;this.proposals=[];
    this.$('imageStage').hidden = true;
    this.$('canvasEmpty').hidden = false;
    this.$('shapeList').replaceChildren();
    this.$('shapeCount').textContent = '0 個';
    this.$('imageDimensions').textContent = '— × — px';
  }
  get hasUncommittedWork() { return !!this.draft.length || !!this.gesture || !!this.candidate; }
  setTool(tool) {
    if (this.locked) return;
    if (this.draft.length && !['polygon','linestrip'].includes(tool)) {
      this.notice('請先按 Enter 完成目前頂點，或按 Esc 取消。'); return;
    }
    this.tool = tool;
    document.querySelectorAll('[data-tool]').forEach(button => button.classList.toggle('active', button.dataset.tool === tool));
    const hints = {
      select:'拖曳物件移動 · Shift 多選 · 拖曳控制點調整 · 雙擊邊線插入頂點 · Alt 點頂點刪除',
      rectangle:'拖曳建立矩形 · 類別取自右側物件類別 · Esc 取消操作',
      obb:'拖曳建立旋轉框 · 選取後可使用右側角度旋轉',
      polygon:'逐點建立多邊形 · Enter 或雙擊完成 · Backspace 撤回頂點 · Esc 取消',
      linestrip:'逐點建立折線 · Enter 或雙擊完成 · Backspace 撤回頂點 · Esc 取消',
      point:'點選建立關鍵點 · 選取後可拖曳位置',
      mask:'拖曳塗畫選取的遮罩；未選遮罩時建立新遮罩 · 半徑使用原圖像素',
      erase:'先選取遮罩，再拖曳擦除 · Ctrl Z 復原',
      pan:'拖曳平移 · 滾輪縮放 · 按「適合視窗」重設',
      'ai-positive':'點選物件內部，加入前景提示點 · 提示點不會直接成為標註',
      'ai-negative':'點選需要排除的背景，加入背景提示點',
    };
    const hint = document.getElementById('toolHint');
    if (hint) hint.textContent = hints[tool] || '';
    this.$('overlay').style.cursor = tool === 'pan' ? 'grab' : tool === 'select' ? 'default' : 'crosshair';
    if(tool.startsWith('ai-'))document.querySelector('.ai-section').open=true;
    this.updateToolPanels();
  }
  label() { return this.$('shapeLabel').value.trim(); }
  selected() { return this.asset?.shapes.filter(shape => this.selection.has(shape.id)) || []; }
  fit(reset = true) {
    if (!this.asset) return;
    const area = this.$('canvasArea');
    if (area.clientWidth < 10 || area.clientHeight < 10) return;
    this.scale = Math.min((area.clientWidth-52) / this.asset.width, (area.clientHeight-52) / this.asset.height);
    if (reset) { this.zoom = 1; this.pan = {x:0,y:0}; }
    this.$('imageStage').style.width = `${this.asset.width * this.scale}px`;
    this.$('imageStage').style.height = `${this.asset.height * this.scale}px`;
    this.transform();
  }
  transform() {
    this.$('imageStage').style.transform = `translate(calc(-50% + ${this.pan.x}px), calc(-50% + ${this.pan.y}px)) scale(${this.zoom})`;
    this.$('zoomLabel').textContent = `${Math.round(this.scale * this.zoom * 100)}%`;
    this.renderCanvas();
  }
  changeZoom(factor, event) {
    const before = this.zoom;
    this.zoom = Math.max(.15, Math.min(24, before * factor));
    if (event) {
      const rect = this.$('canvasArea').getBoundingClientRect();
      const x = event.clientX - rect.left - rect.width / 2, y = event.clientY - rect.top - rect.height / 2;
      this.pan.x = x - (x - this.pan.x) * this.zoom / before;
      this.pan.y = y - (y - this.pan.y) * this.zoom / before;
    }
    this.transform();
  }
  focusPoint(x, y, targetPixelSize = 7) {
    if (!this.asset) return;
    this.fit();
    this.zoom = Math.max(1, Math.min(24, targetPixelSize / Math.max(this.scale, .001)));
    this.pan = {
      x: -(x - this.asset.width / 2) * this.scale * this.zoom,
      y: -(y - this.asset.height / 2) * this.scale * this.zoom,
    };
    this.transform();
    this.notice(`已定位至 X=${Math.round(x)}、Y=${Math.round(y)}；可直接使用遮罩筆刷或橡皮擦。`);
  }
  point(event) {
    const rect = this.$('overlay').getBoundingClientRect();
    return {x: Math.max(0, Math.min(this.asset.width, (event.clientX - rect.left) / rect.width * this.asset.width)),
      y: Math.max(0, Math.min(this.asset.height, (event.clientY - rect.top) / rect.height * this.asset.height))};
  }
  commit(before) {
    if (JSON.stringify(before) === JSON.stringify(this.asset.shapes)) { this.render(); return; }
    this.history.push(before);
    this.onChange(this.asset);
    this.render();
  }
  mutate(action) {
    if (this.locked || !this.asset || this.gesture) return;
    const before = structuredClone(this.asset.shapes);
    try { action(); this.commit(before); }
    catch (error) { this.asset.shapes = before; this.notice(error.message, true); this.render(); }
  }
  add(shape) {
    let created = null;
    this.mutate(() => {
      const next = validateShape({...shape, id:crypto.randomUUID()}, this.asset.width, this.asset.height);
      this.asset.shapes.push(next); this.selection = new Set([next.id]);created=next;
    });
    if(created)this.onCreated?.(created);
  }
  choose(id, multi = false) {
    if (!multi) this.selection.clear();
    if (multi && this.selection.has(id)) this.selection.delete(id); else this.selection.add(id);
    this.render();
  }
  travel(direction) {
    if (this.locked || this.gesture || !this.asset) return;
    const shapes = this.history[direction](this.asset.shapes);
    if (!shapes) return;
    this.asset.shapes = shapes; this.selection.clear(); this.draft = [];
    this.onChange(this.asset); this.render();
  }
  finish() {
    if (!this.asset || !this.draft.length) return;
    const minimum = this.tool === 'polygon' ? 3 : 2;
    if (this.draft.length < minimum) { this.notice(`至少需要 ${minimum} 個頂點。`); return; }
    const draft = this.draft;
    this.draft = [];
    this.add(vector(this.tool, draft, this.label()));
  }
  cancel() {
    if (this.gesture?.before && this.asset) this.asset.shapes = this.gesture.before;
    this.gesture = null; this.draft = []; this.render();
  }
  maskURL(shape, candidate = false) {
    if (this.maskCache.has(shape)) return this.maskCache.get(shape);
    const {width,height} = this.asset;
    const data = decodeMask(shape.counts,width,height), canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d'), pixels = context.createImageData(width,height);
    for (let i=0;i<data.length;i++) if (data[i]) {
      pixels.data[i*4] = candidate ? 239 : 69; pixels.data[i*4+1] = candidate ? 196 : 198;
      pixels.data[i*4+2] = candidate ? 91 : 177; pixels.data[i*4+3] = 110;
    }
    context.putImageData(pixels,0,0);
    const url = canvas.toDataURL(); this.maskCache.set(shape,url); return url;
  }
  drawShape(shape, proposal = false) {
    const t = kind(shape), selected = this.selection.has(shape.id), color = proposal ? '#ffcf67' : colorFor(shape.label);
    const scale = this.scale * this.zoom || 1, stroke = (selected ? 2.4 : 1.5) / scale;
    const attrs = {stroke:color, 'stroke-width':stroke, fill:'none', 'data-shape':shape.id || '', 'pointer-events':proposal?'none':'all'};
    let node;
    if (t === 'mask') node = svgElement('image', {href:this.maskURL(shape,proposal),x:0,y:0,width:this.asset.width,height:this.asset.height,'data-shape':shape.id||'','pointer-events':'none'});
    else if (t === 'point') node = svgElement('circle',{...attrs,cx:shape.points[0][0],cy:shape.points[0][1],r:4/scale,fill:color});
    else if (t === 'rectangle') node = svgElement('rect',{...attrs,x:shape.x,y:shape.y,width:shape.width,height:shape.height,fill:color,'fill-opacity':selected?.12:.035});
    else node = svgElement(t === 'linestrip' ? 'polyline' : 'polygon',{...attrs,points:shape.points.map(p=>p.join(',')).join(' '),fill:t==='linestrip'?'none':color,'fill-opacity':selected?.12:.035});
    if (proposal) node.setAttribute('stroke-dasharray',`${6/scale} ${4/scale}`);
    this.$('overlay').append(node);
    if (!proposal && t !== 'mask' && shape.label) {
      const label = svgElement('text',{x:shape.x,y:Math.max(14/scale,shape.y-5/scale),fill:color,'font-size':12/scale,'font-family':'Segoe UI, Microsoft JhengHei, sans-serif','font-weight':600,'paint-order':'stroke','stroke':'#101820','stroke-width':2/scale,'pointer-events':'none'});
      label.textContent = shape.label; this.$('overlay').append(label);
    }
    if (selected && !proposal && t !== 'mask') {
      const points = shape.points || [[shape.x,shape.y],[shape.x+shape.width,shape.y],[shape.x+shape.width,shape.y+shape.height],[shape.x,shape.y+shape.height]];
      points.forEach((point,index) => this.$('overlay').append(svgElement('rect',{x:point[0]-3.5/scale,y:point[1]-3.5/scale,width:7/scale,height:7/scale,fill:'#e5fff5',stroke:'#175753','stroke-width':1/scale,'data-handle':index,'data-owner':shape.id,style:'cursor:crosshair'})));
    }
  }
  renderCanvas() {
    const svg = this.$('overlay'); svg.replaceChildren();
    if (!this.asset) return;
    for (const shape of this.asset.shapes) if (!shape.hidden && !(this.gesture?.kind==='brush' && this.gesture.target===shape.id)) this.drawShape(shape);
    if (this.gesture?.preview) this.drawShape(this.gesture.preview,true);
    if (this.draft.length) {
      const points = [...this.draft, ...(this.hoverPoint?[[this.hoverPoint.x,this.hoverPoint.y]]:[])];
      const scale = this.scale*this.zoom || 1;
      svg.append(svgElement('polyline',{points:points.map(p=>p.join(',')).join(' '),stroke:'#70f2d1','stroke-width':2/scale,fill:'none','stroke-dasharray':`${5/scale} ${3/scale}`,'pointer-events':'none'}));
      for (const [x,y] of this.draft) svg.append(svgElement('circle',{cx:x,cy:y,r:3/scale,fill:'#bcfff1','pointer-events':'none'}));
    }
    for(const proposal of this.proposals)this.drawShape(proposal,true);
    if (this.candidate) this.drawShape(this.candidate,true);
    for (const [category,points] of Object.entries(this.prompts)) for (const [x,y] of points) {
      const scale = this.scale*this.zoom || 1;
      svg.append(svgElement('circle',{cx:x,cy:y,r:5/scale,fill:category==='positive'?'#54e9b2':'#ff8585',stroke:'#0c1a20','stroke-width':1.5/scale,'pointer-events':'none'}));
      const text = svgElement('text',{x,y:y+3/scale,fill:'#102128','text-anchor':'middle','font-size':12/scale,'font-weight':700,'pointer-events':'none'});
      text.textContent = category === 'positive' ? '+' : '−'; svg.append(text);
    }
  }
  render() {
    this.renderCanvas();
    const list = this.$('shapeList'); list.replaceChildren();
    const shapes = this.asset?.shapes || [];
    this.$('shapeCount').textContent = `${shapes.length} 個`;
    for (const shape of shapes) {
      const item = document.createElement('div');item.className='shape-item';
      const row = document.createElement('button'); row.className = 'shape-row' + (this.selection.has(shape.id)?' active':'');
      const dot = document.createElement('span'); dot.className='shape-dot';dot.style.background=colorFor(shape.label);
      const name = document.createElement('span');name.className='shape-name';name.textContent=shape.label;name.title=shape.label;
      const type = document.createElement('small');type.textContent=(shape.hidden?'隱藏 · ':'')+(shapeNames[kind(shape)]||kind(shape));
      row.append(dot,name,type);row.onclick=event=>{if(!this.locked)this.choose(shape.id,event.shiftKey||event.ctrlKey)};
      const select=document.createElement('select');select.className='shape-class-select';select.setAttribute('aria-label',`${shape.label} 的物件類別`);
      for(const option of this.$('shapeLabel').options)if(option.value)select.append(new Option(option.textContent,option.value));
      if(![...select.options].some(option=>option.value===shape.label))select.append(new Option(shape.label,shape.label));
      select.value=shape.label;select.disabled=this.locked;
      select.onchange=()=>this.mutate(()=>{shape.label=select.value;this.selection=new Set([shape.id])});
      item.append(row,select);list.append(item);
    }
    if (!shapes.length) { const p=document.createElement('p');p.textContent='目前圖片尚無標註物件';list.append(p); }
    const selected = this.selected(), shape = selected[0];
    this.$('shapeGeometry').textContent = selected.length > 1 ? `已選取 ${selected.length} 個物件` : shape ? `${shapeNames[kind(shape)]}  x ${shape.x.toFixed(1)} / y ${shape.y.toFixed(1)}\nw ${shape.width.toFixed(1)} / h ${shape.height.toFixed(1)}` : '選取物件可檢視幾何資訊';
    this.$('undo').disabled=!this.history.canUndo||this.locked;
    this.$('redo').disabled=!this.history.canRedo||this.locked;
    for (const id of ['copyShapes','hideShapes','deleteShapes']) this.$(id).disabled=!selected.length||this.locked;
    this.$('rotateShape').disabled=!shape||!['rectangle','obb'].includes(kind(shape))||this.locked;
    this.$('finishShape').disabled=!this.draft.length||this.locked;
    this.updateToolPanels();
    this.onSelection?.(selected);
  }
  updateToolPanels() {
    const brush=['mask','erase'].includes(this.tool);
    const rotate=this.selected().some(s=>['rectangle','obb'].includes(kind(s)));
    const finish=['polygon','linestrip'].includes(this.tool)||!!this.draft.length;
    this.$('brushSettings').hidden=!brush;this.$('rotateSettings').hidden=!rotate;
    this.$('finishSettings').hidden=!finish;this.$('toolSettings').hidden=!(brush||rotate||finish);
  }
  installEvents() {
    const svg = this.$('overlay'), area = this.$('canvasArea');
    // Screen-space guides stay one pixel wide at every image zoom level.
    const guides = document.createElement('div');
    guides.setAttribute('aria-hidden', 'true');
    guides.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:2';
    guides.hidden = true;
    const horizontal = document.createElement('div'), vertical = document.createElement('div');
    const lineStyle = 'position:absolute;pointer-events:none;background:rgba(112,242,209,.8);box-shadow:0 0 1px 1px rgba(0,0,0,.45);';
    horizontal.style.cssText = lineStyle + 'left:0;right:0;height:1px;top:0';
    vertical.style.cssText = lineStyle + 'top:0;bottom:0;width:1px;left:0';
    guides.append(horizontal, vertical);
    area.append(guides);
    this.crosshair = guides;
    const updateGuides = event => {
      const rect = area.getBoundingClientRect();
      const x = event.clientX - rect.left - area.clientLeft;
      const y = event.clientY - rect.top - area.clientTop;
      guides.hidden = !this.asset || this.locked || event.pointerType === 'touch' ||
        x < 0 || y < 0 || x >= area.clientWidth || y >= area.clientHeight;
      if (guides.hidden) return;
      horizontal.style.transform = `translateY(${Math.round(y)}px)`;
      vertical.style.transform = `translateX(${Math.round(x)}px)`;
    };
    area.addEventListener('pointermove', updateGuides);
    area.addEventListener('pointerenter', updateGuides);
    area.addEventListener('pointerleave', () => { guides.hidden = true; });
    area.addEventListener('pointercancel', () => { guides.hidden = true; });
    window.addEventListener('blur', () => { guides.hidden = true; });
    document.querySelectorAll('[data-tool]').forEach(button => button.onclick=()=>this.setTool(button.dataset.tool));
    svg.addEventListener('pointerdown',event=>this.pointerDown(event));
    svg.addEventListener('pointermove',event=>this.pointerMove(event));
    svg.addEventListener('pointerup',event=>this.pointerUp(event));
    svg.addEventListener('pointercancel',()=>this.cancel());
    svg.addEventListener('dblclick',event=>{
      if(this.locked||!this.asset)return;
      event.preventDefault();
      if(['polygon','linestrip'].includes(this.tool)) {this.finish();return;}
      const shape=this.asset.shapes.find(s=>s.id===event.target.dataset.shape);
      if(this.tool==='select'&&shape&&['polygon','linestrip'].includes(kind(shape))) {
        const edge=closestEdge(shape.points,this.point(event),kind(shape)==='polygon');
        if(edge.distance*this.scale*this.zoom<15)this.mutate(()=>{const p=this.point(event);shape.points.splice(edge.index,0,[p.x,p.y]);Object.assign(shape,bounds(shape.points));});
      }
    });
    area.addEventListener('wheel',event=>{if(!this.asset)return;event.preventDefault();this.changeZoom(event.deltaY<0?1.13:1/1.13,event)},{passive:false});
    this.$('undo').onclick=()=>this.travel('undo');this.$('redo').onclick=()=>this.travel('redo');
    this.$('finishShape').onclick=()=>this.finish();this.$('cancelDrawing').onclick=()=>this.cancel();
    this.$('zoomIn').onclick=()=>this.changeZoom(1.25);this.$('zoomOut').onclick=()=>this.changeZoom(.8);this.$('fitImage').onclick=()=>this.fit();
    this.$('copyShapes').onclick=()=>this.mutate(()=>{
      const copies=this.selected().map(shape=>({...structuredClone(kind(shape)==='mask'?maskMoved(shape,8,8,this.asset.width,this.asset.height):moved(shape,8,8,this.asset.width,this.asset.height)),id:crypto.randomUUID()}));
      this.asset.shapes.push(...copies);this.selection=new Set(copies.map(s=>s.id));
    });
    this.$('hideShapes').onclick=()=>this.mutate(()=>{for(const shape of this.selected())shape.hidden=!shape.hidden});
    this.$('deleteShapes').onclick=()=>this.removeSelected();
    this.$('selectAllShapes').onclick=()=>{if(!this.locked&&this.asset){this.selection=new Set(this.asset.shapes.map(s=>s.id));this.render()}};
    this.$('rotateShape').onclick=()=>this.mutate(()=>{for(const shape of this.selected())if(['rectangle','obb'].includes(kind(shape)))Object.assign(shape,rotated(shape,Number(this.$('rotateDegrees').value)||0,this.asset.width,this.asset.height))});
    this.$('clearPrompts').onclick=()=>{if(!this.locked){this.prompts={positive:[],negative:[]};this.renderCanvas()}};
    this.$('acceptAI').onclick=()=>{
      if(!this.candidate||this.locked)return;
      const shape=this.candidate;this.candidate=null;this.add(shape);this.prompts={positive:[],negative:[]};this.updateCandidate();
      this.$('aiMessage').textContent='候選已加入標註，可使用筆刷進一步修正。';
    };
    this.$('discardAI').onclick=()=>{if(!this.locked){this.candidate=null;this.updateCandidate();this.renderCanvas();this.$('aiMessage').textContent='已捨棄候選，現有標註保持原狀。'}};
    document.addEventListener('keydown',event=>this.keyDown(event));
  }
  pointerDown(event) {
    if(this.locked||!this.asset||event.button!==0)return;
    areaFocus(this.$('canvasArea'));
    event.preventDefault();
    if(event.detail>1&&['polygon','linestrip'].includes(this.tool))return;
    const creates=['rectangle','obb','polygon','linestrip','point'].includes(this.tool)||(this.tool==='mask'&&!this.selected().some(shape=>kind(shape)==='mask'));
    if(creates&&!this.label()){this.notice('請先按「新增／刪除」建立類別，並選擇這個物件的類別。',true);return;}
    const point=this.point(event), shape=this.asset.shapes.find(s=>s.id===event.target.dataset.shape);
    if(this.tool==='pan'||event.altKey&&event.target.dataset.handle===undefined&&event.shiftKey) {
      this.gesture={kind:'pan',start:{x:event.clientX,y:event.clientY},pan:{...this.pan}};
    } else if(this.tool.startsWith('ai-')) {
      const key=this.tool==='ai-positive'?'positive':'negative';
      if(this.prompts.positive.length+this.prompts.negative.length>=64){this.notice('提示點最多 64 個。');return;}
      this.prompts[key].push([point.x,point.y]);this.renderCanvas();return;
    } else if(this.tool==='select') {
      const owner=this.asset.shapes.find(s=>s.id===event.target.dataset.owner),handle=Number(event.target.dataset.handle);
      if(owner&&Number.isInteger(handle)) {
        if(event.altKey&&['polygon','linestrip'].includes(kind(owner))) {
          const min=kind(owner)==='polygon'?3:2;
          if(owner.points.length<=min){this.notice('刪除後頂點不足。');return;}
          this.mutate(()=>{owner.points.splice(handle,1);Object.assign(owner,bounds(owner.points))});return;
        }
        this.gesture={kind:'handle',before:structuredClone(this.asset.shapes),owner:owner.id,handle,start:point};
      } else if(shape) {
        if(event.shiftKey||event.ctrlKey){this.choose(shape.id,true);return;}
        if(!this.selection.has(shape.id))this.choose(shape.id);
        this.gesture={kind:'move',before:structuredClone(this.asset.shapes),start:point};
      } else if(this.selected().some(s=>kind(s)==='mask')) {
        this.gesture={kind:'move',before:structuredClone(this.asset.shapes),start:point};
      } else {this.selection.clear();this.render();return;}
    } else if(['polygon','linestrip'].includes(this.tool)) {
      if(!this.draft.length||Math.hypot(this.draft.at(-1)[0]-point.x,this.draft.at(-1)[1]-point.y)>.05)this.draft.push([point.x,point.y]);
      this.render();return;
    } else if(this.tool==='point') {this.add(vector('point',[[point.x,point.y]],this.label()));return;
    } else if(['mask','erase'].includes(this.tool)) {
      if(this.asset.width*this.asset.height>16777216){this.notice('遮罩編輯上限為 1,677 萬像素。',true);return;}
      const target=this.selected().find(s=>kind(s)==='mask');
      if(this.tool==='erase'&&!target){this.notice('請先從物件清單選取一個遮罩。');return;}
      this.gesture={kind:'brush',before:structuredClone(this.asset.shapes),target:target?.id,last:point,data:target?decodeMask(target.counts,this.asset.width,this.asset.height):new Uint8Array(this.asset.width*this.asset.height)};
      this.paint(point);
    } else this.gesture={kind:'draw',start:point,before:structuredClone(this.asset.shapes)};
    try{this.$('overlay').setPointerCapture(event.pointerId)}catch{}
  }
  pointerMove(event) {
    if(this.locked||!this.asset)return;
    const point=this.point(event);this.hoverPoint=point;
    const gesture=this.gesture;
    if(!gesture){if(this.draft.length)this.renderCanvas();return;}
    if(gesture.kind==='pan') {
      this.pan={x:gesture.pan.x+event.clientX-gesture.start.x,y:gesture.pan.y+event.clientY-gesture.start.y};this.transform();return;
    }
    if(gesture.kind==='draw') {
      const x=Math.min(point.x,gesture.start.x),y=Math.min(point.y,gesture.start.y),w=Math.abs(point.x-gesture.start.x),h=Math.abs(point.y-gesture.start.y);
      gesture.preview=this.tool==='obb'?vector('obb',[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],this.label()):{type:'rectangle',x,y,width:w,height:h,label:this.label()};
    } else if(gesture.kind==='move') {
      const dx=point.x-gesture.start.x,dy=point.y-gesture.start.y;
      this.asset.shapes=gesture.before.map(shape=>this.selection.has(shape.id)?(kind(shape)==='mask'?maskMoved(shape,dx,dy,this.asset.width,this.asset.height):moved(shape,dx,dy,this.asset.width,this.asset.height)):structuredClone(shape));
    } else if(gesture.kind==='handle') {
      const original=gesture.before.find(s=>s.id===gesture.owner),shape=structuredClone(original),i=gesture.handle;
      try {
        if(kind(shape)==='rectangle') {
          const corners=[[shape.x,shape.y],[shape.x+shape.width,shape.y],[shape.x+shape.width,shape.y+shape.height],[shape.x,shape.y+shape.height]],opposite=corners[(i+2)%4];
          Object.assign(shape,{x:Math.min(point.x,opposite[0]),y:Math.min(point.y,opposite[1]),width:Math.abs(point.x-opposite[0]),height:Math.abs(point.y-opposite[1])});
        } else if(kind(shape)==='obb') Object.assign(shape,resizeOBB(shape,i,point,this.asset.width,this.asset.height));
        else {shape.points[i]=[point.x,point.y];Object.assign(shape,bounds(shape.points));}
        validateShape(shape,this.asset.width,this.asset.height);
        this.asset.shapes=this.asset.shapes.map(s=>s.id===shape.id?shape:s);
      } catch {}
    } else if(gesture.kind==='brush')this.paint(point);
    this.renderCanvas();
  }
  paint(point) {
    const gesture=this.gesture,width=this.asset.width,height=this.asset.height;
    const radius=Math.max(1,Math.min(200,Number(this.$('brushRadius').value)||12));
    brush(gesture.data,width,height,gesture.last,point,radius,this.tool==='erase');gesture.last=point;
    const target=gesture.before.find(s=>s.id===gesture.target);
    gesture.preview={...target,type:'mask',label:target?.label||this.label(),x:0,y:0,width,height,counts:encodeMask(gesture.data,width,height)};
    this.renderCanvas();
  }
  pointerUp(event) {
    const gesture=this.gesture;if(!gesture||!this.asset)return;
    this.gesture=null;
    try{this.$('overlay').releasePointerCapture(event.pointerId)}catch{}
    if(gesture.kind==='pan')return;
    if(gesture.kind==='draw') {
      if(!gesture.preview||gesture.preview.width*this.scale*this.zoom<2||gesture.preview.height*this.scale*this.zoom<2){this.renderCanvas();return;}
      this.add(gesture.preview);return;
    }
    if(gesture.kind==='brush') {
      const shape={...gesture.preview,id:gesture.target||crypto.randomUUID()};
      if(!gesture.data.some(n=>n)) {
        if(gesture.target)this.asset.shapes=this.asset.shapes.filter(s=>s.id!==gesture.target);
      } else if(gesture.target)this.asset.shapes=this.asset.shapes.map(s=>s.id===gesture.target?shape:s);
      else this.asset.shapes.push(shape);
      this.selection=new Set([shape.id]);
    }
    this.commit(gesture.before);
  }
  removeSelected() {this.mutate(()=>{this.asset.shapes=this.asset.shapes.filter(shape=>!this.selection.has(shape.id));this.selection.clear()});}
  keyDown(event) {
    if(!this.active||this.locked||document.querySelector('dialog[open]')||event.target.closest?.('input,textarea,select,[contenteditable=true]'))return;
    const key=event.key.toLowerCase();
    if(event.ctrlKey||event.metaKey) {
      if(key==='z'){event.preventDefault();this.travel(event.shiftKey?'redo':'undo');}
      else if(key==='y'){event.preventDefault();this.travel('redo');}
      else if(key==='a'){event.preventDefault();this.$('selectAllShapes').click();}
      return;
    }
    if(key==='escape'){event.preventDefault();this.cancel();return;}
    if(key==='enter'){event.preventDefault();this.finish();return;}
    if(key==='backspace'&&this.draft.length){event.preventDefault();this.draft.pop();this.render();return;}
    if(key==='delete'){event.preventDefault();this.removeSelected();return;}
    if(['arrowright','arrowleft','a','d'].includes(key)){event.preventDefault();this.onAssetNavigation(['arrowright','d'].includes(key)?1:-1);return;}
    const tools={v:'select',r:'rectangle',p:'polygon',l:'linestrip',k:'point',o:'obb',b:'mask',e:'erase',h:'pan'};
    if(tools[key]){event.preventDefault();this.setTool(tools[key]);}
  }
  aiInput() {
    const box=this.selected().find(s=>['rectangle','obb'].includes(kind(s)));
    return {points:this.prompts.positive,negative_points:this.prompts.negative,...(box?{box:[box.x,box.y,box.x+box.width,box.y+box.height]}:{})};
  }
  setCandidate(shape) {
    this.candidate=validateShape({...shape,label:shape.label||this.label()},this.asset.width,this.asset.height);
    this.updateCandidate();this.renderCanvas();
  }
  setProposals(shapes=[]) {
    if(!this.asset)return;this.proposals=shapes.map(shape=>validateShape({...shape},this.asset.width,this.asset.height));this.renderCanvas();
  }
  updateCandidate() {
    this.$('acceptAI').disabled=!this.candidate||this.locked;
    this.$('discardAI').disabled=!this.candidate||this.locked;
    this.$('canvasNotice').hidden=!this.candidate;
    this.$('canvasNotice').textContent='候選遮罩預覽 · 請在右側接受或捨棄';
  }
}
function areaFocus(area) {area.focus({preventScroll:true});}
