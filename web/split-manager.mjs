const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
export class SplitManager {
  constructor({api,onApplied}){
    this.api=api;this.onApplied=onApplied;
    this.dialog=node('dialog',undefined,'smart-split-dialog');this.dialog.id='smartSplitDialog';
    this.dialog.setAttribute('aria-labelledby','smartSplitTitle');document.body.append(this.dialog);
    this.dialog.addEventListener('cancel',e=>{if(this.busy)e.preventDefault()});
  }
  async open(projectId){
    if(this.busy)return;
    this.pid=projectId;this.options={strategy:'smart',ratios:{train:70,val:20,test:10},seed:42,locks:{},group_overrides:{},preserve_test:false};this.plan=null;
    this.dialog.replaceChildren();
    const heading=node('div',undefined,'section-heading'),title=node('h2','資料分割管理');title.id='smartSplitTitle';
    this.close=node('button','關閉','secondary');this.close.onclick=()=>this.dialog.close();heading.append(title,this.close);
    this.dialog.append(heading,node('p','調整目前專案的已核准資料。來源群組優先保持完整，再平衡類別與比例；既有固定版本不會被改寫。','muted'));
    this.message=node('p','','split-manager-message');this.message.setAttribute('role','status');this.dialog.append(this.message);
    this.layout=node('div',undefined,'smart-split-layout');this.dialog.append(this.layout);
    this.settings=node('section',undefined,'smart-split-settings');this.content=node('section',undefined,'smart-split-content');this.layout.append(this.settings,this.content);
    this.select('分割策略','splitStrategy',[['smart','智慧分割 · 來源隔離＋類別平衡'],['class_balanced','獨立圖片 · 類別平衡']],this.options.strategy,v=>{this.options.strategy=v;this.options.locks={}});
    for(const [s,label] of [['train','Train %'],['val','Validation %'],['test','Test %']])this.number(label,`smartRatio-${s}`,this.options.ratios[s],0,100,v=>this.options.ratios[s]=v);
    this.number('隨機種子','smartSplitSeed',42,0,2147483647,v=>this.options.seed=v);
    const keep=node('label',undefined,'check-label'),check=node('input');check.type='checkbox';check.id='smartKeepTest';check.onchange=()=>{this.options.preserve_test=check.checked;this.invalidate()};keep.append(check,document.createTextNode('保留目前 Test（整組鎖定）'));this.settings.append(keep);
    this.preview=node('button','預覽智慧分割','primary full');this.preview.id='previewSmartSplit';this.preview.onclick=()=>this.runPreview();this.settings.append(this.preview);
    this.summary=node('div',undefined,'smart-split-summary');this.groups=node('div',undefined,'training-table-wrap');this.content.append(this.summary,this.groups);
    this.manual=node('details',undefined,'training-monitor-details');this.manual.append(node('summary','手動定義獨立圖片群組'));this.assets=node('div',undefined,'smart-split-assets');this.manual.append(node('p','同名會合併；留空沿用拍攝批次。僅在確認圖片屬於獨立情境時拆分批次，同影片與重複圖片仍保持同組。','muted'),this.assets);this.dialog.append(this.manual);
    this.footer=node('div',undefined,'smart-split-footer');this.apply=node('button','套用到目前專案','primary');this.apply.id='applySmartSplit';this.apply.disabled=true;
    this.apply.onclick=()=>this.runApply(false);this.create=node('button','套用並建立新資料版本','primary');this.create.id='applySmartSplitVersion';this.create.disabled=true;this.create.onclick=()=>this.runApply(true);
    this.footer.append(this.apply,this.create);this.dialog.append(this.footer);if(!this.dialog.open)this.dialog.showModal();
    await this.work(async()=>{
      const info=await this.api(`/api/projects/${this.pid}/split-info`,'POST',{});this.info=info;
      this.message.textContent=`已核准 ${info.assets.length} 張 · ${info.groups.length} 個來源群組。請先預覽方案。`;
      this.renderGroups(info.groups);const ids=new Set(info.assets.map(a=>a.id));const prior=Object.fromEntries(Object.entries(info.previous?.options?.group_overrides||{}).filter(([id])=>ids.has(id)));this.options.group_overrides={...prior};
      for(const asset of info.assets){const row=node('label',undefined,'smart-split-asset'),img=node('img');img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';
        const text=node('span',`${asset.name} · ${asset.batch_id||'未指定來源'}`),input=node('input');input.type='text';input.maxLength=128;input.placeholder='沿用來源批次';input.value=prior[asset.id]||'';input.setAttribute('aria-label',`${asset.name} 的手動分割群組`);
        input.oninput=()=>{this.options.group_overrides[asset.id]=input.value;this.options.locks={};this.invalidate()};row.append(img,text,input);this.assets.append(row)}
    });
  }
  select(label,id,options,value,change){const l=node('label',label);l.htmlFor=id;const input=node('select');input.id=id;for(const [v,t]of options)input.append(new Option(t,v));input.value=value;input.onchange=()=>{change(input.value);this.invalidate()};this.settings.append(l,input)}
  number(label,id,value,min,max,change){const l=node('label',label);l.htmlFor=id;const input=node('input');Object.assign(input,{id,type:'number',min,max,step:1,value});input.oninput=()=>{change(input.value.trim()===''?null:Number(input.value));this.invalidate()};this.settings.append(l,input)}
  invalidate(){this.plan=null;this.apply.disabled=true;this.create.disabled=true;this.message.textContent='設定已變更，請重新預覽。'}
  async work(fn){if(this.busy)return;this.busy=true;const controls=[...this.dialog.querySelectorAll('input,select,button')];const disabled=controls.map(x=>x.disabled);controls.forEach(x=>x.disabled=true);
    try{await fn()}catch(e){this.message.textContent=e.message;this.message.setAttribute('role','alert')}
    finally{controls.forEach((x,i)=>x.disabled=disabled[i]);this.busy=false;this.apply.disabled=this.create.disabled=!this.plan;this.message.setAttribute('aria-live','polite')}
  }
  async runPreview(){this.plan=null;await this.work(async()=>{
    const plan=await this.api(`/api/projects/${this.pid}/split-preview`,'POST',{options:this.options});this.plan=plan;
    this.summary.replaceChildren();for(const s of ['train','val','test'])this.summary.append(node('div',`${s.toUpperCase()} · ${plan.image_counts[s]} 張 · ${plan.actual_ratios[s].toFixed(1)}%`));
    this.renderGroups(plan.groups);this.renderClasses(plan);
    this.message.textContent=`預覽：將調整 ${plan.changed} 張。${plan.warnings.join('；')||'來源群組完整，請確認實際比例。'}`;
  })}
  renderGroups(groups){this.groups.replaceChildren();const table=node('table'),head=node('thead'),tr=node('tr');for(const t of ['來源／群組','圖片','目前集合','建議集合','指定／鎖定'])tr.append(node('th',t));head.append(tr);table.append(head);const body=node('tbody');
    for(const group of groups){const row=node('tr');row.dataset.groupId=group.id;
      const name=node('td'),img=node('img');const asset=this.info?.assets?.find(a=>a.id===group.asset_ids[0]);if(asset){img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';name.append(img)}name.append(document.createTextNode((group.names||group.sources).join(' / ')));
      row.append(name,node('td',String(group.images)),node('td',Object.entries(group.current).map(([s,n])=>`${s} ${n}`).join(' / ')),node('td',group.proposed||'尚未預覽'));
      const cell=node('td'),select=node('select');select.setAttribute('aria-label',(group.names||group.sources).join(' / ')+'指定集合');select.append(new Option('自動分配',''));for(const s of ['train','val','test'])select.append(new Option(s,s));select.value=this.options.locks[group.id]||'';
      select.onchange=()=>{if(select.value)this.options.locks[group.id]=select.value;else delete this.options.locks[group.id];this.invalidate()};cell.append(select);row.append(cell);body.append(row)}table.append(body);this.groups.append(table)
  }
  renderClasses(plan){const table=node('table'),head=node('tr');for(const s of ['類別實例數','Train','Val','Test'])head.append(node('th',s));table.append(head);for(const name of Object.keys(plan.class_totals)){const row=node('tr');row.append(node('td',name));for(const s of ['train','val','test'])row.append(node('td',String(plan.class_counts[s][name]||0)));table.append(row)}this.groups.append(node('h3','類別分布'),table)}
  async runApply(createVersion){if(!this.plan)return;const plan=this.plan;await this.work(async()=>{
    const result=await this.api(`/api/projects/${this.pid}/split-apply`,'POST',{options:plan.options,revision:plan.project_revision,fingerprint:plan.fingerprint});this.plan=null;
    await this.onApplied(result,this.pid,createVersion);this.dialog.close();
  })}
}
