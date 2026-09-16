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
    this.pid=projectId;this.options={purpose:'reviewed_independent',strategy:'multilabel',balance_mode:'hybrid',source_isolation:false,ratios:{train:70,val:20,test:10},seed:42,locks:{},group_overrides:{},preserve_test:false};this.plan=null;
    this.dialog.replaceChildren();
    const heading=node('div',undefined,'section-heading'),title=node('h2','資料分割管理');title.id='smartSplitTitle';
    this.close=node('button','關閉','secondary');this.close.onclick=()=>this.dialog.close();heading.append(title,this.close);
    this.dialog.append(heading,node('p','先選擇資料用途，再以整張圖片分配所有類別。完全相同的圖片永遠不跨集合；預覽不會變更專案。','muted'));
    this.message=node('p','','split-manager-message');this.message.setAttribute('role','status');this.dialog.append(this.message);
    this.layout=node('div',undefined,'smart-split-layout');this.dialog.append(this.layout);
    this.settings=node('section',undefined,'smart-split-settings');this.content=node('section',undefined,'smart-split-content');this.layout.append(this.settings,this.content);
    this.purpose=this.select('資料用途','splitPurpose',[['reviewed_independent','已審核獨立樣本（建議）'],['formal','正式泛化評估'],['experimental','寬鬆實驗'],['all_train','全資料最終訓練']],this.options.purpose,v=>{this.options.purpose=v;this.applyPurpose()});
    this.balance=this.select('多類別平衡目標','splitStrategy',[['hybrid','綜合平衡（建議）'],['presence','類別出現張數'],['instances','物件數量'],['cooccurrence','類別共現配對']],this.options.balance_mode,v=>{this.options.balance_mode=v;this.options.locks={}});
    this.purposeHelp=node('p','','muted');this.settings.append(this.purposeHelp);
    const isolation=node('label',undefined,'check-label'),isolate=node('input');isolate.type='checkbox';isolate.id='splitSourceIsolation';isolate.checked=true;
    isolate.disabled=true;isolation.append(isolate,document.createTextNode('正式模式隔離來源群組；其他模式依用途自動設定'));this.settings.append(isolation);
    this.ratioInputs={};for(const [s,label] of [['train','Train %'],['val','Validation %'],['test','Test %']])this.ratioInputs[s]=this.number(label,`smartRatio-${s}`,this.options.ratios[s],0,100,v=>this.options.ratios[s]=v);
    this.number('隨機種子','smartSplitSeed',42,0,2147483647,v=>this.options.seed=v);
    const keep=node('label',undefined,'check-label'),check=node('input');check.type='checkbox';check.id='smartKeepTest';this.keepTest=check;check.onchange=()=>{this.options.preserve_test=check.checked;this.invalidate()};keep.append(check,document.createTextNode('保留目前 Test（整組鎖定）'));this.settings.append(keep);this.applyPurpose();
    this.preview=node('button','預覽智慧分割','primary full');this.preview.id='previewSmartSplit';this.preview.onclick=()=>this.runPreview();this.settings.append(this.preview);
    this.summary=node('div',undefined,'smart-split-summary');this.metrics=node('div',undefined,'training-table-wrap');this.groups=node('div',undefined,'training-table-wrap split-group-table');this.content.append(this.summary,this.metrics,this.groups);
    this.manual=node('details',undefined,'training-monitor-details');this.manual.append(node('summary','手動定義不可拆分群組'));this.assets=node('div',undefined,'smart-split-assets');this.manual.append(node('p','同名會合併。正式模式留空時沿用拍攝批次與影片來源；其他模式按圖片分配，但相同圖片與手動同名群組永遠保持同組。','muted'),this.assets);this.dialog.append(this.manual);
    this.footer=node('div',undefined,'smart-split-footer');this.apply=node('button','套用到目前專案','primary');this.apply.id='applySmartSplit';this.apply.disabled=true;
    this.apply.onclick=()=>this.runApply(false);this.create=node('button','套用並建立新資料版本','primary');this.create.id='applySmartSplitVersion';this.create.disabled=true;this.create.onclick=()=>this.runApply(true);
    this.footer.append(this.apply,this.create);this.dialog.append(this.footer);if(!this.dialog.open)this.dialog.showModal();
    await this.work(async()=>{
      const info=await this.api(`/api/projects/${this.pid}/split-info`,'POST',{});this.info=info;
      this.message.textContent=`已核准 ${info.assets.length} 張 · ${info.groups.length} 個來源群組 · 樣本獨立確認${info.independence_review?.current?'有效':'未完成或已失效'}。請先預覽方案。`;
      this.renderGroups(info.groups);const ids=new Set(info.assets.map(a=>a.id));const prior=Object.fromEntries(Object.entries(info.previous?.options?.group_overrides||{}).filter(([id])=>ids.has(id)));this.options.group_overrides={...prior};
      for(const asset of info.assets){const row=node('label',undefined,'smart-split-asset'),img=node('img');img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';
        const text=node('span',`${asset.name} · ${asset.batch_id||'未指定來源'}`),input=node('input');input.type='text';input.maxLength=128;input.placeholder='沿用來源批次';input.value=prior[asset.id]||'';input.setAttribute('aria-label',`${asset.name} 的手動分割群組`);
        input.oninput=()=>{this.options.group_overrides[asset.id]=input.value;this.options.locks={};this.invalidate()};row.append(img,text,input);this.assets.append(row)}
    });
  }
  select(label,id,options,value,change){const l=node('label',label);l.htmlFor=id;const input=node('select');input.id=id;for(const [v,t]of options)input.append(new Option(t,v));input.value=value;input.onchange=()=>{change(input.value);this.invalidate()};this.settings.append(l,input);return input}
  number(label,id,value,min,max,change){const l=node('label',label);l.htmlFor=id;const input=node('input');Object.assign(input,{id,type:'number',min,max,step:1,value});input.oninput=()=>{change(input.value.trim()===''?null:Number(input.value));this.invalidate()};this.settings.append(l,input);return input}
  applyPurpose(){
    const purpose=this.options.purpose,formal=purpose==='formal',balanced=['formal','reviewed_independent'].includes(purpose),all=purpose==='all_train';
    this.options.strategy=purpose==='experimental'?'random_loose':'multilabel';this.options.source_isolation=formal;this.options.locks={};
    if(all){this.options.ratios={train:100,val:0,test:0};for(const [s,input]of Object.entries(this.ratioInputs||{})){input.value=this.options.ratios[s];input.disabled=true}}
    else {if(this.options.ratios.val===0){this.options.ratios={train:70,val:20,test:10};for(const [s,input]of Object.entries(this.ratioInputs||{}))input.value=this.options.ratios[s]}for(const input of Object.values(this.ratioInputs||{}))input.disabled=false}
    if(this.balance)this.balance.disabled=!balanced;if(this.keepTest){this.keepTest.disabled=all;this.keepTest.checked=all?false:this.keepTest.checked;this.options.preserve_test=all?false:this.keepTest.checked}
    const isolation=this.dialog.querySelector('#splitSourceIsolation');if(isolation)isolation.checked=formal;
    if(this.purposeHelp)this.purposeHelp.textContent=({formal:'以來源群組隔離 Train／Validation／Test，適合正式泛化評估；類別缺口採嚴格阻擋。',reviewed_independent:'需先在資料審核確認樣本獨立；可按圖片使用四種多類別平衡，Validation 缺類別改為警告。',experimental:'依圖片隨機分配；來源與類別缺口只警告，結果僅供流程或快速實驗。',all_train:'全部核准圖片用於最終訓練，不建立獨立 Validation／Test，也不產生可比較的泛化分數。'})[purpose];
  }
  invalidate(){this.plan=null;this.apply.disabled=true;this.create.disabled=true;this.message.textContent='設定已變更，請重新預覽。'}
  async work(fn){if(this.busy)return;this.busy=true;const controls=[...this.dialog.querySelectorAll('input,select,button')];const disabled=controls.map(x=>x.disabled);controls.forEach(x=>x.disabled=true);
    try{await fn()}catch(e){this.message.textContent=e.message;this.message.setAttribute('role','alert')}
    finally{controls.forEach((x,i)=>x.disabled=disabled[i]);this.busy=false;this.apply.disabled=this.create.disabled=!this.plan||this.plan.ready===false||!!this.plan.blockers?.length;this.message.setAttribute('aria-live','polite')}
  }
  async runPreview(){this.plan=null;await this.work(async()=>{
    const plan=await this.api(`/api/projects/${this.pid}/split-preview`,'POST',{options:this.options});this.plan=plan;
    this.summary.replaceChildren();for(const s of ['train','val','test'])this.summary.append(node('div',`${s.toUpperCase()} · ${plan.image_counts[s]} 張 · ${plan.actual_ratios[s].toFixed(1)}%（目標 ${plan.ratios[s]}%，差 ${(plan.ratio_deviation?.[s]||0).toFixed(1)} 個百分點）`));
    this.renderGroups(plan.groups);this.renderClasses(plan);
    this.message.textContent=`${plan.blockers?.length?'無法套用：'+plan.blockers.map(x=>x.message).join('；'):`預覽：將調整 ${plan.changed} 張。`} ${plan.warnings.join('；')||'來源群組完整，請確認實際比例。'}`;
  })}
  renderGroups(groups){this.groups.replaceChildren();const table=node('table'),head=node('thead'),tr=node('tr');for(const t of ['來源／群組','圖片','目前集合','建議集合','指定／鎖定'])tr.append(node('th',t));head.append(tr);table.append(head);const body=node('tbody');
    for(const group of groups){const row=node('tr');row.dataset.groupId=group.id;
      const name=node('td'),img=node('img');const asset=this.info?.assets?.find(a=>a.id===group.asset_ids[0]);if(asset){img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';name.append(img)}name.append(document.createTextNode((group.names||group.sources).join(' / ')));
      row.append(name,node('td',String(group.images)),node('td',Object.entries(group.current).map(([s,n])=>`${s} ${n}`).join(' / ')),node('td',group.proposed||'尚未預覽'));
      const cell=node('td'),select=node('select');select.setAttribute('aria-label',(group.names||group.sources).join(' / ')+'指定集合');select.append(new Option('自動分配',''));for(const s of ['train','val','test'])select.append(new Option(s,s));select.value=this.options.locks[group.id]||'';
      select.onchange=()=>{if(select.value)this.options.locks[group.id]=select.value;else delete this.options.locks[group.id];this.invalidate()};cell.append(select);row.append(cell);body.append(row)}table.append(body);this.groups.append(table)
  }
  renderClasses(plan){
    this.metrics.replaceChildren();
    const table=node('table'),head=node('tr');
    for(const text of ['類別','總圖片／物件','來源群組／分配群組','Train 圖片／物件','Val 圖片／物件','Test 圖片／物件'])head.append(node('th',text));table.append(head);
    for(const name of Object.keys(plan.class_totals).sort()){
      const row=node('tr');row.append(node('td',name),node('td',`${plan.class_image_totals?.[name]||0} / ${plan.class_totals[name]}`),node('td',`${plan.class_source_counts?.[name]||0} / ${plan.class_group_counts?.[name]||0}`));
      for(const s of ['train','val','test'])row.append(node('td',`${plan.class_image_counts?.[s]?.[name]||0} / ${plan.class_counts[s][name]||0}`));table.append(row);
    }
    this.metrics.append(node('h3','多類別分布'),node('p','同一張圖片可計入多個類別。來源群組由拍攝批次／影片記錄判定；分配群組反映目前隔離與手動設定。','muted'),table);
    const details=node('details');details.append(node('summary',`樣本不足提示（${plan.scarcity?.length||0} 個類別）`));
    for(const item of plan.scarcity||[])details.append(node('p',`${item.label}：${item.messages.join('；')}`));
    if(plan.scarcity?.length)details.open=true;this.metrics.append(details);
    const pairs=node('details');pairs.append(node('summary','常見類別配對分布'));
    const pairTable=node('table'),pairHead=node('tr');for(const text of ['同圖配對','總張數','Train','Val','Test'])pairHead.append(node('th',text));pairTable.append(pairHead);
    for(const pair of plan.pair_distribution||[]){const row=node('tr');row.append(node('td',pair.labels.join(' ＋ ')),node('td',String(pair.total)));for(const s of ['train','val','test'])row.append(node('td',String(pair.counts[s])));pairTable.append(row)}
    pairs.append(node('p','顯示至少出現於集合數量那麼多張圖片的配對，最多 256 組。配對平衡為偏好，並非每個組合都必須覆蓋所有集合。','muted'),pairTable);this.metrics.append(pairs);
  }
  async runApply(createVersion){if(!this.plan||this.plan.ready===false||this.plan.blockers?.length)return;const plan=this.plan;await this.work(async()=>{
    const result=await this.api(`/api/projects/${this.pid}/split-apply`,'POST',{options:plan.options,revision:plan.project_revision,fingerprint:plan.fingerprint});this.plan=null;
    await this.onApplied(result,this.pid,createVersion);this.dialog.close();
  })}
}
