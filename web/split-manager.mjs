const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};

const purposes={
  reviewed_independent:{title:'圖片層級平衡',badge:'建議',description:'按圖片平衡類別與物件數，適合一般訓練。'},
  formal:{title:'依來源分組',badge:'正式評估',description:'同一批次或影片不跨集合，適合評估新來源。'},
  all_train:{title:'全部用於訓練',badge:'最終模型',description:'全部放入 Train，不建立泛化評估。'},
  experimental:{title:'寬鬆實驗',badge:'進階',description:'依圖片隨機分配，只適合快速流程實驗。'},
};

export function visibleScarcityMessages(item,sourceIsolation){
  return (item?.messages||[]).filter(message=>sourceIsolation||!message.includes('來源群組'));
}

export class SplitManager {
  constructor({api,onApplied}){
    this.api=api;this.onApplied=onApplied;
    this.dialog=node('dialog',undefined,'smart-split-dialog');this.dialog.id='smartSplitDialog';
    this.dialog.setAttribute('aria-labelledby','smartSplitTitle');document.body.append(this.dialog);
    this.dialog.addEventListener('cancel',event=>{if(this.busy)event.preventDefault()});
  }

  async open(projectId){
    if(this.busy)return;
    this.pid=projectId;this.plan=null;
    this.options={purpose:'reviewed_independent',strategy:'multilabel',balance_mode:'hybrid',source_isolation:false,ratios:{train:70,val:20,test:10},seed:42,locks:{},group_overrides:{},preserve_test:false};
    this.dialog.replaceChildren();
    this.buildHeader();this.buildSteps();this.buildWorkspace();this.buildFooter();
    if(!this.dialog.open)this.dialog.showModal();

    await this.work(async()=>{
      const info=await this.api(`/api/projects/${this.pid}/split-info`,'POST',{});this.info=info;
      const ids=new Set(info.assets.map(asset=>asset.id));
      const saved=info.previous?.options;
      if(saved){this.options={...this.options,...saved,ratios:{...this.options.ratios,...saved.ratios}};}
      const prior=Object.fromEntries(Object.entries(info.previous?.options?.group_overrides||{}).filter(([id])=>ids.has(id)));
      this.options.group_overrides={...prior};
      for(const asset of info.assets)this.addAssetOverride(asset,prior[asset.id]||'');
      this.datasetSummary.textContent=`${info.assets.length} 張已核准圖片 · ${info.groups.length} 筆來源紀錄`;
      this.message.textContent='正在依目前設定建立預覽…';
    });
    this.purpose.value=this.options.purpose;this.balance.input.value=this.options.balance_mode;
    this.seed.value=this.options.seed;this.keepTest.checked=!!this.options.preserve_test;
    const locks=this.options.locks;this.applyPurpose();this.options.locks=locks;
    await this.runPreview();
  }

  buildHeader(){
    const header=node('header',undefined,'smart-split-header');
    const copy=node('div'),eyebrow=node('div','DATA SPLIT','eyebrow'),title=node('h2','資料分割管理');title.id='smartSplitTitle';
    copy.append(eyebrow,title,node('p','選擇用途、調整比例，再檢查結果。預覽不會更動專案。','muted'));
    this.close=node('button','關閉','secondary');this.close.onclick=()=>this.dialog.close();header.append(copy,this.close);this.dialog.append(header);
  }

  buildSteps(){
    const steps=node('ol',undefined,'split-stepper');
    for(const [number,label] of [['1','選擇方式'],['2','調整比例'],['3','檢查並套用']]){
      const item=node('li');item.append(node('span',number),node('strong',label));steps.append(item);
    }
    this.dialog.append(steps);
  }

  buildWorkspace(){
    this.layout=node('div',undefined,'smart-split-layout');this.dialog.append(this.layout);
    this.settings=node('section',undefined,'smart-split-settings');this.settings.setAttribute('aria-label','分割設定');
    this.content=node('section',undefined,'smart-split-content');this.content.setAttribute('aria-label','分割預覽');this.layout.append(this.settings,this.content);

    this.settings.append(node('h3','1. 選擇資料用途'));
    this.purpose=this.makeSelect('splitPurpose',Object.entries(purposes).map(([value,item])=>[value,item.title]),this.options.purpose,value=>{this.options.purpose=value;this.applyPurpose()});
    this.purpose.className='visually-hidden';this.purpose.setAttribute('aria-label','資料用途');this.settings.append(this.purpose);
    this.purposeCards=node('div',undefined,'split-purpose-cards');this.settings.append(this.purposeCards);
    for(const value of ['reviewed_independent','formal','all_train'])this.purposeCards.append(this.makePurposeCard(value));

    this.settings.append(node('h3','2. 調整比例'));
    this.ratioGrid=node('div',undefined,'split-ratio-grid');this.settings.append(this.ratioGrid);this.ratioInputs={};
    for(const [split,label] of [['train','Train'],['val','Validation'],['test','Test']])this.ratioInputs[split]=this.makeNumber(this.ratioGrid,label,`smartRatio-${split}`,this.options.ratios[split],0,100,value=>this.options.ratios[split]=value,'%');
    this.ratioTotal=node('p','合計 100%','split-ratio-total');this.settings.append(this.ratioTotal);

    this.balance=this.makeLabeledSelect('多類別平衡目標','splitStrategy',[['hybrid','綜合平衡（建議）'],['presence','類別出現張數'],['instances','物件數量'],['cooccurrence','類別共現配對']],this.options.balance_mode,value=>{this.options.balance_mode=value;this.options.locks={}});
    this.settings.append(this.balance.wrapper);

    const advanced=node('details',undefined,'split-advanced');advanced.append(node('summary','進階設定'));
    const isolation=node('label',undefined,'check-label'),isolate=node('input');isolate.type='checkbox';isolate.id='splitSourceIsolation';isolate.disabled=true;isolation.append(isolate,document.createTextNode('依來源隔離（選擇正式評估時自動啟用）'));
    this.seed=this.makeNumber(advanced,'重現種子','smartSplitSeed',42,0,2147483647,value=>this.options.seed=value);
    const keep=node('label',undefined,'check-label'),check=node('input');check.type='checkbox';check.id='smartKeepTest';this.keepTest=check;check.onchange=()=>{this.options.preserve_test=check.checked;this.invalidate()};keep.append(check,document.createTextNode('保留目前 Test（整組鎖定）'));
    this.experimental=this.makePurposeCard('experimental');this.experimental.classList.add('compact');
    advanced.append(isolation,keep,node('p','快速試驗','split-advanced-label'),this.experimental);this.settings.append(advanced);

    this.preview=node('button','重新預覽','secondary full');this.preview.id='previewSmartSplit';this.preview.onclick=()=>this.runPreview();this.settings.append(this.preview);

    const resultHeader=node('div',undefined,'split-result-header'),resultTitle=node('div');resultTitle.append(node('h3','3. 檢查分割結果'));this.datasetSummary=node('p','讀取資料中…','muted');resultTitle.append(this.datasetSummary);resultHeader.append(resultTitle);this.content.append(resultHeader);
    this.message=node('p','','split-manager-message');this.message.setAttribute('role','status');this.content.append(this.message);
    this.summary=node('div',undefined,'smart-split-summary');this.content.append(this.summary);
    this.tabs=node('div',undefined,'split-tabs');this.tabs.setAttribute('role','tablist');this.content.append(this.tabs);
    this.panels={};
    for(const [key,label] of [['overview','總覽'],['issues','需要注意'],['classes','類別分布'],['groups','手動調整']]){
      const button=node('button',label,'split-tab');button.type='button';button.dataset.panel=key;button.setAttribute('role','tab');button.onclick=()=>this.showPanel(key);this.tabs.append(button);
      const panel=node('div',undefined,'split-panel');panel.dataset.panel=key;panel.setAttribute('role','tabpanel');this.panels[key]=panel;this.content.append(panel);
    }
    this.overview=this.panels.overview;this.issues=this.panels.issues;this.metrics=this.panels.classes;this.groups=this.panels.groups;
    this.manual=node('details',undefined,'training-monitor-details split-manual-groups');this.manual.append(node('summary','定義不可拆分群組'));this.assets=node('div',undefined,'smart-split-assets');
    this.manual.append(node('p','輸入相同群組名稱可讓多張圖片一起分配。正式評估若留空，會沿用批次或影片來源。','muted'),this.assets);this.groups.append(this.manual);
    this.showPanel('overview');this.applyPurpose();
  }

  buildFooter(){
    this.footer=node('footer',undefined,'smart-split-footer');this.footerStatus=node('span','先完成預覽','muted');
    const actions=node('div',undefined,'split-footer-actions');this.apply=node('button','套用到目前專案','primary');this.apply.id='applySmartSplit';this.apply.disabled=true;this.apply.onclick=()=>this.runApply(false);
    this.create=node('button','套用並建立新資料版本','secondary');this.create.id='applySmartSplitVersion';this.create.disabled=true;this.create.onclick=()=>this.runApply(true);
    actions.append(this.create,this.apply);this.footer.append(this.footerStatus,actions);this.dialog.append(this.footer);
  }

  makePurposeCard(value){
    const item=purposes[value],button=node('button',undefined,'split-purpose-card');button.type='button';button.dataset.purpose=value;
    const heading=node('span',undefined,'split-purpose-title');heading.append(node('strong',item.title),node('small',item.badge));button.append(heading,node('span',item.description,'muted'));
    button.onclick=()=>{this.purpose.value=value;this.purpose.dispatchEvent(new Event('change'))};return button;
  }

  makeSelect(id,options,value,change){
    const input=node('select');input.id=id;for(const [option,label] of options)input.append(new Option(label,option));input.value=value;
    input.onchange=()=>{change(input.value);this.invalidate()};return input;
  }

  makeLabeledSelect(label,id,options,value,change){
    const wrapper=node('div',undefined,'split-field'),caption=node('label',label);caption.htmlFor=id;const input=this.makeSelect(id,options,value,change);wrapper.append(caption,input);return{wrapper,input};
  }

  makeNumber(parent,label,id,value,min,max,change,suffix=''){
    const wrapper=node('label',undefined,'split-number');wrapper.htmlFor=id;wrapper.append(node('span',label));const line=node('span',undefined,'split-number-input'),input=node('input');
    Object.assign(input,{id,type:'number',min,max,step:1,value});input.oninput=()=>{change(input.value.trim()===''?null:Number(input.value));this.updateRatioTotal();this.invalidate()};line.append(input);if(suffix)line.append(node('span',suffix));wrapper.append(line);parent.append(wrapper);return input;
  }

  addAssetOverride(asset,value){
    const row=node('label',undefined,'smart-split-asset'),img=node('img');img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';
    const text=node('span',`${asset.name} · ${asset.batch_id||'未指定來源'}`),input=node('input');input.type='text';input.maxLength=128;input.placeholder='自動';input.value=value;input.setAttribute('aria-label',`${asset.name} 的手動分割群組`);
    input.oninput=()=>{this.options.group_overrides[asset.id]=input.value;this.options.locks={};this.invalidate()};row.append(img,text,input);this.assets.append(row);
  }

  applyPurpose(){
    const purpose=this.options.purpose,formal=purpose==='formal',balanced=['formal','reviewed_independent'].includes(purpose),all=purpose==='all_train';
    this.options.strategy=purpose==='experimental'?'random_loose':'multilabel';this.options.source_isolation=formal;this.options.locks={};
    if(all)this.options.ratios={train:100,val:0,test:0};
    else if(this.options.ratios.val===0)this.options.ratios={train:70,val:20,test:10};
    for(const [split,input] of Object.entries(this.ratioInputs||{})){input.value=this.options.ratios[split];input.disabled=all}
    if(this.balance)this.balance.input.disabled=!balanced;
    if(this.keepTest){this.keepTest.disabled=all;if(all)this.keepTest.checked=false;this.options.preserve_test=all?false:this.keepTest.checked}
    const isolation=this.dialog.querySelector('#splitSourceIsolation');if(isolation)isolation.checked=formal;
    for(const card of this.dialog.querySelectorAll('.split-purpose-card')){const selected=card.dataset.purpose===purpose;card.classList.toggle('selected',selected);card.setAttribute('aria-pressed',String(selected))}
    this.updateRatioTotal();
  }

  updateRatioTotal(){
    if(!this.ratioTotal)return;const total=Object.values(this.options.ratios).reduce((sum,value)=>sum+(Number(value)||0),0);this.ratioTotal.textContent=`合計 ${total}%`;this.ratioTotal.classList.toggle('invalid',total!==100);
  }

  showPanel(key){
    for(const button of this.tabs.querySelectorAll('.split-tab')){const active=button.dataset.panel===key;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active))}
    for(const [name,panel] of Object.entries(this.panels)){panel.hidden=name!==key}
  }

  invalidate(){
    this.plan=null;this.apply.disabled=true;this.create.disabled=true;this.footerStatus.textContent='設定已變更，請重新預覽';this.summary.replaceChildren();
    this.overview.replaceChildren(node('div','設定已變更。重新預覽後，這裡會顯示比例與分布摘要。','split-empty'));
    this.issues.replaceChildren();this.metrics.replaceChildren();this.resetGroupsPanel();this.message.textContent='設定已變更，請重新預覽。';
  }

  resetGroupsPanel(){this.groups.replaceChildren(this.manual)}

  async work(fn){
    if(this.busy)return;this.busy=true;const controls=[...this.dialog.querySelectorAll('input,select,button')],disabled=controls.map(control=>control.disabled);controls.forEach(control=>control.disabled=true);
    try{await fn()}catch(error){this.message.textContent=error.message;this.message.setAttribute('role','alert')}
    finally{controls.forEach((control,index)=>control.disabled=disabled[index]);this.busy=false;const blocked=!this.plan||this.plan.ready===false||!!this.plan.blockers?.length;this.apply.disabled=this.create.disabled=blocked;this.message.setAttribute('aria-live','polite')}
  }

  async runPreview(){
    this.plan=null;await this.work(async()=>{
      const plan=await this.api(`/api/projects/${this.pid}/split-preview`,'POST',{options:this.options});this.plan=plan;this.renderSummary(plan);this.renderOverview(plan);this.renderIssues(plan);this.renderClasses(plan);this.renderGroups(plan.groups);
      const blockers=(plan.blockers||[]).map(item=>item.message),warnings=plan.warnings||[],unit=plan.source_isolation?'個不可拆分來源群組':'個圖片分配單位';
      this.message.textContent=blockers.length?`無法套用：${blockers.join('；')}`:`預覽完成：${plan.groups.length} ${unit}，將調整 ${plan.changed} 張。`;
      if(warnings.length)this.message.textContent+=` ${warnings.join('；')}`;
      this.footerStatus.textContent=blockers.length?`${blockers.length} 項問題需要處理`:`可套用 · 將調整 ${plan.changed} 張圖片`;
      if(blockers.length)this.showPanel('issues');else this.showPanel('overview');
    });
  }

  renderSummary(plan){
    this.summary.replaceChildren();for(const split of ['train','val','test']){
      const card=node('article',undefined,'split-summary-card');card.dataset.split=split;card.append(node('strong',split==='val'?'VALIDATION':split.toUpperCase()),node('b',`${plan.image_counts[split]} 張`),node('span',`${plan.actual_ratios[split].toFixed(1)}% · 目標 ${plan.ratios[split]}%`));this.summary.append(card);
    }
  }

  renderOverview(plan){
    this.overview.replaceChildren();const heading=node('div',undefined,'split-overview-heading');heading.append(node('h4','預覽摘要'),node('span',plan.source_isolation?'已啟用來源隔離':'圖片可獨立分配','split-mode-badge'));this.overview.append(heading);
    const grid=node('div',undefined,'split-overview-grid');
    const facts=[['分配方式',purposes[plan.purpose||this.options.purpose]?.title||this.options.purpose],['平衡目標',this.balance.input.selectedOptions[0]?.textContent||'—'],['重現種子',String(plan.options?.seed??this.options.seed)],['預計異動',`${plan.changed} 張圖片`]];
    for(const [label,value] of facts){const item=node('div');item.append(node('span',label,'muted'),node('strong',value));grid.append(item)}this.overview.append(grid,node('p','切換上方頁籤可檢查警告、各類別分布，或手動指定圖片集合。','muted'));
  }

  renderIssues(plan){
    this.issues.replaceChildren();const items=[];
    for(const blocker of plan.blockers||[])items.push({kind:'error',text:blocker.message});
    for(const warning of plan.warnings||[])items.push({kind:'warning',text:warning});
    for(const scarcity of plan.scarcity||[])for(const message of visibleScarcityMessages(scarcity,plan.source_isolation))items.push({kind:'warning',text:`${scarcity.label}：${message}`});
    if(!items.length){this.issues.append(node('div','沒有需要處理的問題。分割比例與類別覆蓋可直接套用。','split-empty success'));return}
    const list=node('ul',undefined,'split-issue-list');for(const item of items){const row=node('li',undefined,item.kind);row.append(node('strong',item.kind==='error'?'需修正':'請留意'),node('span',item.text));list.append(row)}this.issues.append(list);
  }

  renderGroups(groups){
    this.resetGroupsPanel();const tableWrap=node('div',undefined,'training-table-wrap split-group-table'),table=node('table'),head=node('thead'),tr=node('tr');for(const text of ['分配單位','圖片','目前集合','建議集合','指定／鎖定'])tr.append(node('th',text));head.append(tr);table.append(head);const body=node('tbody');
    for(const group of groups){const row=node('tr');row.dataset.groupId=group.id;const first=node('td'),img=node('img'),asset=this.info?.assets?.find(item=>item.id===group.asset_ids[0]);if(asset){img.src=asset.url+'?thumbnail=1';img.alt='';img.loading='lazy';first.append(img)}
      const display=this.plan&&!this.plan.source_isolation&&group.images===1?(asset?.name||group.asset_ids[0]):(group.names||group.sources).join(' / ');first.append(document.createTextNode(display));const names={train:'Train',val:'Validation',test:'Test','未分配':'未分配'};
      const current=['train','val','test','未分配'].filter(split=>group.current?.[split]).map(split=>`${names[split]} ${group.current[split]}`).join(' / ');row.append(first,node('td',String(group.images)),node('td',current||'未分配'),node('td',names[group.proposed]||group.proposed||'尚未預覽'));
      const cell=node('td'),select=node('select');select.setAttribute('aria-label',(group.names||group.sources).join(' / ')+'指定集合');select.append(new Option('自動分配',''));for(const split of ['train','val','test'])select.append(new Option(split,split));select.value=this.options.locks[group.id]||'';
      select.onchange=()=>{if(select.value)this.options.locks[group.id]=select.value;else delete this.options.locks[group.id];this.invalidate()};cell.append(select);row.append(cell);body.append(row)}table.append(body);tableWrap.append(table);this.groups.prepend(tableWrap);
  }

  renderClasses(plan){
    this.metrics.replaceChildren();const intro=node('p',plan.source_isolation?'來源群組與分配群組會一起列出，方便檢查泛化評估的隔離條件。':'目前按圖片分配；來源群組數不作為警告條件。','muted');this.metrics.append(intro);
    const wrap=node('div',undefined,'training-table-wrap'),table=node('table'),head=node('thead'),headRow=node('tr');for(const text of ['類別','全部 圖片／物件',plan.source_isolation?'來源／分配群組':'分配群組','Train 圖片／物件','Val 圖片／物件','Test 圖片／物件'])headRow.append(node('th',text));head.append(headRow);table.append(head);const body=node('tbody');
    for(const name of Object.keys(plan.class_totals).sort()){const row=node('tr');row.append(node('td',name),node('td',`${plan.class_image_totals?.[name]||0} / ${plan.class_totals[name]}`),node('td',plan.source_isolation?`${plan.class_source_counts?.[name]||0} / ${plan.class_group_counts?.[name]||0}`:String(plan.class_group_counts?.[name]||0)));for(const split of ['train','val','test'])row.append(node('td',`${plan.class_image_counts?.[split]?.[name]||0} / ${plan.class_counts[split][name]||0}`));body.append(row)}table.append(body);wrap.append(table);this.metrics.append(wrap);
    const pairs=node('details');pairs.append(node('summary','常見類別配對分布'));const pairTable=node('table'),pairHead=node('tr');for(const text of ['同圖配對','總張數','Train','Val','Test'])pairHead.append(node('th',text));pairTable.append(pairHead);
    for(const pair of plan.pair_distribution||[]){const row=node('tr');row.append(node('td',pair.labels.join(' ＋ ')),node('td',String(pair.total)));for(const split of ['train','val','test'])row.append(node('td',String(pair.counts[split])));pairTable.append(row)}pairs.append(pairTable);this.metrics.append(pairs);
  }

  async runApply(createVersion){
    if(!this.plan||this.plan.ready===false||this.plan.blockers?.length)return;const plan=this.plan;await this.work(async()=>{const result=await this.api(`/api/projects/${this.pid}/split-apply`,'POST',{options:plan.options,revision:plan.project_revision,fingerprint:plan.fingerprint});this.plan=null;await this.onApplied(result,this.pid,createVersion);this.dialog.close()});
  }
}
