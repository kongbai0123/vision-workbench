// Estimates use measured progress only, never a timer-generated percentage.
export function duration(seconds) {
  const value=Math.max(0,Math.ceil(seconds));
  if(value<60)return `${value} 秒`;
  if(value<3600)return `${Math.floor(value/60)} 分 ${value%60} 秒`;
  return `${Math.floor(value/3600)} 小時 ${Math.floor(value%3600/60)} 分`;
}
export function taskStage(message='正在處理') {
  return String(message).replace(/\s*\d+\s*\/\s*\d+/g,'').replace(/第\s*\d+\s*(?:幀|張|筆|項)/g,'').replace(/\s*[·（(]?已(?:耗時|進行)\s*[\d.]+\s*秒[）)]?/g,'').trim();
}
export class TaskTiming {
  constructor(clock=()=>performance.now()/1000){this.clock=clock;this.started=clock();this.last=this.started;this.elapsed=0;this.samples=[];this.state='queued';this.phase=null;}
  update(progress,state='running',phase='default'){
    const now=this.clock();
    if(this.state==='running')this.elapsed+=Math.max(0,now-this.last);
    this.last=now;
    if(state!==this.state||phase!==this.phase)this.samples=[];
    this.state=state;this.phase=phase;
    if(state!=='running'||!Number.isFinite(progress)){this.samples=[];return this;}
    const previous=this.samples.at(-1);
    const span=previous?previous.t-this.samples[0].t:0;
    if(previous&&progress>previous.p&&now-previous.t>Math.max(30,span/Math.max(1,this.samples.length-1)*3))this.samples=[];
    if(previous&&progress<previous.p)this.samples=[];
    if(!this.samples.length||progress>this.samples.at(-1).p){
      this.samples.push({t:now,p:progress});
      this.samples=this.samples.slice(-12);
    }
    return this;
  }
  text(scope='目前階段'){
    const now=this.clock();
    if(['succeeded','completed','complete'].includes(this.state))return '已完成';
    if(this.state==='failed')return '處理失敗';
    if(['cancelled','stopped'].includes(this.state))return '已停止';
    if(this.state==='queued')return '排隊等待 · 尚未開始估算';
    if(this.state==='paused')return '已暫停，恢復後重新估算';
    if(this.state==='stopping')return '正在安全停止';
    if(!this.samples.length)return '目前無法估算完成時間';
    const first=this.samples[0],last=this.samples.at(-1),span=last.t-first.t;
    if(last.p>=100)return '正在確認完成';
    if(this.samples.length<4||span<5)return '目前無法估算完成時間（樣本不足）';
    if(now-last.t>Math.max(30,span/(this.samples.length-1)*3))return '目前無法估算完成時間（進度暫未更新）';
    const remaining=(100-last.p)*span/(last.p-first.p);
    return `${scope}預估剩餘約 ${duration(Math.max(1,remaining))}（非整體完成時間）`;
  }
}
