import {duration} from './task-timing.mjs';

const valid=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
const seconds=n=>duration(Math.max(1,n));
export function quickEstimateText(estimate){
  if(!estimate)return '快速估時使用相同資料與設定的完成紀錄，不會啟動訓練。';
  if(!valid(estimate.remaining_seconds))return '尚無完整的相同設定耗時紀錄；開始後由首批實測提供初估。';
  return `歷史初估約 ${seconds(estimate.remaining_seconds)} · 參考 ${estimate.history_runs} 次完成紀錄（波動範圍 ${seconds(estimate.lower_seconds)}～${seconds(estimate.upper_seconds)}）`;
}

export function trainingTimeText(run,now=Date.now()/1000){
  const status=run?.status;
  if(status==='completed')return '已完成';
  if(status==='failed')return '執行失敗';
  if(status==='stopped')return '已停止';
  if(status==='stopping')return '正在安全停止';
  if(status==='queued')return '等待執行';
  const t=run?.timing;
  if(!t)return run?.message||'準備資料與模型';
  const age=Math.max(0,now-t.updated_at),label=t.phase_label||'目前階段';
  if(age+(t.seconds_since_advance||0)>t.stale_after_seconds)return `${label} · 已用時 ${duration(t.elapsed_seconds+age)} · 等待進度回報`;
  if(valid(t.remaining_seconds)){
    const source=t.source==='history'?'歷史初估':t.samples<4?'快速初估':'動態預估';
    return `${source} · 全程剩餘約 ${seconds(Math.max(0,t.remaining_seconds-age))}`;
  }
  if(valid(t.phase_remaining_seconds))return `${t.samples<4?'快速初估':'動態預估'} · ${label}剩餘約 ${seconds(Math.max(0,t.phase_remaining_seconds-age))}`;
  return `${label} · 已用時 ${duration(t.elapsed_seconds+age)}`;
}

export function trainingTimeDetail(run,now=Date.now()/1000){
  const t=run?.timing;if(!t)return '';
  if(['completed','stopped','failed'].includes(run.status))return '';
  if(Math.max(0,now-t.updated_at)+(t.seconds_since_advance||0)>t.stale_after_seconds)return '等待新的工作量回報後，重新校正剩餘時間。';
  if(valid(t.remaining_seconds)&&valid(t.lower_seconds)&&valid(t.upper_seconds))return `波動範圍 ${seconds(t.lower_seconds)}～${seconds(t.upper_seconds)}；依設定輪數估算，早停可能提前結束。`;
  return t.phase_remaining_seconds!=null?'目前顯示此階段剩餘工作；其他階段會分開計時。':'首個工作單位完成後更新時間估算。';
}

export function trainingProgress(run){
  const t=run?.timing;
  if(t)return {label:t.phase_label||'目前階段',value:valid(t.total)&&t.total>0?Math.min(100,Math.max(0,t.completed/t.total*100)):null};
  return {label:'工作進度',value:valid(run?.progress)?run.progress:null};
}
