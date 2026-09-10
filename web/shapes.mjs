// Geometry remains in original-image pixels. Old rectangle records remain valid.
export const kind=s=>s.type||'rectangle';
export function bounds(points){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys);return {x,y,width:Math.max(...xs)-x,height:Math.max(...ys)-y};}
export function vector(type,points,label){return {type,points:points.map(p=>[...p]),...bounds(points),label};}
export function validateShape(s,w,h){
 if(!s||typeof s.label!=='string'||!s.label.trim())throw Error('標註類別無效');
 const t=kind(s);
 if(t==='mask'){
  if(w*h>16777216||!Array.isArray(s.counts)||s.counts.length>2*w*h+1||!s.counts.every(n=>Number.isSafeInteger(n)&&n>=0)||s.counts.reduce((a,b)=>a+b,0)!==w*h)throw Error('Mask RLE 無效或超過 1600 萬像素');
  return {...s,type:t,x:0,y:0,width:w,height:h};
 }
 if(['polygon','linestrip','point','obb'].includes(t)){
  const min=t==='point'?1:t==='linestrip'?2:3;
  if(!Array.isArray(s.points)||s.points.length<min||s.points.length>10000||(t==='point'&&s.points.length!==1)||(t==='obb'&&s.points.length!==4)||!s.points.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[1]>=0&&p[0]<=w&&p[1]<=h))throw Error('頂點數量、座標或範圍無效');
  return {...s,...bounds(s.points)};
 }
 if(t!=='rectangle'||!['x','y','width','height'].every(k=>Number.isFinite(s[k]))||s.x<0||s.y<0||s.width<=0||s.height<=0||s.x+s.width>w+1e-6||s.y+s.height>h+1e-6)throw Error('標註超出圖片邊界或尺寸無效');
 return {...s};
}
export function decodeMask(counts,w,h){
 validateShape({type:'mask',counts,label:'mask'},w,h);
 const data=new Uint8Array(w*h);let offset=0,on=false;
 for(const n of counts){if(on)for(let i=offset;i<offset+n;i++)data[(i%h)*w+Math.floor(i/h)]=1;offset+=n;on=!on;}return data;
}
export function encodeMask(data,w,h){const counts=[];let on=0,n=0;for(let x=0;x<w;x++)for(let y=0;y<h;y++){const v=data[y*w+x]?1:0;if(v===on)n++;else{counts.push(n);n=1;on=v;}}counts.push(n);return counts;}
export function brush(data,w,h,a,b,r,erase=false){
 const steps=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/Math.max(1,r/2)));
 for(let i=0;i<=steps;i++){const cx=a.x+(b.x-a.x)*i/steps,cy=a.y+(b.y-a.y)*i/steps;for(let y=Math.max(0,Math.floor(cy-r));y<Math.min(h,Math.ceil(cy+r));y++)for(let x=Math.max(0,Math.floor(cx-r));x<Math.min(w,Math.ceil(cx+r));x++)if((x+.5-cx)**2+(y+.5-cy)**2<=r*r)data[y*w+x]=erase?0:1;}
}
export function moved(s,dx,dy,w,h){if(kind(s)==='mask')return s;dx=Math.max(-s.x,Math.min(dx,w-s.x-s.width));dy=Math.max(-s.y,Math.min(dy,h-s.y-s.height));return s.points?{...s,...vector(kind(s),s.points.map(([x,y])=>[x+dx,y+dy]),s.label)}:{...s,x:s.x+dx,y:s.y+dy};}
export function rotated(s,degrees,w,h){const p=s.points||[[s.x,s.y],[s.x+s.width,s.y],[s.x+s.width,s.y+s.height],[s.x,s.y+s.height]];const cx=p.reduce((n,v)=>n+v[0],0)/p.length,cy=p.reduce((n,v)=>n+v[1],0)/p.length,a=degrees*Math.PI/180;return validateShape({...s,...vector('obb',p.map(([x,y])=>[cx+(x-cx)*Math.cos(a)-(y-cy)*Math.sin(a),cy+(x-cx)*Math.sin(a)+(y-cy)*Math.cos(a)]),s.label)},w,h);}
export function resizeOBB(s,index,p,w,h){const pts=s.points,o=pts[(index+2)%4],a=pts[(index+1)%4],b=pts[(index+3)%4],unit=q=>{const d=Math.hypot(q[0]-o[0],q[1]-o[1]);return [(q[0]-o[0])/d,(q[1]-o[1])/d]};const u=unit(a),v=unit(b),dx=p.x-o[0],dy=p.y-o[1],du=Math.max(2,dx*u[0]+dy*u[1]),dv=Math.max(2,dx*v[0]+dy*v[1]),next=structuredClone(pts);next[(index+1)%4]=[o[0]+du*u[0],o[1]+du*u[1]];next[(index+3)%4]=[o[0]+dv*v[0],o[1]+dv*v[1]];next[index]=[o[0]+du*u[0]+dv*v[0],o[1]+du*u[1]+dv*v[1]];try{return validateShape({...s,...vector('obb',next,s.label)},w,h)}catch{return s;}}
export function shapeKey(s){return JSON.stringify([kind(s),s.label,s.points||null,s.counts||null,s.x,s.y,s.width,s.height]);}
export function history(limit=40){let undo=[],redo=[];const record=v=>({value:structuredClone(v),size:JSON.stringify(v).length*2});function trim(){while(undo.length>limit||undo.reduce((n,r)=>n+r.size,0)+redo.reduce((n,r)=>n+r.size,0)>64*1024*1024){if(undo.length)undo.shift();else if(redo.length)redo.shift();else break;}}return {push(v){undo.push(record(v));redo=[];trim()},undo(v){if(!undo.length)return null;redo.push(record(v));const value=undo.pop().value;trim();return value},redo(v){if(!redo.length)return null;undo.push(record(v));const value=redo.pop().value;trim();return value},clear(){undo=[];redo=[]},get canUndo(){return !!undo.length},get canRedo(){return !!redo.length}};}
