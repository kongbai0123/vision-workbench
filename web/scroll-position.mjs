export function nearestVisibleScrollTop({scrollTop, viewportHeight, contentHeight, itemTop, itemHeight}) {
  const viewport=Math.max(0,Number(viewportHeight)||0);
  const content=Math.max(viewport,Number(contentHeight)||0);
  const maximum=Math.max(0,content-viewport);
  const current=Math.min(maximum,Math.max(0,Number(scrollTop)||0));
  const top=Math.max(0,Number(itemTop)||0);
  const height=Math.max(0,Number(itemHeight)||0);
  if(top<current)return Math.min(maximum,top);
  if(top+height>current+viewport)return Math.min(maximum,Math.max(0,top+height-viewport));
  return current;
}
