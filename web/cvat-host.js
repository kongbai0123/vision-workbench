/* Desktop integration for the pinned CVAT 2.49 plugin API. */
(() => {
  let context;
  const host=window.workbenchCvatHost={state:'loading',message:''};
  function register() {
    if(context||!window.cvatUI)return;
    window.cvatUI.registerComponent(args=>{
      context=args;host.state='ready';
      return {name:'vision-workbench-sync',destructor:()=>{context=null;}};
    });
  }
  document.addEventListener('plugins.ready',register);register();
  host.save=async function(expectedJob) {
    if(host.state==='saving')return;
    host.state='saving';host.message='';
    try {
      if(!context)throw Error('CVAT 尚未載入完成，請稍候再切換。');
      const {annotation}=context.store.getState(),job=annotation.job.instance;
      if(!job||job.id!==expectedJob)throw Error('請回到此專案的 CVAT 標註工作後再同步。');
      const mode=annotation.canvas.instance?.mode();
      if(mode&&mode!=='idle')throw Error('請先完成或取消目前的繪製／編輯操作，再切換編輯器。');
      if(annotation.annotations.saving.uploading)throw Error('CVAT 正在儲存，請完成後再切換。');
      context.dispatch({type:'SAVE_ANNOTATIONS',payload:{}});
      await job.frames.save();
      await job.annotations.save();
      if(job.annotations.hasUnsavedChanges())throw Error('仍有未儲存的修改，請重試同步。');
      context.dispatch({type:'SAVE_ANNOTATIONS_SUCCESS',payload:{}});
      host.frame=annotation.player.frame.number;
      host.state='saved';
    } catch(error) {
      if(context)context.dispatch({type:'SAVE_ANNOTATIONS_FAILED',payload:{error}});
      host.state='error';host.message=String(error.message||error);
    }
  };
})();
