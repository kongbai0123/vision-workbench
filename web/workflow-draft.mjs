// Serialize writes, preserve edits made in flight, and never cross project boundaries.
export class WorkflowDraft {
  constructor({api,collect,restore,onError=()=>{},onSaved=()=>{}}){
    Object.assign(this,{api,collect,restore,onError,onSaved});this.pid=null;this.revision=0;this.generation=0;this.saved=0;this.inFlight=null;this.timer=null;
  }
  get dirty(){return this.generation!==this.saved}
  async load(pid){
    await this.flush();
    const record=await this.api(`/api/projects/${pid}/workflow-draft`,'POST',{});
    this.pid=pid;this.revision=record.revision;this.generation=0;this.saved=0;
    this.restore(record.payload);this.onSaved();return record;
  }
  changed(){
    if(!this.pid)return;
    this.generation++;clearTimeout(this.timer);
    this.timer=setTimeout(()=>this.flush().catch(this.onError),400);
  }
  async flush(){
    clearTimeout(this.timer);
    if(this.inFlight){await this.inFlight;return this.flush()}
    if(!this.pid||!this.dirty)return;
    const generation=this.generation,payload=this.collect(),pid=this.pid;
    this.inFlight=this.api(`/api/projects/${pid}/workflow-draft`,'POST',{revision:this.revision,payload});
    try{const record=await this.inFlight;this.revision=record.revision;this.saved=generation;this.onSaved()}
    finally{this.inFlight=null}
    if(this.dirty)await this.flush();
  }
}
