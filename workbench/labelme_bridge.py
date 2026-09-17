"""Pinned Labelme 7.4.1 adapter: every save commits to the shared project store."""
from copy import deepcopy
from pathlib import Path
import os
import uuid
import numpy as np
from composer_core.geometry import encode_rle,decode_rle
from .editor_sync import commit_updates,retain_identity


def to_labelme_shape(shape, width, height):
    from labelme._shape import Shape
    kind=shape['type'];mask=None
    if kind=='mask':
        mask=decode_rle(shape['counts'],width,height)>0
        ys,xs=np.nonzero(mask)
        if len(xs):
            left,top,right,bottom=int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
            mask=mask[top:bottom+1,left:right+1]
            points=[[left,top],[right,bottom]]
        else:
            mask=mask[:1,:1];points=[[0,0],[0,0]]
    elif kind=='rectangle':
        points=[[shape['x'],shape['y']],[shape['x']+shape['width'],shape['y']+shape['height']]]
    else: points=shape['points']
    result=Shape(label=shape['label'],shape_type={'obb':'oriented_rectangle'}.get(kind,kind),
        points=points,mask=mask,closed=True,visible=not shape.get('hidden',False),
        other_data={'workbench':deepcopy(shape)})
    extra=shape.get('metadata',{}).get('labelme',{})
    result.flags=extra.get('flags',{});result.group_id=extra.get('group_id');result.description=extra.get('description','')
    return result


def from_labelme_shape(shape, width, height):
    from labelme._utils.shape import shape_to_mask
    old=deepcopy(shape.other_data.get('workbench',{}))
    result={'id':old.get('id',uuid.uuid4().hex),'label':shape.label,'hidden':not shape.visible,
        'metadata':old.get('metadata',{'source':'labelme'})}
    extra={'flags':shape.flags or {},'group_id':shape.group_id,'description':shape.description or ''}
    if any(extra.values()) or 'labelme' in result['metadata']:result['metadata']['labelme']=extra
    kind=shape.shape_type;points=shape.points.tolist()
    if kind=='mask':
        bitmap=np.zeros((height,width),dtype=np.uint8)
        x,y=np.rint(shape.points[0]).astype(int)
        ph,pw=shape.mask.shape
        left,top=max(0,x),max(0,y);right,bottom=min(width,x+pw),min(height,y+ph)
        if right>left and bottom>top:
            bitmap[top:bottom,left:right]=shape.mask[top-y:bottom-y,left-x:right-x]
        result.update(type='mask',x=0,y=0,width=width,height=height,counts=encode_rle(bitmap))
    elif kind=='circle':
        # The shared editor uses pixel masks for circles, preserving the full region.
        result.update(type='mask',x=0,y=0,width=width,height=height,
            counts=encode_rle(shape_to_mask((height,width),points,shape_type='circle')))
    elif kind=='rectangle':
        (x1,y1),(x2,y2)=points
        result.update(type='rectangle',x=min(x1,x2),y=min(y1,y2),width=abs(x2-x1),height=abs(y2-y1))
    elif kind in {'polygon','oriented_rectangle','point','line','linestrip'}:
        result.update(type={'oriented_rectangle':'obb','line':'linestrip'}.get(kind,kind),points=points)
    else: raise ValueError(f'無法同步 Labelme 標註類型：{kind}')
    return result


def create_labelme_editor(store, pid, data_root, asset_id=None):
    # Private Labelme APIs are deliberately isolated in this version-pinned module.
    from labelme._app import MainWindow as LabelmeWindow, _shape_to_dict
    from labelme._label_file import Annotation
    from PySide6.QtCore import Qt,QSignalBlocker
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtGui import QKeySequence

    class ProjectLabelme(LabelmeWindow):
        def __init__(self):
            self.store=store;self.pid=pid;self.current_asset=None;self.last_sync_error=''
            snapshot=store.snapshot(pid)
            self.assets={os.path.normpath(a['image_path']):a['id'] for a in snapshot['assets']}
            self.session_folder=store.directory(pid)/'integrations'/'labelme'
            self.session_folder.mkdir(parents=True,exist_ok=True)
            config=self.session_folder/'config.yaml'
            if not config.exists(): config.write_text('{}\n','utf-8')
            super().__init__(config_file=config,config_overrides={'auto_save':False,'labels':snapshot['classes'],
                'keep_prev':False,'with_image_data':False},output_dir=str(self.session_folder))
            self.setWindowFlags(Qt.WindowType.Widget)
            # Project membership and deletion are managed by the shared workbench.
            for name in ('open','open_dir','change_output_dir','delete_file','close','save_as'):
                action=getattr(self._actions,name,None)
                if action:
                    action.setEnabled(False);action.setVisible(False)
                    action.setShortcut(QKeySequence())
            self.setAcceptDrops(False)
            self._loaded_image_paths=list(self.assets)
            self._refresh_file_list()
            selected=next((path for path,aid in self.assets.items() if aid==asset_id),next(iter(self.assets),None))
            if selected:
                self._load_file(image_or_label_path=selected)
                with QSignalBlocker(self._docks.file_list):
                    matches=self._docks.file_list.findItems(selected,Qt.MatchFlag.MatchExactly)
                    if matches:self._docks.file_list.setCurrentItem(matches[0])

        def _load_file(self, *, image_or_label_path):
            path=os.path.normpath(image_or_label_path)
            if path not in self.assets:
                self.show_error_message(title='專案圖片',message='請從工作台匯入圖片，再由專案清單開啟。')
                return False
            if not self.flush(): return False
            self.loading_asset=store.get_asset(pid,self.assets[path],internal=True)
            marker=self.session_folder/(Path(path).stem+'.json')
            if not marker.exists():marker.write_text('{}','utf-8')
            ok=super()._load_file(image_or_label_path=path)
            if ok:
                self.current_asset=self.loading_asset
                for item,original in zip(self._docks.label_list,self.current_asset['shapes']):
                    item.shape().visible=not original.get('hidden',False)
                    item.setCheckState(Qt.CheckState.Unchecked if original.get('hidden') else Qt.CheckState.Checked)
                self.mark_clean()
            return ok

        def _read_annotation_file(self, *, label_path):
            asset=self.loading_asset
            shapes=[_shape_to_dict(to_labelme_shape(s,asset['width'],asset['height'])) for s in asset['shapes']]
            return Annotation(image_path=asset['image_path'],image_data=Path(asset['image_path']).read_bytes(),
                shapes=shapes,flags={},other_data={})

        def save_labels(self, *, label_path, show_error=True):
            try:
                asset=self.current_asset
                if asset is None:return True
                shapes=[from_labelme_shape(item.shape(),asset['width'],asset['height']) for item in self._docks.label_list]
                # Copy/paste may copy the workbench ID; allocate independent IDs.
                seen=set()
                for shape in shapes:
                    if shape['id'] in seen:shape['id']=uuid.uuid4().hex
                    seen.add(shape['id'])
                shapes=retain_identity(shapes,asset['shapes'])
                recovery=str(self.session_folder/(Path(asset['image_path']).stem+'.json'))
                if not super().save_labels(label_path=recovery,show_error=show_error):return False
                commit_updates(store,pid,[(asset,shapes)],source='labelme')
                self.current_asset=store.get_asset(pid,asset['id'],internal=True)
                self.last_sync_error=''
                self.show_status_message('已同步專案 · 修改影像已送往待審核',delay=3000)
                return True
            except (ValueError,RuntimeError,OSError) as error:
                self.last_sync_error=str(error)
                if show_error:QMessageBox.warning(self,'標註尚未同步',str(error))
                return False

        def flush(self):
            if not self.current_asset:return True
            if self._canvas_widgets.canvas.is_drawing:
                self.last_sync_error='請先完成或取消目前繪製中的物件。'
                return False
            if not self._is_changed:return True
            if self.save_labels(label_path='',show_error=False):
                self.mark_clean();return True
            return False

        def _can_continue(self):
            ok=self.flush()
            if not ok:QMessageBox.warning(self,'標註尚未同步',self.last_sync_error)
            return ok

        def closeEvent(self,event):
            # Labelme's Quit action must close through the host's save barrier.
            event.ignore()
            host=self.window()
            if host is not self:host.close()

    return ProjectLabelme()
