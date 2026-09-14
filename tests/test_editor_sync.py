import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from workbench.store import ProjectStore,ConflictError
from workbench.labelme_bridge import create_labelme_editor,to_labelme_shape,from_labelme_shape
from workbench.editor_sync import commit_updates,retain_identity
from composer_core.geometry import encode_rle,decode_rle
from tests.test_cvat_bridge import MemoryBridge


class EditorSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=ProjectStore(self.root/'projects');self.pid=self.store.create_project('editors')['id']
        self.image=self.root/'image.png';Image.new('RGB',(80,60)).save(self.image)
        self.shape=dict(id='original',type='rectangle',label='part',x=10,y=8,width=35,height=25,metadata={'source':'fixture'})
        self.aid=self.store.add_assets(self.pid,[dict(path=self.image,shapes=[self.shape])])['asset_ids'][0]
        self.editor=None

    def tearDown(self):
        if self.editor:self.editor.deleteLater();self.app.processEvents()
        self.tmp.cleanup()

    def asset(self):return self.store.get_asset(self.pid,self.aid,internal=True)
    def approve(self):
        self.store.review(self.pid,[self.aid],'approved',{self.aid:self.asset()['revision']})

    def test_all_shared_shapes_roundtrip_and_preserve_mask_holes(self):
        pixels=np.ones((60,80),dtype=np.uint8);pixels[10:30,20:50]=0
        shapes=[self.shape,dict(id='m',type='mask',label='part',counts=encode_rle(pixels)),
                dict(id='o',type='obb',label='part',points=[[5,10],[30,5],[35,30],[10,35]])]
        for kind,points in [('polygon',[[1,1],[30,10],[20,40]]),('linestrip',[[1,1],[20,20]]),('point',[[40,30]])]:
            shapes.append(dict(id=kind,type=kind,label='part',points=points))
        for original in shapes:
            with self.subTest(kind=original['type']):
                actual=from_labelme_shape(to_labelme_shape(original,80,60),80,60)
                self.assertEqual(retain_identity([actual],[original]),[original])

    def test_labelme_mask_bounds_follow_object_not_whole_image(self):
        pixels=np.zeros((60,80),dtype=np.uint8);pixels[12:28,21:52]=1;pixels[16:23,30:42]=0
        shape=dict(id='m',type='mask',label='part',counts=encode_rle(pixels))
        converted=to_labelme_shape(shape,80,60)
        self.assertEqual(converted.points.tolist(),[[21,12],[51,27]])
        self.assertEqual(converted.mask.shape,(16,31))
        self.assertEqual(from_labelme_shape(converted,80,60)['counts'],shape['counts'])

    def test_embedded_load_edit_add_class_save_review_and_reopen(self):
        self.approve();before=self.asset()
        self.editor=create_labelme_editor(self.store,self.pid,self.root,self.aid)
        self.assertEqual(len(self.editor._canvas_widgets.canvas.shapes),1)
        self.assertTrue(self.editor.flush());self.assertEqual(self.asset(),before)
        shape=self.editor._canvas_widgets.canvas.shapes[0]
        shape.points[0,0]=6;shape.label='new-label';self.editor.mark_dirty()
        self.assertTrue(self.editor.flush(),self.editor.last_sync_error)
        changed=self.asset();self.assertEqual(changed['shapes'][0]['x'],6)
        self.assertEqual(changed['review_state'],'pending')
        self.assertIn('new-label',self.store.get_project(self.pid)['classes'])
        self.assertEqual(changed['shapes'][0]['id'],'original')
        self.assertTrue(self.editor._load_file(image_or_label_path=changed['image_path']))
        self.assertEqual(self.editor._canvas_widgets.canvas.shapes[0].label,'new-label')

    def test_hidden_flags_and_group_survive_labelme(self):
        a=self.asset();a['shapes'][0]['hidden']=True
        a['shapes'][0]['metadata']['labelme']={'flags':{'verified':True},'group_id':7,'description':'details'}
        self.store.save_asset(self.pid,self.aid,a['shapes'],a['revision']);self.approve()
        self.editor=create_labelme_editor(self.store,self.pid,self.root,self.aid)
        shape=self.editor._canvas_widgets.canvas.shapes[0]
        self.assertFalse(shape.visible);self.assertEqual(shape.group_id,7)
        self.editor.mark_dirty();before=self.asset();self.assertTrue(self.editor.flush());self.assertEqual(self.asset(),before)
        shape.visible=True;shape.description='updated';self.editor.mark_dirty();self.assertTrue(self.editor.flush())
        self.assertFalse(self.asset()['shapes'][0]['hidden'])
        self.assertEqual(self.asset()['shapes'][0]['metadata']['labelme']['description'],'updated')

    def test_conflict_keeps_labelme_dirty_and_recovery_copy(self):
        self.editor=create_labelme_editor(self.store,self.pid,self.root,self.aid)
        self.store.save_asset(self.pid,self.aid,[],self.asset()['revision'])
        self.editor._canvas_widgets.canvas.shapes[0].label='local';self.editor.mark_dirty()
        self.assertFalse(self.editor.flush());self.assertTrue(self.editor._is_changed)
        self.assertEqual(self.asset()['shapes'],[])
        self.assertTrue(list((self.root/'labelme'/self.pid).glob('*.json')))

    def test_atomic_import_rolls_back_earlier_images_and_classes_on_conflict(self):
        second=self.root/'second.png';Image.new('RGB',(80,60),'red').save(second)
        bid=self.store.add_assets(self.pid,[dict(path=second,shapes=[self.shape])])['asset_ids'][0]
        a=self.asset();b=self.store.get_asset(self.pid,bid)
        self.store.save_asset(self.pid,bid,[],b['revision'])
        shapes=[dict(self.shape,label='new')]
        with self.assertRaises(ConflictError):commit_updates(self.store,self.pid,[(a,shapes),(b,shapes)],source='cvat')
        self.assertEqual(self.asset(),a);self.assertNotIn('new',self.store.get_project(self.pid)['classes'])

    def bridge(self):
        bridge=MemoryBridge(self.root);bridge.links[self.pid]=bridge.links.pop('project')
        bridge.links[self.pid].update(mask_sync_version=1)
        bridge.mapping_file.parent.mkdir(parents=True,exist_ok=True)
        bridge._sync_annotations(8,[self.asset()],7,[])
        bridge.mark_synced(self.store.snapshot(self.pid),[])
        return bridge

    def test_builtin_push_cvat_pull_then_labelme_sees_change_and_deletion(self):
        bridge=self.bridge();a=self.asset();a['shapes'][0]['x']=12
        self.store.save_asset(self.pid,self.aid,a['shapes'],a['revision'])
        bridge.synchronize(self.store.snapshot(self.pid),[],self.store)
        self.assertEqual(bridge.annotations['shapes'][0]['points'][0],12)
        self.approve();bridge.annotations['shapes'][0]['points'][0]=14
        bridge.synchronize(self.store.snapshot(self.pid),[],self.store)
        self.assertEqual(self.asset()['shapes'][0]['x'],14);self.assertEqual(self.asset()['review_state'],'pending')
        self.editor=create_labelme_editor(self.store,self.pid,self.root,self.aid)
        self.assertEqual(self.editor._canvas_widgets.canvas.shapes[0].points[0,0],14)
        a=self.asset();self.store.save_asset(self.pid,self.aid,[],a['revision'])
        bridge.synchronize(self.store.snapshot(self.pid),[],self.store)
        self.assertEqual(bridge.annotations['shapes'],[])

    def test_cvat_unchanged_preserves_approval_and_concurrent_edit_is_not_overwritten(self):
        bridge=self.bridge();self.approve();before=self.asset()
        bridge.synchronize(self.store.snapshot(self.pid),[],self.store)
        self.assertEqual(self.asset(),before)
        local=deepcopy(before['shapes']);local[0]['x']=12
        self.store.save_asset(self.pid,self.aid,local,before['revision'])
        bridge.annotations['shapes'][0]['points'][0]=14
        with self.assertRaisesRegex(ValueError,'都有修改'):bridge.synchronize(self.store.snapshot(self.pid),[],self.store)
        self.assertEqual(self.asset()['shapes'][0]['x'],12)
        self.assertEqual(bridge.annotations['shapes'][0]['points'][0],14)


if __name__=='__main__':unittest.main()
