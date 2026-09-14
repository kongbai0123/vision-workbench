import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
import numpy as np

from workbench.cvat_bridge import CvatProjectBridge
from workbench.cvat_masks import to_cvat_mask, from_cvat_mask
from composer_core.geometry import encode_rle, decode_rle


class FakeBridge(CvatProjectBridge):
    def _load(self):
        return {"a"*32:{"project_id":7,"task_id":8,"job_id":9,"asset_count":1}}

    def _request(self, method, path, cookies, body=None, expected=(200,201,202)):
        if path.startswith("/api/labels"):
            return {"results":[{"id":3,"name":"part"}]}
        if path.endswith("/annotations"):
            return {"shapes":[{"type":"rectangle","frame":0,"label_id":3,"points":[2,3,12,18]}]}
        return {}


class CvatBridgeTests(unittest.TestCase):
    def test_mask_crop_uses_rows_and_inclusive_bounds(self):
        pixels=np.zeros((5,7),dtype=np.uint8)
        pixels[1,2:5]=1;pixels[2,2]=1;pixels[2,4]=1;pixels[3,5]=1
        points=to_cvat_mask(encode_rle(pixels),7,5)
        self.assertEqual(points,[0,3,1,1,1,1,4,1,2,1,5,3])
        np.testing.assert_array_equal(decode_rle(from_cvat_mask(points,7,5),7,5)>0,pixels>0)

    def test_mask_roundtrip_handles_edges_holes_empty_and_random_pixels(self):
        rng=np.random.default_rng(12)
        for pixels in [np.zeros((3,8)),np.ones((3,8)),np.eye(5,9),rng.integers(0,2,(13,17))]:
            height,width=pixels.shape
            counts=encode_rle(pixels)
            self.assertEqual(from_cvat_mask(to_cvat_mask(counts,width,height),width,height),counts)
        for points in [[0,1,-1,0,0,0],[0,100,0,0,1,1],[0,1.5,0,0,0,0]]:
            with self.assertRaises(ValueError):from_cvat_mask(points,5,5)

    def test_new_tasks_send_masks_with_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge=MemoryBridge(folder)
            bridge._sync_annotations(8,[self.mask_asset()],7,[])
            self.assertEqual(bridge.annotations['shapes'][0]['type'],'mask')
            self.assertEqual(bridge.annotations['shapes'][0]['points'],[0,1,2,1,1,1,2,2])

    @staticmethod
    def mask_asset():
        pixels=np.zeros((4,5),dtype=np.uint8);pixels[1,1]=1;pixels[2,2]=1
        return {'id':'asset','image_path':'a.png','width':5,'height':4,'shapes':[
            {'id':'original','type':'mask','label':'part','counts':encode_rle(pixels),
             'x':0,'y':0,'width':5,'height':4,'metadata':{'source':'original'},'hidden':False}]}

    def test_legacy_migration_appends_once_and_preserves_cvat_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge=MemoryBridge(folder)
            existing={'id':55,'type':'rectangle','frame':0,'label_id':3,'points':[1,1,3,3]}
            bridge.annotations['shapes']=[existing]
            snapshot={'id':'project','assets':[self.mask_asset()]}
            bridge.ensure_project(snapshot,[])
            self.assertEqual(bridge.annotations['shapes'][0],existing)
            self.assertEqual(len(bridge.annotations['shapes']),2)
            self.assertEqual(bridge.links['project']['skipped_masks'],0)
            self.assertTrue(list((Path(folder)/'cvat/annotation-backups').glob('*.json')))
            # Simulate a retry after PATCH succeeded but before mapping was saved.
            bridge.links['project'].pop('mask_sync_version')
            bridge.ensure_project(snapshot,[])
            self.assertEqual(bridge.patches,1)

    def test_mask_readback_preserves_identity_and_applies_changes_and_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge=MemoryBridge(folder)
            asset=self.mask_asset();snapshot={'id':'project','assets':[asset]}
            bridge.ensure_project(snapshot,[])
            self.assertEqual(bridge.read_annotations(snapshot,[]),[])
            bridge.annotations['shapes'][0]['points']=[0,1,4,3,4,3]
            updates=bridge.read_annotations(snapshot,[])
            self.assertEqual(len(updates[0][1]),1)
            decoded=decode_rle(updates[0][1][0]['counts'],5,4)
            self.assertEqual(np.count_nonzero(decoded),1);self.assertEqual(decoded[3,4],255)
            bridge.annotations['shapes']=[]
            self.assertEqual(bridge.read_annotations(snapshot,[])[0][1],[])

    def test_readback_replaces_vectors_but_preserves_exact_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge=FakeBridge(folder)
            mask={"id":"mask","type":"mask","label":"part","counts":[4],"x":0,"y":0,"width":2,"height":2}
            asset={"id":"b"*32,"revision":1,"image_path":str(Path(folder)/"a.png"),"shapes":[mask,
                {"id":"old","type":"rectangle","label":"part","x":0,"y":0,"width":1,"height":1}]}
            updates=bridge.read_annotations({"id":"a"*32,"assets":[asset]},[])
            self.assertEqual(len(updates),1)
            shapes=updates[0][1]
            self.assertEqual(shapes[0]["id"],"mask")
            self.assertEqual(shapes[1]["x"],2)
            self.assertEqual(shapes[1]["height"],15)
            self.assertEqual(shapes[1]["metadata"]["source"],"cvat")


class MemoryBridge(CvatProjectBridge):
    def __init__(self, folder):
        super().__init__(folder)
        self.links={'project':{'project_id':7,'task_id':8,'job_id':9,'asset_count':1,'skipped_masks':1}}
        self.annotations={'version':1,'tags':[],'shapes':[],'tracks':[]}
        self.patches=0

    def _load(self):return deepcopy(self.links)
    def _save(self, value):self.links=deepcopy(value)
    def _request(self, method, path, cookies, body=None, expected=(200,201,202)):
        if path.startswith('/api/labels'):return {'results':[{'id':3,'name':'part'}]}
        if '/annotations' in path:
            if method=='PATCH':
                self.annotations['shapes'].extend(deepcopy(body['shapes']));self.patches+=1
            elif method=='PUT':self.annotations=deepcopy(body)
            return deepcopy(self.annotations)
        return {'id':8}


if __name__=="__main__": unittest.main()
