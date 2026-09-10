import tempfile
import unittest
from pathlib import Path

from workbench.cvat_bridge import CvatProjectBridge


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


if __name__=="__main__": unittest.main()
