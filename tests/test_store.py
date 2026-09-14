import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image
from workbench.store import ProjectStore, ConflictError


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vision-store-")
        self.root = Path(self.tmp.name)
        self.store = ProjectStore(self.root / "projects")
        self.project = self.store.create_project("工件檢查")
        self.pid = self.project["id"]
        self.image = self.root / "原圖.png"
        Image.new("RGB",(80,60),(110,150,70)).save(self.image)
        self.shape = dict(id="stable-object",type="rectangle",label="工件",x=10,y=8,width=35,height=25,
                          metadata={"source_instance":42})

    def tearDown(self):
        self.tmp.cleanup()

    def add(self):
        result = self.store.add_assets(self.pid,[dict(path=self.image,shapes=[self.shape],batch_id="batch-a")])
        return self.store.get_asset(self.pid,result["asset_ids"][0])

    def define(self, *names):
        project=self.store.get_project(self.pid)
        classes=list(dict.fromkeys(project["classes"]+list(names)))
        return self.store.manage_classes(self.pid,classes,{},project["revision"])

    def test_capture_edit_review_export_snapshot_one_shared_record(self):
        asset = self.add()
        self.store.review(self.pid,[asset["id"]],"approved",{asset["id"]:1})
        asset = self.store.get_asset(self.pid,asset["id"])
        asset["shapes"][0]["x"] = 14
        edited = self.store.save_asset(self.pid,asset["id"],asset["shapes"],asset["revision"])
        self.assertEqual(edited["review_state"],"pending")
        self.assertEqual(edited["shapes"][0]["id"],"stable-object")
        self.store.review(self.pid,[asset["id"]],"approved",{asset["id"]:edited["revision"]})
        snapshot = self.store.snapshot(self.pid)
        self.assertEqual(snapshot["stats"]["approved"],1)
        self.assertEqual(snapshot["assets"][0]["shapes"][0]["x"],14)
        self.assertEqual(hashlib.sha256(self.image.read_bytes()).hexdigest(),snapshot["assets"][0]["sha256"])
        self.assertEqual(Path(snapshot["assets"][0]["image_path"]).read_bytes(),self.image.read_bytes())
        reopened = ProjectStore(self.root/"projects")
        self.assertEqual(reopened.get_asset(self.pid,asset["id"])["shapes"],edited["shapes"])

    def test_conflicting_editor_does_not_overwrite_saved_work(self):
        asset = self.add()
        self.define("A")
        first = json.loads(json.dumps(asset["shapes"]))
        first[0]["label"] = "A"
        self.store.save_asset(self.pid,asset["id"],first,1)
        with self.assertRaises(ConflictError):
            self.store.save_asset(self.pid,asset["id"],asset["shapes"],1)
        self.assertEqual(self.store.get_asset(self.pid,asset["id"])["shapes"][0]["label"],"A")

    def test_stale_review_fails_without_partial_batch_approval(self):
        asset = self.add()
        self.define("different")
        changed = [dict(self.shape,label="different")]
        self.store.save_asset(self.pid,asset["id"],changed,1)
        with self.assertRaises(ConflictError):
            self.store.review(self.pid,[asset["id"]],"approved",{asset["id"]:1})
        self.assertEqual(self.store.get_asset(self.pid,asset["id"])["review_state"],"pending")

    def test_invalid_last_image_leaves_entire_import_unpublished(self):
        before = self.store.get_project(self.pid)["revision"]
        with self.assertRaises(ValueError):
            self.store.add_assets(self.pid,[dict(path=self.image),dict(path=self.root/"missing.png")])
        project = self.store.get_project(self.pid)
        self.assertEqual(project["stats"]["total"],0)
        self.assertEqual(project["revision"],before)

    def test_duplicate_import_reports_conflicting_labels_without_replacing(self):
        asset = self.add()
        result = self.store.add_assets(self.pid,[dict(path=self.image,shapes=[dict(self.shape,label="wrong")])])
        self.assertEqual(result["added"],0)
        self.assertEqual(result["duplicates"],1)
        self.assertTrue(result["conflicts"])
        self.assertEqual(self.store.get_asset(self.pid,asset["id"])["shapes"][0]["label"],"工件")

    def test_rle_holes_survive_review_restart_and_snapshot(self):
        import numpy as np
        mask = np.zeros((60,80),dtype=np.uint8)
        mask[10:50,10:70]=1
        mask[20:40,30:50]=0
        counts=[]; current=0; count=0
        for pixel in mask.T.flatten():
            if pixel==current: count+=1
            else: counts.append(count);count=1;current=int(pixel)
        counts.append(count)
        result=self.store.add_assets(self.pid,[dict(path=self.image,shapes=[dict(type="mask",label="ring",counts=counts)])])
        asset=self.store.get_asset(self.pid,result["asset_ids"][0])
        self.store.review(self.pid,[asset["id"]],"approved")
        self.assertEqual(self.store.snapshot(self.pid)["assets"][0]["shapes"][0]["counts"],counts)

    def test_merge_is_a_copy_of_current_revision(self):
        asset=self.add()
        self.store.review(self.pid,[asset["id"]],"approved")
        target=self.store.create_project("整併")
        result=self.store.merge(target["id"],[self.pid])
        self.assertEqual(result["added"],1)
        merged=self.store.snapshot(target["id"])["assets"][0]
        self.assertEqual(merged["source"]["project_id"],self.pid)
        self.assertEqual(merged["review_state"],"approved")
        self.assertEqual(merged["shapes"][0]["metadata"]["source_instance"],42)
        self.assertNotEqual(merged["image_path"],str(self.store.image_path(self.pid,asset["id"])))

    def test_auto_split_balances_class_counts_and_preserves_source_batch(self):
        self.define("A", "B")
        records = []
        for index in range(12):
            path = self.root / f"balanced-{index}.png"
            Image.new("RGB", (80, 60), (index * 13, 80, 120)).save(path)
            records.append(dict(path=path, batch_id="one-source-batch", shapes=[
                dict(self.shape, id=f"shape-{index}", label="A" if index < 6 else "B")
            ]))
        added = self.store.add_assets(self.pid, records)
        self.store.review(self.pid, added["asset_ids"], "approved")
        result = self.store.auto_split(self.pid, {"train": 60, "val": 20, "test": 20})
        self.assertEqual(result["report"]["class_counts"], {
            "train": {"A": 4, "B": 4}, "val": {"A": 1, "B": 1}, "test": {"A": 1, "B": 1},
        })
        project = self.store.get_project(self.pid)
        self.assertEqual({asset["split"] for asset in project["assets"]}, {"train", "val", "test"})
        self.assertEqual({asset["batch_id"] for asset in project["assets"]}, {"one-source-batch"})
        self.assertTrue(all(asset["class_counts"] for asset in project["assets"]))

    def test_invalid_paths_and_nan_cannot_be_saved(self):
        for value in ("../outside", "", "a"*31, "z"*32):
            with self.assertRaises(ValueError): self.store.directory(value)
        asset=self.add()
        with self.assertRaises(ValueError):
            self.store.save_asset(self.pid,asset["id"],[dict(self.shape,x=float("nan"))],1)
        self.assertEqual(self.store.get_asset(self.pid,asset["id"])["revision"],1)

    def test_project_folder_is_readable_and_follows_renames(self):
        folder=self.store.directory(self.pid)
        self.assertEqual(folder.name,f"工件檢查__{self.pid[:8]}")
        renamed=self.store.update_project(self.pid,name="車把手／左側")
        folder=self.store.directory(self.pid)
        self.assertEqual(renamed["name"],"車把手／左側")
        self.assertEqual(folder.name,f"車把手_左側__{self.pid[:8]}")
        self.assertTrue((folder/"project.sqlite3").is_file())

    def test_class_management_adds_and_removes_unused_classes_exactly(self):
        revision=self.store.get_project(self.pid)["revision"]
        added=self.store.manage_classes(self.pid,["workpiece","temporary"],{},revision)
        self.assertEqual(added["classes"],["workpiece","temporary"])
        removed=self.store.manage_classes(self.pid,["workpiece"],{},added["revision"])
        self.assertEqual(removed["classes"],["workpiece"])
        self.assertEqual(removed["class_update"]["removed"],["temporary"])

    def test_new_project_has_no_default_class_and_editor_cannot_create_one_implicitly(self):
        self.assertEqual(self.project["classes"],[])
        asset=self.add()
        with self.assertRaises(ValueError):
            self.store.save_asset(self.pid,asset["id"],[dict(self.shape,label="系統預設")],asset["revision"])
        self.assertNotIn("系統預設",self.store.get_project(self.pid)["classes"])

    def test_used_class_removal_requires_and_applies_replacement(self):
        asset=self.add()
        project=self.store.get_project(self.pid)
        usage=self.store.class_usage(self.pid)
        self.assertEqual(usage["classes"],[{"name":"工件","object_count":1,"image_count":1}])
        with self.assertRaises(ValueError):
            self.store.update_project(self.pid,classes=["零件"])
        unchanged=self.store.get_asset(self.pid,asset["id"])
        self.assertEqual(unchanged["shapes"][0]["label"],"工件")
        result=self.store.manage_classes(self.pid,["零件"],{"工件":"零件"},project["revision"])
        changed=self.store.get_asset(self.pid,asset["id"])
        self.assertEqual(result["classes"],["零件"])
        self.assertEqual(result["class_update"]["changed_objects"],1)
        self.assertEqual(changed["shapes"][0]["label"],"零件")
        self.assertEqual(changed["revision"],asset["revision"]+1)
        self.assertEqual(changed["review_state"],"pending")
        self.assertEqual(self.store.history(self.pid,asset["id"])[-1]["action"],"class_relabel")

    def test_used_class_can_delete_its_annotation_objects_explicitly(self):
        asset=self.add();project=self.store.get_project(self.pid)
        result=self.store.manage_classes(self.pid,[],{},project["revision"],["工件"])
        changed=self.store.get_asset(self.pid,asset["id"])
        self.assertEqual(result["classes"],[])
        self.assertEqual(result["class_update"]["deleted_objects"],1)
        self.assertEqual(changed["shapes"],[])
        self.assertEqual(changed["review_state"],"pending")
        self.assertEqual(self.store.history(self.pid,asset["id"])[-1]["action"],"class_delete")

    def test_class_management_rejects_stale_project_revision(self):
        revision=self.store.get_project(self.pid)["revision"]
        self.store.update_project(self.pid,name="新專案名稱")
        with self.assertRaises(ConflictError):
            self.store.manage_classes(self.pid,["零件"],{},revision)

    def test_locked_legacy_folder_remains_available(self):
        readable=self.store.directory(self.pid)
        legacy=self.store.root/self.pid
        readable.rename(legacy)
        with patch.object(Path,"rename",side_effect=PermissionError(5,"存取被拒")):
            projects=self.store.list_projects()
        self.assertEqual([project["id"] for project in projects],[self.pid])
        self.assertEqual(self.store.directory(self.pid),legacy)

    def test_locked_folder_does_not_cancel_project_name_change(self):
        original=self.store.directory(self.pid)
        with patch.object(Path,"rename",side_effect=PermissionError(5,"存取被拒")):
            renamed=self.store.update_project(self.pid,name="新名稱")
        self.assertEqual(renamed["name"],"新名稱")
        self.assertEqual(self.store.directory(self.pid),original)

    def test_delete_project_checks_revision_and_removes_only_target(self):
        survivor=self.store.create_project("保留專案")
        folder=self.store.directory(self.pid)
        with self.assertRaises(ConflictError):
            self.store.delete_project(self.pid,self.project["revision"]-1)
        self.assertTrue(folder.is_dir())
        result=self.store.delete_project(self.pid,self.project["revision"])
        self.assertTrue(result["deleted"])
        self.assertFalse(folder.exists())
        self.assertEqual([item["id"] for item in self.store.list_projects()],[survivor["id"]])
        with self.assertRaises(FileNotFoundError):
            self.store.get_project(self.pid)

    def test_restore_creates_new_pending_revision(self):
        asset=self.add()
        self.define("B")
        old=self.store.history(self.pid,asset["id"])[0]["id"]
        edited=self.store.save_asset(self.pid,asset["id"],[dict(self.shape,label="B")],1)
        restored=self.store.restore(self.pid,asset["id"],old,edited["revision"])
        self.assertEqual(restored["shapes"][0]["label"],"工件")
        self.assertEqual(restored["review_state"],"pending")
        self.assertEqual(restored["revision"],3)

    def test_concurrent_saves_have_exactly_one_winner(self):
        asset=self.add();self.define("first","second");outcomes=[]
        def save(label):
            try:
                self.store.save_asset(self.pid,asset["id"],[dict(self.shape,label=label)],1)
                outcomes.append("ok")
            except ConflictError: outcomes.append("conflict")
        threads=[threading.Thread(target=save,args=(name,)) for name in ("first","second")]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertCountEqual(outcomes,["ok","conflict"])

    def test_delete_assets_is_atomic_and_removes_private_images(self):
        first=self.add();first_path=self.store.image_path(self.pid,first["id"])
        second_image=self.root/"第二張.png";Image.new("RGB",(40,30),(20,40,60)).save(second_image)
        added=self.store.add_assets(self.pid,[dict(path=second_image)])
        second=self.store.get_asset(self.pid,added["asset_ids"][0]);second_path=self.store.image_path(self.pid,second["id"])
        self.store.review(self.pid,[second["id"]],"approved",{second["id"]:second["revision"]})
        with self.assertRaises(ConflictError):
            self.store.delete_assets(self.pid,[first["id"],second["id"]],{first["id"]:first["revision"],second["id"]:second["revision"]})
        self.assertEqual(self.store.get_project(self.pid)["stats"]["total"],2)
        self.assertTrue(first_path.is_file());self.assertTrue(second_path.is_file())
        second=self.store.get_asset(self.pid,second["id"])
        result=self.store.delete_assets(self.pid,[first["id"],second["id"]],{first["id"]:first["revision"],second["id"]:second["revision"]})
        self.assertEqual(result["deleted"],2);self.assertEqual(result["project"]["assets"],[])
        self.assertFalse(first_path.exists());self.assertFalse(second_path.exists())


if __name__ == "__main__": unittest.main()
