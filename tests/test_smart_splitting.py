import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from workbench.smart_splitting import smart_split, source_groups
from workbench.store import ProjectStore, ConflictError
from workbench.training import TrainingWorkspace


def samples():
    return [{"id":f"{g}-{i}","sha256":f"{g}-{i}","batch_id":g,"source":{},"split":"train" if i%2 else "val",
             "shapes":[{"label":"part"}]+([{"label":"rare"}] if g=='D' else [])}
            for g,n in [('A',18),('B',12),('C',12),('D',10)] for i in range(n)]


class SmartSplitTests(unittest.TestCase):
    def test_groups_never_split_and_training_keeps_rare_class(self):
        assets=samples();plan=smart_split(assets)
        for batch in 'ABCD':
            self.assertEqual(len({plan['assignments'][a['id']] for a in assets if a['batch_id']==batch}),1)
        self.assertTrue(all(plan['image_counts'].values()))
        self.assertGreater(plan['class_counts']['train']['rare'],0)
        self.assertTrue(any('rare' in w for w in plan['warnings']))
        self.assertEqual(plan,smart_split(list(reversed(assets))))

    def test_source_video_and_duplicates_form_transitive_groups(self):
        assets=samples()
        for a in assets:
            if a['batch_id'] in 'AB':a['source']={'video':'C:/clip.mp4'}
        assets[-1]['sha256']=assets[0]['sha256']
        groups=source_groups(assets)
        self.assertEqual(sorted(g['images'] for g in groups),[12,40])
        with self.assertRaisesRegex(ValueError,'群組'):smart_split(assets)
        plan=smart_split(assets,{'ratios':[80,20,0]})
        self.assertEqual(plan['image_counts']['test'],0)

    def test_locks_are_hard_and_conflicting_locks_fail(self):
        assets=samples();groups=source_groups(assets);group=next(g for g in groups if g['sources']==['B'])
        plan=smart_split(assets,{'locks':{group['id']:'test'}})
        self.assertTrue(all(plan['assignments'][aid]=='test' for aid in group['asset_ids']))
        with self.assertRaises(ValueError):smart_split(assets,{'locks':{g['id']:'test' for g in groups}})

    def test_video_hash_links_copies_and_legacy_path_records(self):
        assets=samples()[:3]
        for index,a in enumerate(assets):a['batch_id']=str(index)
        assets[0]['source']={'video':'C:/old.mp4'}
        assets[1]['source']={'video':'C:/old.mp4','video_sha256':'content'}
        assets[2]['source']={'original':{'video':'D:/copy.mp4','video_sha256':'content'}}
        self.assertEqual([g['images'] for g in source_groups(assets)],[3])

    def test_manual_groups_preserve_batch_metadata(self):
        assets=samples();original=json.dumps(assets)
        overrides={a['id']:a['id'] for a in assets}
        self.assertEqual(len(source_groups(assets,overrides)),52)
        self.assertEqual(json.dumps(assets),original)
        with self.assertRaises(ValueError):smart_split(assets,{'group_overrides':{'unknown':'x'}})

    def test_invalid_options_are_rejected(self):
        for options in ({'ratios':[70,None,10]},{'ratios':[float('nan'),20,10]}, {'ratios':[100,0,0]},
                        {'seed':True},{'seed':-1},{'strategy':'random'},{'locks':{'missing':'test'}}):
            with self.subTest(options=options),self.assertRaises(ValueError):smart_split(samples(),options)


class SmartSplitStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=ProjectStore(self.root/'projects');self.pid=self.store.create_project('smart')['id']
        records=[]
        for i in range(12):
            p=self.root/f'{i}.png';Image.new('RGB',(12,12),(i*17,20,60)).save(p)
            records.append({'path':str(p),'batch_id':f'batch-{i//3}','review_state':'approved','split':('train','val','test')[i%3],
                            'shapes':[{'type':'rectangle','label':'part','x':1,'y':1,'width':5,'height':5}]})
        self.store.add_assets(self.pid,records)
    def tearDown(self):self.tmp.cleanup()
    def test_preview_is_read_only_apply_is_atomic_and_versions_are_immutable(self):
        workspace=TrainingWorkspace(self.root,self.store)
        old=workspace.create_dataset_version(self.pid)
        path=workspace.datasets/self.pid/old['id']/'manifest.json';before=path.read_bytes()
        snapshot=self.store.snapshot(self.pid);plan=self.store.preview_split(self.pid,{})
        unchanged=self.store.snapshot(self.pid)
        self.assertEqual({k:v for k,v in snapshot.items() if k!='snapshot_at'}, {k:v for k,v in unchanged.items() if k!='snapshot_at'})
        result=self.store.apply_split(self.pid,{},plan['project_revision'],plan['fingerprint'])
        after=self.store.snapshot(self.pid)
        for a,b in zip(snapshot['assets'],after['assets']):
            self.assertEqual(a['shapes'],b['shapes']);self.assertEqual(a['batch_id'],b['batch_id']);self.assertEqual(b['review_state'],'approved')
        new=workspace.create_dataset_version(self.pid)
        manifest=json.loads((workspace.datasets/self.pid/new['id']/'manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(manifest['split_plan']['current']);self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(ConflictError):self.store.apply_split(self.pid,{},plan['project_revision'],plan['fingerprint'])
        workspace.close()
    def test_edit_after_preview_invalidates_apply(self):
        plan=self.store.preview_split(self.pid,{})
        asset=self.store.snapshot(self.pid)['assets'][0]
        self.store.assign(self.pid,[asset['id']],split='test')
        with self.assertRaises(ConflictError):self.store.apply_split(self.pid,{},plan['project_revision'],plan['fingerprint'])
