from pathlib import Path
import json
import tempfile
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch

from PIL import Image, ImageDraw
import numpy as np
from composer_core.geometry import encode_rle
from workbench.server import WorkbenchService


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="workbench-api-")
        self.root=Path(self.tmp.name)
        self.service=WorkbenchService(self.root/"data").start()
        source=self.root/"source";source.mkdir()
        image=Image.new("RGB",(160,120),(30,35,40));draw=ImageDraw.Draw(image)
        draw.rectangle((40,25,125,95),fill=(190,190,180));image.save(source/"零件.png")
        (source/"零件.json").write_text(json.dumps({"imagePath":"零件.png","imageWidth":160,"imageHeight":120,
            "shapes":[{"label":"part","shape_type":"rectangle","points":[[40,25],[125,95]]}]}),encoding="utf-8")
        self.source=source

    def tearDown(self):
        self.service.close();self.tmp.cleanup()

    def call(self,path,data=None,method=None,headers=None):
        hdr={"Content-Type":"application/json","X-Workbench":"1","Origin":self.service.url}
        if headers: hdr.update(headers)
        request=Request(self.service.url+path,data=json.dumps(data).encode() if data is not None else None,
                        method=method or ("POST" if data is not None else "GET"),headers=hdr)
        with urlopen(request,timeout=30) as response:
            return json.loads(response.read())

    def job(self,job):
        end=time.monotonic()+40
        while time.monotonic()<end:
            value=self.call('/api/jobs/'+job['id'])
            if value['state']=='succeeded':return value['result']
            if value['state']=='failed':self.fail(value.get('error'))
            time.sleep(.03)
        self.fail('Job timed out')

    def imported(self):
        project=self.call('/api/projects',{'name':'完整流程'})
        pid=project['id']
        result=self.job(self.call(f'/api/projects/{pid}/import',{'paths':[str(self.source)]}))
        self.assertEqual(result['added'],1)
        asset=self.call(f'/api/projects/{pid}/assets/'+result['asset_ids'][0])
        return pid,asset

    def define(self,pid,*names):
        usage=self.call(f'/api/projects/{pid}/classes')
        current=[item['name'] for item in usage['classes']]
        return self.call(f'/api/projects/{pid}/classes',{'classes':list(dict.fromkeys(current+list(names))),
            'replacements':{},'revision':usage['revision']})

    def test_import_edit_approve_export_reimport_without_manual_io(self):
        pid,asset=self.imported();aid=asset['id']
        self.define(pid,'verified-part')
        asset['shapes'][0]['label']='verified-part'
        edited=self.call(f'/api/projects/{pid}/assets/{aid}',{'revision':asset['revision'],'shapes':asset['shapes']},'PUT')
        self.call(f'/api/projects/{pid}/review',{'asset_ids':[aid],'state':'approved','revisions':{aid:edited['revision']}})
        validation=self.job(self.call(f'/api/projects/{pid}/validate',{'format':'native'}))
        self.assertTrue(validation['valid'],validation)
        exported=self.job(self.call(f'/api/projects/{pid}/export',{'format':'native','version':'release-1'}))
        self.assertTrue(Path(exported['path']).is_dir())
        self.assertTrue(Path(exported['zip_path']).is_file())
        target=self.call('/api/projects',{'name':'讀回驗證'})['id']
        result=self.job(self.call(f'/api/projects/{target}/import',{'paths':[exported['path']]}))
        imported=self.call(f'/api/projects/{target}/assets/'+result['asset_ids'][0])
        self.assertEqual(imported['shapes'][0]['label'],'verified-part')
        self.assertEqual(imported['sha256'],asset['sha256'])
        self.assertEqual(imported['review_state'],'approved')
        self.assertEqual(json.loads((self.source/'零件.json').read_text())['shapes'][0]['label'],'part')

    def test_live_edit_conflict_is_http_409(self):
        pid,asset=self.imported();aid=asset['id']
        self.define(pid,'new')
        asset['shapes'][0]['label']='new'
        self.call(f'/api/projects/{pid}/assets/{aid}',{'revision':1,'shapes':asset['shapes']},'PUT')
        with self.assertRaises(HTTPError) as raised:
            self.call(f'/api/projects/{pid}/assets/{aid}',{'revision':1,'shapes':[]},'PUT')
        self.assertEqual(raised.exception.code,409)
        self.assertEqual(self.call(f'/api/projects/{pid}/assets/{aid}')['shapes'][0]['label'],'new')

    def test_batch_delete_checks_revisions_and_returns_updated_project(self):
        pid,asset=self.imported();aid=asset['id']
        with self.assertRaises(HTTPError) as raised:
            self.call(f'/api/projects/{pid}/assets',{'asset_ids':[aid],'revisions':{aid:0}},'DELETE')
        self.assertEqual(raised.exception.code,409)
        result=self.call(f'/api/projects/{pid}/assets',{'asset_ids':[aid],'revisions':{aid:asset['revision']}},'DELETE')
        self.assertEqual(result['deleted'],1)
        self.assertEqual(result['project']['assets'],[])

    def test_class_management_reports_usage_and_relabels_saved_annotations(self):
        pid,asset=self.imported();aid=asset['id']
        usage=self.call(f'/api/projects/{pid}/classes')
        self.assertEqual(usage['classes'],[{'name':'part','object_count':1,'image_count':1}])
        result=self.call(f'/api/projects/{pid}/classes',{
            'classes':['component'],'replacements':{'part':'component'},'revision':usage['revision']})
        self.assertEqual(result['classes'],['component'])
        self.assertEqual(result['class_update']['changed_objects'],1)
        changed=self.call(f'/api/projects/{pid}/assets/{aid}')
        self.assertEqual(changed['shapes'][0]['label'],'component')
        self.assertEqual(changed['review_state'],'pending')

    def test_class_management_can_explicitly_delete_used_annotation_objects(self):
        pid,asset=self.imported();usage=self.call(f'/api/projects/{pid}/classes')
        result=self.call(f'/api/projects/{pid}/classes',{'classes':[],'replacements':{},
            'delete_objects':['part'],'revision':usage['revision']})
        self.assertEqual(result['classes'],[])
        self.assertEqual(result['class_update']['deleted_objects'],1)
        self.assertEqual(self.call(f"/api/projects/{pid}/assets/{asset['id']}")['shapes'],[])

    def test_project_delete_checks_revision_and_removes_project(self):
        project=self.call('/api/projects',{'name':'待刪除專案'})
        pid=project['id']
        with self.assertRaises(HTTPError) as raised:
            self.call(f'/api/projects/{pid}',{'revision':project['revision']-1},'DELETE')
        self.assertEqual(raised.exception.code,409)
        result=self.call(f'/api/projects/{pid}',{'revision':project['revision']},'DELETE')
        self.assertTrue(result['deleted'])
        self.assertNotIn(pid,[item['id'] for item in self.call('/api/projects')['projects']])
        with self.assertRaises(HTTPError) as raised:
            self.call(f'/api/projects/{pid}')
        self.assertEqual(raised.exception.code,404)

    def test_cross_origin_write_and_unknown_host_rejected(self):
        with self.assertRaises(HTTPError) as raised:
            self.call('/api/projects',{'name':'malicious'},headers={'Origin':'https://example.invalid'})
        self.assertEqual(raised.exception.code,403)
        with self.assertRaises(HTTPError) as raised:
            self.call('/api/projects',headers={'Host':'evil.example'})
        self.assertEqual(raised.exception.code,403)
        self.assertEqual(self.call('/api/projects')['projects'],[])

    def test_ai_candidate_requires_acceptance_and_separate_review(self):
        pid,asset=self.imported()
        result=self.job(self.call(f'/api/projects/{pid}/ai',{'asset_id':asset['id'],'revision':asset['revision'],
            'engine':'grabcut','box':[30,15,135,105],'label':'part'}))
        self.assertEqual(result['shape']['type'],'mask')
        unchanged=self.call(f'/api/projects/{pid}/assets/'+asset['id'])
        self.assertEqual(unchanged['shapes'],asset['shapes'])
        saved=self.call(f'/api/projects/{pid}/assets/'+asset['id'],{'revision':asset['revision'],
            'shapes':asset['shapes']+[result['shape']]},'PUT')
        self.assertEqual(saved['review_state'],'pending')
        self.assertEqual(saved['shape_count'],2)

    def test_camera_capture_can_attach_target_as_pending_annotation(self):
        project=self.call('/api/projects',{'name':'相機 ROI'})
        self.define(project['id'],'fixture')
        target={'type':'polygon','label':'fixture','points':[[10,10],[100,10],[80,90]]}
        record={'path':str(self.source/'零件.png'),'name':'相機擷取.png','batch_id':'camera-test','source':{'kind':'camera'}}
        with patch.object(self.service.camera,'snapshot',return_value=record):
            asset=self.call(f"/api/projects/{project['id']}/capture",{'target_shape':target})
        self.assertEqual(asset['review_state'],'pending')
        self.assertEqual(asset['shapes'][0]['type'],'polygon')
        self.assertEqual(asset['shapes'][0]['label'],'fixture')

    def test_dataset_training_model_and_prediction_api(self):
        project=self.call('/api/projects',{'name':'整合訓練 API'});pid=project['id']
        records=[]
        for index,split in enumerate(['train','train','val','val','test','test']):
            pixels=np.full((32,40,3),(20,25,30),np.uint8);pixels[6:26,8+index:25+index]=(185,65,45)
            path=self.root/f'train-{index}.png';Image.fromarray(pixels).save(path)
            mask=np.zeros((32,40),np.uint8);mask[6:26,8+index:25+index]=1
            records.append({'path':str(path),'name':path.name,'split':split,'batch_id':f'{split}-{index}',
                'review_state':'approved','source':{'kind':'api-test'},'shapes':[{'id':f'shape-{index}',
                'type':'mask','label':'handlebar','x':0,'y':0,'width':40,'height':32,'counts':encode_rle(mask)}]})
        added=self.service.store.add_assets(pid,records)
        overview=self.call(f'/api/projects/{pid}/training')
        self.assertTrue(overview['readiness']['ready'],overview)
        dataset=self.call(f'/api/projects/{pid}/dataset-versions',{})
        self.assertEqual(dataset['id'],'D001')
        run=self.call(f'/api/projects/{pid}/training-runs',{'dataset_version_id':'D001',
            'config':{'engine':'pixel_prototype_v1','epochs':5}})
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            overview=self.call(f'/api/projects/{pid}/training')
            run=next(item for item in overview['runs'] if item['run_id']==run['run_id'])
            if run['status'] in {'completed','failed','stopped'}:break
            time.sleep(.04)
        self.assertEqual(run['status'],'completed',run)
        metrics=self.call(f"/api/projects/{pid}/training-runs/{run['run_id']}/metrics")
        self.assertEqual(len(metrics['metrics']),5)
        target=self.root/'unreviewed.png';target_pixels=pixels.copy();target_pixels[0,0]=(21,25,30);Image.fromarray(target_pixels).save(target)
        aid=self.service.store.add_assets(pid,[{'path':str(target),'name':'unreviewed.png',
            'batch_id':'new','source':{'kind':'api-test'}}])['asset_ids'][0]
        prediction=self.job(self.call(f'/api/projects/{pid}/predictions',{
            'model_version_id':run['model_version_id'],'asset_ids':[aid]}))
        self.assertEqual(prediction['assets'][0]['status'],'candidate')
        accepted=self.call(f"/api/predictions/{prediction['candidate_id']}/accept",{'asset_ids':[aid]})
        self.assertEqual(accepted['accepted'],[aid])
        self.assertEqual(self.call(f'/api/projects/{pid}/assets/{aid}')['review_state'],'pending')

    def test_model_catalog_lists_optional_models_without_claiming_support(self):
        catalog=self.call('/api/model-catalog')
        engines={item['key']:item for item in catalog['engines']}
        self.assertIn('fasterrcnn_mobilenet_v3_large_fpn',engines)
        self.assertIn('deeplabv3_mobilenet_v3_large',engines)
        self.assertIn('efficientad',engines)
        self.assertIn('rt_detr_r50',engines)
        self.assertIn('yolo26n_seg',engines)
        self.assertEqual(engines['efficientad']['integration'],'planned')
        self.assertFalse(engines['efficientad']['train'])


if __name__=='__main__':unittest.main()
