import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from workbench import model_trial_worker
from workbench.training import TrainingWorkspace

class FakePredictor:
    def predict(self,rgb,width,height):
        return [{'type':'rectangle','label':'part','x':1,'y':1,'width':width-2,'height':height-2,
                 'metadata':{'confidence':.8}}]

class ModelTrialWorkerTests(unittest.TestCase):
    def test_images_and_every_video_frame_are_processed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);image=root/'one.png';Image.new('RGB',(32,24),'white').save(image)
            video=root/'all.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(32,24))
            self.assertTrue(writer.isOpened())
            for value in (20,80,140):writer.write(np.full((24,32,3),value,np.uint8))
            writer.release();request=root/'request.json';result=root/'result.json'
            request.write_text(json.dumps({'model_path':str(root/'model.json'),'predict_module':'fake','paths':[str(image),str(video)],'output_dir':str(root/'media')}))
            with patch.object(model_trial_worker,'predictor',return_value=(FakePredictor(),np.uint8)),patch.object(sys,'argv',['worker','--request',str(request),'--output',str(result)]):
                self.assertEqual(model_trial_worker.main(),0)
            rows=json.loads(result.read_text())['frames'];self.assertEqual(len(rows),4)
            self.assertEqual([row['frame_index'] for row in rows],[None,0,1,2])
            self.assertTrue(all((root/'media'/row['image']).is_file() for row in rows))
            self.assertEqual([round(row['time_seconds'],1) if row['time_seconds'] is not None else None for row in rows],[None,0,0.2,0.4])

    def test_labeled_comparison_reports_tp_fp_fn_without_changing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'projects'/'p'/'datasets'/'D001';(dataset/'images').mkdir(parents=True)
            Image.new('RGB',(100,80),'white').save(dataset/'images'/'a.png')
            truth={'id':'t','type':'rectangle','label':'part','x':10,'y':10,'width':20,'height':20}
            manifest={'assets':[{'name':'a.png','split':'test','image_file':'images/a.png','shapes':[truth]}]}
            (dataset/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');before=(dataset/'manifest.json').read_bytes()
            class Fake:
                def __init__(self):self.root=root
                def model(self,*_):return {'dataset_version_id':'D001'}
                def datasets_dir(self,*_):return dataset.parent
                def create_model_trial(self,*_args,**_kwargs):
                    media=root/'model-trials'/('b'*32)/'media';media.mkdir(parents=True);Image.new('RGB',(100,80)).save(media/'00000000.jpg')
                    return {'session_id':'b'*32,'frames':[{'width':100,'height':80,'shapes':[
                        {'type':'rectangle','label':'part','x':11,'y':11,'width':20,'height':20},
                        {'type':'rectangle','label':'part','x':60,'y':60,'width':10,'height':10}]}]}
            result=TrainingWorkspace.create_model_comparison(Fake(),'p','M001','test')
            self.assertEqual({k:result['comparison'][k] for k in ('tp','fp','fn')},{'tp':1,'fp':1,'fn':0})
            self.assertEqual((dataset/'manifest.json').read_bytes(),before)

if __name__=='__main__':unittest.main()
