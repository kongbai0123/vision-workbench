"""Explicit small CPU training check in the optional TorchVision environment."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);store=ProjectStore(root/'projects');pid=store.create_project('LR CPU smoke')['id']
        for i,split in enumerate(['train']*4+['val']*2+['test']*2):
            path=root/f'{i}.png';image=Image.new('RGB',(128,128),(220,i*4,180))
            ImageDraw.Draw(image).rectangle((24,24,86,92),fill=(30,130,i*8));image.save(path)
            store.add_assets(pid,[{'path':str(path),'split':split,'batch_id':str(i),'review_state':'approved',
                                  'shapes':[{'type':'rectangle','label':f'part{i%2}','x':24,'y':24,'width':62,'height':68}]}])
        workspace=TrainingWorkspace(root,store);dataset=workspace.create_dataset_version(pid)
        for engine in ['maskrcnn_resnet50_fpn','fasterrcnn_mobilenet_v3_large_320_fpn','deeplabv3_mobilenet_v3_large','resnet18_classification']:
            run=workspace.start_run(pid,dataset['id'],{'engine':engine,'epochs':3,'device':'cpu','image_size':128,
                'batch_size':1,'learning_rate':.0005,'min_learning_rate':.000005,'scheduler':'cosine','warmup_epochs':0})
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                result=workspace.run(pid,run['run_id'])
                if result['status'] in {'completed','failed','stopped'}:break
                time.sleep(.2)
            else:
                workspace.stop_run(pid,run['run_id']);raise TimeoutError(engine)
            assert result['status']=='completed',result
            directory=workspace.runs_dir(pid)/run['run_id']
            rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            rates=[r['train/learning_rate'] for r in rows]
            assert abs(rates[0]-.0005)<1e-12 and abs(rates[-1]-.000005)<1e-12,rates
            assert rates[0]>rates[1]>rates[2],rates
            assert json.loads((directory/'lr-state.json').read_text())['epoch']==3
            print('PASS',engine,rates,flush=True)
            while workspace.processes:time.sleep(.05)
        workspace.close()
        print('LEARNING_RATE_MODELS_SMOKE_OK',flush=True)

if __name__=='__main__':main()

