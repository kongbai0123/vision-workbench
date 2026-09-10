"""Dataset entry files accepted by the unified picker and native drop zone."""
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from workbench.pipeline import import_sources


class DropFormatsTest(unittest.TestCase):
    def test_image_before_coco_keeps_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); image=root/'image.png'; Image.new('RGB',(40,30)).save(image)
            annotation=root/'instances.json'
            annotation.write_text(json.dumps({'images':[{'id':1,'file_name':'image.png','width':40,'height':30}],
                'categories':[{'id':1,'name':'part'}],
                'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[2,2,20,20],'area':400,'iscrowd':0}]}))
            result=import_sources([image,annotation])
            self.assertEqual(len(result['records']),1)
            self.assertEqual(result['records'][0]['shapes'][0]['label'],'part')

    def test_yolo_entry_points_and_single_label_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'images/train').mkdir(parents=True); (root/'labels/train').mkdir(parents=True)
            for name in ['one','two']:
                Image.new('RGB',(40,30)).save(root/f'images/train/{name}.png')
                (root/f'labels/train/{name}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            config=root/'data.yaml'; config.write_text('names: [part]\n')
            result=import_sources([root/'labels/train/one.txt'])
            self.assertFalse(result['issues'])
            self.assertEqual(len(result['records']),1)
            self.assertEqual(result['records'][0]['name'],'one.png')
            self.assertEqual(len(import_sources([config])['records']),2)
            config.unlink()  # Test-owned fixture: legacy labels/classes.txt layout.
            classes=root/'labels/classes.txt'; classes.write_text('part\n')
            result=import_sources([root/'labels/train/one.txt'])
            self.assertFalse(result['issues'])
            self.assertEqual(result['records'][0]['shapes'][0]['label'],'part')
            self.assertEqual(len(import_sources([classes])['records']),2)

    def test_orphan_annotation_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'orphan.txt'; path.write_text('0 0.5 0.5 0.5 0.5')
            result=import_sources([path])
            self.assertFalse(result['records'])
            self.assertEqual(result['issues'][0]['level'],'error')
            self.assertIn('data.yaml',result['issues'][0]['message'])

if __name__=='__main__':unittest.main()
