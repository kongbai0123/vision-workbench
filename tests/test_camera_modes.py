import unittest
from workbench.camera_modes import normalize_modes


class CameraModesTests(unittest.TestCase):
    def test_directshow_interval_bounds_and_rounding(self):
        modes=normalize_modes([{'width':2560,'height':1440,'media_type_str':'MJPG',
                                'min_framerate':59.99988000024,'max_framerate':30.00003000003}])
        self.assertEqual((modes[0]['min_fps'],modes[0]['max_fps']),(30,60))
        self.assertIn(60,modes[0]['fps_options'])
        self.assertNotIn(24,modes[0]['fps_options'])

    def test_modes_retain_encoding_specific_limits(self):
        rows=[{'width':1280,'height':720,'media_type_str':pixel,'min_framerate':fps,'max_framerate':fps}
              for pixel,fps in [('MJPG',60),('YUY2',5)]]
        result=normalize_modes(rows+rows)
        self.assertEqual(len(result),2)
        self.assertEqual({m['pixel_format']:m['fps_options'] for m in result},{'MJPG':[60],'YUY2':[5]})

    def test_invalid_driver_modes_are_not_advertised(self):
        self.assertEqual(normalize_modes([{}, {'width':0,'height':720,'media_type_str':'MJPG','min_framerate':float('nan'),'max_framerate':30}]),[])

if __name__=='__main__':unittest.main()
