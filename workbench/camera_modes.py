"""Read device-advertised Windows modes without starting a video stream."""
import math
import os


def normalize_modes(formats):
    modes=[]
    aliases={'RGB24':'RGB3','YUYV':'YUY2'}
    for item in formats:
        try:
            width,height=abs(int(item['width'])),abs(int(item['height']))
            low,high=sorted((float(item['min_framerate']),float(item['max_framerate'])))
            if not (32<=width<=8192 and 32<=height<=8192 and math.isfinite(low) and math.isfinite(high) and 0<low<=high<=240):continue
            # DirectShow intervals use 100 ns units; 59.99988 represents 60 fps.
            low,high=(round(v) if abs(v-round(v))<.001 else round(v,3) for v in (low,high))
            pixel=aliases.get(item['media_type_str'],item['media_type_str'])
            if len(pixel)!=4 or not pixel.isascii():continue
            rates=sorted({low,high,*[v for v in (5,10,15,20,23.976,24,25,29.97,30,48,50,59.94,60,90,100,120,144,240) if low<=v<=high]})
            mode={'width':width,'height':height,'pixel_format':pixel,'min_fps':low,'max_fps':high,'fps_options':rates}
            if mode not in modes:modes.append(mode)
        except (KeyError,ValueError,TypeError):continue
    return sorted(modes,key=lambda m:(-m['width']*m['height'],m['pixel_format']!='MJPG',-m['max_fps']))


def device_modes(index):
    if type(index) is not int or not 0<=index<=128:raise ValueError('相機編號無效')
    if os.name!='nt':return {'index':index,'modes':[],'error':'此平台未提供 DirectShow 模式清單'}
    import comtypes
    from pygrabber.dshow_graph import FilterGraph
    graph=None
    comtypes.CoInitialize()
    try:
        graph=FilterGraph();names=graph.get_input_devices()
        if index>=len(names):raise ValueError('找不到選取的鏡頭，請重新偵測')
        graph.add_video_input_device(index)
        modes=normalize_modes(graph.get_input_device().get_formats())
        return {'index':index,'name':names[index],'modes':modes,'source':'DirectShow IAMStreamConfig',
                'error':None if modes else '鏡頭未回報可用模式，可改用自訂設定'}
    finally:
        try:
            if graph is not None:
                graph.remove_filters()
                del graph
        finally:comtypes.CoUninitialize()
