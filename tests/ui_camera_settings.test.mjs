import test from 'node:test';
import assert from 'node:assert/strict';
import {controlValues, modesForFormat} from '../web/pages/camera-settings.mjs';

test('format-specific FPS choices cannot leak from MJPG into YUY2', () => {
  const modes = [{pixel_format:'MJPG', min_fps:30, max_fps:60}, {pixel_format:'YUY2', min_fps:5, max_fps:5}];
  assert.deepEqual(modesForFormat(modes,'YUY2').map(mode=>mode.max_fps), [5]);
  assert.equal(modesForFormat(modes,'auto').length,2);
  assert.deepEqual(modesForFormat(modes,'NV12'),[]);
});

test('profiles save current values and mode without stale capability ranges', () => {
  const controls={exposure:{min:-12,max:0,value:-8,auto:false},white_balance:{value:4500,auto:true}};
  const values=controlValues(controls);
  assert.deepEqual(values,{exposure:{value:-8,auto:false},white_balance:{value:4500,auto:true}});
  values.exposure.value=-4;
  assert.equal(controls.exposure.value,-8);
  assert.deepEqual(controlValues(),{});
});
