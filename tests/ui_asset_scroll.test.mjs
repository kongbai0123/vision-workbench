import test from 'node:test';
import assert from 'node:assert/strict';
import {nearestVisibleScrollTop} from '../web/scroll-position.mjs';

test('keeps the current scroll position while the selected asset is visible',()=>{
  assert.equal(nearestVisibleScrollTop({scrollTop:300,viewportHeight:240,contentHeight:1200,itemTop:360,itemHeight:72}),300);
});

test('moves only enough to reveal an asset above or below the viewport',()=>{
  assert.equal(nearestVisibleScrollTop({scrollTop:300,viewportHeight:240,contentHeight:1200,itemTop:210,itemHeight:72}),210);
  assert.equal(nearestVisibleScrollTop({scrollTop:300,viewportHeight:240,contentHeight:1200,itemTop:580,itemHeight:72}),412);
});

test('clamps restored positions to the available scroll range',()=>{
  assert.equal(nearestVisibleScrollTop({scrollTop:900,viewportHeight:240,contentHeight:1000,itemTop:960,itemHeight:80}),760);
});
