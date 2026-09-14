"""Create and reuse a CVAT task for one Vision Workbench project."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import re
import secrets
import time
import uuid
import hashlib
import math
from copy import deepcopy
from collections import Counter

from .cvat_masks import to_cvat_mask, from_cvat_mask
from .editor_sync import retain_identity, commit_updates, geometry_key


class CvatProjectBridge:
    def __init__(self, data_root):
        self.mapping_file = Path(data_root) / "cvat" / "project-links.json"

    def _load(self):
        try:
            value = json.loads(self.mapping_file.read_text("utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, value):
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.mapping_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
        temporary.replace(self.mapping_file)

    @staticmethod
    def _headers(cookies, content_type="application/json"):
        token = next((row["value"] for row in cookies if row["name"] == "csrftoken"), "")
        # CVAT's DRF renderer only advertises its vendor type; plain application/json gets 406.
        return {"Host":"cvat.localhost:8768", "Accept":"application/vnd.cvat+json", "Content-Type":content_type,
                "Cookie":"; ".join(f'{row["name"]}={row["value"]}' for row in cookies),
                "X-CSRFToken":token, "Origin":"http://cvat.localhost:8768", "Referer":"http://cvat.localhost:8768/"}

    def _request(self, method, path, cookies, body=None, expected=(200, 201, 202)):
        connection = http.client.HTTPConnection("127.0.0.1", 8768, timeout=90)
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        try:
            connection.request(method, path, body=payload, headers=self._headers(cookies))
            response = connection.getresponse(); content = response.read()
        finally:
            connection.close()
        if response.status not in expected:
            raise RuntimeError(f"CVAT 專案同步失敗（HTTP {response.status}）。")
        return json.loads(content) if content else {}

    def _upload(self, task_id, images, cookies):
        boundary = "----VisionWorkbench" + secrets.token_hex(12)
        parts = []
        fields = {"image_quality":"70", "use_cache":"true", "sorting_method":"natural"}
        for name, value in fields.items():
            parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode(), None))
        for index, image in enumerate(images):
            header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"client_files[{index}]\"; "
                      f"filename=\"{image.name.replace(chr(34), '_')}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
            parts.append((header, image)); parts.append((b"\r\n", None))
        ending = f"--{boundary}--\r\n".encode()
        length = len(ending) + sum(len(chunk)+(path.stat().st_size if path else 0) for chunk,path in parts)
        connection = http.client.HTTPConnection("127.0.0.1", 8768, timeout=600)
        try:
            connection.putrequest("POST", f"/api/tasks/{task_id}/data", skip_host=True)
            for key,value in self._headers(cookies, f"multipart/form-data; boundary={boundary}").items(): connection.putheader(key,value)
            connection.putheader("Content-Length", str(length)); connection.endheaders()
            for chunk,path in parts:
                connection.send(chunk)
                if path:
                    with path.open("rb") as stream:
                        while block := stream.read(1024*1024): connection.send(block)
            connection.send(ending); response=connection.getresponse(); content=response.read()
        finally:
            connection.close()
        if response.status not in (200,201,202):
            raise RuntimeError(f"CVAT 圖片同步失敗（HTTP {response.status}）。")
        return json.loads(content) if content else {}

    def _wait_request(self, request_id, cookies):
        if not request_id: return
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            result=self._request("GET", f"/api/requests/{request_id}", cookies)
            status=result.get("status")
            if status=="finished": return
            if status=="failed": raise RuntimeError("CVAT 無法建立影像任務。")
            time.sleep(1)
        raise RuntimeError("CVAT 建立影像任務逾時，可稍後重新切換。")

    def _sync_annotations(self, task_id, assets, project_id, cookies):
        response=self._request("GET",f"/api/labels?project_id={project_id}&page_size=1000",cookies)
        rows=response.get("results",response if isinstance(response,list) else [])
        label_ids={row["name"]:row["id"] for row in rows}
        missing=sorted({s['label'] for a in assets for s in a.get('shapes',[])}-label_ids.keys())
        if missing:
            self._request('PATCH',f'/api/projects/{project_id}',cookies,
                {'labels':[{'name':name,'type':'any'} for name in missing]})
            response=self._request('GET',f'/api/labels?project_id={project_id}&page_size=1000',cookies)
            label_ids={row['name']:row['id'] for row in response['results']}
        shapes=[]; skipped_masks=0
        for frame,asset in enumerate(assets):
            for shape in asset.get("shapes",[]):
                kind=shape.get("type","rectangle"); points=None; cvat_kind=kind
                if kind=="rectangle":
                    points=[shape["x"],shape["y"],shape["x"]+shape["width"],shape["y"]+shape["height"]]
                elif kind in {"polygon","linestrip","point","obb"}:
                    cvat_kind={"linestrip":"polyline","point":"points","obb":"polygon"}.get(kind,kind)
                    points=[number for point in shape["points"] for number in point]
                elif kind=="mask":
                    points=to_cvat_mask(shape["counts"],asset["width"],asset["height"])
                else: raise ValueError(f'CVAT 無法接收標註類型：{kind}')
                label_id=label_ids.get(shape["label"])
                if label_id is None: raise ValueError(f'CVAT 缺少類別：{shape["label"]}')
                shapes.append({"type":cvat_kind,"frame":frame,"label_id":label_id,"points":points,
                    "group":0,"source":"manual","occluded":False,"outside":False,"z_order":0,"rotation":0,"attributes":[]})
                shapes[-1].update(shape.get('metadata',{}).get('cvat',{}))
        self._request("PUT",f"/api/tasks/{task_id}/annotations",cookies,
            {"version":0,"tags":[],"shapes":shapes,"tracks":[]},expected=(200,201))
        return skipped_masks

    def _restore_skipped_masks(self, linked, snapshot, cookies, progress):
        if linked.get("mask_sync_version", 0) >= 1:
            return
        task_id=linked["task_id"]
        annotations=self._request("GET",f"/api/tasks/{task_id}/annotations",cookies)
        labels=self._request("GET",f'/api/labels?project_id={linked["project_id"]}&page_size=1000',cookies)
        label_ids={row["name"]:row["id"] for row in labels["results"]}
        # A retry after a successful PATCH must not duplicate masks. Existing CVAT
        # masks (including user edits) count toward each frame/label's instances.
        existing=Counter((row["frame"],row["label_id"]) for row in annotations.get("shapes",[]) if row["type"]=="mask")
        natural=lambda value:[int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)",value)]
        assets=sorted(snapshot.get("assets",[]),key=lambda asset:natural(Path(asset["image_path"]).name))
        missing=[]
        for frame,asset in enumerate(assets):
            for shape in asset.get("shapes",[]):
                if shape.get("type")!="mask": continue
                label_id=label_ids.get(shape["label"])
                if label_id is None:
                    raise ValueError(f'CVAT 缺少遮罩類別「{shape["label"]}」，請先同步類別。')
                key=(frame,label_id)
                if existing[key]:
                    existing[key]-=1
                    continue
                missing.append({"type":"mask","frame":frame,"label_id":label_id,
                    "points":to_cvat_mask(shape["counts"],asset["width"],asset["height"]),
                    "group":0,"source":"manual","occluded":False,"outside":False,"z_order":0,"rotation":0,"attributes":[]})
        if missing:
            progress(f"補入 {len(missing)} 個遮罩標註",95)
            backup=self.mapping_file.parent/"annotation-backups"/f"task-{task_id}-{uuid.uuid4().hex}.json"
            backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_text(json.dumps(annotations,ensure_ascii=False),"utf-8")
            # Append only; never replace annotations already edited inside CVAT.
            self._request("PATCH",f"/api/tasks/{task_id}/annotations?action=create",cookies,
                {"version":annotations.get("version",0),"tags":[],"shapes":missing,"tracks":[]},expected=(200,201))
        linked.update(mask_sync_version=1,skipped_masks=0)

    @staticmethod
    def ordered_assets(snapshot):
        natural=lambda value:[int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)',value)]
        return sorted(snapshot.get('assets',[]),key=lambda a:natural(Path(a['image_path']).name))

    def _baseline_path(self,pid):
        from .store import identifier
        return self.mapping_file.parent/f'sync-{identifier(pid)}.json'

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def mark_synced(self,snapshot,cookies):
        linked=self._load()[snapshot['id']]
        remote=self._request('GET',f'/api/tasks/{linked["task_id"]}/annotations',cookies)
        identities={}
        self.read_annotations(snapshot,cookies,identity_sink=identities)
        links=self._load();links[snapshot['id']]['shape_ids']=identities;self._save(links)
        path=self._baseline_path(snapshot['id']);temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps({'snapshot':snapshot,'remote':self._digest(remote)},ensure_ascii=False),'utf-8')
        temp.replace(path)

    def synchronize(self,snapshot,cookies,store):
        """Merge disjoint asset edits; refuse concurrent edits to the same image."""
        pid=snapshot['id'];linked=self._load()[pid]
        path=self._baseline_path(pid)
        baseline=json.loads(path.read_text('utf-8')) if path.exists() else None
        remote=self._request('GET',f'/api/tasks/{linked["task_id"]}/annotations',cookies)
        if baseline is None:
            if self.read_annotations(snapshot,cookies):
                raise ValueError('CVAT 與專案有尚未同步的既有標註；請先開啟原 CVAT 工作並讀回，避免覆蓋任一版本。')
        elif baseline['remote']!=self._digest(remote):
            current={a['id']:a for a in snapshot['assets']};updates=[]
            for old,shapes in self.read_annotations(baseline['snapshot'],cookies):
                asset=current.get(old['id'])
                if asset is None:raise ValueError('CVAT 修改的圖片已從專案移除，原工作已保留。')
                if asset['shapes']!=old['shapes'] and retain_identity(shapes,asset['shapes'])!=asset['shapes']:
                    raise ValueError('同一張圖片在 CVAT 與工作台都有修改，已保留兩邊內容，請先整合版本。')
                updates.append((asset,retain_identity(shapes,asset['shapes'])))
            commit_updates(store,pid,updates,source='cvat')
            snapshot=store.snapshot(pid)
        # Comparing through the converters preserves OBB identity and ignores IDs.
        if self.read_annotations(snapshot,cookies):
            backup=self.mapping_file.parent/'annotation-backups'/f'task-{linked["task_id"]}-{uuid.uuid4().hex}.json'
            backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_text(json.dumps(remote,ensure_ascii=False),'utf-8')
            self._sync_annotations(linked['task_id'],self.ordered_assets(snapshot),linked['project_id'],cookies)
        self.mark_synced(snapshot,cookies)
        return snapshot

    def ensure_project(self, snapshot, cookies, progress=lambda *_:None, store=None):
        pid=snapshot["id"]; links=self._load(); linked=links.get(pid)
        asset_ids=[a['id'] for a in self.ordered_assets(snapshot)]
        if linked and linked.get("asset_count")==len(asset_ids) and linked.get('asset_ids',asset_ids)==asset_ids:
            try:
                self._request("GET", f'/api/tasks/{int(linked["task_id"])}', cookies)
            except (KeyError, ValueError, RuntimeError):
                links.pop(pid,None)
            else:
                self._restore_skipped_masks(linked,snapshot,cookies,progress)
                linked['asset_ids']=asset_ids
                self._save(links)
                if store is not None:self.synchronize(snapshot,cookies,store)
                return dict(linked, url=f'http://cvat.localhost:8768/tasks/{linked["task_id"]}/jobs/{linked["job_id"]}')
        labels=list(dict.fromkeys(snapshot.get("classes",[])+[shape["label"] for asset in snapshot.get("assets",[]) for shape in asset.get("shapes",[])]))
        if not labels:
            raise ValueError("請先在工作台新增至少一個物件類別，再開啟 CVAT。")
        progress("建立 CVAT 專案與類別",20)
        if linked:
            project={"id":linked["project_id"]}
        else:
            project=self._request("POST","/api/projects",cookies,{"name":snapshot["name"],"labels":[{"name":name,"type":"any"} for name in labels]})
        task_name=snapshot["name"] if not linked else f'{snapshot["name"]} · {len(snapshot.get("assets",[]))} 張'
        task=self._request("POST","/api/tasks",cookies,{"name":task_name,"project_id":project["id"]})
        natural=lambda value:[int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)",value)]
        assets=sorted(snapshot.get("assets",[]),key=lambda asset:natural(Path(asset["image_path"]).name))
        images=[Path(asset["image_path"]) for asset in assets]
        if not images: raise ValueError("目前專案沒有可送入 CVAT 的影像。")
        progress(f"傳送 {len(images)} 張影像至本機 CVAT",45)
        upload=self._upload(task["id"],images,cookies); self._wait_request(upload.get("rq_id") or upload.get("request_id"),cookies)
        progress("取得 CVAT 標註工作",90)
        jobs=self._request("GET",f'/api/jobs?task_id={task["id"]}',cookies)
        rows=jobs.get("results",jobs if isinstance(jobs,list) else [])
        if not rows: raise RuntimeError("CVAT 任務已建立，但尚未產生標註工作。")
        skipped_masks=self._sync_annotations(task["id"],assets,project["id"],cookies)
        linked={"project_id":project["id"],"task_id":task["id"],"job_id":rows[0]["id"],"asset_count":len(images),"asset_ids":asset_ids,"skipped_masks":skipped_masks,"mask_sync_version":1}
        links[pid]=linked;self._save(links)
        if store is not None:self.mark_synced(snapshot,cookies)
        return dict(linked,url=f'http://cvat.localhost:8768/tasks/{task["id"]}/jobs/{rows[0]["id"]}')

    def read_annotations(self, snapshot, cookies, identity_sink=None):
        linked=self._load().get(snapshot["id"])
        if not linked: raise ValueError("目前專案尚未建立 CVAT 工作。")
        response=self._request("GET",f'/api/labels?project_id={linked["project_id"]}&page_size=1000',cookies)
        rows=response.get("results",response if isinstance(response,list) else [])
        names={row["id"]:row["name"] for row in rows}
        annotations=self._request("GET",f'/api/tasks/{linked["task_id"]}/annotations',cookies)
        if annotations.get('tracks') or annotations.get('tags'):
            raise ValueError('目前專案以影像形狀標註同步；CVAT 的軌跡或整張標籤已保留，請先轉成影像形狀後再同步。')
        natural=lambda value:[int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)",value)]
        assets=sorted(snapshot.get("assets",[]),key=lambda asset:natural(Path(asset["image_path"]).name))
        by_frame={};remote_ids={}
        original_by_id={s['id']:s for a in assets for s in a.get('shapes',[])}
        for shape in annotations.get("shapes",[]):
            label=names.get(shape.get("label_id"));points=shape.get("points",[]);kind=shape.get("type")
            if not label or not isinstance(points,list): raise ValueError('CVAT 標註類別或座標無效。')
            result={"id":uuid.uuid4().hex,"label":label,"hidden":False,"metadata":{"source":"cvat"}}
            old=original_by_id.get(linked.get('shape_ids',{}).get(str(shape.get('id'))))
            if old:result.update(id=old['id'],metadata=deepcopy(old.get('metadata',{})),hidden=old.get('hidden',False))
            extra={k:deepcopy(shape[k]) for k in ('group','occluded','z_order','attributes') if shape.get(k)}
            if extra or 'cvat' in result['metadata']:result['metadata']['cvat']=extra
            if kind=="mask":
                frame=int(shape.get("frame",0))
                if not 0<=frame<len(assets): raise ValueError("CVAT 遮罩影格超出專案範圍。")
                asset=assets[frame]
                result.update(type="mask",x=0,y=0,width=asset["width"],height=asset["height"],
                    counts=from_cvat_mask(points,asset["width"],asset["height"]))
            elif kind=="rectangle" and len(points)==4:
                x1,y1,x2,y2=points;result.update(type="rectangle",x=min(x1,x2),y=min(y1,y2),width=abs(x2-x1),height=abs(y2-y1))
                angle=math.radians(shape.get('rotation',0))
                if angle:
                    cx,cy=(x1+x2)/2,(y1+y2)/2
                    result.update(type='obb',points=[[cx+(x-cx)*math.cos(angle)-(y-cy)*math.sin(angle),cy+(x-cx)*math.sin(angle)+(y-cy)*math.cos(angle)] for x,y in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]])
            elif kind in {"polygon","polyline","points"} and len(points)>=2 and len(points)%2==0:
                pairs=[points[index:index+2] for index in range(0,len(points),2)]
                result.update(type={"polyline":"linestrip","points":"point"}.get(kind,kind),points=pairs)
                if kind=='polygon' and len(pairs)==4 and old and old.get('type')=='obb':result['type']='obb'
                if result['type']=='point' and len(pairs)>1:
                    for pair in pairs:
                        by_frame.setdefault(int(shape.get('frame',0)),[]).append(dict(result,id=uuid.uuid4().hex,points=[pair]))
                    continue
            else: raise ValueError(f'CVAT 標註類型「{kind}」尚無對應轉換，已保留原內容。')
            if not 0<=int(shape.get('frame',0))<len(assets):raise ValueError('CVAT 影格超出專案範圍。')
            remote_ids[result['id']]=shape.get('id')
            by_frame.setdefault(int(shape.get("frame",0)),[]).append(result)
        updates=[]
        for frame,asset in enumerate(assets):
            originals=[shape for shape in asset.get("shapes",[]) if shape.get("type")=="mask"]
            shapes=by_frame.get(frame,[])
            if linked.get("mask_sync_version",0)<1:
                # Legacy tasks never received these masks; retain them until migration.
                shapes=originals+shapes
            # CVAT represents an oriented box as a polygon; restore its type.
            for shape in shapes:
                if shape['type']=='polygon':
                    for old in asset.get('shapes',[]):
                        if old['type']=='obb' and old['label']==shape['label'] and old['points']==shape['points']:
                            shape['type']='obb';break
                # Visibility and Labelme-specific properties are local display data.
                for old in asset.get('shapes',[]):
                    comparable=deepcopy(shape);comparable['hidden']=old.get('hidden',False)
                    comparable['metadata']=deepcopy(old.get('metadata',{}))
                    if geometry_key(comparable)==geometry_key(old):
                        shape['hidden']=old.get('hidden',False)
                        if 'labelme' in old.get('metadata',{}):shape['metadata']['labelme']=deepcopy(old['metadata']['labelme'])
                        break
            converted=shapes
            shapes=retain_identity(shapes,asset.get('shapes',[]))
            if identity_sink is not None:
                remaining=list(converted)
                for final in shapes:
                    match=next((s for s in remaining if geometry_key(s)==geometry_key(final)),None)
                    if match is not None:
                        remote_id=remote_ids.get(match['id'])
                        if remote_id is not None:identity_sink[str(remote_id)]=final['id']
                        remaining.remove(match)
            if shapes!=asset.get("shapes",[]): updates.append((asset,shapes))
        return updates
