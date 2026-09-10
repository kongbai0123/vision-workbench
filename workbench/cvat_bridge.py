"""Create and reuse a CVAT task for one Vision Workbench project."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import re
import secrets
import time
import uuid


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
        return {"Host":"cvat.localhost:8768", "Accept":"application/json", "Content-Type":content_type,
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
                    skipped_masks+=1;continue
                else: continue
                label_id=label_ids.get(shape["label"])
                if label_id is None: continue
                shapes.append({"type":cvat_kind,"frame":frame,"label_id":label_id,"points":points,
                    "group":0,"source":"manual","occluded":False,"outside":False,"z_order":0,"rotation":0,"attributes":[]})
        if shapes:
            self._request("PUT",f"/api/tasks/{task_id}/annotations",cookies,
                {"version":0,"tags":[],"shapes":shapes,"tracks":[]},expected=(200,201))
        return skipped_masks

    def ensure_project(self, snapshot, cookies, progress=lambda *_:None):
        pid=snapshot["id"]; links=self._load(); linked=links.get(pid)
        if linked and linked.get("asset_count")==len(snapshot.get("assets",[])):
            try:
                self._request("GET", f'/api/tasks/{int(linked["task_id"])}', cookies)
                return dict(linked, url=f'http://cvat.localhost:8768/tasks/{linked["task_id"]}/jobs/{linked["job_id"]}')
            except (KeyError, ValueError, RuntimeError):
                links.pop(pid,None)
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
        linked={"project_id":project["id"],"task_id":task["id"],"job_id":rows[0]["id"],"asset_count":len(images),"skipped_masks":skipped_masks}
        links[pid]=linked;self._save(links)
        return dict(linked,url=f'http://cvat.localhost:8768/tasks/{task["id"]}/jobs/{rows[0]["id"]}')

    def read_annotations(self, snapshot, cookies):
        linked=self._load().get(snapshot["id"])
        if not linked: raise ValueError("目前專案尚未建立 CVAT 工作。")
        response=self._request("GET",f'/api/labels?project_id={linked["project_id"]}&page_size=1000',cookies)
        rows=response.get("results",response if isinstance(response,list) else [])
        names={row["id"]:row["name"] for row in rows}
        annotations=self._request("GET",f'/api/tasks/{linked["task_id"]}/annotations',cookies)
        by_frame={}
        for shape in annotations.get("shapes",[]):
            label=names.get(shape.get("label_id"));points=shape.get("points",[]);kind=shape.get("type")
            if not label or not isinstance(points,list): continue
            result={"id":uuid.uuid4().hex,"label":label,"hidden":False,"metadata":{"source":"cvat"}}
            if kind=="rectangle" and len(points)==4:
                x1,y1,x2,y2=points;result.update(type="rectangle",x=min(x1,x2),y=min(y1,y2),width=abs(x2-x1),height=abs(y2-y1))
            elif kind in {"polygon","polyline","points"} and len(points)>=2 and len(points)%2==0:
                pairs=[points[index:index+2] for index in range(0,len(points),2)]
                result.update(type={"polyline":"linestrip","points":"point"}.get(kind,kind),points=pairs)
                if result["type"]=="point": result["points"]=pairs[:1]
            else: continue
            by_frame.setdefault(int(shape.get("frame",0)),[]).append(result)
        natural=lambda value:[int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)",value)]
        assets=sorted(snapshot.get("assets",[]),key=lambda asset:natural(Path(asset["image_path"]).name))
        updates=[]
        for frame,asset in enumerate(assets):
            # CVAT does not round-trip the workbench's exact RLE mask representation;
            # retain those masks while replacing the vector annotations it owns.
            masks=[shape for shape in asset.get("shapes",[]) if shape.get("type")=="mask"]
            shapes=masks+by_frame.get(frame,[])
            if shapes!=asset.get("shapes",[]): updates.append((asset,shapes))
        return updates
