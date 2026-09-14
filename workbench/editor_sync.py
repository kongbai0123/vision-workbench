"""One transactional commit path for annotations returned by external editors."""
from copy import deepcopy
from .store import ConflictError, clean_shapes, dump, identifier, timestamp


def geometry_key(shape):
    kind=shape.get('type','rectangle')
    values=[kind,shape['label'],bool(shape.get('hidden',False)),shape.get('metadata',{}).get('labelme'),shape.get('metadata',{}).get('cvat') or None]
    if kind=='mask': values.append(shape['counts'])
    elif kind=='rectangle': values.extend(float(shape[k]) for k in ('x','y','width','height'))
    else: values.append([[float(x),float(y)] for x,y in shape['points']])
    return dump(values)


def retain_identity(shapes, originals):
    """Avoid dirtying approved assets when an editor only normalizes JSON types."""
    remaining=list(originals);result=[]
    for shape in shapes:
        key=geometry_key(shape)
        match=next((old for old in remaining if geometry_key(old)==key),None)
        if match is not None:
            result.append(deepcopy(match));remaining.remove(match)
        else: result.append(shape)
    # An unchanged set must retain its original ordering, metadata and review state.
    if len(result)==len(originals) and sorted(map(geometry_key,result))==sorted(map(geometry_key,originals)):
        return deepcopy(originals)
    return result


def commit_updates(store, pid, updates, *, source):
    """All-or-nothing revision validation, class additions, history and review reset."""
    updated=0
    with store.connection(pid,write=True) as db:
        classes=__import__('json').loads(db.execute('SELECT classes FROM project').fetchone()[0])
        prepared=[]
        for original,shapes in updates:
            row=db.execute('SELECT * FROM assets WHERE id=?',(identifier(original['id']),)).fetchone()
            if row is None or row['revision']!=original['revision']:
                raise ConflictError('專案已有其他修改；已保留編輯內容，請先整合版本再同步。')
            cleaned=clean_shapes(shapes,row['width'],row['height'])
            for shape in cleaned:
                if shape['label'] not in classes: classes.append(shape['label'])
            if dump(cleaned)!=row['shapes']: prepared.append((row,cleaned))
        for row,shapes in prepared:
            revision=row['revision']+1
            db.execute("UPDATE assets SET shapes=?,revision=?,review_state='pending',updated_at=? WHERE id=?",
                (dump(shapes),revision,timestamp(),row['id']))
            store._history(db,row['id'],revision,'edit',{'shapes':shapes,'review_state':'pending','editor':source})
            updated+=1
        if updated:
            db.execute('UPDATE project SET classes=?',(dump(classes),))
            store._touch(db)
    return updated
