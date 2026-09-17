"""One transactional commit path for annotations returned by external editors."""
from copy import deepcopy
from .store import dump


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
    from .annotations import AnnotationService
    return AnnotationService(store).commit(pid, updates, source=source)['updated']
