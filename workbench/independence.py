"""Fingerprint compatibility and evidence-gated migration repair."""
import hashlib
import json


def fingerprint(db, *, strip_metadata=False):
    rows = []
    for row in db.execute("SELECT id,sha256,batch_id,review_state,source,shapes FROM assets WHERE review_state='approved' ORDER BY id"):
        aid, digest, batch, review, raw, shapes = row
        source = json.loads(raw)
        if strip_metadata:
            source.pop('review', None)
            source.pop('quality', None)
        rows.append(dict(id=aid, sha256=digest, batch_id=batch, review_state=review,
                         source=source, annotation_sha256=hashlib.sha256(shapes.encode('utf-8')).hexdigest()))
    raw = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest(), len(rows)


def confirmation(db):
    row = db.execute('SELECT data FROM independence_reviews WHERE id=1').fetchone()
    return json.loads(row[0]) if row else None


def valid(record, signature):
    return bool(record and record.get('asset_fingerprint') == signature[0]
                and record.get('asset_count') == signature[1])


def preserve(db, record, signature, *, migration):
    updated = dict(record, asset_fingerprint=signature[0], asset_count=signature[1])
    updated['migration_audit'] = [*record.get('migration_audit', []), {
        'migration': migration, 'old_fingerprint': record['asset_fingerprint'],
        'new_fingerprint': signature[0]}]
    db.execute('UPDATE independence_reviews SET data=? WHERE id=1',
               (json.dumps(updated, ensure_ascii=False),))
    return updated


def repair(db, backup, *, apply=False):
    """Caller owns the transaction; backup must be an isolated, verified copy.

    Never infer old validity from reconstructed source JSON. Require the original
    confirmation plus unchanged approved content; repeated application is a no-op.
    """
    current, original = confirmation(db), confirmation(backup)
    signature = fingerprint(db)
    if valid(current, signature):
        return {'eligible': False, 'changed': False, 'reason': 'already_current'}
    if not valid(original, fingerprint(backup)):
        reason = 'backup_confirmation_invalid'
    elif current != original:
        reason = 'confirmation_changed'
    elif fingerprint(backup, strip_metadata=True) != signature:
        reason = 'approved_content_changed'
    else:
        if apply:
            preserve(db, original, signature, migration='0002-backup-repair')
        return {'eligible': True, 'changed': apply, 'reason': 'migration_only'}
    return {'eligible': False, 'changed': False, 'reason': reason}
