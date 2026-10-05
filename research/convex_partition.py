"""Merge adjacent equal-material convex fixtures without changing their union.

Only complete shared edges are eligible. A candidate hull must have the same
area as both original pieces together and at most eight strictly convex vertices.
This is an exact representation change, not geometric smoothing or simplification.
"""
import copy
import numpy as np
from scipy.spatial import ConvexHull

from research.random_shapes import moments


def merge(fixtures):
    pieces=copy.deepcopy(fixtures); original=len(pieces)
    while True:
        candidates=[]
        for i,a in enumerate(pieces):
            va=np.asarray(a['vertices']); ma={k:a.get(k,d) for k,d in [('density',1),('friction',.3),('restitution',0),('rolling',0)]}
            for j in range(i+1,len(pieces)):
                b=pieces[j]; mb={k:b.get(k,d) for k,d in [('density',1),('friction',.3),('restitution',0),('rolling',0)]}
                if ma!=mb: continue
                known={'vertices','density','friction','restitution','rolling'}
                if {k:v for k,v in a.items() if k not in known}!={k:v for k,v in b.items() if k not in known}: continue
                vb=np.asarray(b['vertices']); shared=[]
                for x,y in zip(va,np.roll(va,-1,axis=0)):
                    if any((np.array_equal(x,u) and np.array_equal(y,v)) or
                           (np.array_equal(x,v) and np.array_equal(y,u))
                           for u,v in zip(vb,np.roll(vb,-1,axis=0))):
                        shared.append(np.linalg.norm(y-x))
                if not shared: continue
                points=np.unique(np.concatenate((va,vb)),axis=0)
                hull=points[ConvexHull(points).vertices]
                if len(hull)>8: continue
                area=abs(moments(va if moments_signed(va)>0 else va[::-1])[0])+abs(moments(vb if moments_signed(vb)>0 else vb[::-1])[0])
                if abs(moments(hull)[0]-area)>1e-12*max(1.,area): continue
                edge=np.roll(hull,-1,axis=0)-hull; following=np.roll(edge,-1,axis=0)
                if np.min(edge[:,0]*following[:,1]-edge[:,1]*following[:,0])<=1e-8: continue
                candidates.append((max(shared),i,j,hull))
        if not candidates: break
        _,i,j,hull=max(candidates,key=lambda c:c[0])
        pieces[i]['vertices']=hull.tolist(); pieces.pop(j)
    return pieces, {'original_fixtures':original,'merged_fixtures':len(pieces),
                    'geometry_policy':'equal-material shared-edge convex union; no boundary alteration'}


def moments_signed(vertices):
    v=np.asarray(vertices); w=np.roll(v,-1,axis=0)
    return np.sum(v[:,0]*w[:,1]-v[:,1]*w[:,0])/2


def scene_partition(scene):
    result=copy.deepcopy(scene); records=[]
    for index,body in enumerate(result['bodies']):
        if body.get('type','dynamic')!='dynamic' or not body.get('polygons'): continue
        body['polygons'],record=merge(body['polygons']); record['body']=index; records.append(record)
    result['partition_provenance']=records
    return result
