#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Global Localization: Grid search + area constraint + likelihood field

Approach:
  1. FRF filter (keep nearest cluster per angular bin, per frame)
  2. Compute scan convex hull (area + aspect ratio)
  3. Grid search over map free space:
     - Each position × 12 rotations
     - Score: likelihood field + hull area match + wall ratio
  4. Top-5 candidates → refine with finer grid
  5. Output best position

Usage:
  python3 simple_loc.py
  python3 simple_loc.py --data scan_viz/debug_match_data.npz
"""

import os, sys, math, time, argparse
import numpy as np
try: import cv2
except ImportError: print("need opencv-python"); sys.exit(1)
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt; import matplotlib.font_manager as fm
except ImportError: print("need matplotlib"); sys.exit(1)

for name in ['SimHei','Microsoft YaHei','SimSun']:
    if name in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.sans-serif']=[name]; break
plt.rcParams['axes.unicode_minus'] = False

# ===== Load =====
def load(npz_path):
    d=np.load(npz_path,allow_pickle=True)
    info={'resolution':float(d['map_resolution']),'width':int(d['map_width']),
          'height':int(d['map_height']),'origin_x':float(d['map_origin_x']),
          'origin_y':float(d['map_origin_y'])}
    tf_gt=d['tf_odom_to_map']; fts=d['frame_tfs']
    amin=float(d['frame_angle_min']); ainc=float(d['frame_angle_increment'])
    frs=[np.array(d[f'frame_ranges_{i}'],dtype=np.float64)
         for i in range(len(fts))]
    return d['map_data'],info,tf_gt,fts,frs,amin,ainc

# ===== FRF Filter =====
def filter_pts(frame_ranges,frame_tfs,amin,ainc,bin_deg=2.0):
    total,n_removed=0,0; all_pts,all_fids=[],[]
    for fi,(rng,tf) in enumerate(zip(frame_ranges,frame_tfs)):
        valid=(rng>0.1)&(rng<50.0); total+=int(np.sum(valid))
        if not np.any(valid): continue
        angs=amin+np.arange(len(rng))*ainc
        bins=np.round(angs/np.radians(bin_deg)).astype(int)
        keep=np.ones(len(rng),bool)
        for b in np.unique(bins[valid]):
            idx=np.where((bins==b)&valid)[0]
            if len(idx)<2:continue
            si=idx[np.argsort(rng[idx])]; sr=rng[si]
            gaps=np.diff(sr)>0.3
            if not np.any(gaps): continue
            keep[si[int(np.argmax(gaps))+1:]]=False
        n_removed+=int(np.sum(valid&~keep))
        fv=valid&keep
        if not np.any(fv): continue
        xl=rng[fv]*np.cos(angs[fv]); yl=rng[fv]*np.sin(angs[fv])
        c,s=np.cos(tf[2]),np.sin(tf[2])
        all_pts.append(np.column_stack([c*xl-s*yl+tf[0],s*xl+c*yl+tf[1]]))
        all_fids.append(np.full(np.sum(fv),fi,np.int32))
    if not all_pts: return np.empty((0,2)),{'t':total,'r':n_removed,'f':0}
    merged=np.vstack(all_pts)
    print(f"  FRF: {total} -> {len(merged)} ({100*n_removed/max(1,total):.1f}% ghost)")
    return merged,{'t':total,'r':n_removed,'f':len(merged)}

# ===== Grid Search =====
def search(pts,map_data,info,coarse=2.0,n_rot=12):
    res=info['resolution']; ox,oy=info['origin_x'],info['origin_y']
    H,W=map_data.shape; mw, mh = W*res, H*res
    
    # Scan features
    cx,cy=pts[:,0].mean(),pts[:,1].mean(); psc=pts-np.array([cx,cy])
    hull=cv2.convexHull(psc.astype(np.float32).reshape(-1,1,2)).reshape(-1,2)
    hull_area=cv2.contourArea(hull.astype(np.float32))
    hc=hull-hull.mean(0); ev,_=np.linalg.eigh(np.cov(hc.T))
    hull_aspect=math.sqrt(max(ev)/max(min(ev),1e-6))
    print(f"  Scan: {len(pts)}pts hull={hull_area:.0f}m² aspect={hull_aspect:.2f}")
    
    # Likelihood field
    obs=(map_data==100).astype(np.uint8)
    lf=cv2.distanceTransform(((1-obs)*255).astype(np.uint8),cv2.DIST_L2,5)*res
    lf=np.clip(lf,0,15).astype(np.float32)
    
    ds=max(1,len(psc)//500); pds=psc[::ds]
    angles=[math.radians(i*15) for i in range(24)]  # 15-degree steps
    
    xs=np.arange(ox+3,ox+mw-3,coarse); ys=np.arange(oy+3,oy+mh-3,coarse)
    all_res=[]; t0=time.time(); count=0
    
    for ax in xs:
        for ay in ys:
            col=int((ax-ox)/res); row=int(H-1-(ay-oy)/res)
            if not(0<=col<W and 0<=row<H and map_data[row,col]==0): continue
            count+=1
            for yaw in angles:
                c_y,s_y=math.cos(yaw),math.sin(yaw)
                # Hull projection
                hx=c_y*hull[:,0]-s_y*hull[:,1]+ax
                hy=s_y*hull[:,0]+c_y*hull[:,1]+ay
                hcol=((hx-ox)/res+0.5).astype(np.int32)
                hrow=((hy-oy)/res+0.5).astype(np.int32)
                vh=(hcol>=0)&(hcol<W)&(hrow>=0)&(hrow<H)
                if np.sum(vh)<len(hull)*0.5:continue
                mask=np.zeros((H,W),np.uint8)
                cv2.fillPoly(mask,[np.column_stack([hcol[vh],hrow[vh]]).astype(np.int32).reshape(-1,1,2)],255)
                hp=int(np.sum(mask>0))
                if hp<5:continue
                fp=int(np.sum((mask>0)&(map_data==0)))
                up=int(np.sum((mask>0)&(map_data==-1)))
                wp=int(np.sum((mask>0)&(map_data==100)))
                # Reject bad placements
                if hp>0 and (wp/hp>0.15 or (fp+up)/hp<0.7): continue
                
                # Area+aspect
                free_in_hull=(fp+up)*res*res
                ar=min(hull_area,free_in_hull)/max(hull_area,free_in_hull,0.1)
                rr,cc=np.where(mask>0); asp_match=0.5
                if len(rr)>5:
                    ps=np.column_stack([cc,rr]).astype(np.float64)
                    pc=ps-ps.mean(0); cr=np.cov(pc.T); ev2,_=np.linalg.eigh(cr)
                    pa=math.sqrt(max(ev2)/max(min(ev2),1e-6))
                    asp_match=max(0,1-abs(pa-hull_aspect)/max(hull_aspect,1))
                area_sc=fp/hp*0.3+ar*0.35+asp_match*0.35
                
                # Likelihood
                mx=c_y*pds[:,0]-s_y*pds[:,1]+ax
                my=s_y*pds[:,0]+c_y*pds[:,1]+ay
                ca=((mx-ox)/res+0.5).astype(np.int32)
                ra=((my-oy)/res+0.5).astype(np.int32)
                vv=(ca>=0)&(ca<W)&(ra>=0)&(ra<H)
                if np.sum(vv)<len(pds)*0.1:continue
                d=lf[ra[vv],ca[vv]]
                lf_sc=float(np.mean(np.exp(-d**2/0.18))+np.sum(d<0.15)/max(np.sum(vv),1))
                
                all_res.append((lf_sc+area_sc*2.0,ax,ay,yaw,lf_sc,area_sc))
    
    print(f"  Search: {len(xs)}x{len(ys)}={count}pos x{n_rot}rot, {time.time()-t0:.1f}s, {len(all_res)} valid")
    
    # NMS: keep top spatially diverse
    all_res.sort(key=lambda x:x[0],reverse=True)
    nms=[]
    for r in all_res:
        dup=any(math.sqrt((r[1]-p[1])**2+(r[2]-p[2])**2)<3.0 for p in nms)
        if not dup: nms.append(r)
        if len(nms)>=10: break
    return nms,hull_area,hull_aspect

# ===== Refine =====
def refine(pts,nms,map_data,info):
    res=info['resolution']; ox,oy=info['origin_x'],info['origin_y']
    H,W=map_data.shape
    obs=(map_data==100).astype(np.uint8)
    lf=cv2.distanceTransform(((1-obs)*255).astype(np.uint8),cv2.DIST_L2,5)*res
    lf=np.clip(lf,0,15).astype(np.float32)
    cx,cy=pts[:,0].mean(),pts[:,1].mean(); psc=pts-np.array([cx,cy])
    ds=max(1,len(psc)//400); pds=psc[::ds]
    hull=cv2.convexHull(psc.astype(np.float32).reshape(-1,1,2)).reshape(-1,2)
    
    refined=[]
    for _,hx,hy,hyaw,_,_ in nms[:8]:
        best=(-1e9,None,None,None)
        for dx in np.arange(-1.5,1.6,0.5):
            for dy in np.arange(-1.5,1.6,0.5):
                fx,fy=hx+dx,hy+dy
                col=int((fx-ox)/res);row=int(H-1-(fy-oy)/res)
                if not(0<=col<W and 0<=row<H and map_data[row,col]==0):continue
                for da in range(-15,16,5):
                    yaw=hyaw+math.radians(da)
                    c_y,s_y=math.cos(yaw),math.sin(yaw)
                    mx=c_y*pds[:,0]-s_y*pds[:,1]+fx
                    my=s_y*pds[:,0]+c_y*pds[:,1]+fy
                    ca=((mx-ox)/res+0.5).astype(np.int32)
                    ra=((my-oy)/res+0.5).astype(np.int32)
                    vv=(ca>=0)&(ca<W)&(ra>=0)&(ra<H)
                    if np.sum(vv)<len(pds)*0.1:continue
                    d=lf[ra[vv],ca[vv]]
                    sc=float(np.mean(np.exp(-d**2/0.18))+np.sum(d<0.15)/max(np.sum(vv),1))
                    if sc>best[0]: best=(sc,fx,fy,yaw)
        if best[1] is not None:
            refined.append((best[0],best[1],best[2],best[3]))
    refined.sort(key=lambda x:x[0],reverse=True)
    return refined

# ===== Viz =====
def viz(pts,map_data,info,refined,output):
    res,ox,oy=info['resolution'],info['origin_x'],info['origin_y']
    H,W=map_data.shape; ext=[ox,ox+W*res,oy,oy+H*res]
    bg=np.zeros((H,W,3),np.float32)
    bg[map_data==0]=[1,1,1]; bg[map_data==100]=[0.1,0.1,0.1]; bg[map_data==-1]=[0.7,0.7,0.7]
    
    fig,ax=plt.subplots(1,1,figsize=(16,14))
    ax.imshow(bg,origin='lower',extent=ext); ax.set_aspect('equal')
    
    # Best match overlay
    if refined:
        sc,fx,fy,fyaw=refined[0]
        c_y,s_y=math.cos(fyaw),math.sin(fyaw)
        aligned=np.column_stack([c_y*pts[:,0]-s_y*pts[:,1]+fx,
                                  s_y*pts[:,0]+c_y*pts[:,1]+fy])
        ds=aligned[::max(1,len(aligned)//3000)]
        ax.scatter(ds[:,0],ds[:,1],s=2,c='lime',alpha=0.6,label='Scan')
        ax.plot(fx,fy,'r+',ms=15,mew=3)
        ax.annotate('',xy=(fx+2.5*math.cos(fyaw),fy+2.5*math.sin(fyaw)),
                    xytext=(fx,fy),arrowprops=dict(arrowstyle='->',color='red',lw=3))
        for i,(sc2,x2,y2,yaw2) in enumerate(refined[:5]):
            color=['red','orange','green','cyan','magenta'][i]
            ax.plot(x2,y2,'o',color=color,ms=10-i)
            ax.annotate(f'#{i}',(x2,y2),fontsize=8,color=color,fontweight='bold')
        ax.set_title(f"Best: ({fx:.1f},{fy:.1f})@{math.degrees(fyaw):.0f}deg score={sc:.3f}")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.2)
    plt.tight_layout(); plt.savefig(output,dpi=150)
    print(f"\n[Output] {output}"); plt.close()

# ===== Main =====
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  '..','scan_viz','debug_match_data.npz'))
    p.add_argument('--output',default=None)
    args=p.parse_args()
    out=args.output or os.path.dirname(args.data)
    os.makedirs(out,exist_ok=True)
    
    map_data,info,tf_gt,fts,frs,amin,ainc=load(args.data)
    pts,stats=filter_pts(frs,fts,amin,ainc)
    
    print("\n=== Grid Search ===")
    candidates,hull_area,hull_aspect=search(pts,map_data,info)
    
    print("\n=== Refine ===")
    refined=refine(pts,candidates,map_data,info)
    
    print("\n=== Results ===")
    for i,(sc,x,y,yaw) in enumerate(refined[:5]):
        print(f"  #{i}: ({x:.2f},{y:.2f})@{math.degrees(yaw):.0f}deg score={sc:.3f}")
    
    viz(pts,map_data,info,refined,os.path.join(out,'simple_loc_result.png'))

if __name__=='__main__': main()
