"""
图像房间切割 v7 — 墙体线屏障 + 智能连通域
==========================================================
核心算法(针对有清晰墙线的地图):
  1. 三值化: 黑(<50)=墙体线, 灰(50~160)=家具障碍, 白(>160)=自由空间
  2. 提取墙体线(Canny+阈值) → 膨胀形成"薄墙屏障"
  3. 屏障 + 自由空间取反 → 连通域 ≈ 房间
  4. 若走廊仍连通: 用距离变换找"宽区域核心"作为种子,
     再按最近质心分配像素(类似Voronoi但受墙约束)
"""
import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Room:
    id: int; mask: np.ndarray; contour: np.ndarray
    polygon: np.ndarray = None; area: float = 0
    centroid: Tuple[float,float] = (0,0); bbox: Tuple[int,int,int,int] = (0,0,0,0)

def segment_rooms(gray,
                  wall_dark=50, obstacle_high=160, free_low=165,
                  wall_dilate=13, min_area=800):
    h,w = gray.shape

    # ---- 墙体提取: 深色阈值 + Canny边缘 + 中灰区域 ----
    is_wall_dark = gray < wall_dark
    blur = cv2.GaussianBlur(gray, (3,3), 0.5)
    edges = cv2.Canny(blur, 30, 100)
    is_wall_gray = (gray >= wall_dark) & (gray <= 100)
    is_wall = is_wall_dark | (edges > 0) | is_wall_gray

    is_obstacle = (gray > 100) & (gray <= obstacle_high)
    is_free = gray > free_low

    wall_mask = is_wall.astype(np.uint8)*255
    obstacle_mask = is_obstacle.astype(np.uint8)*255
    free_mask = is_free.astype(np.uint8)*255

    nd = int(is_wall_dark.sum()); ne = int((edges>0).sum())
    ng = int(is_wall_gray.sum()); no = int(is_obstacle.sum()); nf = int(is_free.sum())
    print(f"  [wall={is_wall.sum()} dark={nd} edge={ne} gray_w={ng}] obs={no} free={nf}")

    # 清理墙体
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    wall_clean = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # 膨胀形成屏障
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(wall_dilate,wall_dilate))
    barrier = cv2.dilate(wall_clean, kd, iterations=2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(wall_dilate//2+3,wall_dilate//2+3))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kc, iterations=2)

    # 可通行区域
    traversable = free_mask.copy()
    traversable[barrier > 0] = 0
    traversable[obstacle_mask > 0] = 0
    traversable = cv2.morphologyEx(traversable, cv2.MORPH_OPEN, k3, iterations=1)

    # 连通域
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(traversable, connectivity=8)
    print(f"  [CC] {n_cc-1} regions")

    regions = []
    for lid in range(1, n_cc):
        if stats[lid,cv2.CC_STAT_AREA] < min_area: continue
        regions.append({'lid':lid,'area':stats[lid,cv2.CC_STAT_AREA],
                        'ctr':centroids[lid],'mask':(labels==lid).astype(np.uint8)*255})

    if len(regions) <= 1:
        print("  -> corridor connected, using distance-seed split...")
        return _dist_seed_split(gray, wall_mask, obstacle_mask, free_mask, barrier, min_area)

    rooms = []
    for reg in regions:
        m = reg['mask'].copy()
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=3)
        m[free_mask==0] = 0
        cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area*0.5: continue
        M = cv2.moments(m)
        cx = M['m10']/M['m00'] if M['m00']>0 else reg['ctr'][0]
        cy = M['m01']/M['m00'] if M['m00']>0 else reg['ctr'][1]
        xs=np.where(m>0)[1]; ys=np.where(m>0)[0]
        rooms.append(Room(len(rooms),m,c,area=float(cv2.countNonZero(m)),
                          centroid=(cx,cy),bbox=(xs.min(),ys.min(),xs.max()-xs.min(),ys.max()-ys.min())))
    rooms.sort(key=lambda r:r.area, reverse=True)
    for i,r in enumerate(rooms): r.id=i
    return rooms


def _dist_seed_split(gray, wmask, omask, fmask, barrier, min_area):
    """距离变换种子法"""
    h,w = gray.shape
    blocked = ((wmask>0)|(omask>0)|(barrier>0)|(fmask==0))
    freespace = (~blocked).astype(np.uint8)*255
    if cv2.countNonZero(freespace) < min_area: return []

    dist = cv2.distanceTransform(freespace, cv2.DIST_L2, 5)
    dmax = dist.max()
    if dmax < 5: return []

    best_seeds = None
    for ratio in np.arange(0.40, 0.08, -0.04):
        sc = (dist > dmax*ratio).astype(np.uint8)*255
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        sc = cv2.morphologyEx(sc, cv2.MORPH_CLOSE, k5, iterations=2)
        sc = cv2.morphologyEx(sc, cv2.MORPH_OPEN, k5, iterations=1)
        nl,lb,st,ct = cv2.connectedComponentsWithStats(sc, connectivity=8)
        valid = [i for i in range(1,nl) if st[i,cv2.CC_STAT_AREA] >= min_area*0.12]
        if 3 <= len(valid) <= 10:
            best_seeds = [(ct[i][0],ct[i][1]) for i in valid]
            break
        elif len(valid) >= 2 and best_seeds is None:
            best_seeds = [(ct[i][0],ct[i][1]) for i in valid]

    if not best_seeds:
        print("    no valid seeds"); return []
    print(f"    {len(best_seeds)} seeds")

    seeds = np.array(best_seeds)
    ys,xs = np.where(freespace>0)
    assign = np.full((h,w),-1,np.int32)
    for s in range(0,len(xs),50000):
        e=min(s+50000,len(xs))
        chunk=np.column_stack([xs[s:e],ys[s:e]])
        ds=np.sqrt(((chunk[:,None,:]-seeds[None,:,:])**2).sum(axis=2))
        for k,(py,px) in enumerate(zip(ys[s:e],xs[s:e])):
            assign[py,px]=np.argmin(ds[k-s])

    rooms=[]
    k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for si in range(len(seeds)):
        rm=(assign==si).astype(np.uint8)*255; rm[freespace==0]=0
        if cv2.countNonZero(rm)<min_area: continue
        rm=cv2.morphologyEx(rm, cv2.MORPH_CLOSE,k3,iterations=2)
        rm=cv2.morphologyEx(rm, cv2.MORPH_OPEN,k3,iterations=1)
        cs,_=cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        c=max(cs,key=cv2.contourArea)
        if cv2.contourArea(c)<min_area*0.3: continue
        M=cv2.moments(rm); cx=M['m10']/M['m00']if M['m00']else seeds[si][0]; cy=M['m01']/M['m00']if M['m00']else seeds[si][1]
        rx=np.where(rm>0)[1]; ry=np.where(rm>0)[0]
        rooms.append(Room(len(rooms),rm,c,area=float(cv2.countNonZero(rm)),centroid=(cx,cy),
                          bbox=(rx.min(),ry.min(),rx.max()-rx.min(),ry.max()-ry.min())))
    rooms.sort(key=lambda r:r.area,reverse=True)
    for i,r in enumerate(rooms): r.id=i
    return rooms

# ============================================================
# 后处理 & 可视化
# ============================================================

def polygonize(rooms, eps=0.01):
    for r in rooms:
        if r.contour is None or len(r.contour)<4: r.polygon=np.array([]);continue
        peri=cv2.arcLength(r.contour,True)
        if peri<1: r.polygon=np.array([]);continue
        p=cv2.approxPolyDP(r.contour,max(eps*peri,2.0),True)
        r.polygon=p.reshape(-1,2) if len(p)>=3 else np.array([])
    return rooms

def build_graph(rooms, dk=21, thr=30):
    g=defaultdict(list); n=len(rooms)
    if n<2: return dict(g)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dk,dk))
    dils=[cv2.dilate(r.mask,k,iterations=2) for r in rooms]
    for i in range(n):
        for j in range(i+1,n):
            if cv2.countNonZero(cv2.bitwise_and(dils[i],dils[j]))>thr:
                g[i].append(j);g[j].append(i)
    return dict(g)

def visualize(orig,gray,rooms,wbarrier,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    n=len(rooms); cols=plt.cm.Set1(np.linspace(0,1,max(n,9)))
    cbgr=[(int(c[2]*255),int(c[1]*255),int(c[0]*255))for c in cols]

    fig,axes=plt.subplots(2,4,figsize=(26,13))
    axes[0,0].imshow(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')

    # 墙体可视化 (合并后的)
    wv=orig.copy()
    wd=gray<50; blur=cv2.GaussianBlur(gray,(3,3),0.5)
    ed=cv2.Canny(blur,30,100); wg=(gray>=50)&(gray<=100)
    allw=wd|(ed>0)|wg; wv[allw]=[255,0,0]
    axes[0,1].imshow(cv2.cvtColor(wv,cv2.COLOR_BGR2RGB)); axes[0,1].set_title("Walls detected (red)"); axes[0,1].axis('off')
    
    axes[0,2].imshow(wbarrier,cmap='gray'); axes[0,2].set_title("Wall Barrier"); axes[0,2].axis('off')
    axes[0,3].imshow((gray>165).astype(np.uint8)*255,cmap='gray'); axes[0,3].set_title("Free>165"); axes[0,3].axis('off')

    ov=orig.copy(); cf=np.zeros_like(orig,dtype=np.float32)
    for i,r in enumerate(rooms):
        c=cbgr[i%len(cbgr)]; cf[r.mask>0]=c
        cv2.drawContours(ov,[r.contour],-1,c,2)
        cv2.putText(ov,f"R{i}",(int(r.centroid[0])-8,int(r.centroid[1])),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
    axes[1,0].imshow(cv2.cvtColor(ov,cv2.COLOR_BGR2RGB)); axes[1,0].set_title(f"Rooms({n})"); axes[1,0].axis('off')
    axes[1,1].imshow(np.clip(cf,0,255).astype(np.uint8)); axes[1,1].set_title("Fill"); axes[1,1].axis('off')

    pv=orig.copy()*0+245
    for i,r in enumerate(rooms):
        if r.polygon is not None and len(r.polygon)>=3:
            cv2.polylines(pv,[r.polygon.astype(int)],True,cbgr[i%len(cbgr)],2)
            cv2.putText(pv,f"R{i}",(int(r.centroid[0]),int(r.centroid[1])+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
    axes[1,2].imshow(cv2.cvtColor(pv,cv2.COLOR_BGR2RGB)); axes[1,2].set_title("Polygon"); axes[1,2].axis('off')

    g=build_graph(rooms); info=[f"Rooms:{n}",f"Edges:{sum(len(v)for v in g.values())//2}",""]
    for r in rooms:
        nv=len(r.polygon) if(r.polygon is not None and len(r.polygon)>0) else 0
        info.append(f"R{r.id}:{r.area:.0f}px ({r.centroid[0]:.0f},{r.centroid[1]:.0f}){nv}v")
    axes[1,3].text(.05,.95,"\n".join(info),transform=axes[1,3].transAxes,fontsize=8,va='top',family='monospace',bbox=dict(boxstyle='round',facecolor='wheat',alpha=.5))
    axes[1,3].set_title("Stats"); axes[1,3].axis('off')
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(); plt.savefig(out/"seg_overview.png",dpi=150,bbox_inches='tight'); plt.close()

    for r in rooms:
        x,y,bw,bh=r.bbox; p=12
        x1,y1=max(0,x-p),max(0,y-p); x2,y2=min(orig.shape[1],x+bw+p),min(orig.shape[0],y+bh+p)
        cr=orig[y1:y2,x1:x2].copy(); lm=r.mask[y1:y2,x1:x2]; cr[lm==0]=[140,140,140]
        if r.contour.size>0:
            sc=r.contour.astype(float).copy();sc[:,:,0]-=x1;sc[:,:,1]-=y1
            cv2.drawContours(cr,[sc.astype(int)],-1,(0,0,255),2)
        fig,ax=plt.subplots(1,1,figsize=(7,7))
        ax.imshow(cv2.cvtColor(cr,cv2.COLOR_BGR2RGB));ax.set_title(f"Room#{r.id}|{r.area:.0f}px");ax.axis('off')
        plt.tight_layout();plt.savefig(out/f"room_{r.id:03d}.png",dpi=120,bbox_inches='tight');plt.close()
    print(f"[OK] room_*.png ({n})")

    if n>=2:
        fig,ax=plt.subplots(1,1,figsize=(10,10))
        bg=np.ones_like(orig)*240;ax.imshow(cv2.cvtColor(bg,cv2.COLOR_BGR2RGB))
        for i,r in enumerate(rooms):
            ax.scatter(r.centroid[0],r.centroid[1],s=max(100,min(500,r.area/200)),c=[cols[i%len(cols)]],edgecolors='k',lw=1.5,zorder=5)
            ax.text(r.centroid[0],r.centroid[1],f"R{i}\n{r.area:.0f}",ha='center',va='center',fontsize=7,zorder=6)
        ne=sum(len(v)for v in g.values())//2
        for i,nbs in g.items():
            for j in nbs:
                if i<j: ax.plot([rooms[i].centroid[0],rooms[j].centroid[0]],[rooms[i].centroid[1],rooms[j].centroid[1]],'k-',lw=1.2,alpha=.6,zorder=3)
        ax.set_title(f"Graph({n},{ne}e)");ax.axis('off');ax.invert_yaxis()
        plt.tight_layout();plt.savefig(out/"room_graph.png",dpi=150,bbox_inches='tight');plt.close()
        print("[OK] room_graph.png")
    print("[OK] seg_overview.png")

def save_json(rooms,out_dir):
    out=Path(out_dir); g=build_graph(rooms)
    class E(json.JSONEncoder):
        def default(self,o):
            if isinstance(o,np.integer):return int(o)
            if isinstance(o,np.floating):return float(o)
            if isinstance(o,np.ndarray):return o.tolist()
            return super().default(o)
    data={"rooms":[{"id":int(r.id),"area":float(r.area),"centroid":[float(r.centroid[0]),float(r.centroid[1])],
                   "bbox":[int(v)for v in r.bbox],
                   "poly":r.polygon.astype(float).tolist()if(r.polygon is not None and len(r.polygon)>0)else[]}
                  for r in rooms],"graph":{str(k):v for k,v in g.items()}}
    with open(out/"regions_data.json",'w',encoding='utf-8')as f:json.dump(data,f,indent=2,ensure_ascii=False,cls=E)
    print("[OK] json")

def main():
    IMG=r"d:\01-Code\dog_slam\LIO-SAM_MID360_ROS2_PKG\ros2\src\auto_initial_pose_calibrator\scan_viz\aa34812001c24bb3b047ebce08eca6be.png"
    OUT=r"d:\01-Code\dog_slam\LIO-SAM_MID360_ROS2_PKG\ros2\src\auto_initial_pose_calibrator\scan_viz"
    print("="*60);print("  v7: Wall-Aware + Adaptive Split");print("="*60)
    orig=cv2.imread(IMG)
    if orig is None: raise FileNotFoundError(IMG)
    gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    print(f"  {gray.shape}")
    rooms=segment_rooms(gray,wall_dilate=15,min_area=800)
    print(f"=> {len(rooms)} rooms")
    for r in rooms: print(f"   R{r.id}:{r.area:.0f}@({r.centroid[0]:.0f},{r.centroid[1]:.0f})")
    if not rooms: return
    rooms=polygonize(rooms)
    g=build_graph(rooms)
    print(f"G:{len(rooms)}n,{sum(len(v)for v in g.values())//2}e")

    # barrier for viz
    wc=(gray<50).astype(np.uint8)*255
    k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    wc=cv2.morphologyEx(wc,cv2.MORPH_OPEN,k3,iterations=1)
    kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    bar=cv2.dilate(wc,kd,iterations=2)
    visualize(orig,gray,rooms,bar,OUT)
    save_json(rooms,OUT)
    print("\n[DONE]")

if __name__=="__main__":
    main()
