#!/usr/bin/env python3
"""Build a cube mesh from an explicit per-face config (color, letter, rotation),
then render it from 4 corner views to compare against real photos.

Edit CONFIG below to match the physical cube, run, inspect /tmp/cube_cmp.png,
adjust rotations, repeat. When it matches, the textured.obj is the mesh to use.
"""
import os, sys
import numpy as np, torch, trimesh, cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Utils import make_mesh_tensors, nvdiffrast_render

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
COLORS = {"red": (220,20,20), "green": (20,180,60), "blue": (30,70,230),
          "yellow": (245,220,20), "magenta": (225,30,200), "cyan": (20,200,220)}

# slot -> (color_name, letter, rotation_deg_CCW).  Edit to match the real cube.
CONFIG = {
    "+X": ("red",     "A", 0),
    "-X": ("green",   "B", 0),
    "+Y": ("blue",    "C", 0),
    "-Y": ("yellow",  "D", 0),
    "+Z": ("cyan",    "F", 180),
    "-Z": ("magenta", "E", 180),
}
SLOTS = ["+X","-X","+Y","-Y","+Z","-Z"]
NORMALS = {"+X":(1,0,0),"-X":(-1,0,0),"+Y":(0,1,0),"-Y":(0,-1,0),"+Z":(0,0,1),"-Z":(0,0,-1)}
UPS     = {"+X":(0,0,1),"-X":(0,0,1),"+Y":(0,0,1),"-Y":(0,0,1),"+Z":(0,1,0),"-Z":(0,1,0)}
COLS, ROWS = 3, 2


def _font(sz):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", sz)
    except Exception: return ImageFont.load_default()

def _text_color(bg):
    return (0,0,0) if 0.299*bg[0]+0.587*bg[1]+0.114*bg[2] > 140 else (255,255,255)

def draw_tile(size, color, letter, rot):
    img = Image.new("RGB", (size,size), color)
    d = ImageDraw.Draw(img); f=_font(int(size*0.7))
    bb=d.textbbox((0,0),letter,font=f); w,h=bb[2]-bb[0],bb[3]-bb[1]
    d.text(((size-w)/2-bb[0],(size-h)/2-bb[1]),letter,fill=_text_color(color),font=f)
    if rot: img = img.rotate(rot, expand=False)
    return img

def build_atlas(tile=512):
    img = Image.new("RGB",(COLS*tile,ROWS*tile),(0,0,0))
    for i,slot in enumerate(SLOTS):
        cname,letter,rot = CONFIG[slot]
        c,r = i%COLS, i//COLS
        img.paste(draw_tile(tile, COLORS[cname], letter, rot),(c*tile,r*tile))
    return Image.fromarray(np.flipud(np.array(img)))

def build_cube(edge=0.06):
    h=edge/2; verts,faces,uv=[],[],[]; inset=0.04
    for i,slot in enumerate(SLOTS):
        n=np.array(NORMALS[slot],float); v=np.array(UPS[slot],float); u=np.cross(v,n)
        c,r=i%COLS,i//COLS
        u0,u1=(c+inset)/COLS,(c+1-inset)/COLS; v0,v1=(r+inset)/ROWS,(r+1-inset)/ROWS
        corner=lambda s,t:(n+u*s+v*t)*h
        b=len(verts); verts+=[corner(-1,-1),corner(1,-1),corner(1,1),corner(-1,1)]
        uv+=[(u1,v0),(u0,v0),(u0,v1),(u1,v1)]; faces+=[[b,b+1,b+2],[b,b+2,b+3]]
    return np.array(verts),np.array(faces),np.array(uv)


def render(mt, dir_obj, dist=0.22, fx=1100, S=420):
    d=np.asarray(dir_obj,float); d/=np.linalg.norm(d); p=d*dist
    z=-d; up=np.array([0,0,1.0]);
    if abs(np.dot(up,z))>0.95: up=np.array([0,1.0,0])
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],0); T=np.eye(4); T[:3,:3]=R; T[:3,3]=-R@p
    K=np.array([[fx,0,S/2],[0,fx,S/2],[0,0,1]],float)
    Tt=torch.as_tensor(T[None],device="cuda",dtype=torch.float)
    color,depth,_=nvdiffrast_render(K=K,H=S,W=S,ob_in_cams=Tt,mesh_tensors=mt,use_light=True,w_ambient=0.65,w_diffuse=0.6)
    rgb=(color[0].clip(0,1).cpu().numpy()*255).astype(np.uint8); m=depth[0].cpu().numpy()>0
    out=np.full((S,S,3),245,np.uint8); out[m]=rgb[m]; return out


def main():
    atlas=build_atlas(); verts,faces,uv=build_cube()
    mat=trimesh.visual.material.SimpleMaterial(image=atlas)
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv,material=mat,image=atlas),process=False)
    out=os.path.join(CODE_DIR,"mesh","textured.obj"); mesh.export(out)
    mt=make_mesh_tensors(trimesh.load(out))
    # 4 corner views matching the 4 photos
    views=[((1,1,0.55),"photo1: E top, C+A front"),
           ((-1,1,0.55),"photo2: E top, A+D front"),
           ((-1,-1,0.55),"photo3: E top, D+B front"),
           ((-1,1,-0.55),"photo4: F top, D+A front")]
    tiles=[]
    for dirv,lbl in views:
        im=render(mt,dirv);
        cv2.rectangle(im,(0,0),(im.shape[1],26),(30,30,30),-1)
        cv2.putText(im,lbl,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        tiles.append(im)
    top=np.hstack([tiles[0],tiles[1]]); bot=np.hstack([tiles[2],tiles[3]])
    cv2.imwrite("/tmp/cube_cmp.png", np.vstack([top,bot])[...,::-1])
    print("wrote mesh", out, "and /tmp/cube_cmp.png")


if __name__=="__main__":
    main()
