"""
generate_dataset.py
Creates a tiny synthetic image dataset under data/sample_images/{plastic,organic,metal}
Each class gets 50 small images (64x64). These are colored patterns to allow a CNN to learn.
Run: python scripts/generate_dataset.py
"""
from PIL import Image, ImageDraw
import os, random

OUT="data/sample_images"
classes=["plastic","organic","metal"]
os.makedirs(OUT, exist_ok=True)
for cls in classes:
    d=os.path.join(OUT,cls)
    os.makedirs(d,exist_ok=True)

def make_image(cls, path, idx):
    size=(64,64)
    img=Image.new("RGB",(64,64),(255,255,255))
    draw=ImageDraw.Draw(img)
    if cls=="plastic":
        # patterns: colored stripes
        for i in range(0,64,4):
            color=(int(200*(i%3==0)),int(180*(i%3==1)),int(150*(i%3==2)))
            draw.rectangle([i,0,i+3,63], fill=color)
    elif cls=="organic":
        # circular blobs
        for r in range(6,30,6):
            x=random.randint(10,54); y=random.randint(10,54)
            draw.ellipse([x-r,y-r,x+r,y+r], fill=(34,139,34))
    else:
        # metallic: gray gradients + small squares
        for i in range(64):
            g=int(255 * (i/63)**1.5)
            draw.line([(i,0),(i,63)], fill=(g,g,g))
        for i in range(4):
            x=random.randint(5,54); y=random.randint(5,54)
            draw.rectangle([x,y,x+4,y+4], fill=(200,200,200))
    img.save(path,"PNG")

for cls in classes:
    for i in range(50):
        path=os.path.join(OUT,cls,f"{cls}_{i}.png")
        make_image(cls,path,i)

print("Dataset created under data/sample_images with 3 classes x50 images each (64x64).")
