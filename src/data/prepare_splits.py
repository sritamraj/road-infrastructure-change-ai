import argparse, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from .common import load_yaml, seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); c=load_yaml(a.config); seed_everything(c['seed'])
    imgs=Path(c['data']['image_dir']); masks=Path(c['data']['mask_dir']); items=[]
    for p in sorted(imgs.iterdir()):
        if p.suffix.lower() not in {'.jpg','.jpeg','.png','.tif','.tiff'}: continue
        candidates=list(masks.glob(p.stem+'.*')); m=candidates[0] if candidates else masks/p.name
        if m.exists(): items.append({'image':str(p),'mask':str(m)})
    if not items: raise RuntimeError('No image/mask pairs found.')
    train,tmp=train_test_split(items,test_size=.30,random_state=c['seed']); val,test=train_test_split(tmp,test_size=.50,random_state=c['seed'])
    out=Path(c['data']['split_dir']); out.mkdir(parents=True,exist_ok=True)
    for n,x in [('train',train),('val',val),('test',test)]: (out/f'{n}.json').write_text(json.dumps(x,indent=2))
    print({n:len(x) for n,x in [('train',train),('val',val),('test',test)]})
if __name__=='__main__': main()
