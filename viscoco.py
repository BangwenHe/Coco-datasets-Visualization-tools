import argparse
import os
import random

import cv2
import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from tqdm import tqdm


from pycocotools.coco import COCO
matplotlib.use('TkAgg')


def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def showAnns(anns, cat_names):
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    captions = []
    polygons = []
    rectangles = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        if 'segmentation' in ann and type(ann['segmentation']) == list:
            ann['segmentation'] = [ann['segmentation']]
            # polygon
            for seg in ann['segmentation']:
                captions.append(cat_names[ann['category_id']])
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                l_corner, w, h = (ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3]
                rectangles.append(Rectangle(l_corner, w, h))
                polygons.append(Polygon(poly))
                color.append(c)

    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, alpha=1, linestyle='--', linewidths=2)
    ax.add_collection(p)

    for i in range(len(captions)):
        x = rectangles[i].xy[0]
        y = rectangles[i].xy[1]
        ax.text(x, y, captions[i], size=10, verticalalignment='top', color='w', backgroundcolor="none")

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.6)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors='b', linewidths=0.5)
    ax.add_collection(p)


def parse_args():
    parser = argparse.ArgumentParser(description='COCO dataset visualization')
    parser.add_argument('--annfile', type=str, default="D:/Dataset/MSCoCo/annotations/instances_val2017.json", 
        help='COCO annotation file')
    parser.add_argument('--imgroot', type=str, default="D:/Dataset/MSCoCo/val2017", help='COCO image root')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
    parser.add_argument('--num_imgs', type=int, default=10, help='Number of images to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    annfile = args.annfile
    imgroot = args.imgroot
    output_dir = args.output_dir
    num_imgs = args.num_imgs
    
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(annfile)
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    cat_names = {cat['id']: cat['name'] for cat in cats}
    catids = coco.getCatIds(catNms=random.randint(0,len(cat_names)-1))
    imgids = coco.getImgIds(catIds=catids)
    vis_img_ids = random.choices(imgids, k=num_imgs)

    for imgid in tqdm(vis_img_ids):
        img = coco.loadImgs(imgid)[0]
        I = io.imread(os.path.join(imgroot, img['file_name']))
        I = cv2.resize(I, (I.shape[1] // 4, I.shape[0] // 4))
        
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.title(img['file_name'],fontsize=8,color='blue')
        plt.imshow(I, aspect='equal')
        annids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annids)
        showAnns(anns, cat_names)
        plt.savefig(os.path.join(output_dir, f"{img['file_name']}"))
        plt.close()


if __name__ == "__main__":
    main()
