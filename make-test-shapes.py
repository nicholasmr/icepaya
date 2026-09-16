"""
Make idealized test images for icepaya: randomly placed, non-overlapping bright shapes
(circles, ellipses, triangles, or a mix) on a dark background.

The layout is defined in resolution-independent units, so images made with different
resolutions show the same scene; only the number of pixels per shape changes. This lets you
test the pixel-grid error of q_s. A <image>_truth.pkl with exact shape parameters is saved too.

Usage:  python make-test-shapes.py 1024 [--n 60 --aspect 2 --var 0.3 --aa 4 --noise 0 --seed 0 --out .]

Tip: use 1 tile in the GUI, since shapes cut by tile borders are discarded.
"""

import os, argparse, pickle
import numpy as np
import cv2

BG, FG = 40, 210 # grey levels
SMAX = 8

def outline(kind, r, rot, aspect, m=720):
    """Closed outline (rows, cols) around origin for a shape of equal-area radius r; rot in skimage convention."""
    if kind == 'triangle':
        R = r * np.sqrt(4*np.pi/(3*np.sqrt(3))) # circumradius for area pi r^2
        t = rot + 2*np.pi*np.arange(3)/3        # vertex directions
        return R*np.cos(t), R*np.sin(t)
    a, b = (r*np.sqrt(aspect), r/np.sqrt(aspect)) if kind == 'ellipse' else (r, r)
    t = np.linspace(0, 2*np.pi, m, endpoint=False)
    u, v = a*np.cos(t), b*np.sin(t)             # major axis along (cos rot, sin rot) in (row, col)
    return u*np.cos(rot) - v*np.sin(rot), u*np.sin(rot) + v*np.cos(rot)

def exact_q(rows, cols):
    """Exact q_s of a polygon from its edges (outward normals)."""
    dr, dc = np.roll(rows, -1) - rows, np.roll(cols, -1) - cols
    L, phi = np.hypot(dr, dc), np.arctan2(dc, dr) + np.pi/2 # normal angle (up to a constant, irrelevant for |psi_s|)
    return {s: abs(np.sum(L*np.exp(1j*s*phi)))/L.sum() for s in range(2, SMAX+1)}

def place(kinds, radii, rots, aspect, rng, gap=0.4, M=2000, tries=5000):
    """
    Random centres in the unit square such that shapes are at least gap*r apart
    (tested on an M x M occupancy grid, independent of the output resolution).
    """
    occ = np.zeros((M, M), np.uint8)
    cen = np.zeros((len(kinds), 2))
    for i in np.argsort(-radii): # largest first
        rr, cc = outline(kinds[i], radii[i]*M, rots[i], aspect)
        g = max(2, int(np.ceil(gap*radii[i]*M)))
        pad = g + 2
        for _ in range(tries):
            c = rng.uniform(0, M, 2)
            r0, r1, c0, c1 = int(rr.min()+c[0])-pad, int(rr.max()+c[0])+pad, int(cc.min()+c[1])-pad, int(cc.max()+c[1])+pad
            if r0 < 0 or c0 < 0 or r1 >= M or c1 >= M: continue
            pts = np.round(np.column_stack((cc + c[1] - c0, rr + c[0] - r0))).astype(np.int32)
            patch = np.zeros((r1-r0, c1-c0), np.uint8)
            cv2.fillPoly(patch, [pts], 1)
            if np.any(patch & occ[r0:r1, c0:c1]): continue
            cv2.polylines(patch, [pts], True, 1, thickness=2*g) # reserve gap around shape
            occ[r0:r1, c0:c1] |= patch
            cen[i] = c/M
            break
        else: raise RuntimeError('Could not place all shapes; use fewer (--n) or a smaller area fraction (--frac).')
    return cen

def render(res, kinds, cen, radii, rots, aspect, aa):
    """Anti-aliased rendering: each shape is drawn supersampled (aa x aa) in its own patch."""
    cov = np.zeros((res, res), np.float32)
    for k, c, r, rot in zip(kinds, cen*res, radii*res, rots):
        rr, cc = outline(k, r, rot, aspect)
        rr, cc = rr + c[0], cc + c[1]
        r0, c0 = int(np.floor(rr.min())) - 2, int(np.floor(cc.min())) - 2
        r1, c1 = int(np.ceil(rr.max())) + 3, int(np.ceil(cc.max())) + 3
        patch = np.zeros(((r1-r0)*aa, (c1-c0)*aa), np.uint8)
        pts = np.round(np.column_stack(((cc - c0)*aa, (rr - r0)*aa)) * 16).astype(np.int32) # (x, y), 4-bit sub-pixel
        cv2.fillPoly(patch, [pts], 255, lineType=cv2.LINE_8, shift=4)
        small = cv2.resize(patch, (c1-c0, r1-r0), interpolation=cv2.INTER_AREA) if aa > 1 else patch
        cov[r0:r1, c0:c1] = np.maximum(cov[r0:r1, c0:c1], small/255)
    return cov

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('res', type=int, help='image size [px] (square)')
    p.add_argument('--n', type=int, default=60, help='shapes per image (default 60)')
    p.add_argument('--frac', type=float, default=0.2, help='area fraction covered by shapes (default 0.2)')
    p.add_argument('--aspect', type=float, default=2, help='ellipse aspect ratio (default 2)')
    p.add_argument('--var', type=float, default=0.3, help='relative size variation (default 0.3)')
    p.add_argument('--aa', type=int, default=4, help='anti-aliasing supersampling; 1 = hard pixel edges (default 4)')
    p.add_argument('--noise', type=float, default=0, help='Gaussian noise std in grey levels (default 0)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='.', help='output directory')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    r0 = np.sqrt(a.frac/(a.n*np.pi)) # mean equal-area radius (unit square)
    sets = dict(circles=['circle'], ellipses=['ellipse'], triangles=['triangle'], mixed=['circle', 'ellipse', 'triangle'])
    print('%-10s %6s %9s   %s' % ('image', 'shapes', 'r [px]', 'exact q2..q6 per shape type'))
    for n, (name, types) in enumerate(sets.items()):
        rng = np.random.default_rng(a.seed + n) # same scene at every resolution
        kinds = [types[i % len(types)] for i in range(a.n)]
        radii = r0 * rng.uniform(1 - a.var, 1 + a.var, a.n)
        rots = rng.uniform(-np.pi/2, np.pi/2, a.n)
        cen = place(kinds, radii, rots, a.aspect, rng)

        img = BG + (FG - BG)*render(a.res, kinds, cen, radii, rots, a.aspect, a.aa)
        if a.noise > 0: img += np.random.default_rng(a.seed + 100 + n).normal(0, a.noise, img.shape)
        fimg = os.path.join(a.out, 'test-%s-%i.png' % (name, a.res))
        cv2.imwrite(fimg, np.clip(np.round(img), 0, 255).astype(np.uint8))

        # Ground truth (px units, (row, col) centroids, orientation in skimage/icepaya convention)
        qex = {k: exact_q(*outline(k, 1, 0, a.aspect, m=20000)) for k in set(kinds)}
        rot = np.degrees(rots)
        ori = np.where(np.array(kinds) == 'triangle', rot, (rot + 90) % 180 - 90) # triangles: direction of a vertex
        truth = dict(kind=np.array(kinds), centroid=cen*a.res, r=radii*a.res, area=np.pi*(radii*a.res)**2, ori=ori,
                     qi={s: np.array([qex[k][s] for k in kinds]) for s in range(2, SMAX+1)},
                     meta=dict(res=a.res, aspect=a.aspect, aa=a.aa, noise=a.noise, seed=a.seed + n))
        with open(fimg.replace('.png', '_truth.pkl'), 'wb') as f: pickle.dump(truth, f)
        qs = ', '.join('%s: %s' % (k, ' '.join('%.2f' % qex[k][s] for s in range(2, 7))) for k in types)
        print('%-10s %6i %9.1f   %s' % (name, a.n, r0*a.res, qs))
        if r0*a.res*(1 - a.var) < 5: print('           (warning: smallest shapes have r < 5 px)')

if __name__ == '__main__':
    main()
