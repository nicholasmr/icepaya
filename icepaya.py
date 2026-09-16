"""
icepaya: grain and bubble morphometry of ice thin-section images,
including irreducible Minkowski tensors (IMTs) computed with papaya2.

Nicholas M. Rathmann, Niels Bohr Institute, 2020-

Developed as part of
- the Villum Foundation investigator project IceFlow
- the Novo Nordisk Foundation challenge project PRECISE

Scripted use:
    ip = icepaya('section.png')               # defaults, or settings restored from newest section_<mode>.pkl
    ip.set_params('grains', clahe=9, filt=150) # sets mode and its parameters
    ip.skip = {(0, 0)}                         # optional: tiles to skip
    d = ip.process()                           # data dict for current mode
    ip.save(d)                                 # -> section_grains.pkl, settings under 'meta'
    d = icepaya.load_data('section_grains.pkl')
"""

import os, copy, pickle, inspect
from datetime import datetime

import numpy as np
import cv2
from scipy import ndimage
import skimage
from skimage import morphology, measure, color
from skimage.segmentation import clear_border
from matplotlib.figure import Figure  # no pyplot: safe inside Qt and headless
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import pypaya2

MODES = ('grains', 'bubbles')

SMAX   = 8                        # highest IMT order stored
NPX    = 50                       # floor [px^2]: smaller objects are always removed (speckle)

PAR0 = dict(
    grains  = dict(clahe=9, filt=150, amin=NPX), # filt: remove dark structures < filt px
    bubbles = dict(clahe=0, filt=2,   amin=NPX), # filt: no. of erosion+dilation cycles
)                                                # amin: area cut-off [px^2] (>= NPX), objects below are excluded
SIGMA  = 1.5                      # Gaussian smoothing [px] of object masks before IMTs; 0 = off
                                  # (binary masks only have 45-deg boundary segments: q6=q2, q5=q3, q8=1)
NT0    = 2                        # default no. of tiles along the long image edge
BMIN   = 300                      # small structure always removed for bubbles (reflections)
KERNEL = np.ones((5, 5), np.uint8)

LBL = dict(equivdiam='Equiv. diameter [px]', majorax='Major axis [px]', minorax='Minor axis [px]',
           ecc='Eccentricity', area0='Area (regionprops) [px$^2$]', perim0='Perimeter (regionprops) [px]',
           ori0='Orientation (regionprops) [deg]', area='Area [px$^2$]', perim='Perimeter [px]',
           ori='Orientation (IMT) [deg]', **{'q%i' % s: '$q_{%i}$' % s for s in range(2, SMAX+1)})
SCLR = {2: '#e31a1c', 3: '#ff7f00', 4: '#1f78b4', 5: '#6a3d9a', 6: '#33a02c', 7: '#b15928', 8: '#e7298a'} # per IMT order s
CLR  = {'q%i' % s: c for s, c in SCLR.items()} # histogram colours (others black)
SUMKEYS = ('perim', 'majorax', 'ecc', 'area', 'ori', 'q2', 'q3', 'q4', 'q5', 'q6') # plotted histograms
LOGX = ('area', 'perim', 'area0', 'perim0')                         # log x-axis
STATIC = ('area', 'perim', 'majorax')                               # GUI: x-limits fixed to those at amin = NPX
UNIT = ('ecc', *CLR)                                                # x-axis fixed to [0,1]
AGG  = ('qi_agg', 'pi_agg')                                         # aggregate (not per-object) fields

_MAXSIZE = 'max_size' in inspect.signature(morphology.remove_small_objects).parameters

def rm_small(mask, n):
    """Remove objects with < n px (works for old and new skimage API)."""
    if n <= 1: return mask
    if _MAXSIZE: return morphology.remove_small_objects(mask, max_size=n-1)
    return morphology.remove_small_objects(mask, n)

def _segs(ax, x0, y0, x1, y1, **kw):
    """Draw many segments (x0,y0)->(x1,y1) in a single plot call."""
    x0, y0, x1, y1 = [np.ravel(v) for v in np.broadcast_arrays(x0, y0, x1, y1)]
    nan = np.full_like(x0, np.nan)
    ax.plot(np.column_stack((x0, x1, nan)).ravel(), np.column_stack((y0, y1, nan)).ravel(), '-', **kw)


class icepaya:

    def __init__(self, fimg=None, ntiles=None, mode='grains'):
        self.par = copy.deepcopy(PAR0)
        self.mode = mode
        self.ntiles = ntiles or NT0
        self.img, self.frestored = None, None
        self.ij, self.skip = None, set()
        self.t = {} # current tile, {mode: dict(cntr, segm, lbl, data)}
        if fimg is not None: self.load(fimg, ntiles)

    ### I/O

    def load(self, fimg, ntiles=None):
        """
        Load greyscale image with default settings (mode is kept), then restore settings
        from the newest <image>_<mode>.pkl if present (path in self.frestored).
        An explicit ntiles overrides the default and restored tiling.
        """
        img = cv2.imread(fimg, cv2.IMREAD_GRAYSCALE)
        if img is None: raise IOError('Could not read %s' % fimg)
        self.img, self.fimg = img, fimg
        self.fbase = os.path.splitext(fimg)[0]
        self.par = copy.deepcopy(PAR0)
        self.set_tiles(NT0)
        fpkl = [f for f in map(self.fout, MODES) if os.path.isfile(f)]
        self.frestored = max(fpkl, key=os.path.getmtime) if fpkl else None
        if self.frestored: self.load_params(self.frestored)
        if ntiles is not None and ntiles != self.ntiles: self.set_tiles(ntiles)

    def fout(self, name, ext='.pkl'):
        return '%s_%s%s' % (self.fbase, name, ext)

    def settings(self):
        """Current settings (saved under 'meta' in pickles)."""
        return dict(fimg=os.path.basename(self.fimg), mode=self.mode, par=copy.deepcopy(self.par),
                    ntiles=self.ntiles, skip=sorted(self.skip), sigma=SIGMA, time=datetime.now().isoformat(timespec='seconds'),
                    versions=dict(skimage=skimage.__version__, cv2=cv2.__version__, numpy=np.__version__))

    def save(self, d, fname=None):
        """Pickle data dict with current settings under 'meta'; default <image>_<mode>.pkl."""
        fname = fname or self.fout(self.mode)
        with open(fname, 'wb') as f: pickle.dump(dict(d, meta=self.settings()), f)
        return fname

    @staticmethod
    def load_data(fname):
        with open(fname, 'rb') as f: return pickle.load(f)

    ### Parameters

    def set_params(self, mode=None, **kw):
        """Set current mode (optional) and its segmentation parameters."""
        if mode is not None:
            if mode not in MODES: raise ValueError('mode must be one of %s' % (MODES,))
            self.mode = mode
        bad = set(kw) - set(PAR0[self.mode])
        if bad: raise KeyError('Unknown %s parameter(s): %s' % (self.mode, ', '.join(bad)))
        self.par[self.mode].update(kw)

    def load_params(self, fname):
        """Restore parameters, tiling and skipped tiles from a results pickle (mode is kept)."""
        meta = self.load_data(fname)['meta']
        for m, p in meta['par'].items():
            if m in MODES: self.par[m].update({k: v for k, v in p.items() if k in PAR0[m]})
        self.ntiles = int(meta.get('ntiles', self.ntiles))
        if self.img is not None:
            self.set_tiles(self.ntiles)
            self.skip = {tuple(ij) for ij in meta['skip']} & set(self.tiles(skip=False))

    ### Tiles

    def set_tiles(self, n):
        """Split image into n tiles along its long edge, and as many along the short edge as keeps tiles square-ish."""
        self.ntiles = max(1, int(n))
        L, S = max(self.img.shape), min(self.img.shape)
        x = S*self.ntiles/L # ideal no. of tiles on short edge
        ns = min({max(1, int(np.floor(x))), max(1, int(np.ceil(x)))}, key=lambda k: abs(np.log(x/k)))
        n = (self.ntiles, ns) if self.img.shape[0] >= self.img.shape[1] else (ns, self.ntiles)
        self.redg = np.linspace(0, self.img.shape[0], n[0]+1).astype(int) # tile row edges
        self.cedg = np.linspace(0, self.img.shape[1], n[1]+1).astype(int) # tile col edges
        self.nt = tuple(n)
        self.skip, self.ij, self.t = set(), None, {}

    def tiles(self, skip=True):
        return [(i, j) for i in range(self.nt[0]) for j in range(self.nt[1])
                if not (skip and (i, j) in self.skip)]

    def offset(self, ij):
        return self.redg[ij[0]], self.cedg[ij[1]]

    def tile(self, ij):
        (i, j) = ij
        return np.ascontiguousarray(self.img[self.redg[i]:self.redg[i+1], self.cedg[j]:self.cedg[j+1]])

    def toggle_skip(self, ij):
        """Toggle skipping of tile ij; returns True if now skipped."""
        self.skip ^= {tuple(ij)}
        return tuple(ij) in self.skip

    def set_tile(self, ij):
        """Select tile ij (or None) for segment()/identify()."""
        self.ij, self.t = (None if ij is None else tuple(ij)), {}

    ### Processing

    def segment(self, mode=None):
        """Contrast-enhance and segment the current tile (objects = True)."""
        mode = mode or self.mode
        p = self.par[mode]
        img = self.tile(self.ij)
        if p['clahe'] > 0: img = cv2.createCLAHE(clipLimit=p['clahe'], tileGridSize=(10, 10)).apply(img)
        cntr = img
        if mode == 'bubbles': # invert and open to erode away grain boundaries
            img = 255 - img
            if p['filt'] > 0:
                img = cv2.dilate(cv2.erode(img, KERNEL, iterations=p['filt']), KERNEL, iterations=p['filt'])
        _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu
        bg = rm_small(thr == 0, p['filt'] if mode == 'grains' else BMIN)
        self.t[mode] = dict(cntr=cntr, segm=~bg)
        return self.t[mode]

    def identify(self, mode=None):
        """Label and measure objects in the current tile; returns data after area cut-off."""
        mode = mode or self.mode
        if mode not in self.t: self.segment(mode)
        t = self.t[mode]
        mask = rm_small(clear_border(t['segm']), NPX)
        t['lbl'], _ = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
        t['all'] = self.measure(t['lbl'], offset=self.offset(self.ij), tile=self.ij)
        return self.cutoff(mode)

    def cutoff(self, mode=None):
        """(Re)apply area cut-off amin to the measured objects of the current tile."""
        mode = mode or self.mode
        t = self.t[mode]
        amin = max(NPX, self.par[mode]['amin'])
        t['data'] = self.aggregate(self.subset(t['all'], t['all']['area'] >= amin))
        return t['data']

    def floor(self, mode=None):
        """Objects of the current tile at the area floor (amin = NPX), or None if not identified."""
        t = self.t.get(mode or self.mode, {})
        return self.subset(t['all'], t['all']['area'] >= NPX) if 'all' in t else None

    def amax(self):
        """Largest area cut-off [px^2]: half the (mean) tile area, or None without image."""
        if self.img is None: return None
        return 0.5 * np.diff(self.redg).mean() * np.diff(self.cedg).mean()

    @staticmethod
    def xlims(d, keys=STATIC):
        """{key: (min, max)} of fields in d (positive values for log axes)."""
        out = {}
        for k in keys:
            v = d[k][d[k] > 0] if k in LOGX else d[k]
            if len(v) and v.min() < v.max(): out[k] = (v.min(), v.max())
        return out

    @staticmethod
    def subset(d, keep):
        """Per-object fields restricted to objects where keep is True."""
        return {k: ({s: x[keep] for s, x in v.items()} if isinstance(v, dict) else v[keep])
                for k, v in d.items() if k not in ('meta', *AGG)}

    @staticmethod
    def aggregate(d):
        """
        Add network (aggregate) IMTs: pi_agg[s] = sum of psi_s over objects / total perimeter,
        qi_agg[s] = |pi_agg[s]|, indexed by s (0 for s < 2). Randomly oriented objects cancel,
        so qi_agg measures preferred alignment of the microstructure.
        """
        pa = np.zeros(SMAX+1, complex)
        P = np.sum(d['perim'])
        if P > 0:
            for s, v in d['pi'].items(): pa[s] = np.sum(v)/P
        d.update(qi_agg=np.abs(pa), pi_agg=pa)
        return d

    @staticmethod
    def measure(lbl, offset=(0, 0), tile=(0, 0), sigma=SIGMA):
        """Shape descriptors of labelled objects: regionprops (*0 fields) and papaya2 IMTs."""
        props = ('label', 'centroid', 'area', 'perimeter', 'orientation',
                 'major_axis_length', 'minor_axis_length', 'eccentricity')
        rp = measure.regionprops_table(lbl, properties=props)
        n = len(rp['label'])
        d = dict(
            label=rp['label'], tile=np.tile(tile, (n, 1)),
            centroid=np.column_stack((rp['centroid-0'] + offset[0], rp['centroid-1'] + offset[1])), # (row, col) in full image
            majorax=rp['major_axis_length'], minorax=rp['minor_axis_length'], ecc=rp['eccentricity'],
            equivdiam=np.sqrt(4*rp['area']/np.pi),
            perim0=rp['perimeter'], area0=rp['area'], ori0=np.degrees(rp['orientation']),
        )
        S = range(2, SMAX+1)
        area, perim = np.zeros(n), np.zeros(n)
        pi, qi = np.zeros((n, SMAX+1), complex), np.zeros((n, SMAX+1))
        pad = int(np.ceil(4*sigma)) + 1
        for k, sl in enumerate(ndimage.find_objects(lbl)): # labels are 1..n
            bw = np.pad((lbl[sl] == k+1).astype(float), pad)
            if sigma > 0: bw = ndimage.gaussian_filter(bw, sigma) # sub-pixel boundary, see SIGMA
            m = pypaya2.imt_for_image(bw, threshold=0.5)
            area[k], perim[k] = m['area'][0], m['perimeter'][0]
            for s in S: pi[k, s], qi[k, s] = m['psi%i' % s][0], m['q%i' % s][0]
        ori = (np.angle(pi[:, 2])/2 + np.pi) % np.pi - np.pi/2 # elongation axis, skimage convention
        d.update(perim=perim, area=area, ori=np.degrees(ori),
                 qi={s: qi[:, s] for s in S}, pi={s: pi[:, s] for s in S})
        return d

    @staticmethod
    def combine(ds):
        """Concatenate per-tile data dicts (aggregate fields are recomputed)."""
        if not ds: return {}
        cat = lambda k, s=None: np.concatenate([d[k] if s is None else d[k][s] for d in ds])
        return icepaya.aggregate({k: ({s: cat(k, s) for s in v} if isinstance(v, dict) else cat(k))
                                  for k, v in ds[0].items() if k not in AGG})

    def process(self, cb=None, figs=False):
        """
        Process all non-skipped tiles in current mode; returns combined data dict.
        cb(n, N, ij) is called after each tile; figs=True saves tile figures in <image>_tiles/.
        """
        tiles = self.tiles()
        if not tiles: raise ValueError('No tiles to process (all skipped?)')
        ds = []
        for n, ij in enumerate(tiles):
            self.set_tile(ij)
            self.segment()
            ds.append(self.identify())
            if figs: self.save_tile_fig()
            if cb is not None: cb(n+1, len(tiles), ij)
        return self.combine(ds)

    ### Plotting

    def plot_tile(self, ax, key='cntr', mode=None, **kw):
        """Show current tile; key is raw|cntr|segm|lbl. Axes use full-image px coords."""
        t = self.t[mode or self.mode]
        im = self.tile(self.ij) if key == 'raw' else t[key]
        if key == 'lbl': # only objects passing the area cut-off
            im = color.label2rgb(np.where(np.isin(im, t['data']['label']), im, 0), bg_label=0)
        r0, c0 = self.offset(self.ij)
        nr, nc = im.shape[:2]
        ax.imshow(im, cmap='gray', interpolation='nearest', extent=(c0-.5, c0+nc-.5, r0+nr-.5, r0-.5), **kw)
        ax.set_autoscale_on(False) # overlays should not change limits
        ax.set_axis_off()

    @staticmethod
    def plot_ellipses(ax, d, cma='#e31a1c', cmi='#fb9a99', cmid='#1f78b4', lw=2):
        """Major/minor axes of the moment-equivalent ellipses."""
        y0, x0 = d['centroid'].T
        o = np.radians(d['ori0'])
        a, b = d['majorax']/2, d['minorax']/2
        _segs(ax, x0, y0, x0 + np.cos(o)*b, y0 - np.sin(o)*b, color=cmi, lw=lw)
        _segs(ax, x0, y0, x0 - np.sin(o)*a, y0 - np.cos(o)*a, color=cma, lw=lw)
        ax.plot(x0, y0, '.', color=cmid, ms=8)

    @staticmethod
    def plot_symbols(ax, d, svals=(2, 3, 4), scale=3, arms='corners', colors=None, lw=1.5):
        """
        s-fold IMT symbols at centroids; arm length = scale * q_s * equivalent radius.
        arms: 'corners' (s=2 along elongation) or 'normals' (dominant boundary normals).
        colors: dict {s: color} or a single color.
        """
        colors = colors or SCLR
        y0, x0 = d['centroid'][:, :1], d['centroid'][:, 1:]
        R = d['equivdiam'][:, None]/2
        for s in svals:
            L = scale * d['qi'][s][:, None] * R
            p = np.angle(d['pi'][s][:, None])/s + (np.pi/s if arms == 'corners' else 0) + 2*np.pi*np.arange(s)/s
            c = colors.get(s, 'k') if isinstance(colors, dict) else colors
            _segs(ax, x0, y0, x0 + L*np.sin(p), y0 + L*np.cos(p), color=c, lw=lw) # angle from row axis

    @staticmethod
    def plot_hist(ax, d, key, bins=40, color=None, xlim=None, **kw):
        """
        Histogram of a data field; key 'q<s>' selects q_s. Log x-axis for LOGX keys, [0,1] for UNIT keys.
        xlim=(lo, hi) fixes bins and x-limits (data outside is not shown).
        """
        v = np.asarray(d['qi'][int(key[1:])] if key[0] == 'q' else d[key])
        if key in LOGX: v = v[v > 0]
        if key in UNIT: xlim = (0, 1)
        elif xlim is None and len(v) and v.min() < v.max(): xlim = (v.min(), v.max())
        if xlim is not None:
            lo, hi = xlim
            if key in LOGX:
                ax.set_xscale('log')
                bins = np.logspace(np.log10(lo), np.log10(hi), bins+1)
                subs = (1,) if hi/lo > 100 else (1, 3) if hi/lo > 10 else (1, 2, 5) # plain-number ticks
                ax.xaxis.set_major_locator(LogLocator(subs=subs))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '%g' % x))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlim(lo/1.08, hi*1.08)
            else:
                bins = np.linspace(lo, hi, bins+1)
                pad = 0 if key in UNIT else 0.03*(hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
        elif key in LOGX: ax.set_xscale('log')
        if len(v): ax.hist(v, bins=bins, color=color or CLR.get(key, 'k'), **kw)
        ax.set_xlabel(LBL.get(key, key))
        ax.set_ylabel('#')

    @staticmethod
    def plot_agg(ax, d, svals=range(2, 7), colors=None):
        """Bar plot of aggregate q_s (preferred alignment of the microstructure)."""
        colors, svals = colors or SCLR, list(svals)
        q = d['qi_agg'][svals]
        ax.bar(svals, q, width=0.7, color=[colors.get(s, 'k') for s in svals])
        ax.set_xticks(svals)
        ax.set_xticklabels(['$q_{%i}$' % s for s in svals])
        ax.set_ylim(0, max(0.01, 1.2*q.max()))
        ax.set_ylabel('Aggregate $q_s$')

    @staticmethod
    def plot_summary(axs, d, keys=SUMKEYS, **kw):
        """Histograms of keys on axs, aggregate q_s bars in the last axis; unused axes are hidden."""
        axs = np.ravel(axs)
        for ax, k in zip(axs, keys): icepaya.plot_hist(ax, d, k, **kw)
        for ax in axs[len(keys):-1]: ax.set_axis_off()
        icepaya.plot_agg(axs[-1], d)

    def save_tile_fig(self, fname=None, dpi=150):
        """Current tile: enhanced image + IMT symbols | labels + ellipses."""
        t = self.t[self.mode]
        if fname is None:
            os.makedirs(self.fbase + '_tiles', exist_ok=True)
            fname = '%s_tiles/%i_%i_%s.png' % (self.fbase, *self.ij, self.mode)
        nr, nc = t['segm'].shape
        fig = Figure(figsize=(10, 5*nr/nc), dpi=dpi)
        ax1, ax2 = fig.subplots(1, 2)
        fig.subplots_adjust(0, 0, 1, 1, 0.01, 0)
        self.plot_tile(ax1, 'cntr'); self.plot_symbols(ax1, t['data'])
        self.plot_tile(ax2, 'lbl');  self.plot_ellipses(ax2, t['data'], cmid='w', cma='0.4', cmi='0.8', lw=1)
        fig.savefig(fname)
        return fname

    def save_summary_fig(self, d, fname=None):
        """Histograms of data dict d; default <image>_<mode>_stats.png."""
        fname = fname or self.fout(self.mode + '_stats', '.png')
        fig = Figure(figsize=(14, 9))
        axs = fig.subplots(3, 4)
        self.plot_summary(axs, d, bins=40)
        amin = self.par[self.mode]['amin']
        if amin > 0 and 'area' in SUMKEYS: axs.flat[SUMKEYS.index('area')].axvline(amin, color='r', ls='--', lw=1.5)
        fig.suptitle('%s (%i %s)' % (os.path.basename(self.fimg), len(d['label']), self.mode))
        fig.tight_layout()
        fig.savefig(fname)
        return fname


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]: # batch: python icepaya.py img1.png img2.png ...
        ip = icepaya(f)
        for m in MODES:
            ip.set_params(m)
            d = ip.process(cb=lambda n, N, ij: print('%s %s: tile %i/%i %s' % (f, m, n, N, ij)))
            ip.save(d); ip.save_summary_fig(d)
