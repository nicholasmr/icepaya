"""
GUI for icepaya: tile selection, segmentation calibration and batch processing.
Usage: python GUI.py [image.png]

Nicholas M. Rathmann, Niels Bohr Institute, 2020-
"""

import sys, os, traceback
import numpy as np
import cv2
from skimage.draw import ellipse, polygon

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QKeySequence, QPalette
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QSpinBox, QPushButton, QSlider, QRadioButton,
                             QButtonGroup, QCheckBox, QGridLayout, QHBoxLayout, QVBoxLayout, QFileDialog,
                             QSizePolicy, QShortcut, QScrollArea, QFrame)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from icepaya import icepaya, MODES, SUMKEYS, SCLR, NPX

SLIDERS = dict( # {mode: {param: (label, min, max, step)}}; amin range is set from identified objects
    grains  = dict(clahe=('CLAHE clip limit: %i', 0, 21, 1),
                   filt=('Filter small-scale structure < %i px', 0, 400, 20),
                   amin=('Area cut-off [px²]: %i', 0, 0, 1)),
    bubbles = dict(clahe=('CLAHE clip limit: %i', 0, 21, 1),
                   filt=('Erosion/dilation cycles: %i', 0, 10, 1),
                   amin=('Area cut-off [px²]: %i', 0, 0, 1)),
)
SLW    = 260      # slider width [px]
NCOL   = 4        # histogram columns
SVALS  = (2, 3, 4) # IMT symbols shown
SLN    = 400      # area cut-off slider positions (log scale)
OVSIZE = 440      # max size of image overview [px]

GUIDE = """<b>icepaya</b> segments greyscale thin-section images of ice into grains or bubbles,
and measures their size and shape, including Minkowski tensors computed with papaya2.
<ol style="margin-left:-20px">
<li><b>Browse</b> to open an image, and set the number of <b>tiles</b> along its long edge.</li>
<li>Choose <b>grain</b> or <b>bubble</b> mode.</li>
<li><b>Click</b> a tile above and adjust the sliders until the segmentation looks right;
<b>Identify (F5)</b> to check. The area cut-off (log scale, from %i px<sup>2</sup> to half a tile)
excludes small objects; these are shown in light grey in the area histogram.</li>
<li><b>Shift+click</b> tiles to skip them.</li>
<li><b>Process tiles</b>: saves results and settings to &lt;image&gt;_&lt;mode&gt;.pkl.
Settings are restored from it when the image is opened again.</li>
</ol>""" % NPX

IMTHELP = """<b>Minkowski tensors.</b> The boundary of each grain consists of segments
of length L<sub>k</sub> and outward normal angle &phi;<sub>k</sub>, giving
&psi;<sub>s</sub> = &Sigma;<sub>k</sub> L<sub>k</sub> exp(i&nbsp;s&nbsp;&phi;<sub>k</sub>).
The magnitude q<sub>s</sub> = |&psi;<sub>s</sub>| / perimeter (0 to 1) measures how
s-fold symmetric the grain is, and arg(&psi;<sub>s</sub>)/s its orientation.
An ellipse-like grain is best described by q<sub>2</sub>, a triangle-like grain by q<sub>3</sub>,
a square-like grain by q<sub>4</sub>, and so on:"""

MARKERS = """Markers: arms along the %s directions, with length &prop; q<sub>s</sub>.""" % \
    ', '.join('<span style="color:%s">s=%i</span>' % (SCLR[s], s) for s in SVALS)

AGGHELP = """<b>Aggregate metrics.</b> The lower-right bar plot shows aggregate q<sub>s</sub>:
&psi;<sub>s</sub> summed over all objects and divided by their total perimeter. Randomly oriented
grains cancel out, so it measures preferred alignment of the microstructure rather than the shape
of individual grains."""

CREDITS = '<span style="color:gray">NM Rathmann, NBI, UCPH</span>'


def fig2pix(fig):
    """Render matplotlib figure to QPixmap (keeps transparency)."""
    cv = FigureCanvasAgg(fig)
    cv.draw()
    a = np.asarray(cv.buffer_rgba())
    h, w = a.shape[:2]
    return QPixmap.fromImage(QImage(a.data, w, h, 4*w, QImage.Format_RGBA8888).copy())


def imt_demo(fg='k', width=410):
    """Pixmap of ellipse (q2) and regular polygon (q3 to q6) grains of equal area, with IMT markers."""
    n = 140
    grains = np.zeros((5, n, n), int)
    grains[0][ellipse(70, 70, 50, 25)] = 1
    for i, (s, r) in enumerate(((3, 55), (4, 45.3), (5, 41), (6, 39.5)), start=1):
        a = np.pi/2 + 2*np.pi*np.arange(s)/s + (np.pi/4 if s == 4 else 0) # vertex up (square: axis aligned)
        grains[i][polygon(70 - r*np.sin(a), 70 + r*np.cos(a))] = 1
    fig = Figure(figsize=(width/100, 0.25*width/100), dpi=100, facecolor='none')
    axs = fig.subplots(1, 5)
    fig.subplots_adjust(0, 0.02, 1, 0.78, 0.12)
    for ax, lbl, s in zip(axs, grains, range(2, 7)):
        d = icepaya.measure(lbl)
        ax.imshow(np.ma.masked_equal(lbl, 0), cmap=ListedColormap(['#c8c8c8']), interpolation='nearest')
        icepaya.plot_symbols(ax, d, svals=(s,), scale=1.3, lw=1.8)
        ax.set_xlim(10, 130); ax.set_ylim(128, 12) # same scale for all grains
        ax.set_axis_off()
        ax.set_title('$q_%i$ = %.2f' % (s, d['qi'][s][0]), color=fg, fontsize=9)
    return fig2pix(fig)


class Canvas(FigureCanvas):
    """Matplotlib canvas using the window background and text colours."""

    def __init__(self, hist=False):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        if hist: self.fig.subplots_adjust(left=0.2, bottom=0.34, top=0.95, right=0.95)
        else:    self.fig.subplots_adjust(0, 0, 1, 1)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pal = QApplication.palette() # (matplotlib sets a white palette on the canvas)
        self.bg, self.fg = pal.color(QPalette.Window).name(), pal.color(QPalette.WindowText).name()
        self.fig.patch.set_facecolor(self.bg)
        self.hist = hist
        self.clear()

    def clear(self, draw=True):
        self.ax.clear()
        self.ax.set_facecolor('none')
        if self.hist: self.style()
        else:         self.ax.set_axis_off()
        if draw: self.draw()

    def style(self):
        for sp in self.ax.spines.values(): sp.set_color(self.fg)
        self.ax.tick_params(colors=self.fg, labelsize=9)
        for lbl in (self.ax.xaxis.label, self.ax.yaxis.label):
            lbl.set_color(self.fg)
            lbl.set_fontsize(11)


class Overview(QLabel):
    """Down-scaled full image with tile grid; click selects a tile, shift+click toggles skip."""

    clicked = pyqtSignal(tuple)

    def __init__(self, ip):
        super().__init__('No image loaded')
        self.ip, self.base = ip, None
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(OVSIZE, int(0.75*OVSIZE))
        self.setStyleSheet('QLabel {background: #ddd;}')

    def set_image(self):
        img = self.ip.img
        self.f = min(OVSIZE/img.shape[1], OVSIZE/img.shape[0])
        wh = (max(1, round(img.shape[1]*self.f)), max(1, round(img.shape[0]*self.f)))
        self.base = cv2.resize(img, wh, interpolation=cv2.INTER_AREA)
        self.setFixedSize(*wh)
        self.setStyleSheet('')
        self.redraw()

    def edges(self):
        return [np.round(e*self.f).astype(int) for e in (self.ip.redg, self.ip.cedg)]

    def redraw(self):
        if self.base is None: return
        rgb = cv2.cvtColor(self.base, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        re, ce = self.edges()
        for (i, j) in self.ip.skip: # dim and cross out
            r0, r1, c0, c1 = re[i], re[i+1]-1, ce[j], ce[j+1]-1
            rgb[r0:r1, c0:c1] //= 3
            cv2.line(rgb, (c0, r0), (c1, r1), (230, 230, 230), 1, cv2.LINE_AA)
            cv2.line(rgb, (c1, r0), (c0, r1), (230, 230, 230), 1, cv2.LINE_AA)
        for r in re[1:-1]: cv2.line(rgb, (0, r), (w-1, r), (255, 150, 150), 1) # light red grid
        for c in ce[1:-1]: cv2.line(rgb, (c, 0), (c, h-1), (255, 150, 150), 1)
        if self.ip.ij is not None: # dark red frame
            i, j = self.ip.ij
            cv2.rectangle(rgb, (ce[j]+2, re[i]+2), (ce[j+1]-3, re[i+1]-3), (170, 0, 0), 4)
        self.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()))

    def mousePressEvent(self, ev):
        if self.base is None: return
        re, ce = self.edges()
        ij = (int(np.clip(np.searchsorted(re, ev.y(), 'right')-1, 0, self.ip.nt[0]-1)),
              int(np.clip(np.searchsorted(ce, ev.x(), 'right')-1, 0, self.ip.nt[1]-1)))
        if ev.modifiers() & Qt.ShiftModifier: self.ip.toggle_skip(ij)
        elif ij not in self.ip.skip:          self.clicked.emit(ij)
        self.redraw()


class TileView(QWidget):
    """Calibration view of the current tile, for the current mode."""

    def __init__(self, ip):
        super().__init__()
        self.ip = ip
        g = QGridLayout()
        g.setSpacing(15)
        bold = QFont()
        bold.setBold(True)
        W = 3*NCOL # grid columns: images span NCOL, histograms span 3

        self.cv = dict(cntr=Canvas(), segm=Canvas(), lbl=Canvas())
        self.ttl = {}
        for n, k in enumerate(self.cv):
            self.ttl[k] = QLabel()
            self.ttl[k].setFont(bold)
            g.addWidget(self.ttl[k], 0, n*NCOL, 1, NCOL, Qt.AlignCenter)
            g.addWidget(self.cv[k], 1, n*NCOL, 1, NCOL)
        self.ttl['cntr'].setText('Contrast-enhanced tile')
        self.ttl['segm'].setText('Segmented tile')

        self.sl = {}
        for n, k in enumerate(('clahe', 'filt', 'amin')): # one slider under each image
            txt, s = QLabel(), QSlider(Qt.Horizontal)
            txt.setMinimumWidth(SLW + 40) # room for changing values
            txt.setAlignment(Qt.AlignCenter)
            s.setTickPosition(QSlider.TicksBelow)
            s.setTracking(False) # process on release only
            s.setFixedWidth(SLW)
            s.sliderMoved.connect(lambda v, k=k: self.set_txt(k, self.pos2a(v) if k == 'amin' else v))
            s.valueChanged.connect(lambda v, k=k: self.on_slider(k, v))
            self.sl[k] = (s, txt)
            if k == 'amin':
                s.setRange(0, SLN); s.setPageStep(SLN//20); s.setTickInterval(SLN//10)
            else:
                g.addWidget(txt, 2, n*NCOL, 1, NCOL, Qt.AlignCenter)
                g.addWidget(s, 3, n*NCOL, 1, NCOL, Qt.AlignCenter)

        self.btn = QPushButton() # left of area cut-off slider
        self.btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn.clicked.connect(self.identify)
        box = QWidget()
        G = QGridLayout(box)
        G.setContentsMargins(0, 0, 0, 0)
        G.setVerticalSpacing(g.verticalSpacing())
        G.setHorizontalSpacing(10)
        G.addWidget(self.btn, 0, 0, 2, 1, Qt.AlignVCenter)
        G.addWidget(self.sl['amin'][1], 0, 1, Qt.AlignCenter)
        G.addWidget(self.sl['amin'][0], 1, 1, Qt.AlignCenter)
        g.addWidget(box, 2, 2*NCOL, 2, NCOL, Qt.AlignCenter)

        self.hc = {} # histograms, then aggregate q_s bars in the last slot
        for n, key in enumerate(SUMKEYS):
            r, c = divmod(n, NCOL)
            self.hc[key] = Canvas(hist=True)
            g.addWidget(self.hc[key], 4+r, 3*c, 1, 3)
        self.hc['agg'] = Canvas(hist=True)
        g.addWidget(self.hc['agg'], 4+2, 3*(NCOL-1), 1, 3)
        for r in range(3): g.setRowStretch(4+r, 1)
        for c in range(W): g.setColumnStretch(c, 1)
        g.setRowStretch(1, 3)
        self.setLayout(g)
        self.set_mode(ip.mode)

    def set_txt(self, k, v):
        self.sl[k][1].setText(SLIDERS[self.ip.mode][k][0] % v)

    def set_mode(self, mode):
        """Switch mode: update labels and sliders (without processing), then redraw."""
        self.ip.set_params(mode)
        self.ttl['lbl'].setText('Identified %s' % mode)
        self.btn.setText('Identify %s (F5)' % mode)
        self.btn.setFixedWidth(self.btn.sizeHint().width() + 2*self.btn.fontMetrics().averageCharWidth()) # ~1 char padding
        self.sync()
        self.refresh()

    def sync(self):
        """Set slider ranges and values from ip.par."""
        mode = self.ip.mode
        for k, (s, txt) in self.sl.items():
            if k == 'amin': continue # see amin_range()
            fmt, vmin, vmax, step = SLIDERS[mode][k]
            s.blockSignals(True)
            s.setRange(vmin, vmax)
            s.setSingleStep(step); s.setPageStep(step); s.setTickInterval(step)
            s.setValue(self.ip.par[mode][k])
            s.blockSignals(False)
            self.set_txt(k, s.value())
        self.amin_range()

    def pos2a(self, p):
        """Slider position -> area cut-off (log scale from NPX to ip.amax())."""
        amax = self.ip.amax() or NPX
        return int(round(NPX * (amax/NPX)**(p/SLN))) if amax > NPX else NPX

    def a2pos(self, a):
        amax = self.ip.amax() or NPX
        return int(round(SLN * np.log(max(a, NPX)/NPX) / np.log(amax/NPX))) if amax > NPX else 0

    def amin_range(self):
        """Area cut-off slider: NPX to half a tile's area (clamps and re-applies the cut-off if needed)."""
        ip, (s, txt) = self.ip, self.sl['amin']
        amax = ip.amax()
        ok = bool(amax is not None and amax > NPX)
        if ok:
            amin = int(np.clip(ip.par[ip.mode]['amin'], NPX, amax))
            if amin != ip.par[ip.mode]['amin']:
                ip.set_params(amin=amin)
                if 'all' in ip.t.get(ip.mode, {}): ip.cutoff()
        s.blockSignals(True)
        s.setValue(self.a2pos(ip.par[ip.mode]['amin']))
        s.blockSignals(False)
        s.setEnabled(ok)
        s.setToolTip('Log scale: %i px² to %i px² (half a tile)' % (NPX, amax) if ok else 'Open an image first')
        self.set_txt('amin', ip.par[ip.mode]['amin'])

    def on_slider(self, k, v):
        if k == 'amin': v = self.pos2a(v)
        self.set_txt(k, v)
        self.ip.set_params(**{k: v})
        t = self.ip.t.get(self.ip.mode, {})
        if k == 'amin':
            if 'all' in t: self.ip.cutoff() # fast: no re-measuring
        elif self.ip.ij is not None:
            self.ip.segment() # discards identified objects
        self.refresh()

    def refresh(self):
        """Draw current tile (segmenting if needed), with identified objects if available."""
        ip = self.ip
        if ip.ij is None:
            for c in [*self.cv.values(), *self.hc.values()]: c.clear()
            self.amin_range()
            return
        if ip.mode not in ip.t: ip.segment()
        self.amin_range() # may re-apply a clamped cut-off
        d = ip.t[ip.mode].get('data')
        for k, c in self.cv.items():
            c.clear(draw=False)
            if d is not None or k != 'lbl':
                ip.plot_tile(c.ax, k)
                if d is not None: ip.plot_symbols(c.ax, d, svals=SVALS)
            c.draw()
        amin = ip.par[ip.mode]['amin']
        fl = ip.floor() # all objects, i.e. at amin = NPX
        lims = ip.xlims(fl) if d is not None else {} # static x-limits
        for key, c in self.hc.items():
            c.clear(draw=False)
            if d is not None:
                if key == 'agg': ip.plot_agg(c.ax, d)
                else:
                    if key == 'area': ip.plot_hist(c.ax, fl, key, xlim=lims.get(key), color='0.8') # incl. discarded
                    ip.plot_hist(c.ax, d, key, xlim=lims.get(key))
                if key == 'area': c.ax.axvline(amin, color='r', ls='--', lw=1.5)
                c.style()
            c.draw()

    def identify(self):
        if self.ip.ij is None: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.ip.identify()
            self.refresh()
        finally:
            QApplication.restoreOverrideCursor()


class Main(QWidget):

    def __init__(self):
        super().__init__()
        self.ip = icepaya()
        self.view = TileView(self.ip)
        self.ov = Overview(self.ip)
        self.ov.clicked.connect(self.select_tile)

        L = QVBoxLayout()
        L.setSpacing(8)
        ttl = QLabel('Full image')
        bold = QFont()
        bold.setBold(True)
        ttl.setFont(bold)
        L.addWidget(ttl, 0, Qt.AlignCenter)
        L.addWidget(self.ov, 0, Qt.AlignHCenter | Qt.AlignTop)

        btn = QPushButton('Browse')
        btn.clicked.connect(lambda: self.open_img())
        L.addWidget(btn)

        row = QHBoxLayout()
        self.inp_nt = QSpinBox()
        self.inp_nt.setRange(1, 50)
        self.inp_nt.setValue(self.ip.ntiles)
        btn = QPushButton('Set tiles')
        btn.clicked.connect(self.set_tiles)
        row.addWidget(QLabel('Tiles on long edge:')); row.addWidget(self.inp_nt); row.addWidget(btn, 1)
        L.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Mode:'))
        self.rb = {}
        grp = QButtonGroup(self)
        for m in MODES:
            self.rb[m] = QRadioButton(m.capitalize())
            self.rb[m].toggled.connect(lambda on, m=m: on and self.view.set_mode(m))
            grp.addButton(self.rb[m])
            row.addWidget(self.rb[m])
        row.addStretch(1)
        self.rb[self.ip.mode].setChecked(True)
        L.addLayout(row)

        row = QHBoxLayout()
        btn = QPushButton('Process tiles')
        btn.clicked.connect(self.process)
        self.chk_figs = QCheckBox('Save tile figures')
        self.chk_figs.setChecked(True)
        row.addWidget(btn, 1); row.addWidget(self.chk_figs)
        L.addLayout(row)

        self.lbl_file = QLabel('Image: none')
        self.lbl_status = QLabel('')
        self.lbl_status.setWordWrap(True)
        L.addWidget(self.lbl_file)
        L.addWidget(self.lbl_status)
        L.addWidget(self.help_box(), 1)

        panel = QWidget()
        panel.setLayout(L)
        panel.setFixedWidth(OVSIZE + 20)
        lay = QHBoxLayout()
        lay.addWidget(panel)
        lay.addWidget(self.view, 1)
        self.setLayout(lay)

        QShortcut(QKeySequence(Qt.Key_F5), self, self.view.identify)
        self.setWindowTitle('icepaya')
        self.resize(1800, 1000)
        self.show()

    def help_box(self):
        """Scrollable guide with IMT illustration and credits."""
        box = QWidget()
        box.setObjectName('help')
        box.setAttribute(Qt.WA_StyledBackground)
        box.setStyleSheet('QWidget#help {background: #f4f4f4;}')
        V = QVBoxLayout(box)
        V.setContentsMargins(10, 8, 10, 8)
        for html in (GUIDE, IMTHELP):
            lbl = QLabel(html)
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            V.addWidget(lbl)
        img = QLabel()
        img.setPixmap(imt_demo())
        V.addWidget(img, 0, Qt.AlignCenter)
        for html in (MARKERS, AGGHELP):
            lbl = QLabel(html)
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            V.addWidget(lbl)
        V.addStretch(1)
        V.addWidget(QLabel(CREDITS))
        scroll = QScrollArea()
        scroll.setWidget(box)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def status(self, msg, clr='black'):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet('QLabel {font-weight: bold; color: %s;}' % clr)
        QApplication.processEvents()

    def open_img(self, fimg=None):
        fimg = fimg or QFileDialog.getOpenFileName(self, 'Open file', './', 'Image files (*.png *.bmp *.tif *.tiff *.jpg)')[0]
        if not fimg: return
        try:
            self.ip.load(fimg)
        except Exception as e:
            return self.status(str(e), 'red')
        self.lbl_file.setText('Image: ' + os.path.basename(fimg))
        self.inp_nt.setValue(self.ip.ntiles)
        self.ov.set_image()
        self.view.set_mode(self.ip.mode)
        msg = 'Loaded %s (%i x %i px, %i x %i tiles)' % (os.path.basename(fimg), *self.ip.img.shape, *self.ip.nt)
        if self.ip.frestored: msg += '. Settings restored from ' + os.path.basename(self.ip.frestored)
        self.status(msg)

    def set_tiles(self):
        if self.ip.img is None: return
        self.ip.set_tiles(self.inp_nt.value())
        self.ov.redraw()
        self.view.refresh()
        tr, tc = np.diff(self.ip.redg).mean(), np.diff(self.ip.cedg).mean()
        self.status('%i x %i tiles (~%i x %i px)' % (*self.ip.nt, tr, tc))

    def select_tile(self, ij):
        self.ip.set_tile(ij)
        self.ov.redraw()
        self.view.refresh()

    def process(self):
        ip = self.ip
        if ip.img is None: return self.status('Open an image first', 'red')
        ij = ip.ij
        self.setEnabled(False) # avoid changing state while processing
        try:
            d = ip.process(figs=self.chk_figs.isChecked(),
                           cb=lambda n, N, ij: self.status('Processing %s: tile %i/%i %s' % (ip.mode, n, N, ij), 'orange'))
            fpkl = ip.save(d)
            ip.save_summary_fig(d)
            self.status('Finished: %i %s saved to %s' % (len(d['label']), ip.mode, os.path.basename(fpkl)), 'green')
        except Exception as e:
            traceback.print_exc()
            self.status('Error: %s' % e, 'red')
        finally:
            self.setEnabled(True)
            self.select_tile(None if ij in ip.skip else ij) # restore calibration tile


def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(11)
    app.setFont(font)
    w = Main()
    if len(sys.argv) > 1: w.open_img(sys.argv[1])
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
