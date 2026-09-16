"""
Nicholas M. Rathmann, Niels Bohr Institute, 2020-

Developed as part of
- the Villum Foundation investigator project IceFlow
- the Novo Nordisk Foundation challenge project PRECISE
"""

import sys, os, copy, glob
import pandas as pd
import configparser
from datetime import datetime
import code # code.interact(local=locals())

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QIntValidator
from PyQt5.QtWidgets import (QMainWindow, QWidget, QSizePolicy, QLabel, QLineEdit, QTextEdit, QGridLayout, QHBoxLayout, QApplication, QFileDialog, QSlider, QPushButton, QSizePolicy, QSpacerItem)

import cv2
from scipy import ndimage

import skimage
if float(skimage.__version__[0:4]) < 0.16: exit('Please install "skimage" version 0.16 or higher.')
from skimage import morphology
from skimage.segmentation import clear_border
from skimage import measure, color #, io

#from scipy import ndimage as nd
from scipy.ndimage import generate_binary_structure

import pypaya2

#from matplotlib import cm
#from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

class OverView(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        #self.open_img(blankimg=True)
        self.tileview = TileView(self, TYPE=T_TILEVIEW_GRAINS) 
        self.tileview_bubbles = TileView(self, TYPE=T_TILEVIEW_BUBBLES) 
        self.raise_()
        self.activateWindow()

    def initUI(self):
        
        self.tileres = 2000 # Tile max resolution (x or y). If this is too large, processing on laptops is not possible.
        
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        btnwidth = 320
        ROWNUM = 0
        
        self.btn_open = QPushButton("Browse")
        self.btn_open.clicked.connect(self.open_img)
        self.btn_open.setMinimumWidth(btnwidth)
        self.btn_open.setMaximumWidth(btnwidth)
        self.grid.addWidget(self.btn_open, ROWNUM,0)

        ROWNUM+=1
        self.ROWTILERES = ROWNUM
        self.subgrid = QHBoxLayout()
        self.subgrid.setSpacing(6)
        onlyInt = QIntValidator()
        self.inp_tileres = QLineEdit()
        self.inp_tileres.setValidator(onlyInt)    
        self.inp_tileres.setMinimumWidth(90)
        self.inp_tileres.setMaximumWidth(90)
        self.inp_tileres.setText(str(self.tileres))
        self.subgrid.addWidget(self.inp_tileres, Qt.AlignLeft)
        #
        self.unit_tileres = QLabel("px")
        self.unit_tileres.setMaximumWidth(30)
        self.subgrid.addWidget(self.unit_tileres, Qt.AlignLeft)
        #
        self.btn_tileres = QPushButton("Set tile size")
        self.btn_tileres.clicked.connect(lambda: self.set_tileres(int(self.inp_tileres.text())))
        self.btn_tileres.clicked.connect(self.set_tiles)
        self.btn_tileres.setMinimumWidth(180)
        self.btn_tileres.setMaximumWidth(180) 
        self.subgrid.addWidget(self.btn_tileres, Qt.AlignLeft)
        #
        self.grid.addLayout(self.subgrid, self.ROWTILERES,0, Qt.AlignLeft)
        
        ROWNUM+=1
        self.btn_proc = QPushButton("Process tiles")
        self.btn_proc.clicked.connect(lambda: self.process_tiles())
        self.btn_proc.setMinimumWidth(btnwidth)
        self.btn_proc.setMaximumWidth(btnwidth)
        self.grid.addWidget(self.btn_proc, ROWNUM, 0)
        
        ROWNUM+=1
        self.lbl_fopen = QLabel("Image: none")
        self.grid.addWidget(self.lbl_fopen,ROWNUM,0, Qt.AlignLeft)
        
        ROWNUM+=1
        self.lbl_status = QLabel("")
        self.lbl_status.setMinimumWidth(btnwidth)
        self.grid.addWidget(self.lbl_status,ROWNUM,0, Qt.AlignLeft)
        
        #ROWNUM+=1
        helpstr = """Help:\n--------
  1) Open image (browse) 
  2) Set tile size (<5000px is recommended)
  3) Select calibration tile below
  4) Pick calibration settings in 'tile viewer'
  5) Shift+click tiles to skip them
  6) Process tiles"""
        self.lbl_howto = QLabel(helpstr)
        self.grid.addWidget(self.lbl_howto, 0, 1, ROWNUM, 1, Qt.AlignLeft)

        ROWNUM+=1        
        self.ROWTILESGRID = ROWNUM
        self.tiles = QLabel('')
        self.grid.addWidget(self.tiles, self.ROWTILESGRID, 0, 1, 2)

        ROWNUM+=1        
        self.pady = QLabel('')
        self.grid.addWidget(self.pady,ROWNUM,0)
        self.padx = QLabel('')
        self.grid.addWidget(self.padx,ROWNUM,2)

        self.grid.setRowStretch(0, 0)
        self.grid.setRowStretch(1, 0)
        self.grid.setRowStretch(2, 0)
        self.grid.setRowStretch(3, 0)
        self.grid.setRowStretch(4, 0)
        self.grid.setRowStretch(5, 0)
        self.grid.setRowStretch(6, 1)
        
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 0)
        self.grid.setColumnStretch(2, 1)

        self.setLayout(self.grid)
        self.setWindowTitle('LASM processing')
        self.show()

    def open_img(self, blankimg=None):
        if blankimg:
            self.img = np.zeros((1e2,1e2),dtype=np.uint8)
        else:
            self.filename = QFileDialog.getOpenFileName(self, 'Open file', "./", "Image files (*.png *.bmp)")[0]
            self.lbl_fopen.setText('Image: '+os.path.basename(self.filename))
            self.img = cv2.imread(self.filename, 0)
            print('Loaded %s'%(self.filename))
            self.dumppath = self.filename[:-4] + '_tiles'
            self.fcombcsv_grains  = self.filename[:-4] + '_grains'  + '.csv'
            self.fcombcsv_bubbles = self.filename[:-4] + '_bubbles' + '.csv'
            self.fsettings = self.filename[:-4] + '.ini'
            
            if not os.path.exists(self.dumppath): os.mkdir(self.dumppath)
            self.config_load()
            self.set_tiles()
            
    def set_tiles(self):
        nx = np.shape(self.img)[0]
        ny = np.shape(self.img)[1]
        self.Nx, self.Ny = 1,1
        for N in np.power(2,np.arange(20)):
            if nx/N >= self.tileres: self.Nx = N
            if ny/N >= self.tileres: self.Ny = N
        self.dx = int( nx/self.Nx )
        self.dy = int( ny/self.Ny )
        
        self.grid.removeWidget(self.tiles)
        self.tiles.deleteLater()
        self.tiles = MyTilesWidget(self.Nx,self.Ny, self.dx,self.dy, self.img, self.tileview, self.tileview_bubbles)
        self.grid.addWidget(self.tiles, self.ROWTILESGRID, 0, 1, 2)
        self.grid.update()
        
        
    def process_tiles(self):
        print('Processing all tiles...')
        self.lbl_status.setText("Processing tiles...")
        self.lbl_status.setStyleSheet("QLabel {font-weight: bold; color: orange;}");
        for iy in np.arange(self.Ny):
            for ix in np.arange(self.Nx):
                if (ix,iy) in self.tiles.skiplist:
                    print('%i %i (skipped)'%(ix,iy))
                    continue
                print('%i %i'%(ix,iy))
                self.tiles.labelsgrid[iy,ix].mousePressEvent(0)
                self.tiles.repaint()
                #
                self.tileview.find_grains()
                self.tileview.repaint()
                self.tileview.save_csv(tiley0=ix*self.dx, tilex0=iy*self.dy)
                self.tileview.save_img_raw()
                self.tileview.save_img()
                #
                self.tileview_bubbles.find_grains()
                self.tileview_bubbles.repaint()
                self.tileview_bubbles.save_csv(tiley0=ix*self.dx, tilex0=iy*self.dy)
                self.tileview_bubbles.save_img()
                
        self.combine_csv_tiles(T_TILEVIEW_GRAINS)
        self.combine_csv_tiles(T_TILEVIEW_BUBBLES)
        self.plot_summary()
        self.config_save()
        self.lbl_status.setText('Finished: output is "%s"'%(self.fcombcsv_grains))
        self.lbl_status.setStyleSheet("QLabel {font-weight: bold; color: green;}");
                
    def combine_csv_tiles(self, TYPE):
        files = [i for i in glob.glob('%s/*_%s.csv'%(self.dumppath, 'grains' if TYPE==T_TILEVIEW_GRAINS else 'bubbles'))]
        combined_csv = pd.concat([pd.read_csv(f) for f in files])
        fcomb = self.fcombcsv_grains if TYPE==T_TILEVIEW_GRAINS else self.fcombcsv_bubbles
        combined_csv.to_csv(fcomb, index=False, encoding='utf-8-sig')
        print('Saved %s'%(fcomb))
                
    def set_tileres(self, res):
        self.tileres = res

    def config_save(self):
        
        config = configparser.ConfigParser()
        config['settings'] = {
                'clahe':    self.tileview.slider_cntr.value(), \
                'ssfilter': self.tileview.slider_segm.value(), \
                'clahebubbles': self.tileview_bubbles.slider_cntr.value(), \
                'cvopening':    self.tileview_bubbles.slider_segm.value(), \
                'tileres':   int(self.inp_tileres.text()), \
                'timestamp': datetime.now() \
                }
        with open(self.fsettings, 'w') as configfile: config.write(configfile)
        print('Saved config %s'%(self.fsettings))

    def config_load(self):
        
        if os.path.isfile(self.fsettings):
            config = configparser.ConfigParser()
            config.read(self.fsettings)        
            self.tileview.slider_cntr.setValue(int(config['settings']['clahe']))
            self.tileview.slider_segm.setValue(int(config['settings']['ssfilter']))
            self.tileview_bubbles.slider_cntr.setValue(int(config['settings']['clahebubbles']))
            self.tileview_bubbles.slider_segm.setValue(int(config['settings']['cvopening']))
            self.inp_tileres.setText(str(config['settings']['tileres']))
            self.set_tileres(int(config['settings']['tileres']))
            print('Loaded config %s'%(self.fsettings))
    
    def plot_summary(self):
        
        for TYPE in [T_TILEVIEW_GRAINS, T_TILEVIEW_BUBBLES]:
        
            fname = self.fcombcsv_grains if TYPE==T_TILEVIEW_GRAINS else self.fcombcsv_bubbles
            df = pd.read_csv(fname, sep=',')
            
            f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12,7))
            f.suptitle(fname + ' (%i %s)'%(len(df['grain_number']), 'grains' if TYPE==T_TILEVIEW_GRAINS else 'bubbles') )
            bins = 40
            
            ax1.hist(df['equivalent_diameter'], bins=bins, color='k')
            ax1.set_xlabel('Equiv. diameter [px]')
            
            ax2.hist(df['area'], bins=bins, color='0.4')
            ax2.set_xlabel('Area [px^2]')
            
            ax4.hist(df['major_axis_length'], bins=bins, color='#1f78b4')
            ax4.set_xlabel('Major axis [px]')
            
            ax5.hist(df['eccentricity'], bins=bins, color='#6a3d9a')
            ax5.set_xlabel('Eccentricity')
            
            ax3.hist(df['perimeter'], bins=bins, color='#33a02c')
            ax3.set_xlabel('Perimeter [px]')
            
            ax6.hist(df['orientation'], bins=bins, color='#b15928')
            ax6.set_xlabel('Orientation [deg. btw. y- and major-axis]')
            
            plt.tight_layout()
            plt.savefig(fname[:-4]+'_stats.png')
            
            plt.close(f)
    
###############################################################################
###############################################################################
        
class MyTilesWidget(QWidget):

    def __init__(self, Nx,Ny, dx,dy, img, tileview, tileview_bubbles):
        super().__init__()
        self.tileview = tileview
        self.tileview_bubbles = tileview_bubbles
        self.imgxmax, self.imgymax = 1000, 800# Resolution for the gridded tiles view. If too large, low res monitors can't show the full image.
        self.xscale = int(self.imgxmax/Nx)
        self.yscale = int(self.imgymax/Ny)
        self.childInFocus = None
        self.skiplist = []
        
        grid = QGridLayout()
        grid.setSpacing(2)
        
        self.labelsgrid =  np.array([ [None for ix in np.arange(Nx)] for iy in np.arange(Ny) ], dtype=object) # @TODO test for SKEW GRID
        
        for iy in np.arange(Ny):
            for ix in np.arange(Nx):
                frame = np.array(img[ix*dx:(ix+1)*dx,iy*dy:(iy+1)*dy]) 
                self.labelsgrid[iy,ix] = ClickableLabel(frame, (ix,iy), self)
                grid.addWidget(self.labelsgrid[iy,ix], ix,iy)
        
        vspacer = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Expanding)
        grid.addItem(vspacer, Ny, 0, 1, -1)
        hspacer = QSpacerItem(QSizePolicy.Expanding, QSizePolicy.Minimum)
        grid.addItem(hspacer, 0, Nx, -1, 1)
        
        self.setLayout(grid)
        self.show()
        
        #skip frame
        skipbg = frame*0 + 255
        np.fill_diagonal(skipbg, 0)
        kernel = np.ones((5,5),np.uint8)
        self.skipbg = cv2.erode(skipbg,kernel,iterations = 12)
        #self.skipbg = frame*0 + 255
        
###############################################################################
###############################################################################
        
class ClickableLabel(QLabel):
    
    clicked = QtCore.Signal(str)

    def __init__(self, frame, tile_ij, parent):
        super(ClickableLabel, self).__init__()
        self.frame = frame        
        self.tile_ij = tile_ij
        self.parent = parent
        self.setbg(self.frame)
        
    def setbg(self, frame, Qimageformat=QImage.Format_Grayscale8):
        height, width = frame.shape
        #https://stackoverflow.com/questions/41596940/qimage-skews-some-images-but-not-others
        totalBytes = frame.nbytes         # calculate the total number of bytes in the frame 
        bytesPerLine = int(totalBytes/height)  # divide by the number of rows
        Qimg = QImage(frame, width, height, bytesPerLine, Qimageformat)
        pixmap = QPixmap.fromImage(Qimg)
        self.setPixmap(pixmap.scaled(self.parent.xscale,self.parent.yscale,Qt.KeepAspectRatio, Qt.FastTransformation))

    def mousePressEvent(self, event):
       
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            if self.tile_ij not in self.parent.skiplist:  # add skip tile
                self.parent.skiplist.append(self.tile_ij)
                self.setbg(self.parent.skipbg)
            else: # remove skip tile
                self.parent.skiplist.remove(self.tile_ij)
                self.setbg(self.frame)
                
        elif self.tile_ij not in self.parent.skiplist:
            brt = 50
            img = copy.deepcopy(self.frame)
            img[img < 255-brt] += brt  
            self.setbg(img)
            self.parent.tileview.set_tile(self.frame, self.tile_ij)
            self.parent.tileview_bubbles.set_tile(self.frame, self.tile_ij)
            if (self.parent.childInFocus is not None) and (self.parent.childInFocus is not self): self.parent.childInFocus.setbg(self.parent.childInFocus.frame)        
            self.parent.childInFocus = self
        
###############################################################################
###############################################################################
        
T_TILEVIEW_GRAINS  = 1
T_TILEVIEW_BUBBLES = 2
        
class TileView(QWidget):

    def __init__(self, parent, TYPE=T_TILEVIEW_GRAINS):
        super().__init__()
        self.parent = parent
        self.rad2deg = 57.2958
        self.type = TYPE
        self.typestr = 'grains' if self.type==T_TILEVIEW_GRAINS else 'bubbles'
        self.initUI()

    def initUI(self):
        
        grid = QGridLayout()
        grid.setSpacing(20)

        self.fig_cntr = PlotCanvas()
        self.fig_segm = PlotCanvas()
        self.fig_lbld = PlotCanvas()
        
        self.fig_area = PlotCanvas(plottype='hist', histcolor='0.4', title='Area [px^2]')
        self.fig_edia = PlotCanvas(plottype='hist', histcolor='0.0', title='Equivalent diameter [px]')
        self.fig_ecce = PlotCanvas(plottype='hist', histcolor='#6a3d9a', title='Eccentricity')
        self.fig_orie = PlotCanvas(plottype='hist', histcolor='#b15928', title='Orientation [deg. btw. y- and major-axis]')
        self.fig_peri = PlotCanvas(plottype='hist', histcolor='#33a02c', title='Perimeter [px]')
        self.fig_maax = PlotCanvas(plottype='hist', histcolor='#1f78b4', title='Major axis [px]')
        
        self.fig_edia.plotstats([]); grid.addWidget(self.fig_edia, 6, 0)        
        self.fig_area.plotstats([]); grid.addWidget(self.fig_area, 6, 1)
        self.fig_peri.plotstats([]); grid.addWidget(self.fig_peri, 6, 2)
        self.fig_maax.plotstats([]); grid.addWidget(self.fig_maax, 7, 0)
        self.fig_ecce.plotstats([]); grid.addWidget(self.fig_ecce, 7, 1)
        self.fig_orie.plotstats([]); grid.addWidget(self.fig_orie, 7, 2)
            
        #------------------
        
        myBold=QFont()
        myBold.setBold(True)
        objwidth = 300
        
        txt_cntr_title = QLabel('Raw tile')
        txt_cntr_title.setFont(myBold)
        grid.addWidget(txt_cntr_title, 0, 0, Qt.AlignCenter)
        self.slider_cntr = QSlider(Qt.Horizontal, self)
        self.slider_cntr.setMinimum(0)
        self.slider_cntr.setMaximum(21)
        self.slider_cntr.setValue(9 if self.type==T_TILEVIEW_GRAINS else 0)
        self.slider_cntr.setTickPosition(QSlider.TicksBelow)
        self.slider_cntr.setTickInterval(1)
        self.slider_cntr.setSingleStep(1)
        self.slider_cntr.setTracking(False)
        self.slider_cntr.valueChanged[int].connect(lambda: txt_cntr.setText('CLAHE clip limit: %+i'%(self.slider_cntr.value())))
        self.slider_cntr.valueChanged[int].connect(self.update_cntr)
        self.slider_cntr.valueChanged[int].connect(lambda: self.fig_cntr.plot(self.tile_cntr))
        self.slider_cntr.valueChanged[int].connect(lambda: self.update_segm(self.slider_segm.value()))
        self.slider_cntr.valueChanged[int].connect(lambda: self.fig_segm.plot(self.tile_segm))
        self.slider_cntr.setMinimumWidth(objwidth)
        self.slider_cntr.setMaximumWidth(objwidth)
        grid.addWidget(self.slider_cntr, 3, 0, Qt.AlignLeft)
        txt_cntr = QLabel('CLAHE contrast limit: %i'%(self.slider_cntr.value()))
        grid.addWidget(txt_cntr, 2, 0, Qt.AlignLeft)
        grid.addWidget(self.fig_cntr, 1, 0)
        
        txt_segm_title = QLabel('Segmented tile')
        txt_segm_title.setFont(myBold)
        grid.addWidget(txt_segm_title, 0, 1, Qt.AlignCenter)
        self.slider_segm = QSlider(Qt.Horizontal, self)
        ##
        if self.type == T_TILEVIEW_GRAINS:
            self.slider_segm.setMinimum(0)
            self.slider_segm.setMaximum(400)
            self.slider_segm.setValue(150)
            self.slider_segm.setTickPosition(QSlider.TicksBelow)
            self.slider_segm.setTickInterval(20)
            self.slider_segm.setSingleStep(20)
            self.slider_segm.setTracking(False)
            self.slider_segm.valueChanged[int].connect(lambda: txt_segm.setText('Filter small-scale structure <= %i px'%(self.slider_segm.value())))
            self.slider_segm.valueChanged[int].connect(self.update_segm)
            self.slider_segm.valueChanged[int].connect(lambda: self.fig_segm.plot(self.tile_segm))
            self.slider_segm.setMinimumWidth(objwidth)
            self.slider_segm.setMaximumWidth(objwidth)
            grid.addWidget(self.slider_segm, 3, 1, Qt.AlignLeft)
            txt_segm = QLabel('Filter small-scale structure <= %i px'%(self.slider_segm.value()))
        ##
        if self.type == T_TILEVIEW_BUBBLES:
            self.slider_segm.setMinimum(0)
            self.slider_segm.setMaximum(10)
            self.slider_segm.setValue(2)
            self.slider_segm.setTickPosition(QSlider.TicksBelow)
            self.slider_segm.setTickInterval(1)
            self.slider_segm.setSingleStep(1)
            self.slider_segm.setTracking(False)
            self.slider_segm.valueChanged[int].connect(lambda: txt_segm.setText('Number of erosions and dilations: %s'%(self.slider_segm.value())))
            self.slider_segm.valueChanged[int].connect(self.update_segm)
            self.slider_segm.valueChanged[int].connect(lambda: self.fig_segm.plot(self.tile_segm))
            self.slider_segm.setMinimumWidth(objwidth)
            self.slider_segm.setMaximumWidth(objwidth)
            grid.addWidget(self.slider_segm, 3, 1, Qt.AlignLeft)
            txt_segm = QLabel('Number of erosion and dilation cycles: %s'%(self.slider_segm.value()))
        ##
        grid.addWidget(txt_segm, 2, 1, Qt.AlignLeft)
        grid.addWidget(self.fig_segm, 1, 1)

        txt_findgrains_title = QLabel('Identified %s'%(self.typestr))
        txt_findgrains_title.setFont(myBold)
        grid.addWidget(txt_findgrains_title, 0, 2, Qt.AlignCenter)
        self.btn_findgrains = QPushButton('Identify %s (F5)'%(self.typestr))
        self.btn_findgrains.clicked.connect(lambda: self.find_grains())
        self.btn_findgrains.setMinimumWidth(objwidth)
        self.btn_findgrains.setMaximumWidth(objwidth)
        grid.addWidget(self.btn_findgrains, 2, 2, Qt.AlignCenter)
        self.btn_savegrains = QPushButton('Save tile %s to .csv'%(self.typestr))
        self.btn_savegrains.clicked.connect(lambda: self.save_csv())
        self.btn_savegrains.setMinimumWidth(objwidth)
        self.btn_savegrains.setMaximumWidth(objwidth)
        grid.addWidget(self.btn_savegrains, 3, 2, Qt.AlignCenter)

        grid.addWidget(self.fig_lbld, 1, 2)

        self.set_tile(np.zeros((5,5),dtype=np.uint8), (0,0)) # Initialize blank
        
        grid.setRowStretch(1, 2)
        grid.setRowStretch(6, 1)
        grid.setRowStretch(7, 1)
        
        self.setLayout(grid)
        self.setWindowTitle('Tile viewer -- %s'%(self.typestr))
        self.show()
        
    def set_tile(self, tile, ij):
    
        self.tile = tile
        self.tile_ij = ij
        
        self.update_cntr(self.slider_cntr.value())
        self.fig_cntr.plot(self.tile_cntr)
        self.update_segm(self.slider_segm.value())
        self.fig_segm.plot(self.tile_segm)
        self.fig_lbld.plot(0*self.tile)

    def update_cntr(self, value):
        if value == 0:
            self.tile_cntr = copy.deepcopy(self.tile) 
        else:
            clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(10,10)) 
            equ = clahe.apply(self.tile)
            self.tile_cntr = copy.deepcopy(equ)

    def update_segm(self, value):

        tile = self.tile_cntr
    
        if self.type == T_TILEVIEW_BUBBLES:  # Erode away grain boundaries
#            if self.type == T_TILEVIEW_BUBBLES: self.tile = 255 - self.tile # invert image
            if self.type == T_TILEVIEW_BUBBLES: tile = 255 - tile # invert image
            kernel = np.ones((5,5),np.uint8)
            tile = cv2.erode(tile,  kernel, iterations=value)
            tile = cv2.dilate(tile, kernel, iterations=value)
    
        ret, thresh0 = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Automatic segmentation
        k = 0
        thresh1bool = (thresh0 < ret+k) 
        if self.type == T_TILEVIEW_GRAINS:  thresh1 = morphology.remove_small_objects(thresh1bool, value) 
        if self.type == T_TILEVIEW_BUBBLES: thresh1 = morphology.remove_small_objects(thresh1bool, 300) # Always remove a bit of the small scale structure due to reflections over bubbles.
        thresh1 = np.uint8(np.logical_not(thresh1)*255)
        self.tile_segm = thresh1

    def find_grains(self):
        
        mask = (self.tile_segm==255)  #Sets TRUE for all 255 valued pixels and FALSE for 0
        mask = clear_border(mask)   #Removes edge touching grains. 
        mask = morphology.remove_small_objects(mask, 500)
                
        s = generate_binary_structure(2,2)
        labeled_mask, num_labels = ndimage.label(mask, structure=s)

        self.tile_lbld = color.label2rgb(labeled_mask, bg_label=0)
        self.fig_lbld.plot(self.tile_lbld)

        #--------------------

        #self.clusters = measure.regionprops(labeled_mask, self.tile, coordinates='xy')
        self.clusters = measure.regionprops(labeled_mask, self.tile) # assumes rc-coords in skimage>=0.16
        getGrainsProp = lambda prop: np.array([ cluster_props[prop] for cluster_props in self.clusters])
        
        self.gnum = getGrainsProp('label')
        self.edia = getGrainsProp('equivalent_diameter')
        self.area = getGrainsProp('area')
        self.ecce = getGrainsProp('eccentricity')
        self.orie = getGrainsProp('orientation') * self.rad2deg # rad to deg
        self.peri = getGrainsProp('perimeter')
        self.cent = getGrainsProp('centroid')
        self.maax = getGrainsProp('major_axis_length')
        self.miax = getGrainsProp('minor_axis_length')
        
        self.fig_area.plotstats(self.area)
        self.fig_edia.plotstats(self.edia)
        self.fig_ecce.plotstats(self.ecce)
        self.fig_maax.plotstats(self.maax)
        self.fig_peri.plotstats(self.peri)
        self.fig_orie.plotstats(self.orie)
        
        self.fig_segm.plotellipses(self.cent,self.orie/self.rad2deg,self.maax,self.miax)
        
    def save_csv(self, tilex0=0, tiley0=0):
        
        fname = self.parent.dumppath + '/%i_%i_%s.csv'%(self.tile_ij[0], self.tile_ij[1], self.typestr)
        
        propList = {'gnum':'grain_number', \
                    'cntx':'centroid_x', \
                    'cnty':'centroid_y', \
                    'edia':'equivalent_diameter', \
                    'area':'area', \
                    'ecce':'eccentricity', \
                    'orie':'orientation', \
                    'peri':'perimeter', \
                    'maax':'major_axis_length', \
                    'miax':'minor_axis_length'}
    
        output_file = open(fname, 'w')
        output_file.write(","+",".join(propList.values()) + '\n') 
        
        for ii,gnumii in enumerate(self.gnum):
            for propkey in propList.keys():
                fieldstr = ''
                if   propkey == 'cntx':    fieldstr = str(self.cent[ii][1] + tilex0);
                elif propkey == 'cnty':    fieldstr = str(self.cent[ii][0] + tiley0);
                else:                       fieldstr = str(getattr(self,propkey)[ii]);
                output_file.write(',' + fieldstr)
            output_file.write('\n')
            
        output_file.close() 
    
    def save_img(self):
        
        fname_base = self.parent.dumppath + '/%i_%i_%s'%(self.tile_ij[0], self.tile_ij[1], self.typestr)
        fname = fname_base + '.png'
        
        fig = plt.figure(dpi=150)
        fig.tight_layout()
        ax = fig.gca()
        ax.imshow(self.tile_lbld)
        plot_ellipses(ax,self.cent,self.orie/self.rad2deg,self.maax,self.miax, cmid='w', cma='0.4', cmi='0.8') 
        plt.savefig(fname)
        plt.close(fig)
        
    def save_img_raw(self):
        
        fname_base = self.parent.dumppath + '/%i_%i'%(self.tile_ij[0], self.tile_ij[1])
        fname = fname_base + '_raw.png'
        
        fig = plt.figure(dpi=150)
        fig.tight_layout()
        ax = fig.gca()
        ax.imshow(self.tile_cntr, 'Greys_r')
        plt.savefig(fname)
        plt.close(fig)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.find_grains()
        event.accept()
        
###############################################################################
###############################################################################

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, plottype='img', title='TITLE', cmap='Greys_r', histcolor='k'):
        
        self.title = title
        self.cmap  = cmap 
        self.histcolor = histcolor
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        if plottype=='img':
            a = 0 # 0.05
            self.fig.subplots_adjust(left=a, right=1-a, bottom=a, top=1-2*a)
        if plottype=='hist':
            self.fig.subplots_adjust(bottom=0.3, top=1-0.05)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def plot(self, img):
        self.ax.clear()
        self.ax.imshow(img, self.cmap)
        self.ax.axis('off')
        self.draw()        
        
    def plotstats(self, stat):
        self.ax.clear()
        self.ax.hist(stat, bins=50, color=self.histcolor)
        self.ax.set_ylabel('#')
        self.ax.set_xlabel(self.title)
        self.draw()        

    def plotellipses(self, centriods, orientations, majoraxes, minoraxes):
        plot_ellipses(self.ax, centriods, orientations, majoraxes, minoraxes)
        self.draw()

###############################################################################
###############################################################################
        
def plot_ellipses(ax, centriods, orientations, majoraxes, minoraxes, cma='#e31a1c', cmi="#fb9a99", cmid="#1f78b4"):
    for ii in np.arange(len(centriods)):        
        y0, x0 = centriods[ii]
        ori = orientations[ii]
        minor_axis_length = minoraxes[ii] # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
        major_axis_length = majoraxes[ii]
        x1 = x0 + np.cos(ori) * 0.5 * minor_axis_length
        y1 = y0 - np.sin(ori) * 0.5 * minor_axis_length
        x2 = x0 - np.sin(ori) * 0.5 * major_axis_length
        y2 = y0 - np.cos(ori) * 0.5 * major_axis_length
        ax.plot((x0, x1), (y0, y1), '-', color=cmi, linewidth=2.0)
        ax.plot((x0, x2), (y0, y2), '-', color=cma, linewidth=2.0)
        ax.plot(x0, y0, '.', color=cmid, markersize=8)
    
###############################################################################
###############################################################################
       
def main():
    app = QApplication(sys.argv)
    custom_font = QFont()
    custom_font.setPointSize(10);
    app.setFont(custom_font, "QLabel")
    app.setFont(custom_font, "QPushButton")
    overview = OverView()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
