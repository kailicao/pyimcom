import numpy as np
from astropy.io import fits
from copy import deepcopy
from ..config import Config
import sys
import time

# specific compression tools we need
from .i24 import i24compress, i24decompress

class CompressedOutput:
   """Class for compressing pyimcom output files.

   Properties
   ----------
   gzip : is gzipped? (True/False)
   ftype : file type? current options are 'fits'
   cprstype : compression type
   origfile : original file name (when opened)
   hdul : HDU List of information (initially a deep copy of the input file)
   cfg : configuration file used to generate this image

   Methods
   -------
   __init__ : Constructor.
   compress_2d_image : wrapper for 2d image compression (staticmethod)
   decompress_2d_image : wrapper for 2d image decompression (staticmethod)
   compress_layer : compresses a layer
   decompress : decompresses the whole file
   recompress : recompress previously compressed layers
   to_file : saves to a file.
   close : close associated file

   Comments
   --------
   At some point, we may add file formats other than FITS, but not yet.

   This object is big since it stores a deep copy of the whole file.
   Therefore, it is not a good idea to open lots of compressed files at once.
   """

   def __init__(self, fname, format=None):
      """Builds output from a specified file name and format.

      format options are:
          'orig' = original (.fits or .fits.gz, no lossy compression)
          None = determine from file name
      """

      self.origfile = fname

      # figure out what type of file it is, and if it is gzipped
      self.gzip = False
      if fname[-3:]=='.gz': self.gzip = True
      if format is None:
         pref = fname[:-3] if self.gzip else fname

         # right now supports fits files
         if pref[-5:]=='.fits':
            self.ftype = 'fits'
            self.hdul = fits.open(fname, mode='readonly', decompress_in_memory=True)

            if 'CPRSTYPE' in self.hdul[0].header:
               self.cprstype = self.hdul[0].header['CPRSTYPE']
            else:
               self.cprstype = ''
               self.hdul[0].header['CPRSTYPE'] = ''
            self.cfg = Config(fname, inmode='block')

         else:
            raise Exception("unrecognized file type")

   @staticmethod
   def compress_2d_image(im, scheme, pars):
      """Wrapper to compress a 2D image"""

      if scheme[:3]=='I24':
         imout, ovflow = i24compress(im,scheme,pars)
         return imout, ovflow

      # unrecognized scheme or NULL
      return np.copy(im), None

   @staticmethod
   def decompress_2d_image(im, scheme, pars, overflow=None):
      """Wrapper to decompress a 2D image; overflow table optional"""

      if scheme[:3]=='I24':
         return i24decompress(im,scheme,pars,overflow=overflow)

      # unrecognized scheme or NULL
      return np.copy(im)

   def compress_layer(self, layerid, scheme=None, pars={}):
      """Compresses the given layerid with the indicated scheme.

      The pars is a dictionary of parameters that go with that scheme.
      It must be in a format that supports a FITS header.

      If scheme is None, then the algorithm will re-compress the layer in the same way was
      done previously (if it was compressed before), otherwise it will do the NULL compression.
      """

      # this failure to write the EXTNAME shouldn't happen, but just in case
      if layerid>=16**4: return

      # make a blank table if there isn't compression information yet
      if 'CPRESS' not in [hdu.name for hdu in self.hdul]:
         hdu = fits.TableHDU.from_columns([fits.Column(name='text', format='A512', array=[], ascii=True)])
         hdu.name = 'CPRESS'
         self.hdul.append(hdu)
      # now get compression information
      rows = self.hdul['CPRESS'].data['text'].tolist()

      # None means we should check if this data has been compressed
      # before and use that method.
      if scheme is None:
         compressiondict = {}
         for kv in rows:
            layer_, key_, val_ = kv.strip().split(':')
            key_ = key_.strip()
            val_ = val_.strip()
            if int(layer_, 0x10)==layerid:
               compressiondict[key_] = val_
         if 'SCHEME' in compressiondict.keys():
            # this was done before
            # re-compress without adding new keywords
            cd_data, cd_overflow = CompressedOutput.compress_2d_image(self.hdul[0].data[0,layerid,:,:],
                      compressiondict['SCHEME'], compressiondict)
            self.hdul[0].data[0,layerid,:,:] = 0
            newhdu = fits.ImageHDU(cd_data)
            for p in compressiondict.keys():
               newhdu.header[p] = compressiondict[p]
            newhdu.name = 'HSHX{:04X}'.format(layerid)
            self.hdul.append(newhdu)
            cd_overflow.name = 'HSHV{:04X}'.format(layerid)
            self.hdul.append(cd_overflow)

            #print('re-compressed', compressiondict)
            return

         # we will do the null compression in this case
         scheme = 'NULL'

      cd_data, cd_overflow = CompressedOutput.compress_2d_image(self.hdul[0].data[0,layerid,:,:], scheme, pars)
      self.hdul[0].data[0,layerid,:,:] = 0
      newhdu = fits.ImageHDU(cd_data)
      for p in pars.keys():
         newhdu.header[p] = pars[p]
         rows += ['{:04X}:{:8s}:{:s}'.format(layerid, p, str(pars[p]))]
      newhdu.header['SCHEME'] = scheme
      rows += ['{:04X}:{:8s}:{:s}'.format(layerid, 'SCHEME', scheme)]
      newhdu.name = 'HSHX{:04X}'.format(layerid)
      self.hdul.append(newhdu)
      cd_overflow.name = 'HSHV{:04X}'.format(layerid)
      self.hdul.append(cd_overflow)

      # which HDU has the compression data?
      j_ = -1
      for j in range(len(self.hdul)):
         if self.hdul[j].name=='CPRESS':
            j_ = j
      if j_==-1: raise Exception("Can't find CPRESS: this shouldn't happen.")
      # now overwrite that one
      hdu = fits.TableHDU.from_columns([fits.Column(name='text', format='A512', array=rows, ascii=True)])
      hdu.name = 'CPRESS'
      self.hdul[j_] = hdu

   def decompress(self):
      """Decompresses all the layers that were compressed by compress_layer."""

      j=0
      while j<len(self.hdul):
         if self.hdul[j].name[:4] == 'HSHX':
            layer_target = int(self.hdul[j].name[-4:], 0x10)
            self.hdul[0].data[0,layer_target,:,:] = \
               CompressedOutput.decompress_2d_image(\
               self.hdul[j].data, self.hdul[j].header['SCHEME'], self.hdul[j].header,
               overflow=self.hdul['HSHV'+self.hdul[j].name[-4:]])
            del self.hdul[j]
         else:
            # we only increment j if the we didn't remove anything, since if we removed an HDU then
            # what was hdul[j+1] is now hdul[j]
            j=j+1

      # remove the overflow tables
      j=0
      while j<len(self.hdul):
         if self.hdul[j].name[:4] == 'HSHV':
            del self.hdul[j]
         else:
            j=j+1

   def recompress(self):
      """Recompresses all the layers that were previously compressed by compress_layer."""

      # if this wasn't compressed before, nothing to do
      if 'CPRESS' not in [hdu.name for hdu in self.hdul]: return

      # figure out which layers were previously compressed
      nlayer = np.shape(self.hdul[0].data)[-3]
      wascompressed = np.zeros(nlayer, dtype=bool)
      for compressnote in self.hdul['CPRESS'].data['text'].tolist():
         ilayer = int(compressnote.split(':')[0], 16) # which layer this referred to
         wascompressed[ilayer]=True
      #print(wascompressed)
      for ilayer in range(nlayer):
         if wascompressed[ilayer]:
            self.compress_layer(ilayer)

   def to_file(self, fname, overwrite=False):
      """Saves to a file. Specify file name (fname) and whether to overwrite (True/False)."""

      self.hdul.writeto(fname,overwrite=overwrite)

   def close(self):
      """Closes the associated file."""
      self.hdul.close()

   def __enter__(self):
      return self

   def __exit__(self, exc_type, exc_val, exc_tb):
      self.close()
      return False # do not suppress exception

def ReadFile(fname):
   """Wrapper to read a compressed file.

   This should read a file just like astropy.io.fits.open(fname, mode='readonly'),
   but works even if the file is compressed using compressutils.

   It can also be used with the Python context manager, e.g.:
   with ReadFile('my.fits.gz') as f:
      ...
   """

   # if this file hasn't been compressed, just pass the handle:
   f = fits.open(fname)
   if 'CPRESS' not in [hdu.name for hdu in f]:
      return f
   else:
      f.close()

   # otherwise, make a decompressed version
   x = CompressedOutput(fname)
   x.decompress()
   return fits.HDUList(x.hdul)

### Test functions below here ###

def test(argv):

   t_ = [time.time()]
   with CompressedOutput(argv[1]) as f:
      print('ftype =', f.ftype)
      print('gzip =', f.gzip)
      print('cprstype =', f.cprstype)
      print(f.hdul.info())
      print(f.cfg.to_file(fname=None))

      for j in range(1,len(f.cfg.extrainput)):
         if f.cfg.extrainput[j][:6].lower()=='gsstar' or f.cfg.extrainput[j][:5].lower()=='cstar'\
               or f.cfg.extrainput[j][:8].lower()=='gstrstar' or f.cfg.extrainput[j][:8].lower()=='gsfdstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='nstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1500., 'VMAX': 10500., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='gsext':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:8].lower()=='labnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -5, 'VMAX': 5, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:10].lower()=='whitenoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -8, 'VMAX': 8, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:7].lower()=='1fnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -32, 'VMAX': 32, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
      f.to_file(argv[2], overwrite=True)
   t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

   with CompressedOutput(sys.argv[2]) as g:
      print(g.hdul.info())
      g.decompress()
      print(g.hdul.info())
      #print(g.hdul['CPRESS'].data['text'])
      g.to_file(argv[3], overwrite=True)
   t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

   with CompressedOutput(sys.argv[3]) as h:
      h.recompress()
      h.to_file(argv[4], overwrite=True)
   t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

   with ReadFile(sys.argv[4]) as i:
      print('-- COMPRESSED/DECOMPRESSED/RECOMPRESSED --')
      print(i.info())
      t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

      ior = ReadFile(sys.argv[1])
      print('-- ORIGINAL INPUT --')
      print(ior.info())

      print('')
      for j in range(np.shape(ior[0].data)[-3]):
         print('slice {:3d} max {:11.5E} maxerr {:11.5E}'.format(j,
            np.amax(np.abs(ior[0].data[0,j,:,:])),
            np.amax(np.abs(i[0].data[0,j,:,:]-ior[0].data[0,j,:,:]))))
      ior.close()

def test1(fname):
   """Test compression of a file."""

   fout = fname[:-5] + '.cpr.fits.gz'
   frec = fname[:-5] + '_recovered.fits'

   t_ = [time.time()]
   with CompressedOutput(fname) as f:
      for j in range(1,len(f.cfg.extrainput)):
         if f.cfg.extrainput[j][:6].lower()=='gsstar' or f.cfg.extrainput[j][:5].lower()=='cstar'\
               or f.cfg.extrainput[j][:8].lower()=='gstrstar' or f.cfg.extrainput[j][:8].lower()=='gsfdstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='nstar':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1500., 'VMAX': 10500., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:5].lower()=='gsext':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -1./64., 'VMAX': 7./64., 'BITKEEP': 20, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:8].lower()=='labnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -5, 'VMAX': 5, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:10].lower()=='whitenoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -8, 'VMAX': 8, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
         if f.cfg.extrainput[j][:7].lower()=='1fnoise':
            f.compress_layer(j, scheme='I24B', pars={'VMIN': -32, 'VMAX': 32, 'BITKEEP': 16, 'DIFF': True, 'SOFTBIAS': -1})
      f.to_file(fout, overwrite=True)
   t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

   ReadFile(fout).writeto(frec, overwrite=True)
   t_.append(time.time()); print('Delta t = {:6.2f} s'.format(t_[-1]-t_[-2]))

   with ReadFile(fname) as f1, ReadFile(frec) as f2:
      print('')
      for j in range(np.shape(f1[0].data)[-3]):
         print('slice {:3d} max {:11.5E} maxerr {:11.5E}'.format(j,
            np.amax(np.abs(f1[0].data[0,j,:,:])),
            np.amax(np.abs(f1[0].data[0,j,:,:]-f2[0].data[0,j,:,:]))))

if __name__ == "__main__":
   """with 1 argument: makes compressed version of a file. .fits input only (no gz)"""
   if len(sys.argv)==2:
      test1(sys.argv[1])

   """with 5 arguments: main program arguments are:
   input_file out_file_compressed recovered recompressed ncycle
   ncycle=1 is sufficient to test functionality, but can do more than once to
   test for memory leaks.
   """
   if len(sys.argv)==6:
      for j in range(int(sys.argv[5])):
         test(sys.argv[:5])
