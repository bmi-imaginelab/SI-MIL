import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat
import math

Image.MAX_IMAGE_PIXELS = None

if os.name == 'nt':
	_dll_path = os.getenv('OPENSLIDE_PATH')
	if _dll_path is not None:
		if hasattr(os, 'add_dll_directory'):
			# Python >= 3.8
			with os.add_dll_directory(_dll_path):
				import openslide
		else:
			# Python < 3.8
			_orig_path = os.environ.get('PATH', '')
			os.environ['PATH'] = _orig_path + ';' + _dll_path
			import openslide
			os.environ['PATH'] = _orig_path
else:
	import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'

class TileWorker(Process):
	"""A child process that generates and writes tiles."""

	def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
				threshold, lower_mag_level = None):
		Process.__init__(self, name='TileWorker')
		self.daemon = True
		self._queue = queue
		self._slidepath = slidepath
		self._tile_size = tile_size
		self._overlap = overlap
		self._limit_bounds = limit_bounds
		self._threshold = threshold
		self.lower_mag_level = lower_mag_level # 5X, if none i.e single scale
		
		self._slide = None

	def run(self):
		self._slide = open_slide(self._slidepath)
		last_associated = None
		dz = self._get_dz()
		
		while True:
			data = self._queue.get()
			if data is None:
				self._queue.task_done()
				break
			associated, level, address, outfile = data
			if last_associated != associated:
				dz = self._get_dz(associated)
				last_associated = associated
			try:
				tile = dz.get_tile(level, address)
				edge = tile.filter(ImageFilter.FIND_EDGES)
				edge = ImageStat.Stat(edge).sum
				edge = np.mean(edge)/(self._tile_size**2)
				w, h = tile.size

				if self.lower_mag_level is None:
					if edge > self._threshold:
						# if not (w==self._tile_size and h==self._tile_size):
							# tile = tile.resize((self._tile_size, self._tile_size))
						if w==self._tile_size and h==self._tile_size:
							tile.save(outfile)
				
				else:
					if level == dz.level_count - self.lower_mag_level - 1:
						if edge > self._threshold:
							# if not (w==self._tile_size and h==self._tile_size):
								# tile = tile.resize((self._tile_size, self._tile_size))
							if w==self._tile_size and h==self._tile_size:
								tile.save(outfile)
					else:
						# if not required lower label generate patches for all. As we wan't lower mag level patch
						# to contain all the corresponding childrens/(higher mag level patch).

						if w==self._tile_size and h==self._tile_size:
							tile.save(outfile)                   

			except:
				pass
			self._queue.task_done()
			

	def _get_dz(self, associated=None):
		if associated is not None:
			image = ImageSlide(self._slide.associated_images[associated])
		else:
			image = self._slide
		return DeepZoomGenerator(image, self._tile_size, self._overlap,
					limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
	"""Handles generation of tiles and metadata for a single image."""

	def __init__(self, dz, basename, format, associated, queue, tile_size_, required_levels):
		self._dz = dz
		self._basename = basename
		self._format = format
		self._associated = associated
		self._queue = queue
		self._processed = 0
				
		self.required_level_deepzoom_list = []
		
		for temp_level in required_levels:
			self.required_level_deepzoom_list.append(self._dz.level_count-1-temp_level)

	def run(self):
		self._write_tiles()

	def _write_tiles(self):
		# for level in range(self._dz.level_count):
		for level in self.required_level_deepzoom_list:
			tiledir = os.path.join("%s_files" % self._basename, str(level))
			if not os.path.exists(tiledir):
				os.makedirs(tiledir)
			cols, rows = self._dz.level_tiles[level]
			for row in range(rows):
				for col in range(cols):
					tilename = os.path.join(tiledir, '%d_%d_%d.%s' % (
									level, col, row, self._format))
					if not os.path.exists(tilename):
						self._queue.put((self._associated, level, (col, row),
									tilename))
					self._tile_done()

	def _tile_done(self):
		self._processed += 1
		count, total = self._processed, self._dz.tile_count
		if count % 100 == 0 or count == total:
			print("Tiling %s: wrote %d/%d tiles" % (
					self._associated or 'slide', count, total),
					end='\r', file=sys.stderr)
			if count == total:
				print(file=sys.stderr)


class DeepZoomStaticTiler(object):
	"""Handles generation of tiles and metadata for all images in a slide."""

	def __init__(self, slidepath, basename, format, tile_size, overlap,
				limit_bounds, workers, threshold, required_levels):
		self._slide = open_slide(slidepath)
		self._basename = basename
		self._format = format
		self._tile_size = tile_size
		self._overlap = overlap
		self._limit_bounds = limit_bounds
		self._queue = JoinableQueue(2 * workers)
		self._workers = workers
		self.required_levels = required_levels
		
		if len(self.required_levels) == 1:
			lower_mag_level = None
		else:
			lower_mag_level = required_levels[-1]
			
		# self._dzi_data = {}
		for _i in range(workers):
			TileWorker(self._queue, slidepath, tile_size, overlap,
						limit_bounds, threshold, lower_mag_level).start()

	def run(self):
		self._run_image()
		self._shutdown()

	def _run_image(self, associated=None):
		"""Run a single image from self._slide."""
		if associated is None:
			image = self._slide
			basename = self._basename
		else:
			image = ImageSlide(self._slide.associated_images[associated])
			basename = os.path.join(self._basename, self._slugify(associated))
		dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
					limit_bounds=self._limit_bounds)
		tiler = DeepZoomImageTiler(dz, basename, self._format, associated,
					self._queue, self._tile_size, self.required_levels)
		tiler.run()

	def _url_for(self, associated):
		if associated is None:
			base = VIEWER_SLIDE_NAME
		else:
			base = self._slugify(associated)
		return '%s.dzi' % base

	def _copydir(self, src, dest):
		if not os.path.exists(dest):
			os.makedirs(dest)
		for name in os.listdir(src):
			srcpath = os.path.join(src, name)
			if os.path.isfile(srcpath):
				shutil.copy(srcpath, os.path.join(dest, name))

	@classmethod
	def _slugify(cls, text):
		text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
		return re.sub('[^a-z0-9]+', '_', text)

	def _shutdown(self):
		for _i in range(self._workers):
			self._queue.put(None)
		self._queue.join()

def organize_patches(img_slide, out_base, level=(0,), tile_size=224, overlap=0, limit_bounds=True):
	print('Organizing enter')

	n_levels = DeepZoomGenerator(open_slide(img_slide), tile_size, overlap,
		limit_bounds=limit_bounds)
	n_levels = n_levels.level_count
	
	img_name = img_slide.split(os.sep)[-1].split('.')[0]
	bag_path = os.path.join(out_base, img_name)
	os.makedirs(bag_path, exist_ok=True)
	if len(level)==1:
		patches = glob.glob(os.path.join('WSI_temp_organ_files', str(n_levels-level[-1]-1), '*.jpg'))
		for i, patch in enumerate(patches):
			patch_name = patch.split(os.sep)[-1]
			shutil.move(patch, os.path.join(bag_path, patch_name))
			sys.stdout.write('\r Organizing patches [%d/%d]' % (i+1, len(patches)))
		print('Done.')
	else:
		level_factor = 2**int(level[1]-level[0])
		low_patches = glob.glob(os.path.join('WSI_temp_organ_files', str(n_levels-level[-1]-1), '*.jpg'))
		print(len(low_patches))
		for i, low_patch in enumerate(low_patches):
			low_patch_name = low_patch.split(os.sep)[-1]
			shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
			low_patch_folder = low_patch_name.split('.')[0]
			high_patch_path = os.path.join(bag_path, low_patch_folder)
			os.makedirs(high_patch_path, exist_ok=True)
			low_x = int(low_patch_folder.split('_')[1])//tile_size
			low_y = int(low_patch_folder.split('_')[2])//tile_size
			high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
			high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
			for x_pos in high_x_list:
				for y_pos in high_y_list:
					high_patch = glob.glob(os.path.join('WSI_temp_organ_files', str(n_levels-level[0]-1), '{}_{}_{}.jpg'.format(n_levels-level[0]-1, x_pos*tile_size, y_pos*tile_size)))
					if len(high_patch)!=0:
						high_patch = high_patch[0]
						shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
			try:
				os.rmdir(os.path.join(bag_path, low_patch_folder))
				os.remove(low_patch)
			except:
				pass
			sys.stdout.write('\r Organizing patches dee[%d/%d]' % (i+1, len(low_patches)))
		print('Done.')

if __name__ == '__main__':
	Image.MAX_IMAGE_PIXELS = None
	parser = argparse.ArgumentParser(description='Patch extraction for WSI')
	parser.add_argument('-d', '--dataset', type=str, default='test_dataset/slides', help='Dataset name')
	parser.add_argument('--save_path', type=str, default='test_dataset/patches', help='save_path name')
	parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
	parser.add_argument('-f', '--format', type=str, default='jpg', help='image format for tiles [png]')
	parser.add_argument('-v', '--slide_format', type=str, default='svs', help='image format for tiles [svs]')
	parser.add_argument('-j', '--workers', type=int, default=10, help='number of worker processes to start [4]')
	parser.add_argument('-s', '--tile_size', type=int, default=224, help='tile size [224]')
	parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(3, ), help='Levels for patch extraction [0]')  # 3 for 5x, 2 for 10x, 1 for 20x, 0 for 40x.
	parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [10]')  
	args = parser.parse_args()
	levels = tuple(args.magnifications)
	assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'


	out_base = args.save_path
	if not os.path.isdir(out_base):
		os.mkdir(out_base)
	
	all_slides = []
	
	for i in os.listdir(args.dataset):
		if 'DX' in i:
			all_slides.append(args.dataset +'/' + i)
	
	print(len(all_slides), len(os.listdir(args.dataset)))
	
	# pos-i_pos-j -> x, y
	for idx, c_slide in enumerate(all_slides):
		levels = tuple(args.magnifications)
		print(levels, np.array(levels))   # level = 3, we want 5x when max magnification is 40 (40/2^^3)
		print(c_slide)
		if not os.path.isdir(out_base + '/' + c_slide.split('/')[-1].split('.')[0]):
			print('Process slide {}/{}'.format(idx+1, len(all_slides)))

			try:
				temp_slide = open_slide(c_slide)
			except:
				continue
			try:
				base_level_mag = int(temp_slide.properties['aperio.AppMag'][:2])
			except:
				continue
			print(base_level_mag)
# 			if base_level_mag != 40:
# 				 # can be done above. Shown here for reference

# 				# then level 0 means that particular WSI's base level magnification

# 				# but here if we give 1, 3 means we want 20X and 5X according to this paper. That is it should give 
# 				# output base on 40X only. 

# 				# therefore there is need of some substraction factor in line abc and abc, so that if level=1, 3 is 
# 				# given and base mag is 20X, it should be converted to level=0, 2 to be able to extract 20X and 5X 
# 				# patches. 

# 				# To be noted that this approach is not very robust, as there is no means of giving level=0 and 20X 
# 				# base mag. According to this method it will convert it to level = -1, which doesnt make sense

# 				levels = np.array(levels)

# 				div_temp = base_level_mag/2
# 				substraction_factor = int(math.log(40/div_temp, base_level_mag/div_temp) - 1)
# 				levels -= substraction_factor

			if base_level_mag != 40:
				continue  # we will just work on 40x images for now
			
			print(levels, base_level_mag)

			DeepZoomStaticTiler(c_slide, 'WSI_temp_organ', args.format, args.tile_size, args.overlap, True, args.workers, args.background_t, levels).run()

			organize_patches(c_slide, out_base, levels, args.tile_size, args.overlap, True)
			shutil.rmtree('WSI_temp_organ_files')
		print('Patch extraction done for {} slides.'.format(len(all_slides)))

