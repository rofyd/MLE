from matplotlib import pyplot as plt
import numpy as np
import h5py as h5
from scipy.interpolate import interpn
from pathlib import Path
import sys
import getopt

def generate_specific_rows(filePath, row_indices = []):
	with open(filePath) as f:

		# using enumerate to track line no.
		for i, line in enumerate(f):
			#if line no. is in the row index list, then return that line
			if i in row_indices:
				yield line

# function to load data from path
def loadData(path, index):
	print("loading data ...")
	gen = generate_specific_rows(path, [int(index)])
	data = np.genfromtxt(gen, skip_header = 0, dtype = str, comments = 'comment')
	print("finished loading data")
	#print(data)
	#exit(1)
	return data
	

# puts data into usable format
def adjustData(store, odlimit):
	print("adjusting data for evaluation ...")
	vals1 = store[3].split('#') # values after splitting the '#'
	vals2 = [] # values after splitting '_'

	for i in range(len(vals1)):
		vals2.append(vals1[i].split('_'))
	
	vals2.pop() # remove last element, which is generated as '' because the list ends on '#'

	vals3 = [] # values after splitting '-' for position and results
	for i in range(len(vals2)):
		pos = [int(val) for val in vals2[i][0].split('-')]
		res = list(filter(None, vals2[i][1].split('-')))
		res = [float(val) for val in res]

		if(res[0] < odlimit):
			continue
		vals3.append([pos, res])
	
	print("finished adjusting")
	return vals3

# checks if two points are neighbors
def arePointsNeighbors(point1, point2):
	if  (abs(point1[0] - point2[0]) == 1 and point1[1] == point2[1] and point1[2] == point2[2]):
		return True
	elif(point1[0] == point2[0] and abs(point1[1] - point2[1]) == 1 and point1[2] == point2[2]):
		return True
	elif(point1[0] == point2[0] and point1[1] == point2[1] and abs(point1[2] - point2[2]) == 1):
		return True

	return False


# get all groups that a points neighbors
def getNeighboringGroups(groups, point):
	if not groups:
		return []

	groupNumbers = []

	for g in range(len(groups)):
		for i in range(len(groups[g])):
			if(arePointsNeighbors(point, groups[g][i])):
				groupNumbers.append(g)
				break
			
	return groupNumbers

# group neighboring points together
def groupPoints(positions):
	print("grouping points ...")
	positions_len = len(positions)
	groups = []

	for i in range(positions_len):
 #       if(i % np.floor(positions_len / 100) == 0):
 #           print(np.round(i / positions_len, 2) * 100, '%')

		neighbors = getNeighboringGroups(groups, positions[i])
		if not neighbors:
			groups.append([positions[i]])
		elif (len(neighbors) > 1):
			groups[neighbors[0]].append(positions[i])
	print("finihsed grouping")
	return groups

def makeNestedList(div):
	lis = []
	for i in range(div):
		lis.append([])
		for j in range(div):
			lis[i].append([])
			for k in range(div):
				lis[i][j].append([])
	return lis

# group points based on position
def spatialSort(positions, div, num):
	print("spatially sorting points ...")
	divd = np.ceil(num / div) # division distance

	spatially_sorted = makeNestedList(div)
	
	positions_len = len(positions)
	for i in range(positions_len):
		if(i % np.floor(positions_len / 100) == 0):
			print(np.round(i / positions_len, 2) * 100, '%')
		posi = int(np.floor(positions[i][0] / divd))
		posj = int(np.floor(positions[i][1] / divd))
		posk = int(np.floor(positions[i][2] / divd))

		spatially_sorted[posi][posj][posk].append(positions[i])

	print("finished spatial sort")
	return spatially_sorted

# extract only the position data
def getOnlyPosition(values):
	print("extracting positions ...")
	pos = []
	for i in range(len(values)):
		pos.append(values[i][0])
	print("finished extracting")
	return pos

def getOnlyOverdensities(values):
	ods = []
	for i in range(len(values)):
		ods.append(values[i][1][0])
	return ods

def makeHistogram(values, binSize):
	maxV = int(np.ceil(max(values)))
	bins = np.arange(0, maxV, binSize)

	hist, bins = np.histogram(values, bins = bins)

	plt.bar(bins[:-1], hist, width = binSize)
	plt.yscale('log')
	plt.autoscale(enable = True, axis = 'y')
	plt.autoscale(enable = True, axis = 'x')
	
	plt.savefig("hist.png")
	plt.close()
#    plt.xlim([0, np.ceil(maxV)])
#    plt.hist(values, bins = bins, alpha = 0.5)
#    plt.show()

def plotODhist(values, binSize):
	print("extracting overdensities ...")
	ods = getOnlyOverdensities(values)
	print("finished extraction")

	makeHistogram(ods, binSize)


def transferFunction(x):

	# values per peak. They are the same for all colors at that peak
	means = [7.0, 15.0, 80.0]
	sigmas = [1.0, 4.0, 20.0]

	# amplitudes for each peak and color (rgba)
	amplitudes = [[0.1, 0.1, 1.0, 0.05], [0.1, 1.0, 0.1, 0.1], [1.0, 0.1, 0.1, 0.6]]
	numPeaks = len(means)
	r, g, b, a = 0, 0, 0, 0
	for i in range(numPeaks):
		scale = np.exp( -(x - means[i])**2 / (2 * sigmas[i]**2))
		r += amplitudes[i][0] * scale
		g += amplitudes[i][1] * scale
		b += amplitudes[i][2] * scale
		a += amplitudes[i][3] * scale

	# amplitudes = [[0.0, 0.0, 0.1, 0.1], [0.0, 0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.6]]

	# for i in range(numPeaks):
	# 	scale = 2 * np.heaviside(x - means[i], 1)
	# 	r += amplitudes[i][0] * scale
	# 	g += amplitudes[i][1] * scale
	# 	b += amplitudes[i][2] * scale
	# 	a += amplitudes[i][3] * scale

	return r,g,b,a



def volumeRenderer(datacube):
	# Datacube Grid
	Nx, Ny, Nz = datacube.shape
	x = np.linspace(-Nx/2, Nx/2, Nx)
	y = np.linspace(-Ny/2, Ny/2, Ny)
	z = np.linspace(-Nz/2, Nz/2, Nz)
	points = (x, y, z)
	
	# Do Volume Rendering at Different Viewing Angles
	Nangles = 1
	for i in range(Nangles):
		
		print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
	
		# Camera Grid / Query Points -- rotate camera view
		angle = 2 * np.pi * i / Nangles
		N = Nx
		c = np.linspace(-N/2, N/2, N)
		qx, qy, qz = np.meshgrid(c,c,c)
		qxR = qx
		qyR = qy * np.cos(angle) - qz * np.sin(angle) 
		qzR = qy * np.sin(angle) + qz * np.cos(angle)
		qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
		
		# Interpolate onto Camera Grid
		#camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
		camera_grid = datacube

		# Do Volume Rendering
		image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
	
		count = 0
		for dataslice in camera_grid:
			count += 1
			print("Status: ", count, "/", Nx)
			r,g,b,a = transferFunction(dataslice)#np.log(dataslice))
			image[:,:,0] = a*r + (1-a)*image[:,:,0]
			image[:,:,1] = a*g + (1-a)*image[:,:,1]
			image[:,:,2] = a*b + (1-a)*image[:,:,2]
		
		image = np.clip(image, 0.0, 1.0)
		
		# Plot Volume Rendering
		print("Plotting volume render ...")
		plt.figure(figsize=(4,4), dpi=80)
		
		plt.imshow(image)
		plt.axis('off')
		
		# Save figure
		plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
		plt.close()

def main(argv):
	num = 0
	index = 0

	try:
		opts, args = getopt.getopt(argv, "hn:i:", ["ifile="])
	except getopt.GetoptError:
		print('test.py -i <index> -n <N>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -i <index> -n <N>')
			sys.exit()
		elif opt == "-i":
			index = int(arg)
		elif opt == "-n":
			num = int(arg)

	print("Index: ", index, " num: ", num)
	if(num == 0):
		print("Please insert valid num <N>")
		exit(1)


	#num = 512 # amount of points per side
	div = 3 # amount of divisions per side for spatial sort
	odlimit = 1.0 # all overdensities below this value are cut

	# load data
	base_path = Path(__file__).parent
	file_path = (base_path / "../overdensities_0.dat").resolve()
	
	df = loadData(file_path, index)
	#df = np.array([['3330.350000', '2.000006', '0.000259', '0-5-37_5.073954--0.202656-2.765944#0-5-38_5.134136--0.064139-2.782694#0-5-39_9.0-0.054905-2.787238#2-5-39_2.0-0.054905-2.787238#2-5-40_5.149941-0.054905-2.787238#']])

	# put data into usable format
	values = adjustData(df, odlimit)

	# extract positions
	positions = getOnlyPosition(values)
	print("amount of points:", len(positions))
	print("fraction of total volume:", np.format_float_positional(len(positions) / num**3 * 100, precision = 4), '%')


	plotODhist(values, 0.5)
	# prepare datacube for volume rendering
	datacube = np.zeros(shape = (num, num, num), dtype = float)

	for i in range(num):
		for j in range(num):
			for k in range(num):
				datacube[i, j, k] = 1

	for i in range(len(values)):
		posx, posy, posz = values[i][0]
		#print(posx, posy, posz)
		datacube[posx, posy, posz] = values[i][1][0]

	volumeRenderer(datacube)

if __name__ == '__main__':

	main(sys.argv[1:])
	exit(1)


	# posx = [pos[0] for pos in positions]
	# posy = [pos[1] for pos in positions]
	# posz = [pos[2] for pos in positions]
	
	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.scatter3D(posx, posy, posz, c = posz, cmap = 'viridis')

	# plt.show()

	# exit(1)

	# spatially_sorted = spatialSort(positions, div, num)

	# for i in range(div):
	# 	for j in range(div):
	# 		for k in range(div):
	# 			try:
	# 				print(len(spatially_sorted[i][j][k]))
	# 			except:
	# 				print(i, j, k)

	# groups = makeNestedList(div)
	# for i in range(div):
	# 	for j in range(div):
	# 		for k in range(div):
	# 			groups[i][j][k] = groupPoints(spatially_sorted[i][j][k])
	# 			print("finished ", i * div * div + j * div + k, " / ", div**3)

	# for i in range(div):
	# 	for j in range(div):
	# 		for k in range(div):
	# 			try:
	# 				print(len(groups[i][j][k]))
	# 			except:
	# 				print(i, j, k)

	# print(len(groups))
