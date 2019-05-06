import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb
class Point(object):
	def __init__(self, p):
		super(Point, self).__init__()
		#self.x = int(np.round(p[0]))
		#self.y = int(np.round(p[1]))
		self.x = p[0]
		self.y = p[1]


	def __add__(self, other):
		return Point((self.x + other.x, self.y + other.y))

	def __repr__(self):
		return '(%.2f, %.2f)' % (self.x, self.y)
	
	@staticmethod
	def get_slope(p1,p2):
		m = float(p2.y-p1.y)/(p2.x-p1.x)
		return m
	
	@staticmethod
	def get_distance(p1, p2):
		return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5
	
	@staticmethod
	def to_numpy():
		return np.float32([[pt.x, pt.y] for pt in pt_arr])

	@staticmethod
	def from_numpy(np_arr):
		return [Point(p) for p in np_arr]
		
	def to_tuple(self):
		return (self.x, self.y)


class Line(object):
	def __init__(self, p1=None, p2=None, m=None, b=None):
		super(Line, self).__init__()
		if m is None:
			assert(p1 is not None and p2 is not None)
			m = Point.get_slope(p1, p2)
		if b is None:
			b = -1*m*p1.x + p1.y
		self.m = m
		self.b = b
	
	def __repr__(self):
		return "y = %.4f*x + %.4f" % (self.m, self.b)

	def get_y(self, x):
		#return int(np.round(self.m * x + self.b))
		return self.m * x + self.b
	
	def get_x(self, y):
		return (y-self.b)/self.m
	
	def intersect(self, other):
		x = float(self.b - other.b)/(other.m - self.m)
		y = self.get_y(x)
		#print(x)
		#print(y, other.get_y(x))
		#return Point((int(np.round(x)),int(np.round(y))))
		return Point((x,y))


def get_clicks(img_path, num_clicks=None, text=None, print_pos=False):
	if text is not None:
		print(text)
	print('Press any key when finished clicking (press "r" to restart clicking).')
	img = cv2.imread(img_path)
	h = img.shape[0]
	w = img.shape[1]
	img = cv2.resize(img, (int(w/2), int(h/2)))
	clicks = []
	def draw_circle(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONUP:
			if print_pos:
				print('(%d, %d)' % (x, y))
			clicks.append((x, y))

			cv2.circle(img, (x,y), 5, (255, 0, 0), -1) 
			cv2.imshow('image', img)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', draw_circle)
	while(True):
		cv2.imshow('image', img)
		key = cv2.waitKey(0)
		if key == ord("r"):
			clicks = []
			img = cv2.imread(img_path)
			cv2.imshow('image', img)
		elif num_clicks is not None and len(clicks) < num_clicks:
			print('Keep clicking to get to %d clicks.' % num_clicks)
		else:
			if num_clicks is not None and len(clicks) > num_clicks:
				print('Using the first %d clicks.' % num_clicks)
				clicks = clicks[:num_clicks]
			break
	cv2.destroyAllWindows()
	return clicks

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("img_path", type=str, help="path to image")
	parser.add_argument("ref_height", type=float)
	opts = parser.parse_args()
	return opts

opts = parse_args()

clicks = get_clicks(img_path=opts.img_path, num_clicks=14,print_pos=True)

ax = Point.from_numpy(clicks[0:4])
ax.__repr__()
ay = Point.from_numpy(clicks[4:8])
az = Point.from_numpy(clicks[8:12])
#ay = Point.from_numpy(clicks[2:4])
#az = Point.from_numpy(clicks[4:6])

linex1 = Line(ax[0],ax[1])
vpx = Line(ax[2],ax[3]).intersect(Line(ax[0],ax[1]))
vpy = Line(ay[2],ay[3]).intersect(Line(ay[0],ay[1]))
vpz = Line(az[2],az[3]).intersect(Line(az[0],az[1]))

vl = Line(vpx,vpy)


rbot = Point(clicks[8])
mbot = Point(clicks[12])
rtop = Point(clicks[9])
mtop = Point(clicks[13])
mcenter = Point(((mbot.x+mtop.x)/2, (mbot.y+mtop.y)/2))
vpz_mcenter = Line(mcenter, vpz)

mbot = Point((vpz_mcenter.get_x(clicks[12][1]), clicks[12][1]))
mtop = Point((vpz_mcenter.get_x(clicks[13][1]), clicks[13][1]))

ref = Line(rbot, rtop)
mea = Line(mtop, mbot)
bot_line = Line(rbot, mbot)
u = bot_line.intersect(vl)

mtop_u = Line(mtop, u)
mea_parallel = Line(rbot, Point((rbot.x+1.0, rbot.y+mea.m)))
mtop_rtop = Line(rtop, mtop)
t1_hat = mtop_u.intersect(ref)
t2 = mtop_rtop.intersect(mea_parallel)
t1 = mea_parallel.intersect(mtop_u)



rd = t2.get_distance(t2, rbot)
md = t1.get_distance(t1, rbot)
print(rd)
print(md)
height = (md/rd)*opts.ref_height
print(height)






