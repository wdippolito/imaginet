import cv2

class VideoProcessor:
	def __init__(self, input_path, output_path):
		self.cap = cv2.VideoCapture(input_path)
		self.frame_width = int(self.cap.get(3))
		self.frame_height = int(self.cap.get(4))

	def get_next_frame(self):
		ret, frame = self.cap.read()
		return ret, frame

	def get_frame_width(self):
		return self.frame_width
	
	def get_frame_height(self):
		return self.frame_height
	