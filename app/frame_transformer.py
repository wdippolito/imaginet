import cv2
import numpy as np

class FrameTransformer:
	def __init__(self):
		pass
	
	def to_grayscale(self, frame):
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray_frame

	def calculate_optical_flow(self, prev_frame, next_frame):
		# next_gray = self.to_grayscale(next_frame)
		# prev_gray = self.to_grayscale(prev_frame)
		flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 1, 40, 7, 5, 1.1, 0)
		return flow

	def apply_optical_flow(self, prev_frame, flow):
		h,w = prev_frame.shape[:2]
		print(h)
		flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
		flow_map = np.stack((flow_map_x, flow_map_y), axis=2) + flow
		flow_map = flow_map.astype(np.float32)
		interpolated_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_LINEAR)
		return interpolated_frame
	