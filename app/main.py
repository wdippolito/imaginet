from video_processor import VideoProcessor
from frame_transformer import FrameTransformer
import cv2
import os
import base64
import requests
from io import BytesIO




def main():
    input_video_path = 'clips/clip2.mov'
    output_video_path = 'clips/processed/'

    vp = VideoProcessor(input_video_path, output_video_path)
    ft = FrameTransformer()

    prev_frame = None
    count = 0
    while True:

        ret, frame = vp.get_next_frame()

        if not ret:
            break

        frame = ft.to_grayscale(frame)

        # display frame as is
        if count % 2 == 0:
            output_frame = frame

        # interpolate every other frame     
        else:
            next_ret, next_frame = vp.get_next_frame()
            next_frame = ft.to_grayscale(next_frame)
            if next_ret:
                flow = ft.calculate_optical_flow(prev_frame, next_frame)
                output_frame = ft.apply_optical_flow(prev_frame, flow)
                prev_frame = next_frame  # Update the prev_frame for the next iteration
                count += 1  # Increment frame_count for the skipped frame
            else:
                break


        cv2.imshow('video player', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1
        prev_frame = frame

    cv2.destroyAllWindows()

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

if __name__ == "__main__":
    main()
