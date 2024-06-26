import sys
import time
import numpy as np
import math
import random
import typing as t

from ndi_to_osc.utils import fade_rgb

if t.TYPE_CHECKING:
    ndi: t.Any = None
else:
    import NDIlib as ndi

def main():

    if not ndi.initialize():
        return 0

    ndi_send = ndi.send_create()

    if ndi_send is None:
        return 0

    img = np.zeros((90, 160, 4), dtype=np.uint8)

    video_frame = ndi.VideoFrameV2()

    video_frame.data = img
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA

    video_length = 10 #seconds
    frame_rate = 12 #fps

    n_frames = math.ceil(frame_rate*video_length)

    n_frames_red_to_green = math.floor(n_frames/3)
    n_frames_green_to_blue = n_frames_red_to_green
    n_frames_blue_to_red = n_frames - n_frames_green_to_blue - n_frames_red_to_green
    fades_red_green = fade_rgb((255,0,0),(0,255,0),steps=n_frames_red_to_green)
    fades_green_blue = fade_rgb(fades_red_green[-2],(0,0,255),steps=n_frames_green_to_blue)
    fades_blue_red =  fade_rgb(fades_green_blue[-2],(255,0,0),steps=n_frames_blue_to_red)

    fades = np.concatenate((fades_red_green,fades_green_blue,fades_blue_red),
                           axis=0)

    # fades = np.array([*[(255,50,50)]*20,*[(0,50,50)]*20]*5)


    try:
        while True:
            start_send = time.time()

            send_correction = 0
            for idx in range(n_frames):
                t1 = time.time()
                # time.sleep(random.randint(0,4)*0.005)
                # img.fill(255 if idx % 2 else 0)
                img[:,:,0] = fades[idx,0]
                img[:,:,1] = fades[idx,1]
                img[:,:,2] = fades[idx,2]
                img[:,:,3] = 255
                t2 = time.time()
                td = t2-t1

                time.sleep(max((1/frame_rate)-td-send_correction,0))

                t1_send = time.time()
                ndi.send_send_video_v2(ndi_send, video_frame)
                t2_send = time.time()
                send_correction = t2_send - t1_send

            print(f'{n_frames} frames sent, at {(n_frames / (time.time() - start_send)):.3f}fps')
    finally:
        ndi.send_destroy(ndi_send)
        ndi.destroy()
        return 0

if __name__ == "__main__":
    sys.exit(main())
