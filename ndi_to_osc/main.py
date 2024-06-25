from __future__ import annotations
from collections import defaultdict, deque
import logging
from multiprocessing import Queue
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter, sleep
import typing as t

import click
import math
from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
from pythonosc.udp_client import SimpleUDPClient
import questionary
from ruamel.yaml import YAML

from ndi_to_osc.segments import Segment

if t.TYPE_CHECKING:
    ndi: t.Any = None
else:
    import NDIlib as ndi

log = logging.getLogger(__name__)

OSCAdress = str


class OSCPayload(t.NamedTuple):
    adress: OSCAdress
    value: int | float


class Config(BaseModel):
    osc_server_ip: str
    osc_server_port: int

    blackout_threshold: int = 0

    max_change_rate_per_frame: float = 10

    segments: list[Segment]


class OSC:
    """
    Manage OSC protocol / connection
    """

    def __init__(self, osc_server_ip: str, osc_server_port: int, blackout_threshold: int):
        self.client: SimpleUDPClient = SimpleUDPClient(osc_server_ip,
                                                       osc_server_port)
        self.blackout_threshold = blackout_threshold

    @classmethod
    def from_config(cls, config: Config) -> OSC:
        return cls(config.osc_server_ip, config.osc_server_port, config.blackout_threshold)

    def send(self, addr: str, value: float | int = 1):
        """Send message to OSC bus"""
        self.client.send_message(addr, value)  # Send float message

    def send_paths(self):
        for path in "RGB":
            self.send(path, 1)
            sleep(0.05)

    def send_rgb(self, rgb: np.ndarray, segment: str):
        log.debug(f"{rgb}, {rgb.dtype}, {rgb.shape}")

        if sum(rgb <= self.blackout_threshold) >= 2:
            rgb = np.array((0,0,0))
            log.debug(f"[OSC] Send RGB values for segment '{segment}': R={rgb[0]}, G={rgb[1]}, B={rgb[2]} [Blackout Override]")
        else:
            log.debug(f"[OSC] Send RGB values for segment '{segment}': R={rgb[0]}, G={rgb[1]}, B={rgb[2]}")


        self.send(f"segment_{segment}_R", int(rgb[0]))
        self.send(f"segment_{segment}_G", int(rgb[1]))
        self.send(f"segment_{segment}_B", int(rgb[2]))


class NDI:

    def __init__(self):
        log.info('[NDI] Initializing...')
        if not ndi.initialize():
            return 0

        ndi_find = ndi.find_create_v2()

        if ndi_find is None:
            return 0

        sources = []
        while True:
            log.info('[NDI] Looking for sources..')
            ndi.find_wait_for_sources(ndi_find, 1000)
            sources = ndi.find_get_current_sources(ndi_find)
            if not sources:
                continue

            ndi_source = questionary.select(
                "Please select an ndi device",
                choices=[
                    questionary.Choice(
                        f"{ndi_source.ndi_name} [{ndi_source.url_address}]",
                        ndi_source) for ndi_source in sources
                ]).ask()
            if not ndi_source:
                log.info('[NDI] No source choosen..')
                continue
            else:
                log.info(f'[NDI] Source {ndi_source} chosen..')
                break
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA

        self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)

        if self.ndi_recv is None:
            return 0

        log.info('[NDI] Connecting..')
        ndi.recv_connect(self.ndi_recv, ndi_source)

        ndi.find_destroy(ndi_find)

        log.info('[NDI] Initializing complete..')

    def __del__(self):
        log.info('[NDI] Deinitalizing..')
        ndi.recv_destroy(self.ndi_recv)
        ndi.destroy()
        log.info('[NDI] Deinitalizing complete!')

    def recv_frame(self, rate: int = 250) -> np.ndarray | None:
        t, v, a, m = ndi.recv_capture_v2(self.ndi_recv, rate)

        match t:
            case ndi.FRAME_TYPE_VIDEO:
                log.debug(f'[NDI] Video data received ({v.xres},{v.yres})')
                frame = np.copy(v.data)
                ndi.recv_free_video_v2(self.ndi_recv, v)
                return frame
            case ndi.FRAME_TYPE_AUDIO:
                ndi.recv_free_audio_v2(self.ndi_recv, a)
            case ndi.FRAME_TYPE_METADATA | ndi.FRANE_TYPE_STATUS_CHANGE:
                if m.length:
                    metadata = m.data
                else:
                    metadata = None
                log.debug(f'[NDI] Metadata received: {metadata} ')
                ndi.recv_free_metadata(self.ndi_recv, m)
            case ndi.FRAME_TYPE_NONE:
                log.debug('[NDI] No new data received')
            case _:
                log.debug('[NDI] Other data received')

    def queue_frames(self, queue: Queue):
        while True:
            frame = self.recv_frame()
            if frame is None:
                continue
            queue.put(frame)


def background_frame_to_osc(config: Config, frame_queue: Queue[np.ndarray]):
    osc = OSC.from_config(config)

    segement_last_rgb: dict[str,np.ndarray] = {}

    max_change_rate_per_frame = (config.max_change_rate_per_frame
                                 * 0.01 * np.array((255,255,255,255)))

    last_frame = None

    while True:
        t1 = perf_counter()

        frame = frame_queue.get()
        if frame is None:
            if last_frame is not None:
                frame = last_frame
            else:
                continue

        t2_framequeue = perf_counter()

        t_segment_1 = perf_counter()
        for segment in config.segments:

            rgb = segment.frame_to_rgb(frame)

            if segment.name in segement_last_rgb:
                last_rgb = segement_last_rgb[segment.name]
                current_change = rgb - last_rgb
                log.debug(f"[COMPUTE] {current_change=},{last_rgb=},{rgb=},{max_change_rate_per_frame=}")
                if (np.abs(current_change) >= max_change_rate_per_frame).any():
                    limited_rgb = last_rgb + (max_change_rate_per_frame * np.sign(current_change))
                    limited_rgb = np.minimum(np.maximum(limited_rgb,0),255)
                    log.debug(f"[COMPUTE] Color-Jump: Limiting change from {rgb} to {limited_rgb}")

                    rgb = limited_rgb

            segement_last_rgb[segment.name] = rgb

            osc.send_rgb(rgb, segment=segment.name)

        t_segment_2 = perf_counter()

        last_frame = frame

        t2 = perf_counter()

        log.debug(f"[COMPUTE] Timing {t2-t1:.2f}s\n"
                  f"          Frame-Queue {t2_framequeue-t1:.2f}s\n"
                  f"          Segments {t_segment_2-t_segment_1:.2f}s")



@click.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--send-paths",
    is_flag=True,
    help="Send all OSC paths, usefull with qlcplus profile wizard.",
)
@click.option(
    "--display",
    is_flag=True,
)
def main(
    config_path: Path,
    send_paths: bool,
    display: bool,
):
    basic_format: str = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=basic_format,
                        stream=sys.stdout)

    # Load config
    with open(config_path) as f:
        config = Config.model_validate(YAML().load(f))

    if send_paths:
        osc = OSC.from_config(config)
        while True:
            osc.send_paths()
            sleep(0.5)

    # Init ndi
    ndi_provider = NDI()

    if display:
        frame = None
        while frame is None:
            frame = ndi_provider.recv_frame()

        dim = list(frame.shape)
        dim[2] = 4
        frame = np.resize(frame,dim)
        log.info(str(frame.shape))
        frame[:,:,3] = 100

        for segment in config.segments:
            mask = segment.gen_mask((dim[1],dim[0])).transpose()
            frame[mask,3] = 255

        plt.imshow(frame, interpolation=None)
        plt.show()
        return

    frame_queue = Queue(8)

    # Start background osc worker process
    frame_process = mp.Process(target=background_frame_to_osc,
                               args=(config, frame_queue),
                               daemon=True)
    frame_process.start()

    while True:
        log.debug(f"QUEUE-SIZE: {frame_queue.qsize()}")
        frame = ndi_provider.recv_frame()
        frame_queue.put(frame)


if __name__ == "__main__":
    main()
