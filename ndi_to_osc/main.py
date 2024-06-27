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
        if sum(rgb <= self.blackout_threshold) >= 2:
            log.debug(f"[OSC] Send RGB values for segment '{segment}': R={rgb[0]}|0, G={rgb[1]}|0, B={rgb[2]}|0 [Blackout Override]")
            rgb = np.array((0,0,0))
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

    def recv_frame(self, rate: int = 100) -> np.ndarray | None:
        t, v, a, m = ndi.recv_capture_v3(self.ndi_recv, rate)

        match t:
            case ndi.FRAME_TYPE_VIDEO:
                log.debug(f'[NDI] Video data received ({v.xres},{v.yres})')
                frame = np.copy(v.data)
                ndi.recv_free_video_v2(self.ndi_recv, v)
                return frame
            case ndi.FRAME_TYPE_AUDIO:
                ndi.recv_free_audio_v3(self.ndi_recv, a)
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

BASE_CHANGE_RATE = np.array((0.1,0.1,0.1,0)) * 1
def smoothed(last_rgbs: np.ndarray):
    max_var = 65000 #
    changes = last_rgbs[1:] - last_rgbs[:-1]
    var = np.var(changes**2,axis=0)
    factor = 100 - (var / max_var)
    last_rgb = last_rgbs[-1]
    log.debug(f"[COMPUTE] {last_rgb=}, {var=}, {factor=}")

    if (factor[:3] > 90).all():
        log.debug(f"[COMPUTE] Skipping smoothing")
        return last_rgb

    change = np.round(BASE_CHANGE_RATE * factor * np.sign(changes[-1]))
    limited = np.minimum(np.maximum(last_rgbs[-2] + change,0),255)
    log.debug(f"[COMPUTE] Smooth {last_rgb} from {last_rgb[-2]} to {limited} with change {change}")

    return limited


def background_frame_to_osc(config: Config,
                            frame_queue: Queue[np.ndarray]):

    osc = OSC.from_config(config)

    segment_last_rgbs: dict[str,deque[np.ndarray]] \
            = defaultdict(lambda: deque([np.array((0.,0.,0.,0.))]*2))
    segment_last_rgbs_alt: dict[str,deque[np.ndarray]] \
            = defaultdict(lambda: deque([np.array((0.,0.,0.,0.))]*2))

    while True:
        start_frame_time = perf_counter()

        frame = frame_queue.get()

        for segment in config.segments:
            last_rgbs = segment_last_rgbs[segment.name]
            last_rgbs_alt = segment_last_rgbs_alt[segment.name]

            if frame is not None:
                rgb, rgb_alt = segment.frame_to_rgb(frame)
                last_rgbs.append(rgb)
                if rgb_alt is not None and segment.alt_mode:
                    last_rgbs_alt.append(rgb_alt)
            else:
                rgb = last_rgbs[-1]
                if segment.alt_mode:
                    rgb_alt = last_rgbs_alt[-1]

            if len(last_rgbs) > 20:
                last_rgbs.popleft()

            if segment.smooth:
                rgb = smoothed(np.array(last_rgbs))
                last_rgbs[-1] = rgb
                if segment.alt_mode:
                    rgb_alt = smoothed(np.array(last_rgbs_alt))
                    last_rgbs_alt[-1] = rgb_alt

            osc.send_rgb(rgb, segment=segment.name)

            if segment.alt_mode:
                osc.send_rgb(rgb, segment=f"{segment.name}_alt")

        stop_frame_time = perf_counter()
        log.debug(f"[COMPUTE] Frame took {stop_frame_time-start_frame_time:.3f}")


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
        if frame is not None:
            frame_queue.put(frame)


if __name__ == "__main__":
    main()
