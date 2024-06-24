from __future__ import annotations
import time
import logging
from multiprocessing import Queue
import multiprocessing as mp
from pathlib import Path
import sys
from time import sleep
import typing as t

import click
import math
import numpy as np
from pydantic import BaseModel
from pythonosc.udp_client import SimpleUDPClient
import questionary
from ruamel.yaml import YAML

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


class OSC:
    """
    Manage OSC protocol / connection
    """

    def __init__(self, osc_server_ip: str, osc_server_port: int):
        self.client: SimpleUDPClient = SimpleUDPClient(osc_server_ip,
                                                       osc_server_port)

    @classmethod
    def from_config(cls, config: Config) -> OSC:
        return cls(config.osc_server_ip, config.osc_server_port)

    def send(self, addr: str, value: float | int = 1):
        """Send message to OSC bus"""
        self.client.send_message(addr, value)  # Send float message

    def send_paths(self):
        for path in "RGB":
            self.send(path, 1)
            sleep(0.05)

    def send_rgb(self, r: float, g: float, b: float):
        log.info(f"[OSC] Send RGB values: R={r:.2f}, G={g:.2f}, B={b:.2f}")
        self.send("R", int(r))
        self.send("G", int(g))
        self.send("B", int(b))


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

            ndi_source = questionary.select("Please select an ndi device",
                                            choices=[
                                                questionary.Choice(
                                                    f"{ndi_source.ndi_name} [{ndi_source.url_address}]",
                                                    ndi_source)
                                                for ndi_source in sources
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

        if t == ndi.FRAME_TYPE_VIDEO:
            log.debug(f'[NDI] Video data received ({v.xres},{v.yres})')
            frame = np.copy(v.data)
            ndi.recv_free_video_v2(self.ndi_recv, v)
            return frame
        elif t == ndi.FRAME_TYPE_AUDIO:
            ndi.recv_free_audio_v3(self.ndi_recv, a)
        elif t == ndi.FRAME_TYPE_METADATA:
            metadata = m.data
            log.info(f'[NDI] Metadata received: {metadata} ')
            ndi.recv_free_metadata(self.ndi_recv, m)
        elif t == ndi.FRAME_TYPE_NONE:
            log.debug('[NDI] No new data received')
        else:
            log.debug('[NDI] Other data received')

    def queue_frames(self, queue: Queue):
        while True:
            frame = self.recv_frame()
            if frame is None:
                continue
            queue.put(frame)


def background_frame_to_osc(config: Config, frame_queue: Queue[np.ndarray]):
    osc = OSC.from_config(config)

    x_old = -1
    y_old = -1
    mask = np.zeros(0, dtype=bool)
    while True:
        frame = frame_queue.get()

        per = 0.20
        # Yes. y than x
        y, x, _ = frame.shape
        if (x_old, y_old) != (x, y):
            y_b = math.floor(y * per)
            x_b = math.floor(x * per)
            mask = np.ones((y, x), dtype=bool)
            mask[y_b:y - y_b + 1, x_b:x - x_b + 1] = False

        avg_rgb = np.average(frame[mask], axis=0).round()
        osc.send_rgb(avg_rgb[0], avg_rgb[1], avg_rgb[2])


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
def main(
    config_path: Path,
    send_paths: bool,
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

    frame_queue = Queue(8)

    # Start background osc worker process
    frame_process = mp.Process(target=background_frame_to_osc,
                               args=(config, frame_queue),
                               daemon=True)
    frame_process.start()

    while True:
        frame = ndi_provider.recv_frame()
        if frame is None:
            continue
        log.debug(f"QUEUE-SIZE: {frame_queue.qsize()}")
        frame_queue.put(frame)


if __name__ == "__main__":
    main()
