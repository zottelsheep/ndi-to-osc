from __future__ import annotations
from abc import abstractmethod
import logging
from math import ceil
from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator
import numpy as np

log = logging.getLogger(__name__)

class Segment(BaseModel):
    name: str
    mask: list[Annotated[Circle | Block, Field(discriminator="type")]]
    delay: int = 0
    mode: Literal["average","average-invert"] = "average"

    def model_post_init(self, __context) -> None:
        self._mask_cache = {}
        return super().model_post_init(__context)

    def gen_mask(self, dimensions: tuple[int,int]):
        if dimensions in self._mask_cache:
            return self._mask_cache[dimensions]

        log.debug(f"[SEGMENT] Gen Mask for {self.name}")
        mask = np.zeros(dimensions, dtype=bool)

        for sub_mask_config in self.mask:
            sub_mask = sub_mask_config.gen_mask(dimensions)
            mask = mask | sub_mask

        self._mask_cache[dimensions] = mask

        return mask

    def frame_to_rgb(self, frame:np.ndarray) -> np.ndarray:

        # Yes. y than x
        y, x, _ = frame.shape
        mask = self.gen_mask((x,y)).transpose()

        match self.mode:
            case "average":
                rgb = np.average(frame[mask], axis=0).round()
            case "average-invert":
                rgb = 255 - np.average(frame[mask], axis=0).round()

        return rgb


class Mask(BaseModel):
    x: int = 0
    y: int = 0

    @abstractmethod
    def gen_mask(self, dim: tuple[int,int]) -> np.ndarray:
        pass

class Circle(Mask):
    type: Literal["circle"]
    radius: int
    inner_radius: int = 0

    def gen_mask(self, dim: tuple[int,int]) :
        radius = int(dim[0] * self.radius*0.01)
        inner_radius = int(dim[0] * self.inner_radius*0.01)

        # use relative coords from center
        x = int((50 + self.x)*0.01*dim[0])
        y = int((50 + self.y)*0.01*dim[1])

        mask = self.create_circular_mask(dim=dim,
                                         radius=radius,
                                         center=(x,y))

        if inner_radius:
            inner_mask = self.create_circular_mask(dim=dim,
                                                   radius=inner_radius,
                                                   center=(x,y))

            mask = mask ^ inner_mask

        return mask

    @staticmethod
    def create_circular_mask(dim: tuple[int,int],
                             radius: int | None = None,
                             center: tuple[int,int] | None =None):

        if center is None: # use the middle of the image
            center = (int(dim[0]/2), int(dim[0]/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], dim[0]-center[0], dim[1]-center[1])

        Y, X = np.ogrid[:dim[1], :dim[0]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask.transpose()

class Block(Mask):
    type: Literal["block"]
    height: int
    width: int
    location: Literal["top","bottom","left", "right"] | None = None
    inset: int = 0

    @model_validator(mode='after')
    def parse_location_setting(self):
        match self.location:
            case "top":
                self.x = 0
                self.y = -45
            case "bottom":
                self.x = 0
                self.y = +45
            case "left":
                self.x = -45
                self.y = 0
            case "right":
                self.x = +45
                self.y = 0
        return self

    def gen_mask(self, dim: tuple[int, int]) -> np.ndarray:

        # use relative coords from center
        x = ceil((50 + self.x)*0.01*dim[0])
        y = ceil((50 + self.y)*0.01*dim[1])

        width = ceil(self.width*0.01*dim[0])
        height = ceil(self.height*0.01*dim[1])

        mask = self._generate_rectangle(dim,
                                        (x,y),
                                        height,
                                        width)
        # log.info(f"{self.location=},{x=},{y=},{width,}{height,}\n{mask}")

        if self.inset:
            width = int((self.width-self.inset)*0.1*dim[0])
            height = int((self.height-self.inset)*0.1*dim[1])

            inset_mask = self._generate_rectangle(dim,
                                                  (x,y),
                                                  height,
                                                  width)

            mask = mask ^ inset_mask

        return mask

    @staticmethod
    def _generate_rectangle(dim: tuple[int,int],
                            center: tuple[int,int],
                            height: int,
                            width: int
                            ):
        mask = np.zeros(dim, dtype=bool)

        half_width = ceil(width / 2)
        half_height = ceil(height / 2)

        mask[max(0,center[0]-half_width):min(dim[0],center[0]+half_width),
             max(0,center[1]-half_height):min(dim[1],center[1]+half_height)-1,
             ] = True

        return mask



