from __future__ import annotations

from pydantic import BaseModel


class CameraIn(BaseModel):
    camera_id: str
    name: str
    rtsp_url: str
    location: str = ""
    enabled: bool = True


class CameraOut(CameraIn):
    status: str = "offline"


