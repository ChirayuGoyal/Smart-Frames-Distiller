"""
server/routers/streams.py — RTSP stream management endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, status
from server.schemas import StreamCreateRequest, StreamResponse

router = APIRouter(prefix="/v1/streams", tags=["streams"])


@router.post("", response_model=StreamResponse, status_code=status.HTTP_201_CREATED)
async def create_stream(request: Request, body: StreamCreateRequest) -> StreamResponse:
    """Start processing a live RTSP camera stream."""
    return request.app.state.stream_manager.start_stream(
        rtsp_url=body.rtsp_url,
        site_id=body.site_id,
        camera_id=body.camera_id,
        options=body.options,
    )


@router.get("", response_model=list[StreamResponse])
async def list_streams(request: Request) -> list[StreamResponse]:
    """List all active and stopped RTSP streams and their stats."""
    return request.app.state.stream_manager.list_streams()


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(request: Request, stream_id: str) -> StreamResponse:
    """Get status and live stats of a specific RTSP stream."""
    return request.app.state.stream_manager.get_stream(stream_id)


@router.delete("/{stream_id}", response_model=StreamResponse)
async def delete_stream(request: Request, stream_id: str) -> StreamResponse:
    """Stop processing an active RTSP stream and return its final stats."""
    return request.app.state.stream_manager.stop_stream(stream_id)
