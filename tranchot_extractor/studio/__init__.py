"""
Tranchot Label Studio Web Application.
"""

def launch_studio(host: str = "127.0.0.1", port: int = 8000):
    """Launches the Label Studio server using uvicorn with auto-reload."""
    import uvicorn
    uvicorn.run("tranchot_extractor.studio.server:app", host=host, port=port, reload=False)

__all__ = ["launch_studio"]
