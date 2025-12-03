"""
Shared FastAPI Dependencies

Common dependency injection functions used across routers.
"""

from fastapi import Request


def get_api_manager(request: Request):
    """
    Get API manager from app state.

    This is the shared dependency for accessing the APIManager instance.
    Use with FastAPI's Depends() in route functions.

    Example:
        @router.get("/status")
        async def get_status(api_manager = Depends(get_api_manager)):
            return api_manager.get_status()
    """
    return request.app.state.api_manager
