from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from util.auth import verifyAccessToken

class AuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS": # preflight pass
            return Response(status_code=200)

        # list of unprotected routes
        skipMid = ["/api/user/login", "/api/user/register"]

        # skip login and sign in routes
        if request.url.path in skipMid:
            return await call_next(request) 
        
        # extract token from the cookie
        token = request.cookies.get("access_token")
        
        # if token is not present in the cookies
        if not token:
            raise HTTPException(status_code=401, detail="Missing authentication token")
        
        # verify the token
        try:
            payload = verifyAccessToken(token)
                
            # store user information in the request.state so routes can access it
            request.state.user = payload
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # continue to the requested route
        response = await call_next(request)
        return response