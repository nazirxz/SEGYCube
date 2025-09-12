@echo off
echo Starting SEG-Y FastAPI Server...
call SEGYEnv\Scripts\activate.bat
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8100
pause