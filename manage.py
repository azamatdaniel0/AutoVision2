from fastapi import FastAPI, WebSocket, Depends
from web.media_handler.routers import media_router

#If MQTT
# from .broker_handler import mqttc
# mqttc.loop_forever()


app = FastAPI()
app.include_router(media_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
