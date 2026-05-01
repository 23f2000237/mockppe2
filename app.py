from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import joblib
import pandas as pd

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# -----------------------------
# 🔭 Tracing Setup
# -----------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# -----------------------------
# 🪵 Structured Logging Setup
# -----------------------------
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# 🚀 FastAPI App
# -----------------------------
app = FastAPI(title="🌸 Iris Classifier API")

# -----------------------------
# 📦 Model Loading
# -----------------------------
model = None

app_state = {
    "is_ready": False,
    "is_alive": True
}

@app.on_event("startup")
async def startup_event():
    global model
    try:
        time.sleep(2)  # simulate loading delay
        model = joblib.load("model.joblib")
        app_state["is_ready"] = True

        logger.info(json.dumps({
            "event": "startup_complete",
            "status": "model_loaded"
        }))
    except Exception as e:
        app_state["is_alive"] = False
        logger.exception(json.dumps({
            "event": "startup_failed",
            "error": str(e)
        }))

# -----------------------------
# 📊 Input Schema
# -----------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# -----------------------------
# ❤️ Health Probes
# -----------------------------
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# -----------------------------
# ⏱ Middleware (Latency)
# -----------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# -----------------------------
# ❌ Global Exception Handler
# -----------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "trace_id": trace_id
        },
    )

# -----------------------------
# 🏠 Root Endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

# -----------------------------
# 🔮 Prediction Endpoint
# -----------------------------
@app.post("/predict/")
def predict_species(data: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = data.dict()
            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]

            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "prediction": str(prediction),
                "latency_ms": latency,
                "status": "success"
            }))

            return {
                "predicted_class": prediction
            }

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
