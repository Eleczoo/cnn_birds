import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import (
    FieldDescriptionType,
    ExecutionUnitTagName,
    ExecutionUnitTagAcronym,
)
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
# TODO: 1. ADD REQUIRED IMPORTS (ALSO IN THE REQUIREMENTS.TXT)
import tensorflow as tf
from scipy import signal
import numpy as np
from PIL import Image
import librosa

settings = get_settings()


class MyService(Service):
    # TODO: 2. CHANGE THIS DESCRIPTION
    """
    Service analyzing audio signal to detect bird species.
    Sampling rate HAS to be 22050 Hz and the audio HAS to be 3 seconds long.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            # TODO: 3. CHANGE THE SERVICE NAME AND SLUG
            name="Bird Detection",
            slug="bird-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            # TODO: 4. CHANGE THE INPUT AND OUTPUT FIELDS, THE TAGS AND THE HAS_AI VARIABLE
            data_in_fields=[
                FieldDescription(
                    name="audio",
                    type=[
                        FieldDescriptionType.AUDIO_MP3,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            tags=[
                # ExecutionUnitTag(
                #    name=ExecutionUnitTagName.
                #    acronym=ExecutionUnitTagAcronym.AUDIO_PROCESSING,
                # ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

        # TODO: 5. INITIALIZE THE MODEL (BY IMPORTING IT FROM A FILE)
        self._model = tf.keras.models.load_model("../model.keras")

    # TODO: 6. CHANGE THE PROCESS METHOD (CORE OF THE SERVICE)
    def process(self, data):
        class_names = [
            "American_Robin",
            "Bewick's_Wren",
            "Northern_Cardinal",
            "Northern_Mockingbird",
            "Song_Sparrow",
        ]

        def scale_minmax(X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled

        raw_audio = data["audio"].data
        with open("file.mp3", "wb") as f:
            f.write(raw_audio)

        audio, sr = librosa.load("file.mp3")

        # self._logger.info(message=f"audio len : {data['audio']}")
        self._logger.info(message=f"audio len : {len(audio)}")
        self._logger.info(message=f"audio sr : {sr}")
        # Convert the bytes into numpy array
        # audio = np.frombuffer(audio, dtype=np.int16)
        # self._logger.info(message=f"audio len after: {len(audio)}")
        sampling_rate = 22050

        frequencies, times, spectrogram = signal.spectrogram(audio, sampling_rate)
        spectrogram = 255 - scale_minmax(np.log(spectrogram), 0, 255).astype(np.uint8)

        height, width = spectrogram.shape
        self._logger.info(message=f"spectrogram shape : {spectrogram.shape}")

        predictions = self._model.predict(spectrogram.reshape((1, height, width, 1)))
        predicted_class = class_names[np.argmax(predictions)]

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(
                data=predicted_class,
                type=FieldDescriptionType.TEXT_PLAIN,
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


# TODO: 7. CHANGE THE API DESCRIPTION AND SUMMARY
api_description = """Bird Detection Service
You NEED to provide an audio file with a sampling rate of 22050 Hz and a duration of 3 seconds.
"""
api_summary = """My service
bla bla bla...
"""

# Define the FastAPI application with information
# TODO: 8. CHANGE THE API TITLE, VERSION, CONTACT AND LICENSE
app = FastAPI(
    lifespan=lifespan,
    title="Bird detection API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
