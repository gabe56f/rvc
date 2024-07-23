import logging
from warnings import filterwarnings

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
from starlette.datastructures import UploadFile
from strawberry.file_uploads import Upload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.schema.types import Query, Mutations, Subscriptions
from src.schema.audio import router as audio_router


filterwarnings("ignore")

schema = strawberry.Schema(
    query=Query,
    mutation=Mutations,
    subscription=Subscriptions,
    scalar_overrides={UploadFile: Upload},
)
gql_router = GraphQLRouter(
    schema=schema,
    path="/graphql",
    subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL],
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route("/graphql", gql_router)
app.include_router(audio_router, tags=["audio"])
app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")


if __name__ == "__main__":
    logging.getLogger("uvicorn.asgi").setLevel("DEBUG")
    uvicorn.run(app, host="0.0.0.0", port=8000)
