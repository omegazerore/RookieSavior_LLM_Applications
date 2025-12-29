import importlib

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import chain


story_telling_module = importlib.import_module("tutorial.LLM+Langchain.Week-8.logic.story_telling")
story_telling_pipeline = story_telling_module.story_pipeline(story_telling_module.system_template)

image_generation_module = importlib.import_module("tutorial.LLM+Langchain.Week-8.logic.image_generation")
image_create_pipeline = image_generation_module.image_create_pipeline(image_generation_module.system_template)
image_edit_pipeline = image_generation_module.image_edit_pipeline(image_generation_module.system_template)
# image_input = image_generation_module.Input
image_output = image_generation_module.Output


audio_generation_module = importlib.import_module("tutorial.LLM+Langchain.Week-8.logic.audio_generation")
tts_synthesizer = audio_generation_module.tts_synthesizer
tts_input = audio_generation_module.Input


@chain
def image_router(kwargs):

    if len(kwargs.get('image_io')) == 0:
        pipeline = image_create_pipeline
        print("***create***")
    else:
        pipeline = image_edit_pipeline
        print("***edit***")

    return pipeline
    
app = FastAPI(title="Audio book application",
              version="1.0",
              description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    story_telling_pipeline,
    path="/story_telling"
)

image_generation_pipeline = image_router.with_types(
                                                    #input_type=image_input, 
                                                    output_type=image_output
)

add_routes(
    app,
    image_generation_pipeline,
    path="/image_generation"
)

audio_pipeline = tts_synthesizer.with_types(input_type=tts_input)

add_routes(
    app,
    tts_synthesizer,
    path="/audio_generation"
)


if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=8080)