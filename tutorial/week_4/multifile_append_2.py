from pydantic import BaseModel, Field

class Output(BaseModel):
    name: str = Field(description="The revised article in traditional Chinese (繁體中文), please do not include the title.")