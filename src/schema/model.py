from strawberry import type, field


@type
class Model:
    label: str = field(description="The name of the model.")
    value: str = field(description="The value of the model.")
