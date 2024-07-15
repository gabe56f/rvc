from strawberry import type, field


@type
class Entry:
    label: str = field(description="The name of the entry.")
    value: str = field(description="The value of the entry.")
