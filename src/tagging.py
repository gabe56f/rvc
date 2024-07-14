from mutagen.flac import FLAC


def write_tags_to_flac(file_path, tags: dict):
    audio = FLAC(file_path)
    for k, v in tags.items():
        audio[k] = v
    audio.save()


def get_tags(file_path):
    tags = FLAC(file_path)
    return tags.tags
