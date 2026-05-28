import asyncio
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(r"c:/Users/aadeshpande/Desktop/Sarla-Project/Backend/Backend").resolve()))

from local_detection import analyze_pid_image


class _DummyEngine:
    def ocr(self, _image, cls=True):
        return []


async def main() -> None:
    if os.getenv("OCR_SMOKE_FAST", "0") == "1":
        import local_detection

        local_detection.get_ocr_engine.cache_clear()
        local_detection.get_ocr_engine = lambda: _DummyEngine()  # type: ignore[assignment]

    image = Image.new("RGB", (800, 800), color="white")
    result = await asyncio.to_thread(analyze_pid_image, image)
    print("Counts:", result["counts"])
    print("Industry:", result["industry"])
    print("Coordinates:", len(result["coordinates"]["root"]["children"]))


if __name__ == "__main__":
    asyncio.run(main())
