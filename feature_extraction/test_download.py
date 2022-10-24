from PIL import Image
from PIL import ImageFile
import io
import requests
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

response = requests.get(
    "https://pbs.twimg.com/media/FMvYg4GXsAASjfU?format=png&name=large"
)
if response.status_code == 200:
    if response.headers["content-type"].split("/")[0] == "image":
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        plt.imshow(image)
        plt.show()
    else:
        print("not donwloaded")
else:
    print("status code not 200!")
