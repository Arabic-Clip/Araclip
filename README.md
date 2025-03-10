# Araclip

## How to use 
```
pip install git+https://github.com/Arabic-Clip/Araclip.git
```
```python
from araclip import AraClip
model = AraClip.from_pretrained("Arabic-Clip/araclip")
```

```python
import numpy as np
from PIL import Image
labels = ["قطة جالسة", "قطة تقفز" ,"كلب", "حصان"]
image = Image.open("cat.png")

image_features = model.embed(image=image)
text_features = np.stack([model.embed(text=label) for label in labels])

similarities = text_features @ image_features
best_match = labels[np.argmax(similarities)]

print(f"The image is most similar to: {best_match}")
# قطة جالسة
```
![alt text](assets/image.png)
