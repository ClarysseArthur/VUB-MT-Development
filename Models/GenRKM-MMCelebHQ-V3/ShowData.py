import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dataset_source = "./VUB-MT-Development/Datasets/MMCelebAHQ/DATASET/"

images = []
sketches = []
labels = []

for i in range(2):  # two examples -> rows
    image = np.array(Image.open(dataset_source + f"image/{i}.jpg").convert("RGB")).astype(np.double)
    sketch = np.array(Image.open(dataset_source + f"sketch/{i}.jpg").convert("1")).astype(np.double)
    label = np.array(open(dataset_source + f"label/{i}.txt").read().split(",")).astype(np.double)

    images.append(image)
    sketches.append(sketch)
    labels.append(label)

def to_display_img(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] != x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    x = x.astype(np.float32)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax > xmin:
        x = (x - xmin) / (xmax - xmin)

    return x

fig, ax = plt.subplots(2, 3, figsize=(14, 8), dpi=150)

# --- add padding between subplots + a bit around the whole figure ---
fig.subplots_adjust(wspace=0.25, hspace=0.25, left=0.01, right=0.99, top=0.92, bottom=0.01)

for r in range(2):
    im = to_display_img(images[r])
    sk = to_display_img(sketches[r])
    lab = np.asarray(labels[r]).reshape(-1)

    ax[r, 0].imshow(im)
    ax[r, 0].set_title(f"Image {r}")
    ax[r, 0].axis("off")

    ax[r, 1].imshow(sk, cmap="gray")
    ax[r, 1].set_title(f"Sketch {r}")
    ax[r, 1].axis("off")

    ones = np.where(lab > 0.5)[0].tolist()

    llll = []
    classes = ['_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling ','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']

    for i, l in enumerate(ones):
        llll.append(classes[l])
        if i > 5:
            break

    text = "Subset of Labels:\n- " + ("\n- ".join(map(str, llll)) if len(llll) else "(none)")
    ax[r, 2].text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")
    ax[r, 2].set_title(f"Labels {r}")
    ax[r, 2].set_xlim(0, 1)
    ax[r, 2].set_ylim(0, 1)
    ax[r, 2].axis("off")

# (optional) keep tight_layout off since we're manually controlling padding
plt.show()