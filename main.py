# Define your model
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet50
from torchcam.methods import SmoothGradCAMpp


model = resnet50(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model)

# Get your input
img = read_image("/Users/chongzhang/PycharmProjects/CAM-Test/bird1.jpg")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255.,
                         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

with SmoothGradCAMpp(model) as cam_extractor:
    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = extractor(out.squeeze(0).argmax().item(), out)

# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy())
plt.axis('off')
plt.tight_layout()
plt.show()
