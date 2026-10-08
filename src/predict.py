import sys

import torch
from PIL import Image
from torchvision import transforms

from cnn_model import ConventionalCNN
from resnet_model import ResNetClassifier


# =========================================================
# Configuration
# =========================================================

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CLASSES = [
    "healthy",
    "leaf_spot",
    "powdery_mildew",
    "rust"
]


# =========================================================
# Image Transformation
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Prediction Function
# =========================================================

def predict(model, image):

    model.eval()


    image = transform(
        image
    ).unsqueeze(0)


    image = image.to(DEVICE)


    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted = (
            probabilities.max(dim=1)
        )


    predicted_class = CLASSES[
        predicted.item()
    ]


    confidence_value = (
        confidence.item() * 100
    )


    return (
        predicted_class,
        confidence_value
    )


# =========================================================
# Main
# =========================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/predict.py image.jpg"
        )

        return


    image_path = sys.argv[1]


    image = Image.open(
        image_path
    ).convert("RGB")


    # -----------------------------------------------------
    # CNN
    # -----------------------------------------------------

    cnn = ConventionalCNN(
        num_classes=4
    )

    cnn.load_state_dict(
        torch.load(
            "results/cnn_model.pth",
            map_location=DEVICE
        )
    )

    cnn = cnn.to(DEVICE)


    cnn_class, cnn_confidence = predict(
        cnn,
        image
    )


    # -----------------------------------------------------
    # ResNet
    # -----------------------------------------------------

    resnet = ResNetClassifier(
        num_classes=4
    )

    resnet.load_state_dict(
        torch.load(
            "results/resnet_model.pth",
            map_location=DEVICE
        )
    )

    resnet = resnet.to(DEVICE)


    resnet_class, resnet_confidence = predict(
        resnet,
        image
    )


    # =====================================================
    # Output
    # =====================================================

    print()
    print("=" * 50)
    print("PLANT DISEASE PREDICTION")
    print("=" * 50)

    print()

    print("Image:", image_path)

    print()

    print("Conventional CNN")
    print("----------------")

    print(
        "Prediction:",
        cnn_class
    )

    print(
        f"Confidence: {cnn_confidence:.2f}%"
    )

    print()

    print("ResNet")
    print("----------------")

    print(
        "Prediction:",
        resnet_class
    )

    print(
        f"Confidence: {resnet_confidence:.2f}%"
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    main()