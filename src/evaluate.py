import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from prepare_data import load_datasets
from cnn_model import ConventionalCNN
from resnet_model import ResNetClassifier


# =========================================================
# Evaluate Model
# =========================================================

def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )


    return (
        accuracy,
        precision,
        recall,
        f1,
        matrix
    )


# =========================================================
# Main
# =========================================================

def main():

    # Load dataset
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = load_datasets()


    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # Number of classes
    num_classes = len(
        train_dataset.classes
    )


    # -----------------------------------------------------
    # CNN
    # -----------------------------------------------------

    cnn = ConventionalCNN(
        num_classes=num_classes
    )

    cnn.load_state_dict(
        torch.load(
            "results/cnn_model.pth",
            map_location=device
        )
    )

    cnn = cnn.to(device)


    cnn_results = evaluate_model(
        cnn,
        test_loader,
        device
    )


    # -----------------------------------------------------
    # ResNet
    # -----------------------------------------------------

    resnet = ResNetClassifier(
        num_classes=num_classes
    )

    resnet.load_state_dict(
        torch.load(
            "results/resnet_model.pth",
            map_location=device
        )
    )

    resnet = resnet.to(device)


    resnet_results = evaluate_model(
        resnet,
        test_loader,
        device
    )


    # =====================================================
    # Print Results
    # =====================================================

    print()
    print("=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)


    print()
    print("Conventional CNN")
    print("----------------")

    print(
        f"Accuracy : {cnn_results[0]:.4f}"
    )

    print(
        f"Precision: {cnn_results[1]:.4f}"
    )

    print(
        f"Recall   : {cnn_results[2]:.4f}"
    )

    print(
        f"F1 Score : {cnn_results[3]:.4f}"
    )


    print()
    print("ResNet")
    print("----------------")

    print(
        f"Accuracy : {resnet_results[0]:.4f}"
    )

    print(
        f"Precision: {resnet_results[1]:.4f}"
    )

    print(
        f"Recall   : {resnet_results[2]:.4f}"
    )

    print(
        f"F1 Score : {resnet_results[3]:.4f}"
    )


    print()
    print("Classes:")

    print(
        train_dataset.classes
    )


    print()
    print("CNN Confusion Matrix:")

    print(
        cnn_results[4]
    )


    print()
    print("ResNet Confusion Matrix:")

    print(
        resnet_results[4]
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    main()