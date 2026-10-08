import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cnn_model import ConventionalCNN
from prepare_data import load_datasets


# =========================================================
# Training Function
# =========================================================

def train_model():

    # Load dataset
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader
    ) = load_datasets()


    # Select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)


    # Create model
    model = ConventionalCNN(
        num_classes=len(
            train_dataset.classes
        )
    )

    model = model.to(device)


    # Loss function
    criterion = nn.CrossEntropyLoss()


    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # Number of epochs
    epochs = 15


    # Store results
    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []


    # =====================================================
    # Epoch Loop
    # =====================================================

    for epoch in range(epochs):

        # -----------------------------
        # Training
        # -----------------------------

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # Clear gradients
            optimizer.zero_grad()


            # Forward pass
            outputs = model(images)


            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            # Statistics
            running_loss += (
                loss.item() *
                images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        train_loss = (
            running_loss / total
        )


        train_accuracy = (
            correct / total
        )


        train_losses.append(
            train_loss
        )

        train_accuracies.append(
            train_accuracy
        )


        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        validation_loss = 0.0

        validation_correct = 0

        validation_total = 0


        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)

                labels = labels.to(device)


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                validation_loss += (
                    loss.item() *
                    images.size(0)
                )


                predictions = outputs.argmax(
                    dim=1
                )


                validation_correct += (
                    predictions == labels
                ).sum().item()


                validation_total += labels.size(0)


        validation_loss /= validation_total


        validation_accuracy = (
            validation_correct /
            validation_total
        )


        validation_losses.append(
            validation_loss
        )

        validation_accuracies.append(
            validation_accuracy
        )


        # -----------------------------
        # Print Results
        # -----------------------------

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {validation_loss:.4f} "
            f"Val Acc: {validation_accuracy:.4f}"
        )


    # =====================================================
    # Save Model
    # =====================================================

    torch.save(
        model.state_dict(),
        "results/cnn_model.pth"
    )


    # =====================================================
    # Plot Training Curves
    # =====================================================

    plt.figure()

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Conventional CNN Loss"
    )

    plt.legend()

    plt.savefig(
        "results/cnn_loss_curve.png"
    )

    plt.close()


    plt.figure()

    plt.plot(
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        validation_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "Conventional CNN Accuracy"
    )

    plt.legend()

    plt.savefig(
        "results/cnn_accuracy_curve.png"
    )

    plt.close()


    print()
    print("CNN training completed!")
    print("Model saved to:")
    print("results/cnn_model.pth")


# =========================================================
# Run Training
# =========================================================

if __name__ == "__main__":

    train_model()