"""Train the simplified E-GraphSAGE model on the toy graph.

This file demonstrates the main PyTorch training cycle:

1. Make predictions.
2. Measure the error.
3. Calculate gradients.
4. Update the model parameters.
5. Repeat.

Important:
The graph has only five edges. This is useful for learning how
training works, but it is not a valid research experiment.
"""

import torch
from torch import nn

from build_toy_graph import build_graph, FLOWS
from normalize_features import standardize_features
from toy_egraphsage import ToyEGraphSAGE

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim = 1 )
    currect_predictions = predictions == labels
    acurracy = currect_predictions.float().mean()

    return acurracy.item()

def main() -> None:
    torch.manual_seed(42)
    graph,_ = build_graph(FLOWS)
    normalize_features,_,_ = standardize_features(graph.edge_attr)
    graph.edge_attr = normalize_features
    model = ToyEGraphSAGE(edge_feature_size=graph.edge_attr.shape[1], hidden_size=8, number_of_classes=2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    number_of_epochs = 200

    for epoch in range(1, number_of_epochs + 1):
        # Enable training behavior.
        model.train()
        # Remove gradients from the previous training round.
        optimizer.zero_grad()
         # Forward pass:
        # ask the model to produce scores for every edge.
        logits,_,_ = model(graph)
        # Measure how wrong the predictions are.
        loss = loss_function(logits, graph.edge_y)
        # backward pass:
        # calculate gradients for every model parameter.
        loss.backward()
        # Update model parameters using the gradients.
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0:
            acurracy = calculate_accuracy(logits, graph.edge_y)
            print(f"Epoch {epoch} | " f"loss = {loss.item():.4f} |" f"accuracy = {acurracy:.4f}")

    #evaluate the model on the training data.
    model.eval()
    with torch.no_grad():
        final_logits,_,_ = model(graph)
        final_predictions = final_logits.argmax(dim=1)
        final_probabilities = torch.softmax(final_logits, dim=1)

    print("\nFinal probabilities :")
    print(final_probabilities)

    print("\nFinal predictions :")
    print(final_predictions)

    print("\nCorrect lables :")
    print(graph.edge_y)

if __name__ == "__main__":
    main()
