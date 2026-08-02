"""Train the toy E-GraphSAGE model using flows loaded from CSV.

This verifies the complete data pipeline:

CSV file
    -> pandas DataFrame
    -> PyTorch Geometric graph
    -> normalized edge features
    -> model training
"""

import torch
from torch import nn

from dataframe_to_graph import dataframe_to_graph
from load_flows_csv import TOY_DATASET_PATH, load_flow_csv
from normalize_features import standardize_features
from toy_egraphsage import ToyEGraphSAGE

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = logits.argmax(dim=1)
    correct_predictions = prediction == labels
    return correct_predictions.float().mean().item()

def main()->None:
    torch.manual_seed(42)

    flows = load_flow_csv(TOY_DATASET_PATH)
    graph,_ = dataframe_to_graph(flows)
    normalized_features,_,_ = standardize_features(graph.edge_attr)
    graph.edge_attr = normalized_features
    model = ToyEGraphSAGE(edge_feature_size=graph.edge_attr.shape[1], hidden_size=8, number_of_classes=2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    number_of_epochs = 200

    for epoch in range(1, number_of_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits,_,_ = model(graph)
        loss = loss_function(logits, graph.edge_y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0:
            accuracy = calculate_accuracy(logits, graph.edge_y)
            print(f"Epoch {epoch} | loss = {loss.item():.4f} | accuracy = {accuracy:.4f}")
    model.eval()
    with torch.no_grad():
        final_logits,_,_ = model(graph)
        final_predictions = final_logits.argmax(dim=1)
        final_probabilities = torch.softmax(final_logits, dim=1)

    print("\nFinal probabilities:")
    print(final_probabilities)

    print("\nFinal predictions:")
    print(final_predictions)

    print("\nCorrect edge labels:")
    print(graph.edge_y)

if __name__ == "__main__":
    main()