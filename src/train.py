from __future__ import annotations
import argparse, json, os, tqdm
import torch

from .graphs import barbell_data, add_edges_balanced_forman
from .models import GCN4
from .utils  import layer_condition, energy


# ---------- helpers numéricos ---------- #
def jacobian_layer(layer, x_in: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Devuelve el Jacobiano J = d(layer(x))/d(x) evaluado en x_in.
    Salida con shape (N * d_out, N * d_in).
    """
    x = x_in.clone().requires_grad_(True)
    y = layer(x, edge_index)
    J = torch.autograd.functional.jacobian(
        lambda z: layer(z, edge_index), x, create_graph=False
    )
    # reshape para κ(J) con SVD
    return J.reshape(y.numel(), x.numel()).detach()


# ---------- bucle de entrenamiento ---------- #
def run(n=200, epochs=200, lr=1e-2, device="cpu", log_every=10, outdir="results"):
    data = barbell_data(n=n, to_device=device)
    model = GCN4(in_dim=n).to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    log  = {"kappa": [], "energy": []}

    for ep in tqdm.trange(epochs):
        model.train()
        opt.zero_grad()

        out, inputs, outputs = model(
            data.x, data.edge_index, return_io=True
        )                                  # <-- nuevas listas

        loss = torch.nn.functional.cross_entropy(out, data.y)
        loss.backward()
        opt.step()

        # ----- métricas cada 'log_every' epochs ----- #
        if ep % log_every == 0:
            with torch.no_grad():
                kappa_layers, energy_layers = [], []
                for l, layer in enumerate(model.convs):
                    x_in = inputs[l]              # entrada de la capa l
                    J = jacobian_layer(layer, x_in, data.edge_index)
                    kappa_layers.append(layer_condition(J))
                    energy_layers.append(energy(x_in))
                log["kappa"].append(kappa_layers)
                log["energy"].append(energy_layers)

    # ---------- guardado ---------- #
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/baseline_gnn.pt")
    with open(f"{outdir}/baseline_log.json", "w") as f:
        json.dump(log, f)

    return log

def run_rewired(
    n=200,
    epochs=200,
    lr=1e-2,
    device="cpu",
    log_every=10,
    outdir="results",
    k=2,                     # nº de aristas a insertar por curvatura
):
    """
    Igual que `run`, pero entrena sobre un grafo con rewiring
    basado en Balanced Forman curvature y guarda 'rewired_log.json'.
    """
    # --- 1. Generar grafo base y aplicar rewiring ---
    data_base = barbell_data(n=n, to_device=device)
    data = add_edges_balanced_forman(data_base, k=k)          # ⬅️ rewiring

    model = GCN4(in_dim=n).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    log = {"kappa": [], "energy": []}

    for ep in tqdm.trange(epochs):
        model.train()
        opt.zero_grad()

        out, inputs, outputs = model(
            data.x, data.edge_index, return_io=True
        )

        loss = torch.nn.functional.cross_entropy(out, data.y)
        loss.backward()
        opt.step()

        if ep % log_every == 0:
            with torch.no_grad():
                kappa_layers, energy_layers = [], []
                for l, layer in enumerate(model.convs):
                    x_in = inputs[l]
                    J = jacobian_layer(layer, x_in, data.edge_index)
                    kappa_layers.append(layer_condition(J))
                    energy_layers.append(energy(x_in))
                log["kappa"].append(kappa_layers)
                log["energy"].append(energy_layers)

    # --- 2. Guardar artefactos específicos del rewiring ---
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/rewired_gnn.pt")
    with open(f"{outdir}/rewired_log.json", "w") as f:
        json.dump(log, f)

    return log

# ---------- CLI ---------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN oversquash baseline")
    parser.add_argument("--n", type=int, default=200, help="tamaño total del grafo")
    parser.add_argument("--epochs", type=int, default=200, help="número de epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--device", default="cpu", help="cpu o mps/cuda")
    parser.add_argument("--log_every", type=int, default=10,
                        help="frecuencia de log de Jacobianos")
    args = parser.parse_args()

    run(n=args.n,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every)
