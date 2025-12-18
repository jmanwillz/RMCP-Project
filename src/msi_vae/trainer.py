import math

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def train(
    model,
    dataloader,
    optimizer,
    writer,
    samples_processed,
    batches_processed,
    num_checkpoints,
    device,
):
    checkpoints = [
        math.floor(round((len(dataloader) * i / num_checkpoints) / 100) * 100)
        for i in range(1, num_checkpoints + 1)
    ]
    checkpoints.append(0)
    checkpoints.append(len(dataloader) - 1)

    model.train()
    overall_loss = 0

    for batch_idx, data in enumerate(tqdm(dataloader)):
        batch_size = data.shape[0]
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = output.loss
        overall_loss += loss
        loss.backward()

        if batch_idx in checkpoints:
            writer.add_scalar("Loss/Train", loss.item(), batches_processed)
            writer.add_scalar(
                "Loss/Train/MSE", output.loss_recon.item(), batches_processed
            )
            writer.add_scalar(
                "Loss/Train/KLD", output.loss_kl.item(), batches_processed
            )

            print(
                f"Step: {batch_idx} (Samples: {samples_processed}), Loss: {loss.item():.12f} (Recon: {output.loss_recon.item():.12f}, KL: {output.loss_kl.item():.12f})"
            )

            for i in range(min(2, batch_size)):
                fig, ax = plt.subplots()
                ax.plot(range(len(data[i].cpu())), data[i].cpu(), label="Actual")
                ax.plot(
                    range(len(output.x_recon[i].detach().cpu())),
                    output.x_recon[i].detach().cpu(),
                    label="Predicted",
                )
                ax.set_xlabel("m/z")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.set_title(f"Actual vs Predicted Spectrum (Sample {i})")
                writer.add_figure(
                    f"Spectrum/Checkpoint_{batches_processed}_Sample_{i}",
                    fig,
                    global_step=batches_processed,
                )
                plt.close(fig)

        optimizer.step()
        samples_processed += batch_size
        batches_processed += 1

    average_epoch_loss = overall_loss / len(dataloader.dataset)
    return batches_processed, samples_processed, average_epoch_loss
