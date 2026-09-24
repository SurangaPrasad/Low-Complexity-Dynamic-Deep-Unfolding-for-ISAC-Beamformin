import numpy as np
import matplotlib.pyplot as plt
import inspect
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAIN_LR = learning_rate
TRAIN_SCHEDULER_FACTOR = 0.1
TRAIN_SCHEDULER_PATIENCE = 3
TRAIN_MIN_LR = 1e-7
TRAIN_GRAD_CLIP_MAX_NORM = 1.0


def build_optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    scheduler_kwargs = {
        'mode': 'min',
        'factor': TRAIN_SCHEDULER_FACTOR,
        'patience': TRAIN_SCHEDULER_PATIENCE,
        # 'min_lr': TRAIN_MIN_LR,
    }
    if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
        scheduler_kwargs['verbose'] = True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs
    )
    return optimizer, scheduler


def clip_gradients(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP_MAX_NORM)

# ---- training and test the models ----

# Load training data
H_train, H_test0 = get_data_tensor(data_source)
print(H_train.shape)
H_test = H_test0[:, :test_size, :, :]
torch.manual_seed(3407)

def run_UPGA(step_size_UPGA):
    model_UPGA = PGA_Unfold_JX(step_size_UPGA)
    optimizer = torch.optim.Adam(model_UPGA.parameters(), lr=learning_rate)
    epoch_losses = [] # To store average loss per epoch
    for i_epoch in range(n_epoch):
        batch_losses = [] # To store loss of each batch in current epoch
        
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]  # balanced per-SNR
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)
            
            rate, __, F, W= model_UPGA.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, step_size_UPGA.shape[0], track_metrics=False)
            
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA.state_dict(), directory_model + f'UPGA_J{step_size_UPGA.shape[0]}.pth')


def run_UPGA_decay(step_size_UPGA_decay):
    model_UPGA_decay = PGA_Unfold_JX_decay(step_size_UPGA_decay)
    optimizer = torch.optim.Adam(model_UPGA_decay.parameters(), lr=learning_rate)
    epoch_losses = []  # store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = []  # loss of each batch in current epoch
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)

            __, __, F, W = model_UPGA_decay.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, step_size_UPGA_decay.shape[0], track_metrics=False)

            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_decay.state_dict(), directory_model + f'UPGA_J{step_size_UPGA_decay.shape[0]}.pth')
# ====================================================== Conventional PGA ====================================
if run_conv_PGA == 1:
    # Object defining
    model_conv_PGA = PGA_Conv(step_size_conv_PGA)

    # executing classical PGA on the test set
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    rate_iter_conv, beam_iter_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
    rate_conv = [r.detach().cpu().numpy() for r in (sum(rate_iter_conv) / len(H_test[0]))]
    beam_error_conv = [e.detach().cpu().numpy() for e in (sum(beam_iter_conv) / (len(H_test[0])))]
    iter_number_conv = np.array(list(range(n_iter_outer + 1)))

# ====================================================== Proposed Unfolding PGA ====================================

if run_UPGA_J1 == 1:
    run_UPGA(step_size_UPGA_J1)

if run_UPGA_J20 == 1:
    run_UPGA(step_size_UPGA_J20)

if run_UPGA_J10 == 1:
    run_UPGA(step_size_UPGA_J10)

if run_UPGA_J5 == 1:
    run_UPGA(step_size_UPGA_J5)

if run_UPGA_J4 == 1:
    run_UPGA(step_size_UPGA_J4)
if run_UPGA_J6 == 1:
    run_UPGA(step_size_UPGA_J6)

# ====================================================== Proposed unfolding PGA with decaying inner iterations =============================
if run_UPGA_J5_decay == 1:
    run_UPGA_decay(step_size_UPGA_J5_decay)

if run_UPGA_J10_decay == 1:
    run_UPGA_decay(step_size_UPGA_J10_decay)

if run_UPGA_J20_decay == 1:
    run_UPGA_decay(step_size_UPGA_J20_decay)
