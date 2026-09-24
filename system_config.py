import numpy as np
import os
import torch

# ////////////////////////////////////////////// GLOBAL DTYPES //////////////////////////////////////////////
# Use single-precision complex tensors to keep memory usage manageable unless a
# specific routine requires doubles.
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#/////////////////////////// CONSIONDER SCHEMES /////////////////////////////////////////////////////////
run_conv_PGA = 0           # Conventional PGA without unfolding
run_conv_PGA_J5 = 1        # Conventional PGA with setting J = 5
run_conv_PGA_J10 = 1       # Conventional PGA with setting J = 10
run_conv_PGA_J20 = 0
run_conv_PGA_J10_PC = 0    # Conventional PGA with J = 10 and partial coupling (PC) 
run_UPGA_J1 = 0            # Unfolded PGA without any modification (J = 1)
run_UPGA_J4 = 0
run_UPGA_J5 = 1            # Unfolded PGA with setting J = 5
run_UPGA_J6 = 0            # Unfolded PGA with setting J = 6 (for ablation on inner iteration number)
run_UPGA_J10 = 1           # Unfolded PGA with setting J = 10
run_UPGA_J20 = 0           # Unfolded PGA with setting J = 20

run_UPGA_J5_decay = 1        # Unfolded PGA with decaying inner iterations (J_max=5 → 1)
run_UPGA_J10_decay = 1       # Unfolded PGA with decaying inner iterations (J_max=10 → 1)
run_UPGA_J20_decay = 0       # Unfolded PGA with decaying inner iterations (J_max=20 → 1)


# ////////////////////////////////////////////// SYSTEM PARAMS //////////////////////////////////////////////
Nt = 64                 # Num of Tx antennas
M = 4                   # Num of Users
Nrf = 4                 # Num of RF chains (must be >= M)
K = 1                   # Num of frequency bands
n_target = 3            # Num of sensing targets
theta_desire = 45 # Angles of sensing targets

snr_dB = 12                 # SNR for training and showing the convergences
snr = 10 ** (snr_dB / 10)   # transmit power
sigma2 = 1                  # normalized noise power
snr_dB_list = np.array([0, 2, 4, 6, 8, 10, 12], dtype='float64') # SNR for showing the rate and MSEs

init_W = 'ZF' # initialization scheme
initial_normalization = 0  # normalization for initialization
data_source = 'matlab'  # data generate by matlab or python
init_scheme = 'svd'  # proposed initialization for best convergence


system_config = str(Nt) + "TX_" + str(M) + "UE_" + str(Nrf) + "RF"

OMEGA = 0.25



# ////////////////////////////////////////////// MODEL PARAMS //////////////////////////////////////////////
train_size = 112 * 2    # size of training set
test_size = 10     # size of testing set
batch_size = len(snr_dB_list) * 4
n_epoch = 20         # number of training epochs
learning_rate = 0.0005 # learning 
# learning_rate = 0.00002

n_iter_outer = 120      # Number of outer iterations (I)
n_iter_inner_J1 = 1     # Number of inner iterations (J = 1)
n_iter_inner_J4 = 4     # Number of inner iterations (J = 4)
n_iter_inner_J5 = 5     # Number of inner iterations (J = 5)
n_iter_inner_J6 = 6     # Number of inner iterations (J = 6)
n_iter_inner_J10 = 10  # Number of inner iterations (J = 10)
n_iter_inner_J20 = 20   # Number of inner iterations (J = 20)


# ============================ TUNING PARAMETERS ===========================
WEIGHT_F_RAD = OMEGA  # fixed
WEIGHT_W_RAD = OMEGA / Nt * K
WEIGHT_F_COM = OMEGA  
WEIGHT_W_COM = OMEGA 
WEIGHT_F_CRB = 1
WEIGHT_W_CRB = 1

# ========================= CRB PARAMETERS =========================
# xi_0 = 10 ** (-40 / 10) ## path loss at reference distance (1 m) in linear scale
xi_0 = 1
lambda_wave = 1 # wavelength normalized
delta = lambda_wave / 2 # antenna spacing
desired_angle_rad = np.radians(theta_desire) # desired angles in radians
n_indices = torch.arange(Nt, dtype=torch.float32)
desired_angle_rad_torch = torch.tensor(desired_angle_rad, dtype=torch.float32)
phase = 1j* 2 * torch.pi * delta * torch.sin(desired_angle_rad_torch) * n_indices
a_phi_0 = torch.exp(phase)  # shape: (Nt, 1)
a_dot_phi_0 = ((1j * 2 * torch.pi * delta * torch.cos(desired_angle_rad_torch) * n_indices) * a_phi_0)

a_phi_0 = a_phi_0.unsqueeze(1)  # shape: (Nt, 1)
a_dot_phi_0 = a_dot_phi_0.unsqueeze(1)  # shape: (Nt, 1)

A_dot = (a_dot_phi_0 @ a_phi_0.transpose(0, 1) + a_phi_0 @ a_dot_phi_0.transpose(0, 1)).to(COMPLEX_DTYPE).to(device)

R_N = torch.eye(Nt)  # noise covariance matrix
R_N_inv = torch.linalg.inv(R_N).to(COMPLEX_DTYPE).to(device)  # pre-cast to complex64 and move to device


# ========================== initiate step sizes as tensor for training ================
step_size_fixed = 1e-2  # step size of conventional PGA
step_size_conv_PGA = torch.full([n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J1 = torch.full([n_iter_inner_J1, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J4 = torch.full([n_iter_inner_J4, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J5 = torch.full([n_iter_inner_J5, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J6 = torch.full([n_iter_inner_J6, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J10 = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J20 = torch.full([n_iter_inner_J20, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J10_PC = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
# J_decay uses the same shape as J10 (max_inner=10) but the class uses fewer steps per outer iter dynamically
step_size_UPGA_J5_decay = torch.full([n_iter_inner_J5, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J10_decay = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
step_size_UPGA_J20_decay = torch.full([n_iter_inner_J20, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)
# J_GradReuse uses the same shape as J10; gradient reuse logic is handled inside execute_PGA
step_size_UPGA_J_GradReuse = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, device=device, requires_grad=True)

# # ========================== Initialize step sizes seperately for lambda and mu ============
# step_size_lambda = torch.diag([Nt, M], step_size_fixed, requires_grad=True)
# step_size_mu = torch.diag([Nrf, M], step_size_fixed, requires_grad=True)
# step_size_UPGA_J10_lambda = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_lambda, requires_grad=True)
# step_size_UPGA_J10_mu = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_mu, requires_grad=True)
# ////////////////////////////////////////////// SAVING RESULTS AND DATA //////////////////////////////////////////////
directory_data = "./dataset/" + system_config + "/"
if not os.path.exists(directory_data):
    os.makedirs(directory_data)
directory_benchmark = directory_data  # To save benchmark results

if data_source == 'python':
    train_data_file_name = "train_data.mat"
    test_data_file_name = "test_data.mat"
else:  # matlab
    train_data_file_name = "train_data_matlab.mat"
    test_data_file_name = "test_data_matlab.mat"

data_path_train = directory_data + train_data_file_name
data_path_test = directory_data + test_data_file_name

# To save trained model
directory_model = "./model/" + system_config + "/"
directory_model03 = "./model/" + system_config  + "/"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

model_file_name_UPGA_J1 = directory_model + 'UPGA_J1.pth'
model_file_name_UPGA_J5 = directory_model + 'UPGA_J5.pth'
model_file_name_UPGA_J10 = directory_model + 'UPGA_J10.pth'
model_file_name_UPGA_J10_PRCDN = directory_model + 'UPGA_J10_PRCDN.pth'
model_file_name_UPGA_J20 = directory_model + 'UPGA_J20.pth'
model_file_name_UPGA_J10_PC = directory_model + 'UPGA_J10_PC.pth'
model_file_name_UPGA_J5_decay = directory_model + 'UPGA_J5_decay.pth'
model_file_name_UPGA_J10_decay = directory_model + 'UPGA_J10_decay.pth'
model_file_name_UPGA_J20_decay = directory_model + 'UPGA_J20_decay.pth'
model_file_name_UPGA_J_GradReuse = directory_model + 'UPGA_J_GradReuse.pth'
model_file_name_UPGA_J10_PC_omega03 = directory_model03 + 'UPGA_J10_PC.pth'
# To save result figures
directory_result = "./sim_results/" + system_config + "/"
if not os.path.exists(directory_result):
    os.makedirs(directory_result)

# define labels in figures
label_conv = 'Conventional PGA'
label_conv_PGA = 'Conventional PGA ' + '$(J = 1)$'
label_conv_PGA_J5 = 'Conventional PGA ' + '$(J = ' + str(n_iter_inner_J5) + ')$'
label_conv_PGA_J10 = 'Conventional PGA ' + '$(J = ' + str(n_iter_inner_J10) + ')$'
label_conv_PGA_J20 = 'Conventional PGA ' + '$(J = ' + str(n_iter_inner_J20) + ')$'

label_PGA_J10 = 'PGA ' + '$(J = ' + str(n_iter_inner_J10) + ')$'

label_UPGA_J1 = r'Fixed-UPGA,$120$ inner layers'
label_UPGA_J4 = r'Fixed-UPGA, $480$ inner layers'
label_UPGA_J5 = r'Fixed-UPGA, $600$ inner layers'
label_UPGA_J6 = r'Fixed-UPGA, $720$ inner layers'
label_UPGA_J10 = r'Fixed-UPGA, $1200$ inner layers'
label_UPGA_J20 = r'Fixed-UPGA, $2400$ inner layers'

label_UPGA_J10_PC = r'UPGA ' + '$(J = ' + str(n_iter_inner_J10) + ', PC)$'
label_conv_PGA_J10_PC = 'Conventional PGA ' + '$(J = ' + str(n_iter_inner_J10) + ', PC)$'

label_UPGA_J5_decay = r'Dynamic-UPGA, $421$ inner layers'
label_UPGA_J10_decay = r'Dynamic-UPGA, $722$ inner layers'
label_UPGA_J20_decay = r'Dynamic-UPGA, $20$ inner layers'
label_UPGA_J_GradReuse = r'UPGA ' + r'$(J=' + str(n_iter_inner_J10) + r', \mathrm{GradReuse})$'
label_ZF = 'ZF (digital, comm. only)'
label_SCA = 'SCA-ManOpt (converged)'

Conv_PGA_J1 = r'Conv. PGA, $120$ inner iterations'
Conv_PGA_J5 = r'Conv. PGA, $600$ inner iterations'
Conv_PGA_J10 = r'Conv. PGA, $1200$ inner iterations'
