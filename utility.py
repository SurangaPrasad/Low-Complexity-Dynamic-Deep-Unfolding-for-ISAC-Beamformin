import torch
import torch.nn as nn
import sys
import h5py
import scipy.io
from system_config import *
import matplotlib.pyplot as plt


def randn_complex(shape, device=None):
    """Sample a complex tensor with IID unit-variance entries."""
    real = torch.randn(shape, dtype=REAL_DTYPE, device=device)
    imag = torch.randn(shape, dtype=REAL_DTYPE, device=device)
    return torch.complex(real, imag)


# ========================= COMPUTE A_dot FROM A GIVEN ANGLE =========================
# theta_desire_list: the pool of angles from which theta_desire is selected per batch
theta_desire_list = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype='float64')

def compute_A_dot(theta_desire):
    """Compute the A_dot matrix for the given desired angle(s) in degrees.

    Args:
        theta_desire: scalar or array of desired angles in degrees.
    Returns:
        A_dot tensor of shape (Nt, Nt) on the configured device/dtype.
    """
    desired_angle_rad = np.radians(np.atleast_1d(theta_desire))
    desired_angle_rad_torch = torch.tensor(desired_angle_rad, dtype=torch.float32)
    phase = 1j * 2 * torch.pi * delta * torch.sin(desired_angle_rad_torch) * n_indices
    a_phi_0 = torch.exp(phase)  # shape: (Nt,)
    a_dot_phi_0 = ((1j * 2 * torch.pi * delta * torch.cos(desired_angle_rad_torch) * n_indices) * a_phi_0)

    a_phi_0 = a_phi_0.unsqueeze(1)  # shape: (Nt, 1)
    a_dot_phi_0 = a_dot_phi_0.unsqueeze(1)  # shape: (Nt, 1)

    A_dot = (a_dot_phi_0 @ a_phi_0.transpose(0, 1) + a_phi_0 @ a_dot_phi_0.transpose(0, 1)).to(COMPLEX_DTYPE).to(device)
    return A_dot


# ==================================== initialize F and W ===========================
def initialize(H, Pt, normalization, pc=False):
    if init_scheme == 'conv':
        # randomizing F
        F = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_scheme == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        # if K == 1:
        #     F = H / torch.abs(H)
        #     F = torch.transpose(F, 2, 3)
        #     Hp = H.conj()
        #     Q = torch.linalg.pinv(Hp)
        #     FQ = torch.linalg.pinv(F) @ Q
        #     W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
        # else:
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, len(H[0]), Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H,K//2,snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, len(H[0]), Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        else:
            sys.stderr.write('Error: Wrong RF chain configuration....\n')
        F = F * generage_partial_connection_mask(Nt, Nrf, device=F.device) if pc else F
    elif init_scheme == 'svd':
        U, S, V_H = torch.linalg.svd(H)
        V = V_H
        # V = torch.transpose(V_H, 2, 3).conj()
        F = V[:, :, :, :Nrf]
        F = F / torch.abs(F)
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # rate_0 = get_sum_rate(H, F, W)
    # print(rate_0)
    if normalization == 1:
        # normalize both F and W
        F, W = normalize(F, W, H, Pt)
    else:
        # only normalize W for power constraint
        B = len(H[0])
        norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)  # (B,)
        if torch.is_tensor(Pt) and Pt.dim() >= 1:
            Pt_vec = Pt.to(dtype=norm2_FW.dtype, device=F.device)
        else:
            Pt_vec = torch.full((B,), float(Pt), dtype=norm2_FW.dtype, device=F.device)
        W = torch.sqrt(Pt_vec / norm2_FW).view(B, 1, 1) * W
    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]), device=H.device)
    # beam_error_init = torch.zeros(1, len(H[0]))
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    # beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, F, W


# ==================================== initialize F and W with different methods for comparison ===========================
def initialize_schemes(H, R, Pt, init_method):
    if init_method == 'conv':
        # randomizing F
        F = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_method == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
    elif init_method == 'svd':
        if Nrf == M:
            U, S, V_H = torch.linalg.svd(H)
            V = V_H
            # V = torch.transpose(V_H, 2, 3).conj()
            F = V[:, :, :, :Nrf]
            F = F / torch.abs(F)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k, :, :, :]
                Hp = Hk.conj()
                Q = torch.linalg.pinv(Hp)
                FQ = torch.linalg.pinv(F) @ Q
                fro_norm = torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = FQ / fro_norm
        elif Nrf > M:
            # Determine G
            G = get_mat_G_SVD(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = randn_complex((K, test_size, Nrf, M), device=H.device)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)

    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # only normalize W for power constraint
    B = len(H[0])
    norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(dtype=norm2_FW.dtype, device=F.device)
    else:
        Pt_vec = torch.full((B,), float(Pt), dtype=norm2_FW.dtype, device=F.device)
    W = torch.sqrt(Pt_vec / norm2_FW).view(B, 1, 1) * W

    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]), device=H.device)
    beam_error_init = torch.zeros(1, len(H[0]), device=H.device)
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, beam_error_init, F, W


# ================== get matrix G for initalization when Nrf > K
def get_mat_G(H,fre_indx,snr_dB):
    G = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
    Htmp = torch.transpose(H[fre_indx, :, :, :], 1, 2)
    G[:, :, :M] = Htmp

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G

def get_mat_G_SVD(H,fre_indx,snr_dB):
    G = randn_complex((len(H[0]), Nt, Nrf), device=H.device)
    U, S, V_H = torch.linalg.svd(H)
    V = V_H
    G[:, :, :M] = V[:, :, :, :M]

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G
# ==================================== compute sum rate of MU-MISO system for each subcarrier ===========================
def get_sum_rate(H, F, W, Pt):

    # Normalize
    F, W = normalize(F, W, H, Pt)

    # ================= Power constraint check =================
    power_high_threshold = Pt + 1e-3

    power = torch.linalg.matrix_norm(F @ W, dim=(-2, -1)) ** 2  # (K, B)
    sum_power = torch.mean(power)

    # if torch.any(sum_power > power_high_threshold):
        # sys.stderr.write('Error: power constraint violated\n')

    # ================= Precompute =================
    F_H = F.conj().transpose(-2, -1)             # (K, B, Nrf, Nt)
    W_H = W.conj().transpose(-2, -1)             # (K, B, M, Nrf)

    V = W @ W_H                                  # (K, B, Nrf, Nrf)

    # ================= Build V_m =================
    Mval = W.shape[-1]

    mask = (1 - torch.eye(Mval, device=W.device, dtype=W.dtype))  # (M, M)

    # Apply mask: zero each column m
    W_m_all = W.unsqueeze(2) * mask.view(1, 1, Mval, 1, Mval)
    # shape: (K, B, M, Nrf, M)

    V_m_all = W_m_all @ W_m_all.conj().transpose(-1, -2)
    # (K, B, M, Nrf, Nrf)

    # ================= Channel outer products =================
    # H: (K, B, M, Nt)
    h = H.unsqueeze(-1)                          # (K, B, M, Nt, 1)
    Htilde = h @ h.conj().transpose(-1, -2)      # (K, B, M, Nt, Nt)

    # ================= Trace 1 =================
    FVF_H = F @ V @ F_H                          # (K, B, Nt, Nt)

    trace_1 = (
        (FVF_H.unsqueeze(2) @ Htilde)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )  # (K, B, M)

    # ================= Trace 2 =================
    FVmFH = F.unsqueeze(2) @ V_m_all @ F_H.unsqueeze(2)

    trace_2 = (
        (FVmFH @ Htilde)
        .diagonal(dim1=-1, dim2=-2)
        .sum(-1)
    )  # (K, B, M)

    # ================= Rate =================
    rate = (
        torch.log2(trace_1 + sigma2)
        - torch.log2(trace_2 + sigma2)
    ).real.sum(dim=-1)  # (K, B)

    sum_rate = torch.mean(rate)

    return sum_rate


# ==================================== compute tau function ===========================
def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2 / torch.linalg.matrix_norm(R, ord='fro') ** 2
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    sum_error = torch.mean(error)
    return sum_error

# ==================================== compute CRB  fishery equation function ===========================
def get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt):
    F, W = normalize(F, W, H, Pt)
    
    A_dot = A_dot.unsqueeze(0).unsqueeze(0) # [1, 1, Nt, Nt]
    R_N_inv = R_N_inv.unsqueeze(0).unsqueeze(0) # [1, 1, Nr, Nr]

    A_dot_H = A_dot.conj().transpose(-2, -1)
    W_H = W.conj().transpose(-2, -1)
    F_H = F.conj().transpose(-2, -1)
    
    M = A_dot_H @ R_N_inv @ A_dot
    inner_mat = W_H @ F_H @ M @ F @ W
    batch_trace = (torch.diagonal(inner_mat, dim1=-2, dim2=-1).sum(-1))  # [K, batch_size]

    fim = batch_trace.sum(0).real  # sum over K users -> [batch_size]

    crb = torch.log(fim) + torch.log(2*torch.tensor(xi_0) ** 2)  # [batch_size]

    # crb_real = 1 / (2 * (xi_0 ** 2) * fim)
    # print(f"CRB (real): {crb_real}")

    return crb

# ====================================compute power of F and W ==========================

def get_power(F, W):
    # power = frobinus_norm(F * W)^2
    power = torch.linalg.matrix_norm(F @ W, ord='fro') ** 2
    return power
# ==================================== compute MSE ===========================
def get_MSE(F, W, at, R, Pt):
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    at_H = torch.transpose(at, 2, 3).conj()
    beampattern = torch.real(torch.diagonal(at_H @ X @ X_H @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_mean = torch.mean(beampattern,0)
    # benchmark beampatter
    beam_bm = torch.real(torch.diagonal(at_H @ R @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_bm_mean = torch.mean(beam_bm,0)

    MSE = (torch.abs(beam_bm_mean - beam_mean)) ** 2
    MSE_mean = 10 * torch.log10(torch.mean(torch.mean(MSE, 1)))  # average over channel and get sum
    return MSE_mean


# ==================================== compute trace of matrix A ===========================
def get_trace(A):
    trace_A = torch.diagonal(A, offset=0, dim1=-1, dim2=-2).sum(-1)  # sum all diagonal elements
    return trace_A


# ======== normalization to meet constant modulus and power constraint ===========================
def normalize(F, W, H, Pt):
    B = len(H[0])

    # ================= Constant modulus =================
    F = F / (torch.abs(F) + 1e-12)

    # ================= Power computation =================
    power = torch.linalg.matrix_norm(F @ W, dim=(-2, -1)) ** 2  # (K, B)

    # Total power per batch
    sum_norm_BB = torch.sum(power, dim=0)  # (B,)
    sum_norm_BB = torch.clamp(sum_norm_BB, min=1e-6)

    # ================= Handle Pt =================
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(device=F.device, dtype=sum_norm_BB.dtype)
    else:
        Pt_vec = torch.full((B,), float(Pt), device=F.device, dtype=sum_norm_BB.dtype)

    # ================= Normalize W =================
    normalize_factor = torch.sqrt(K * Pt_vec / sum_norm_BB).view(B, 1, 1)
    W = normalize_factor * W

    return F, W


# ========================= normalize F based on power constraint =====================
def normalize_power(F, W, H, Pt):
    """Normalize F to meet the per-sample power constraint.

    Pt can be a scalar or a 1-D tensor of shape (B,).
    """
    B = len(H[0])
    sum_norm_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)  # (B,)
    sum_norm_power = torch.clamp(sum_norm_power, min=1e-6)
    if torch.is_tensor(Pt) and Pt.dim() >= 1:
        Pt_vec = Pt.to(dtype=sum_norm_power.real.dtype if sum_norm_power.is_complex() else sum_norm_power.dtype,
                       device=F.device)
    else:
        Pt_vec = torch.full((B,), float(Pt), dtype=sum_norm_power.real.dtype if sum_norm_power.is_complex() else sum_norm_power.dtype,
                            device=F.device)
    normalize_factor = torch.sqrt(Pt_vec / sum_norm_power).view(B, 1, 1)
    F = normalize_factor * F
    return F

# ========================= generate PC mask =====================
def generage_partial_connection_mask(N, M, device=None):
    mask = torch.zeros((N, M), dtype=COMPLEX_DTYPE, device=device)
    antennas_per_rf = N // M
    for rf in range(M):
        start_idx = rf * antennas_per_rf
        end_idx = start_idx + antennas_per_rf
        mask[start_idx:end_idx, rf] = 1.0 + 0j  # connect these antennas to this RF chain
    return mask


# ======================== generate channels =============================================================
def array_response(N, phi, theta):
    # Generate array response vectors
    a = np.zeros([N, 1], dtype='complex_')
    for n in range(N):
        a[n] = (1 / np.sqrt(N)) * np.exp(1j * np.pi * (n * np.sin(phi)))
    return a


def gen_channel(train_batch_size):
    batch_H = np.zeros([K, train_batch_size, M, Nt],
                       dtype='complex64')  # use to save testing data, used latter in Matlab

    for k in range(K):
        for ii in range(train_batch_size):

            # randomly generate azimuth and elevation angles
            AoD = np.zeros([2, Ncluster * Nray], dtype='complex64')
            AoA = np.zeros([2, Ncluster * Nray], dtype='complex64')

            for cc in range(Ncluster):
                AoD_m = np.random.uniform(0, 2 * np.pi, 2)
                AoA_m = np.random.uniform(0, 2 * np.pi, 2)

                AoD[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[0], angle_sigma, Nray)
                AoD[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[1], angle_sigma, Nray)
                AoA[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[0], angle_sigma, Nray)
                AoA[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[1], angle_sigma, Nray)

            alpha = np.sqrt(sigma / 2) * (
                    np.random.normal(0, 1, Ncluster * Nray) + 1j * np.random.normal(0, 1, Ncluster * Nray))

            # generate channel matrix
            H = np.zeros([M, Nt], dtype='complex_')
            At = np.zeros([Nt, Ncluster * Nray], dtype='complex64')

            for j in range(Ncluster * Nray):
                at = array_response(Nt, AoD[0, j], AoD[1, j])  # UPA array response
                ar = array_response(M, AoA[0, j], AoA[1, j])  # UPA array response
                H = H + alpha[j] * ar * at.conj().T
            H = gamma * H
            batch_H[k, ii, :, :] = H

    return batch_H


# =================================== save generated data ==================================================
def save_data(data_train, data_test):
    # write data
    with h5py.File(data_path_train, 'w') as hf:
        hf.create_dataset('train_set', data=data_train)
    with h5py.File(data_path_test, 'w') as hf:
        hf.create_dataset('test_set', data=data_test)
    # scipy.io.savemat('./channel.mat', {'channel':data_test})


# =================================== load data generated in Matlab ==================================================
def load_data_matlab():
    data_train = scipy.io.loadmat(data_path_train)
    data_train_array = data_train['H_train']
    data_test = scipy.io.loadmat(data_path_test)
    data_test_array = data_test['H_test']
    return data_train_array, data_test_array


# =================================== load data generated in python ==================================================
def load_data():
    # read data
    with h5py.File(data_path_train, 'r') as hf:
        data_train = list(hf.keys())[0]
        data_train_array = hf[data_train][()]
    with h5py.File(data_path_test, 'r') as hf:
        data_test = list(hf.keys())[0]
        data_test_array = hf[data_test][()]
    return data_train_array, data_test_array


# =================================== load data and convert to tensor for trainign=================================
def get_data_tensor(data_source):
    # first load the saved data
    if data_source == 'python':
        data_train_array, data_test_array = load_data()
    else:  # use matlab data
        data_train_array, data_test_array = load_data_matlab()
    # then convert numpy to tensor
    max_train = min(train_size, data_train_array.shape[1])
    max_test = min(test_size, data_test_array.shape[1])

    train_slice = np.ascontiguousarray(data_train_array[:, :max_train, :, :])
    test_slice = np.ascontiguousarray(data_test_array[:, :max_test, :, :])

    H_train_tensor = torch.from_numpy(train_slice).to(COMPLEX_DTYPE).contiguous().to(device)
    H_test_tensor = torch.from_numpy(test_slice).to(COMPLEX_DTYPE).contiguous().to(device)
    return H_train_tensor, H_test_tensor


# =================================== load radar data generated in Matlab ==================================================
def get_radar_data(snr_dB, H):
    """Load pre-computed radar covariance matrix R and array-steering vectors.

    snr_dB can be:
      - a scalar (float / int / 0-d array) — same R replicated over the batch (original behaviour)
      - a 1-D array of shape (B,)          — per-sample R assembled from the lookup table

    Returns
    -------
    R          : torch tensor  (K, B, Nt, Nt)
    at         : torch tensor  (K, B, Nt, n_angles)
    theta      : np.ndarray    angle grid
    ideal_beam : np.ndarray    ideal beampattern
    """
    radar_data_file_name = directory_data + 'radar_data.mat'
    radar_data = scipy.io.loadmat(radar_data_file_name)
    R0_4D = radar_data['J']          # shape: (Nt, Nt, K_freq, n_snr)  or (Nt, Nt, n_snr) when K==1

    at_2D = radar_data['a']
    theta = radar_data['theta']
    ideal_beam = radar_data['Pd_theta']

    B = len(H[0])

    # ------------------------------------------------------------------ #
    # Determine whether snr_dB is a scalar or per-sample array
    # ------------------------------------------------------------------ #
    snr_dB_arr = np.atleast_1d(np.asarray(snr_dB, dtype=float)).ravel()
    per_sample = snr_dB_arr.size > 1   # True  → per-sample path
                                        # False → replicate-scalar path

    if K == 1:
        if per_sample:
            # Build R per sample: shape (1, B, Nt, Nt)
            R_list = []
            for s in snr_dB_arr:
                idx = np.where(snr_dB_list == s)[0]
                R_s = np.squeeze(R0_4D[:, :, 0, idx])   # (Nt, Nt)
                R_list.append(R_s[None, :, :])           # (1, Nt, Nt)
            R_stack = np.stack(R_list, axis=0)           # (B, Nt, Nt)
            R_array = R_stack[None, :, :, :]             # (1, B, Nt, Nt)
        else:
            idx_snr = np.where(snr_dB_list == snr_dB_arr[0])
            R0_2D = np.squeeze(R0_4D[:, :, 0, idx_snr])
            R_array = np.tile(R0_2D, [1, B, 1, 1])

        at0 = np.expand_dims(at_2D, axis=0)
        at_array1 = np.tile(at0, (B, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))
    else:
        if per_sample:
            R_list = []
            for s in snr_dB_arr:
                idx = np.where(snr_dB_list == s)[0]
                R_s = np.squeeze(R0_4D[:, :, :, idx])   # (Nt, Nt, K)
                R_s_k = np.transpose(R_s, (2, 0, 1))    # (K, Nt, Nt)
                R_list.append(R_s_k[:, None, :, :])     # (K, 1, Nt, Nt)
            R_stack = np.concatenate(R_list, axis=1)    # (K, B, Nt, Nt)
            R_array = R_stack
        else:
            idx_snr = np.where(snr_dB_list == snr_dB_arr[0])
            R0_2D = np.squeeze(R0_4D[:, :, :, idx_snr])
            R_array0 = np.transpose(R0_2D, (2, 0, 1))
            R_array1 = np.tile(R_array0, [B, 1, 1, 1])
            R_array = np.transpose(R_array1, (1, 0, 2, 3))

        at0 = np.transpose(at_2D, (2, 0, 1))
        at_array1 = np.tile(at0, (B, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))

    R = torch.from_numpy(R_array).to(COMPLEX_DTYPE).contiguous().to(device)
    at = torch.from_numpy(at_array).to(COMPLEX_DTYPE).contiguous().to(device)

    return R, at, theta, ideal_beam[0, :]


# =================================== get the array of beampattern values ==================================================
def get_beampattern(F, W, at, Pt):
    Q = F @ W
    at_H = torch.transpose(at, 2, 3).conj()
    Q_H = torch.transpose(Q, 2, 3).conj()
    B = at_H @ Q @ Q_H @ at
    # print(torch.linalg.matrix_norm(B, ord='fro') ** 2)
    Bdiag = torch.diagonal(B, offset=0, dim1=-1, dim2=-2) / Pt
    # Bmean = 10 * torch.log10(torch.real(torch.mean(Bdiag, 1)))
    Bmean = torch.real(torch.mean(torch.mean(Bdiag, 1), 0))
    B_array = Bmean.detach().cpu().numpy()
    return B_array

# if __name__ == '__main__':
#     # generate data
#     channel_train = gen_channel(train_size)
#     channel_test = gen_channel(test_size)
#
#     # save data
#     save_data(channel_train, channel_valid, channel_test)
#     data_train_array, data_test_array = load_data()
#
#     get_data_tensor()

# print(channel_train[0][0])
# print(data_train_array[0][0])
# print('------------------------------')
# print(channel_test[0][0])
# print(data_test_array[0][0])
# print('------------------------------')

def extract_active_elements(F):
    """
    F: (B, 1, Nt, Nrf) complex
    Returns:
        F_active: (B, Nt, 1) complex
    """
    B, _, Nt, Nrf = F.shape
    aprf = Nt // Nrf
    F_2d = F[:, 0, :, :]                               # (B, Nt, Nrf)
    F_blocks = F_2d.reshape(B, Nrf, aprf, Nrf)          # (B, Nrf, aprf, Nrf)
    F_diag = torch.diagonal(F_blocks, dim1=1, dim2=3)   # (B, aprf, Nrf)
    F_active = F_diag.permute(0, 2, 1).reshape(B, Nt, 1)
    return F_active
    
def safe_legend(**kwargs):
    """Add legend only when labeled artists exist to avoid Matplotlib warnings."""
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    valid = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith('_')]
    if not valid:
        return
    valid_handles, valid_labels = zip(*valid)
    # Use opaque legend frame to keep EPS exports warning-free.
    kwargs.setdefault('framealpha', 1.0)
    ax.legend(valid_handles, valid_labels, **kwargs)