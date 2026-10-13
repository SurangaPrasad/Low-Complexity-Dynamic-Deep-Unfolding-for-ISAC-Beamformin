
from utility import *
from PGA_models import *

# load at for old MSE-based models (conv_PGA, UPGA_J10_PC, etc.)
_, H_test_tmp = get_data_tensor(data_source)
H_test_tmp1 = H_test_tmp[:, :test_size, :, :]
R_tmp, at0, _, ideal_beam = get_radar_data(snr_dB, H_test_tmp1)
at = at0[:, : test_size, :, :]

def execute_conv_PGA(model_conv_PGA, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_conv_PGA.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J1)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_conv_PGA_J5(conv_PGA_J5, H_test, Pt):
    # Conventional PGA with J=5 inner iterations (CRB-based sensing metric)
    rates, crbs, _, F, W, _, _ = conv_PGA_J5.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J5)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_conv_PGA_J10(conv_PGA_J10, H_test, Pt):
    # Conventional PGA with J=10 inner iterations (CRB-based sensing metric)
    rates, crbs, _, F, W, _, _ = conv_PGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J4(model_UPGA_J4, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_UPGA_J4.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J4)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J5(model_UPGA_J5, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_UPGA_J5.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J5)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J6(model_UPGA_J6, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_UPGA_J6.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J6)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J10(model_UPGA_J10, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_UPGA_J10.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J20(model_UPGA_J20, H_test, Pt):
    rates, crbs, _, F, W, _, _ = model_UPGA_J20.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J20)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr



def execute_UPGA_J10_PC(model_UPJA_J10_PC, H_test, R, Pt):
    rate, tau, F, W = model_UPJA_J10_PC.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr

def execute_conv_PGA_J10_PC(conv_PGA_J10_PC, H_test, R, Pt):
    # executing classical PGA on the test set
    rate, tau, F, W = conv_PGA_J10_PC.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    return rate_avr

def execute_UPGA_J5_decay(model_UPGA_J5_decay, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J5_decay.execute_PGA(H_test, xi_0, theta_desire, R_N_inv, Pt, n_iter_outer, n_iter_inner_J5)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J10_decay(model_UPGA_J10_decay, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J10_decay.execute_PGA(H_test, xi_0, theta_desire, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J20_decay(model_UPGA_J20_decay, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J20_decay.execute_PGA(H_test, xi_0, theta_desire, R_N_inv, Pt, n_iter_outer, n_iter_inner_J20)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J_GradReuse(model_UPGA_J_GradReuse, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J_GradReuse.execute_PGA(H_test, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr

def execute_UPGA_J_GradReuse_SC(model_UPGA_J_GradReuse_SC, H_test, Pt):
    rates, crbs, _, F, W = model_UPGA_J_GradReuse_SC.execute_PGA(
        H_test, xi_0, theta_desire, R_N_inv, Pt, n_iter_outer, n_iter_inner_J5
    )
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rates) / len(H_test[0]))][-1]
    crb_avr = [r.detach().item() for r in (sum(crbs) / len(H_test[0]))][-1]
    return rate_avr, crb_avr