from algorithms import *
import scipy.io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from utility import safe_legend

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 0
load_saved_plot_data = 0

benchmark = 1


def get_snr_plot_cache_file_name():
    return directory_result + 'snr_plot_cache_' + str(Nt) + '_' + str(OMEGA) + '.mat'


def save_snr_plot_cache(file_path, namespace):
    """Persist SNR sweep arrays so figures can be redrawn without rerunning models."""
    prefixes = (
        'rate_',
        'CRB_',
        'layers_',
    )
    payload = {'snr_dB_list': np.asarray(namespace['snr_dB_list'])}
    for key, value in namespace.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            payload[key] = np.asarray(value)
    scipy.io.savemat(file_path, payload)
    print(f'Saved SNR plot cache to {file_path}')


def load_snr_plot_cache(file_path):
    """Load cached SNR sweep arrays saved by save_snr_plot_cache."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'SNR plot cache file not found: {file_path}')

    cache = scipy.io.loadmat(file_path, simplify_cells=True)
    loaded_data = {}
    for key, value in cache.items():
        if key.startswith('__'):
            continue
        loaded_data[key] = np.asarray(value) if isinstance(value, list) else value
    return loaded_data


# ///////////////////////////////////////// SHOW RATES VS. SNRs ///////////////////////////////////
snr_dB_list = np.array([0, 2, 4, 6, 8, 10, 12], dtype='float64')

if run_program == 1:
    # Load training data
    _, H_test0 = get_data_tensor(data_source)
    H_test = H_test0[:, :test_size, :, :]

    # Create new model for conventional PGA
    if run_conv_PGA == 1:
        model_conv_PGA = PGA_Unfold_JX(step_size_UPGA_J1)
    if run_conv_PGA_J5 == 1:
        conv_PGA_J5 = PGA_Unfold_JX(torch.full_like(step_size_UPGA_J5, step_size_fixed, requires_grad=False, ))
    if run_conv_PGA_J10 == 1:
        conv_PGA_J10 = PGA_Unfold_JX(torch.full_like(step_size_UPGA_J10, step_size_fixed, requires_grad=False, ))

    # Create new model and load states for unfolded PGA with different J values
    if run_UPGA_J1 == 1:
        model_UPGA_J1 = PGA_Unfold_JX(step_size_UPGA_J1)
        model_UPGA_J1.load_state_dict(torch.load(load_model_file('UPGA_J1'), map_location=device))
    if run_UPGA_J4 == 1:
        model_UPGA_J4 = PGA_Unfold_JX(step_size_UPGA_J4)
        model_UPGA_J4.load_state_dict(torch.load(load_model_file('UPGA_J4'), map_location=device))
    if run_UPGA_J5 == 1:
        model_UPGA_J5 = PGA_Unfold_JX(step_size_UPGA_J5)
        model_UPGA_J5.load_state_dict(torch.load(load_model_file('UPGA_J5'), map_location=device))
    if run_UPGA_J6 == 1:
        model_UPGA_J6 = PGA_Unfold_JX(step_size_UPGA_J6)
        model_UPGA_J6.load_state_dict(torch.load(load_model_file('UPGA_J6'), map_location=device))
    if run_UPGA_J10 == 1:
        model_UPGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(load_model_file('UPGA_J10'), map_location=device))
    if run_UPGA_J20 == 1:
        model_UPGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
        model_UPGA_J20.load_state_dict(torch.load(load_model_file('UPGA_J20'), map_location=device))

    # Create new model and load states for unfolded PGA with decay
    if run_UPGA_J5_decay == 1:
        model_UPGA_J5_decay = PGA_Unfold_JX_decay(step_size_UPGA_J5_decay)
        model_UPGA_J5_decay.load_state_dict(torch.load(load_model_file('UPGA_J5_decay'), map_location=device))
    if run_UPGA_J10_decay == 1:
        model_UPGA_J10_decay = PGA_Unfold_JX_decay(step_size_UPGA_J10_decay)
        model_UPGA_J10_decay.load_state_dict(torch.load(load_model_file('UPGA_J10_decay'), map_location=device))
    if run_UPGA_J20_decay == 1:
        model_UPGA_J20_decay = PGA_Unfold_JX_decay(step_size_UPGA_J20_decay)
        model_UPGA_J20_decay.load_state_dict(torch.load(load_model_file('UPGA_J20_decay'), map_location=device))

    rate_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_conv_PGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)

    rate_UPGA_J4 = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J6 = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)

    rate_UPGA_J5_decay = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
    rate_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)
    layers_UPGA_J5_decay = []
    layers_UPGA_J10_decay = []
    layers_UPGA_J20_decay = []

    # CRB-based sensing metrics
    CRB_conv_PGA = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_conv_PGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_conv_PGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)

    CRB_UPGA_J4 = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J5 = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J6 = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype=float)

    CRB_UPGA_J5_decay = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J10_decay = np.zeros([len(snr_dB_list), ], dtype=float)
    CRB_UPGA_J20_decay = np.zeros([len(snr_dB_list), ], dtype=float)

    for ss in range(len(snr_dB_list)):
        snr_dB = snr_dB_list[ss]
        snr_ss = 10 ** (snr_dB / 10)
        print('---------------------- snr = ' + str(snr_dB))

        # load data
        R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

        # Conventional PGA
        if run_conv_PGA == 1:
            rate_conv_PGA[ss], CRB_conv_PGA[ss] = execute_conv_PGA(model_conv_PGA, H_test, snr_ss)
        if run_conv_PGA_J5 == 1:
            rate_conv_PGA_J5[ss], CRB_conv_PGA_J5[ss] = execute_conv_PGA_J5(conv_PGA_J5, H_test, snr_ss)
        if run_conv_PGA_J10 == 1:
            rate_conv_PGA_J10[ss], CRB_conv_PGA_J10[ss] = execute_conv_PGA_J10(conv_PGA_J10, H_test, snr_ss)

        if run_UPGA_J4 == 1:
            rate_UPGA_J4[ss], CRB_UPGA_J4[ss] = execute_UPGA_J4(model_UPGA_J4, H_test, snr_ss)
        if run_UPGA_J5 == 1:
            rate_UPGA_J5[ss], CRB_UPGA_J5[ss] = execute_UPGA_J5(model_UPGA_J5, H_test, snr_ss)
        if run_UPGA_J6 == 1:
            rate_UPGA_J6[ss], CRB_UPGA_J6[ss] = execute_UPGA_J6(model_UPGA_J6, H_test, snr_ss)
        if run_UPGA_J10 == 1:
            rate_UPGA_J10[ss], CRB_UPGA_J10[ss] = execute_UPGA_J10(model_UPGA_J10, H_test, snr_ss)
        if run_UPGA_J20 == 1:
            rate_UPGA_J20[ss], CRB_UPGA_J20[ss] = execute_UPGA_J20(model_UPGA_J20, H_test, snr_ss)

        if run_UPGA_J5_decay == 1:
            rate_UPGA_J5_decay[ss], CRB_UPGA_J5_decay[ss] = execute_UPGA_J5_decay(model_UPGA_J5_decay, H_test, snr_ss)
            layers_UPGA_J5_decay.append(total_inner_layers(model_UPGA_J5_decay.inner_iter_history))
        if run_UPGA_J10_decay == 1:
            rate_UPGA_J10_decay[ss], CRB_UPGA_J10_decay[ss] = execute_UPGA_J10_decay(model_UPGA_J10_decay, H_test, snr_ss)
            layers_UPGA_J10_decay.append(total_inner_layers(model_UPGA_J10_decay.inner_iter_history))
        if run_UPGA_J20_decay == 1:
            rate_UPGA_J20_decay[ss], CRB_UPGA_J20_decay[ss] = execute_UPGA_J20_decay(model_UPGA_J20_decay, H_test, snr_ss)
            layers_UPGA_J20_decay.append(total_inner_layers(model_UPGA_J20_decay.inner_iter_history))

    if run_UPGA_J5_decay == 1 and layers_UPGA_J5_decay:
        n_layers = int(round(float(np.mean(layers_UPGA_J5_decay))))
        label_UPGA_J5_decay = make_dynamic_upga_label(n_layers)
        print(f'Dynamic-UPGA (J_max=5) mean total inner layers over SNR: {n_layers}')
        print(f'  label: {label_UPGA_J5_decay}')
    if run_UPGA_J10_decay == 1 and layers_UPGA_J10_decay:
        n_layers = int(round(float(np.mean(layers_UPGA_J10_decay))))
        label_UPGA_J10_decay = make_dynamic_upga_label(n_layers)
        print(f'Dynamic-UPGA (J_max=10) mean total inner layers over SNR: {n_layers}')
        print(f'  label: {label_UPGA_J10_decay}')
    if run_UPGA_J20_decay == 1 and layers_UPGA_J20_decay:
        n_layers = int(round(float(np.mean(layers_UPGA_J20_decay))))
        label_UPGA_J20_decay = make_dynamic_upga_label(n_layers)
        print(f'Dynamic-UPGA (J_max=20) mean total inner layers over SNR: {n_layers}')
        print(f'  label: {label_UPGA_J20_decay}')

    # Always refresh SNR plot cache after expensive run.
    print('Saving SNR plot data...')
    save_snr_plot_cache(get_snr_plot_cache_file_name(), locals())

if plot_figure == 1:
    if load_saved_plot_data == 1:
        snr_cache_file_name = get_snr_plot_cache_file_name()
        print(f'Loading SNR plot data from {snr_cache_file_name}...')
        globals().update(load_snr_plot_cache(snr_cache_file_name))

    # Rebuild Dynamic-UPGA labels from measured (or cached) per-SNR layer counts.
    if run_UPGA_J5_decay == 1 and 'layers_UPGA_J5_decay' in globals() and len(np.atleast_1d(layers_UPGA_J5_decay)) > 0:
        label_UPGA_J5_decay = make_dynamic_upga_label(int(round(float(np.mean(np.atleast_1d(layers_UPGA_J5_decay))))))
    if run_UPGA_J10_decay == 1 and 'layers_UPGA_J10_decay' in globals() and len(np.atleast_1d(layers_UPGA_J10_decay)) > 0:
        label_UPGA_J10_decay = make_dynamic_upga_label(int(round(float(np.mean(np.atleast_1d(layers_UPGA_J10_decay))))))
    if run_UPGA_J20_decay == 1 and 'layers_UPGA_J20_decay' in globals() and len(np.atleast_1d(layers_UPGA_J20_decay)) > 0:
        label_UPGA_J20_decay = make_dynamic_upga_label(int(round(float(np.mean(np.atleast_1d(layers_UPGA_J20_decay))))))

    # ========================== plot rate vs SNR ==========================
    plt.figure(figsize=(8, 4.2))
    if run_conv_PGA == 1:
        plt.plot(snr_dB_list, rate_conv_PGA, '--', color='black', linewidth=3, markersize=8, label=Conv_PGA_J1)
    if run_conv_PGA_J5 == 1:
        plt.plot(snr_dB_list, rate_conv_PGA_J5, '--', color='blue', linewidth=3, markersize=8, label=Conv_PGA_J5)
    if run_conv_PGA_J10 == 1:
        plt.plot(snr_dB_list, rate_conv_PGA_J10, '-*', color='blue', linewidth=3, markersize=8, label=Conv_PGA_J10)
    if run_UPGA_J4 == 1:
        plt.plot(snr_dB_list, rate_UPGA_J4, '--', color='orange', linewidth=3, markersize=8, label=label_UPGA_J4)
    if run_UPGA_J5 == 1:
        plt.plot(snr_dB_list, rate_UPGA_J5, '--', color='red', linewidth=3, markersize=8, label=label_UPGA_J5)
    if run_UPGA_J6 == 1:
        plt.plot(snr_dB_list, rate_UPGA_J6, '-d', color='orange', linewidth=3, markersize=8, label=label_UPGA_J6)
    if run_UPGA_J10 == 1:
        plt.plot(snr_dB_list, rate_UPGA_J10, '-*', color='red', linewidth=3, markersize=8, label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(snr_dB_list, rate_UPGA_J20, '-*', color='red', linewidth=3, markersize=8, label=label_UPGA_J20)
        plt.plot(snr_dB_list, rate_conv_PGA, ':', color='black', linewidth=3, markersize=8, label=label_conv)
    if run_UPGA_J5_decay == 1:
        plt.plot(snr_dB_list, rate_UPGA_J5_decay, '--', color='green', linewidth=3, markersize=8, label=label_UPGA_J5_decay)
    if run_UPGA_J10_decay == 1:
        plt.plot(snr_dB_list, rate_UPGA_J10_decay, '-*', color='green', linewidth=3, markersize=8, label=label_UPGA_J10_decay)
    if run_UPGA_J20_decay == 1:
        plt.plot(snr_dB_list, rate_UPGA_J20_decay, ':p', color='green', linewidth=3, markersize=8, label=label_UPGA_J20_decay)

    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.grid()
    plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)

    # ======================= Plot CRLB vs SNR with inset =======================
    plt.figure(figsize=(8, 4.2))
    ax = plt.gca()

    def crlb_from_log_inv(x):
        return (1 / torch.exp(torch.tensor(x))).detach().cpu().numpy()

    def plot_curve(x, y, style, color, label=None, lw=3, ms=7, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(x, y, style, color=color, linewidth=lw, markersize=ms, label=label)

    curves = []

    if run_conv_PGA == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA), '--', 'black', Conv_PGA_J1))
    if run_conv_PGA_J5 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA_J5), '--', 'blue', Conv_PGA_J5))
    if run_conv_PGA_J10 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_conv_PGA_J10), '-*', 'blue', Conv_PGA_J10))
    if run_UPGA_J4 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J4), '--', 'orange', label_UPGA_J4))
    if run_UPGA_J5 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J5), '--', 'red', label_UPGA_J5))
    if run_UPGA_J6 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J6), '-d', 'orange', label_UPGA_J6))
    if run_UPGA_J10 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J10), '-*', 'red', label_UPGA_J10))
    if run_UPGA_J20 == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J20), ':s', 'red', label_UPGA_J20))
    if run_UPGA_J5_decay == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J5_decay), '--', 'green', label_UPGA_J5_decay))
    if run_UPGA_J10_decay == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J10_decay), '-*', 'green', label_UPGA_J10_decay))
    if run_UPGA_J20_decay == 1:
        curves.append((snr_dB_list, crlb_from_log_inv(CRB_UPGA_J20_decay), ':p', 'green', label_UPGA_J20_decay))

    for x, y, style, color, label in curves:
        plot_curve(x, y, style, color, label=label, ax=ax)

    ax.set_xlabel('SNR [dB]', fontsize=14)
    ax.set_ylabel(r'$\mathrm{CRLB}$', fontsize=14)
    ax.grid(True)

    # Inset zoom
    axins = inset_axes(ax, width="42%", height="42%", loc='upper right', borderpad=1.2)
    for x, y, style, color, label in curves:
        plot_curve(x, y, style, color, ax=axins, lw=2, ms=5)

    axins.set_xlim(10, 12)
    zoom_vals = []
    for x, y, style, color, label in curves:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        mask = (x_arr >= 10) & (x_arr <= 12)
        if np.any(mask):
            zoom_vals.extend(y_arr[mask])

    if len(zoom_vals) > 0:
        ymin = np.min(zoom_vals)
        ymax = np.max(zoom_vals)
        ypad = 0.15 * (ymax - ymin) if ymax > ymin else 0.1 * ymax
        axins.set_ylim(ymin - ypad, ymax + ypad)

    axins.grid(True)
    axins.tick_params(labelsize=8)
    axins.patch.set_alpha(1.0)
    axins.patch.set_facecolor("white")

    mark_inset(ax, axins, loc1=2, loc2=4, fc="white", ec="0.4", linewidth=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=12, labelspacing=0.1, ncol=2, frameon=False, columnspacing=0.6)
    plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(directory_result + 'CRB_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps', bbox_inches='tight', pad_inches=0.02)
