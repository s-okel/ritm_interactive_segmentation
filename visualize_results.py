import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import regex as re
import scipy.stats as st
from termcolor import colored
from isegm.inference.utils import compute_noc_metric


def plots(ious_array, n_clicks, lw=0.5, save=False, font_size=12):
    # statistics
    mean = np.mean(ious_array, axis=0)
    std = np.std(ious_array, axis=0)
    ci = st.t.interval(0.95, ious_array.shape[0], loc=mean, scale=st.sem(ious_array, axis=0))

    # boxplots
    f, ax = plt.subplots()
    ax.boxplot(ious_array[:, :n_clicks], showfliers=False, medianprops=dict(color="#e28743"))
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if save:
        plt.savefig("epoch_evaluations/individual_boxplot.pdf", dpi=300)
    else:
        plt.show()

    # ci
    ''' 
    f, ax = plt.subplots()
    ax.plot(mean, color="#e28743", linewidth=lw)
    ax.plot(ci[0], color="#eab676", linewidth=lw)
    ax.plot(ci[1], color="#eab676", linewidth=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.fill_between(range(50), ci[0], ci[1], color="#f3cfb4", alpha=0.5)
    plt.xlabel("Amount of clicks")
    plt.ylabel("mIoU")
    plt.ylim([0, 1])
        if save:
        plt.savefig("epoch_evaluations/individual_ci.pdf", dpi=300)
    else:
        plt.show()
    '''

    # with +- one std
    f, ax = plt.subplots()
    ax.plot(range(1, 51), mean, color="#e28743", linewidth=lw)
    ax.plot(range(1, 51), mean - std, color="#eab676", linewidth=lw)
    ax.plot(range(1, 51), mean + std, color="#eab676", linewidth=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.fill_between(range(1, 51), mean - std, mean + std, color="#f3cfb4", alpha=0.5)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, 50])
    if save:
        plt.savefig("epoch_evaluations/individual_std.pdf", dpi=300)
    else:
        plt.show()

    # histogram: how many clicks to 80, 90, 95
    f, ax = plt.subplots()

    nocs = []

    for i in range(ious_array.shape[0]):
        noc = get_noc(ious_array[i], 0.8, ious_array.shape[1])
        nocs.append(noc)

    print(f"NoC@80: {np.mean(nocs)}")

    # noc_list, over_max_list = compute_noc_metric(ious_array, [0.8], max_clicks=50)
    # print(noc_list)

    ax.hist(nocs, bins=ious_array.shape[1], histtype='step', color="#e28743", lw=0.5)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.xlim([0, 51])
    if save:
        plt.savefig("epoch_evaluations/individual_histogram.pdf", dpi=300)
    else:
        plt.show()


def get_noc(iou_arr, iou_thr, max_clicks):
    vals = iou_arr >= iou_thr
    return np.argmax(vals) + 1 if np.any(vals) else max_clicks


def combined_plot(data_dict, save=False, font_size=12):
    f, ax = plt.subplots()
    colors = ['#332288', '#88CCEE', '#117733', "#e28743", '#DDCC77', '#44AA99', '#CC6677', '#882255', '#AA4499']
    # colors = ['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']
    selected_colors = colors[0:len(data_dict)]
    for k, color in zip(data_dict, selected_colors):
        ax.plot(range(1, 51), np.mean(data_dict[k], axis=0), linewidth=1, label=k, color=color)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, 50])
    plt.legend(prop={'size': font_size})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig("epoch_evaluations/mIoU_combined.pdf", dpi=300)
    else:
        plt.show()

    f2, ax2 = plt.subplots()
    threshold = 0.8

    for k, color in zip(data_dict, selected_colors):
        nocs = []

        for i in range(data_dict[k].shape[0]):
            noc = get_noc(data_dict[k][i], threshold, data_dict[k].shape[1])
            nocs.append(noc)

        ax2.hist(nocs, bins=data_dict[k].shape[1], histtype='step', label=k, color=color, density=True)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    plt.xlabel(f"Amount of clicks for NoC of {threshold}")
    plt.ylim([0, 1])
    plt.xlim([0, 51])
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_avg_mask_influence(data_dict, structures_dict, save=False, font_size=12):
    avg_mask_list = []
    avg_nocs = []
    keys = []
    noc_threshold = 0.8

    for k in data_dict:
        keys.append(k)
        # return to the original labels
        label = k.lower()
        label = label.replace("superior mesenteric artery", "arteria mesenterica superior")
        label = label.replace(" ", "_")

        nocs = []
        avg_mask_list.append(structures_dict[label]['avg_mask'])
        for i in range(data_dict[k].shape[0]):
            noc = get_noc(data_dict[k][i], noc_threshold, data_dict[k].shape[1])
            nocs.append(noc)
        avg_nocs.append(np.mean(np.array(nocs)))

    f, ax = plt.subplots()
    ax.plot(avg_mask_list, avg_nocs, 'x', color="#e28743")
    for i in range(len(avg_mask_list)):
        if "Tumour" in keys[i]:
            plt.text(avg_mask_list[i] + 70, avg_nocs[i] - 1, keys[i], fontsize=font_size)
        else:
            plt.text(avg_mask_list[i] + 70, avg_nocs[i], keys[i], fontsize=font_size)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.ylim([0, math.ceil(np.max(avg_nocs)) + 1])
    plt.xlabel("Average mask size", fontsize=font_size)
    plt.ylabel(f"NoC@{int(100 * noc_threshold)}", fontsize=font_size)
    plt.xlim([0, 3500])
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig("epoch_evaluations/noc_vs_mask_size.pdf", dpi=300)
    else:
        plt.show()


def load_data_to_plot(data_dict):
    iou_dict = {}

    for model in os.listdir(experiments_path):
        model_path = experiments_path + model + "/"
        for model_try in os.listdir(model_path):
            model_try_path = model_path + model_try + "/"
            evaluation_path = model_try_path + "evaluation_logs/others/"
            if os.path.exists(evaluation_path):
                for epoch in os.listdir(evaluation_path):
                    plots_path = evaluation_path + epoch + "/plots/"
                    if os.path.exists(plots_path):
                        for file in os.listdir(plots_path):
                            for k in data_dict:
                                if ".pickle" in file:
                                    if data_dict[k]['try'] in plots_path and str(
                                            data_dict[k]['epoch']) in plots_path and k in plots_path and '1' + str(
                                            data_dict[k]['epoch']) not in plots_path and '2' + str(
                                            data_dict[k]['epoch']) not in plots_path:
                                        print(colored(f"Key: {k}", "green"))
                                        print(f"Loading {plots_path + file}")
                                        with open(plots_path + file, "rb") as f:
                                            label = k.replace("_", " ")
                                            label = label.capitalize()
                                            label = label.replace("Arteria mesenterica superior",
                                                                  "Superior mesenteric artery")
                                            iou_dict[label] = np.array(pickle.load(f)['all_ious'])

    return iou_dict


def process_results_txt():
    # finds all epoch evaluations and saves them in different formats
    results = {}
    txt_file = "epoch_evaluations/epoch_evaluations.txt"

    if os.path.exists(txt_file):
        os.remove(txt_file)
    for model in os.listdir(experiments_path):
        model_path = experiments_path + model + "/"
        model_name = model.replace("_hrnet64_iter", "")
        print(colored(f"Model name: {model_name}", "green"))
        with open(txt_file, "a") as write_file:
            write_file.write(f"Model name: {model_name}\n")
        results[model_name] = {}
        for try_folder in os.listdir(model_path):
            results_per_try = []
            epochs_per_try = []
            try_nr = try_folder.split("_")[0]
            print(colored(f"\tTry: {try_nr}", "red"))
            with open(txt_file, "a") as write_file:
                write_file.write(f"\tTry: {try_nr}\n")
            results[model_name][try_nr] = {}
            epochs_path = model_path + try_folder + "/evaluation_logs/others/"
            if os.path.exists(epochs_path):
                for epoch in os.listdir(epochs_path):
                    epoch_path = epochs_path + epoch + "/"
                    epoch = epoch.split("-")[1]
                    print(colored(f"\t\tEpoch: {epoch}", "yellow"))
                    with open(txt_file, "a") as write_file:
                        write_file.write(f"\t\tEpoch: {epoch}\n")
                    results[model_name][try_nr][epoch] = {}
                    for file in os.listdir(epoch_path):
                        if ".txt" in file:
                            with open(epoch_path + file, "r") as f:
                                text = f.read()
                                values = re.findall(r"([0-9]{1,2}\.[0-9]{2})", text)
                                values = values[:3] + values[4:]

                                # just print to terminal
                                print(f"\t\t{values}")

                                # save in comprehensive dict
                                results[model_name][try_nr][epoch] = values

                                # write to file
                                with open(txt_file, "a") as write_file:
                                    write_file.write(f"\t\t{values}\n")

                                # add to list/array
                                results_per_try.append([float(item) for item in values])
                                epochs_per_try.append(float(epoch))

            results_per_try = np.array(results_per_try)
            epochs_per_try = np.array(epochs_per_try)[:, None]

            results_per_try = results_per_try.reshape((-1, 9))

            df = pd.DataFrame(np.concatenate((epochs_per_try, results_per_try), axis=1))
            df.to_excel(f"epoch_evaluations/model_{model}_try_{try_nr}.xlsx", index=False, header=False)


def plot_delta(data_dict, save=False, font_size=12):
    colors = ['#332288', '#88CCEE', '#117733', "#e28743", '#DDCC77', '#44AA99', '#CC6677', '#882255', '#AA4499']
    # colors = ['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']
    selected_colors = colors[0:len(data_dict)]

    f, ax = plt.subplots()
    for k, color in zip(data_dict, selected_colors):
        improvement = np.mean(data_dict[k] - np.mean(data_dict[k][:, 0]), axis=0)
        ax.plot(range(1, 51), improvement, linewidth=1, label=k, color=color)
        print(f"{improvement[49]:.2f} improvement after 50 clicks for {k}")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=font_size)
    plt.ylabel(f"Increase in mIoU", fontsize=font_size)
    plt.xlim([1, 50])
    plt.legend(prop={'size': font_size})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig("epoch_evaluations/delta.pdf", dpi=300)
    else:
        plt.show()

    f, ax = plt.subplots()
    plt.hlines(0, 0, 49, colors=['k'], lw=1)
    for k, color in zip(data_dict, selected_colors):
        ax.plot(range(1, 50), np.diff(np.mean(data_dict[k], axis=0)), linewidth=1, label=k, color=color)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks")
    plt.ylabel(f"Delta")
    plt.xlim([1, 49])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    experiments_path = "Z:/Pancreas/interactivity/repos/ritm_interactive_segmentation/experiments/iter_mask/"
    fs = 13
    process_results_txt()

    # final: aorta, SMA, PD
    structures = {'aorta': {'try': '001', 'epoch': 169, 'avg_mask': 3001},
                  'arteria_mesenterica_superior': {'try': '001', 'epoch': 119, 'avg_mask': 252},
                  'common_bile_duct': {'try': '001', 'epoch': 30, 'avg_mask': 501},
                  'gastroduodenalis': {'try': '001', 'epoch': 149, 'avg_mask': 61},
                  'pancreas': {'try': '002', 'epoch': 149, 'avg_mask': 979},
                  'pancreatic_duct': {'try': '000', 'epoch': 79, 'avg_mask': 162},
                  'tumour': {'try': '000', 'epoch': 29, 'avg_mask': 75}}
    # data = load_data_to_plot(structures)
    # plot_delta(data, save=True, font_size=fs)
    # combined_plot(data, save=True, font_size=fs)
    # plot_avg_mask_influence(data, structures, save=True, font_size=fs)
    # plots(data['Pancreas'], n_clicks=20, lw=0.5, save=False, font_size=fs)
