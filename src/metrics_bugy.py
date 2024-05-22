from torchvision.ops import box_area, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import wandb
import copy
import numpy as np
import matplotlib.pyplot as plt

class MetricModule():

    def __init__(self, cfg_scores, cfg, nb_classes_model, nb_classes_dataset, known_classes, class_metric=True):

        self.nb_classes_model = nb_classes_model
        self.nb_classes_dataset = nb_classes_dataset
        self.cfg = cfg
        self.threshold_score_minimum = cfg_scores.threshold_score_minimum 
        self.score_threshold = cfg_scores.threshold_score
        self.iou_threshold = cfg_scores.iou_threshold
        self.known_classes = known_classes

        #print("Class metrics is :", class_metric)
        if self.cfg.mAP:
            self.known_maps = MeanAveragePrecision(box_format="xyxy", class_metric=class_metric)
            self.unknown_maps = MeanAveragePrecision(box_format="xyxy", class_metric=class_metric)
        self.reset()

    def reset(self):

        self.true_positives = torch.tensor(0)
        self.false_positives = torch.tensor(0)
        self.false_negatives = torch.tensor(0)
        self.unknown_true_positives = torch.tensor(0)
        self.unknown_false_positives = torch.tensor(0)
        self.unknown_false_negatives = torch.tensor(0)

        self.nb_gt = torch.tensor(0)
        self.nb_known_gt = torch.tensor(0)
        self.nb_unknown_gt = torch.tensor(0)

        self.nb_multiple_match_for_gt_boxes = 0

        self.keep_all_custom_area_targets = np.array([], dtype=int)
        self.keep_all_custom_area_prediction_bad = np.array([], dtype=int)
        self.keep_all_custom_area_prediction_good = np.array([], dtype=int)
        self.keep_all_custom_area_randoms = np.array([], dtype=int)
        self.keep_all_custom_area_true_randoms = np.array([], dtype=int)

        self.keep_all_custom_scores_targets =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_good_predictions_TP =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_good_predictions_FP = [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_unknown_predictions_TP =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_unknown_predictions_FP = [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_randoms =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_true_randoms =  [np.array([], dtype=float), np.array([], dtype=float)]

        if self.cfg.mAP:
            self.known_maps.reset()
            self.unknown_maps.reset()


        self.intersection_unknown_FP_TP_on_CC = 0
        self.intersection_random_and_targets_on_ED = 0


    def get_known_recall(self):
        known_recall = self.true_positives/(self.nb_known_gt + 1e-10)
        return known_recall.item()

    def scatter_hist(self, x, y, label=""):

 
        # Start with a square Figure.
        fig = plt.figure(figsize=(6, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        mask_x = x < 20
        mask_y = y < 100000
        ax.scatter(x[mask_x & mask_y], y[mask_x & mask_y], alpha=0.5)
        fig.suptitle(label)

        ax_histx.hist(x[mask_x & mask_y])
        ax_histy.hist(y[mask_x & mask_y], orientation='horizontal')



    def log_histograms(self, score_id, limits_high=10000000, density=False, normalize=False, cumsum=False):


        # Calculate histograms
        hist_targets, bin_edges = np.histogram(self.keep_all_custom_scores_targets[score_id][self.keep_all_custom_scores_targets[score_id] < limits_high], bins=100, density=density)
        hist_good_prediction_FP, bin_edges = np.histogram(self.keep_all_custom_scores_good_predictions_FP[score_id][self.keep_all_custom_scores_good_predictions_FP[score_id] < limits_high], bins=100, density=density)
        hist_unknown_prediction_FP, bin_edges = np.histogram(self.keep_all_custom_scores_unknown_predictions_FP[score_id][self.keep_all_custom_scores_unknown_predictions_FP[score_id] < limits_high], bins=100, density=density)
        hist_good_prediction_TP, bin_edges = np.histogram(self.keep_all_custom_scores_good_predictions_TP[score_id][self.keep_all_custom_scores_good_predictions_TP[score_id] < limits_high], bins=100, density=density)
        hist_unknown_prediction_TP, bin_edges = np.histogram(self.keep_all_custom_scores_unknown_predictions_TP[score_id][self.keep_all_custom_scores_unknown_predictions_TP[score_id] < limits_high], bins=100, density=density)
        hist_randoms, bin_edges = np.histogram(self.keep_all_custom_scores_randoms[score_id][self.keep_all_custom_scores_randoms[score_id]< limits_high], bins=100, density=density)
        hist_true_randoms, bin_edges = np.histogram(self.keep_all_custom_scores_true_randoms[score_id][self.keep_all_custom_scores_true_randoms[score_id]< limits_high], bins=100, density=density)


        if normalize:
            hist_targets = hist_targets / len(self.keep_all_custom_scores_targets[score_id])
            hist_good_prediction_FP = hist_good_prediction_FP / len(self.keep_all_custom_scores_good_predictions_FP[score_id])
            hist_unknown_prediction_FP = hist_unknown_prediction_FP / len(self.keep_all_custom_scores_unknown_predictions_FP[score_id])
            hist_good_prediction_TP = hist_good_prediction_TP / len(self.keep_all_custom_scores_good_predictions_TP[score_id])
            hist_unknown_prediction_TP = hist_unknown_prediction_TP / len(self.keep_all_custom_scores_unknown_predictions_TP[score_id])
            hist_randoms = hist_randoms / len(self.keep_all_custom_scores_randoms[score_id])
            hist_true_randoms = hist_true_randoms / len(self.keep_all_custom_scores_true_randoms[score_id])

            if score_id == 0:
                self.intersection_unknown_FP_TP_on_CC = np.sum(np.minimum(hist_unknown_prediction_FP, hist_unknown_prediction_TP))
            elif score_id == 1:
                self.intersection_random_and_targets_on_ED = np.sum(np.minimum(hist_targets, hist_randoms))

        if cumsum:
            hist_targets = np.cumsum(hist_targets)
            hist_good_prediction_FP = np.cumsum(hist_good_prediction_FP)
            hist_good_prediction_TP = np.cumsum(hist_good_prediction_TP)
            hist_unknown_prediction_FP = np.cumsum(hist_unknown_prediction_FP)
            hist_unknown_prediction_TP = np.cumsum(hist_unknown_prediction_TP)
            hist_randoms = np.cumsum(hist_randoms)
            hist_true_randoms = np.cumsum(hist_true_randoms)


    def log_target_background(self):

        fig = plt.figure()
        plt.plot(hist_targets, label="targets (" + str( len(self.keep_all_custom_scores_targets[score_id])) + " boxes)", c="darkblue")
        target_heatmap_area_cc = np.zeros((nb_interval , nb_interval ), dtype=int)

        for i, area in enumerate(self.keep_all_custom_area_targets):

            cc = self.keep_all_custom_scores_targets[0][i]
            edge = self.keep_all_custom_scores_targets[1][i]

            if cc == -1 or edge == -1:
                continue


            cc_index = np.searchsorted(cc_intervals, cc) 
            edge_index = np.searchsorted(edge_intervals, edge) 
            area_index = np.searchsorted(area_intervals, area) 

            print(cc_index, edge_index, area_index)

            target_heatmap_cc_edge[cc_index, edge_index] += 1
            target_heatmap_area_edge[area_index, edge_index] += 1
            target_heatmap_area_cc[area_index, cc_index] += 1

        label_each = 8

        # Plot the first heatmap
        axs[0][0].imshow(target_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[0][0].set_title(" cc over edge")
        axs[0][0].set_xlabel("Color contrast")
        axs[0][0].set_ylabel("Edge")
        axs[0][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])
        axs[0][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[0][1].imshow(target_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[0][1].set_title(" area over edge")
        axs[0][1].set_xlabel("Area")
        axs[0][1].set_ylabel("Edge")
        axs[0][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[0][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[0][2].imshow(target_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[0][2].set_title(" area over cc")
        axs[0][2].set_xlabel("Area")
        axs[0][2].set_ylabel("Color contrast")
        axs[0][2].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[0][2].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])


        random_heatmap_cc_edge = np.zeros((nb_interval , nb_interval ), dtype=int)
        random_heatmap_area_edge = np.zeros((nb_interval , nb_interval ), dtype=int)
        random_heatmap_area_cc = np.zeros((nb_interval , nb_interval ), dtype=int)

        for i, area in enumerate(self.keep_all_custom_area_randoms):

            cc = self.keep_all_custom_scores_randoms[0][i]
            edge = self.keep_all_custom_scores_randoms[1][i]

            if cc == -1 or edge == -1:
                continue

            cc_index = np.searchsorted(cc_intervals, cc) 
            edge_index = np.searchsorted(edge_intervals, edge) 
            area_index = np.searchsorted(area_intervals, area) 

            print(cc_index, edge_index, area_index)

            random_heatmap_cc_edge[cc_index, edge_index] += 1
            random_heatmap_area_edge[area_index, edge_index] += 1
            random_heatmap_area_cc[area_index, cc_index] += 1


        # Plot the first heatmap
        axs[1][0].imshow(random_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[1][0].set_title(" cc over edge")
        axs[1][0].set_xlabel("Color contrast")
        axs[1][0].set_ylabel("Edge")
        axs[1][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])
        axs[1][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[1][1].imshow(random_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[1][1].set_title(" area over edge")
        axs[1][1].set_xlabel("Area")
        axs[1][1].set_ylabel("Edge")
        axs[1][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[1][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[1][2].imshow(random_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[1][2].set_title(" area over cc")
        axs[1][2].set_xlabel("Area")
        axs[1][2].set_ylabel("Color contrast")
        axs[1][2].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[1][2].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])

        random_cc_intervals = get_interval(self.keep_all_custom_scores_randoms[0], nb_interval)
        random_edge_intervals = get_interval(self.keep_all_custom_scores_randoms[1], nb_interval)
        random_area_intervals = get_interval(self.keep_all_custom_area_randoms, nb_interval)



        for i, area in enumerate(self.keep_all_custom_area_randoms):

            cc = self.keep_all_custom_scores_randoms[0][i]
            edge = self.keep_all_custom_scores_randoms[1][i]

            if cc == -1 or edge == -1:
                continue

            cc_index = np.searchsorted(random_cc_intervals, cc) 
            edge_index = np.searchsorted(random_edge_intervals, edge) 
            area_index = np.searchsorted(random_area_intervals, area) 

            print(cc_index, edge_index, area_index)

            random_heatmap_cc_edge[cc_index, edge_index] += 1
            random_heatmap_area_edge[area_index, edge_index] += 1
            random_heatmap_area_cc[area_index, cc_index] += 1

        print(cc_intervals, random_cc_intervals)
        print(edge_intervals, random_edge_intervals)
        print(area_intervals, random_area_intervals)

        # Plot the first heatmap
        axs[2][0].imshow(random_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[2][0].set_title(" cc over edge")
        axs[2][0].set_xlabel("Color contrast")
        axs[2][0].set_ylabel("Edge")
        axs[2][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_cc_intervals[0::label_each])
        axs[2][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_edge_intervals[0::label_each])

        axs[2][1].imshow(random_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[2][1].set_title(" area over edge")
        axs[2][1].set_xlabel("Area")
        axs[2][1].set_ylabel("Edge")
        axs[2][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_area_intervals[0::label_each])
        axs[2][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_edge_intervals[0::label_each])

        axs[2][2].imshow(random_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        axs[2][2].set_title(" area over cc")
        axs[2][2].set_xlabel("Area")
        axs[2][2].set_ylabel("Color contrast")
        axs[2][2].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_area_intervals[0::label_each])
        axs[2][2].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_cc_intervals[0::label_each])
        plt.show()
        exit()


    def log_histograms(self, score_id, limits_high=10000000, density=False, normalize=False, cumsum=False):


        # Calculate histograms
        hist_targets, bin_edges = np.histogram(self.keep_all_custom_scores_targets[score_id][self.keep_all_custom_scores_targets[score_id] < limits_high], bins=100, density=density)
        hist_good_prediction_FP, bin_edges = np.histogram(self.keep_all_custom_scores_good_predictions_FP[score_id][self.keep_all_custom_scores_good_predictions_FP[score_id] < limits_high], bins=100, density=density)
        hist_unknown_prediction_FP, bin_edges = np.histogram(self.keep_all_custom_scores_unknown_predictions_FP[score_id][self.keep_all_custom_scores_unknown_predictions_FP[score_id] < limits_high], bins=100, density=density)
        hist_good_prediction_TP, bin_edges = np.histogram(self.keep_all_custom_scores_good_predictions_TP[score_id][self.keep_all_custom_scores_good_predictions_TP[score_id] < limits_high], bins=100, density=density)
        hist_unknown_prediction_TP, bin_edges = np.histogram(self.keep_all_custom_scores_unknown_predictions_TP[score_id][self.keep_all_custom_scores_unknown_predictions_TP[score_id] < limits_high], bins=100, density=density)
        hist_randoms, bin_edges = np.histogram(self.keep_all_custom_scores_randoms[score_id][self.keep_all_custom_scores_randoms[score_id]< limits_high], bins=100, density=density)
        hist_true_randoms, bin_edges = np.histogram(self.keep_all_custom_scores_true_randoms[score_id][self.keep_all_custom_scores_true_randoms[score_id]< limits_high], bins=100, density=density)


        if normalize:
            hist_targets = hist_targets / len(self.keep_all_custom_scores_targets[score_id])
            hist_good_prediction_FP = hist_good_prediction_FP / len(self.keep_all_custom_scores_good_predictions_FP[score_id])
            hist_unknown_prediction_FP = hist_unknown_prediction_FP / len(self.keep_all_custom_scores_unknown_predictions_FP[score_id])
            hist_good_prediction_TP = hist_good_prediction_TP / len(self.keep_all_custom_scores_good_predictions_TP[score_id])
            hist_unknown_prediction_TP = hist_unknown_prediction_TP / len(self.keep_all_custom_scores_unknown_predictions_TP[score_id])
            hist_randoms = hist_randoms / len(self.keep_all_custom_scores_randoms[score_id])
            hist_true_randoms = hist_true_randoms / len(self.keep_all_custom_scores_true_randoms[score_id])

            if score_id == 0:
                self.intersection_unknown_FP_TP_on_CC = np.sum(np.minimum(hist_unknown_prediction_FP, hist_unknown_prediction_TP))
            elif score_id == 1:
                self.intersection_random_and_targets_on_ED = np.sum(np.minimum(hist_targets, hist_randoms))

        if cumsum:
            hist_targets = np.cumsum(hist_targets)
            hist_good_prediction_FP = np.cumsum(hist_good_prediction_FP)
            hist_good_prediction_TP = np.cumsum(hist_good_prediction_TP)
            hist_unknown_prediction_FP = np.cumsum(hist_unknown_prediction_FP)
            hist_unknown_prediction_TP = np.cumsum(hist_unknown_prediction_TP)
            hist_randoms = np.cumsum(hist_randoms)
            hist_true_randoms = np.cumsum(hist_true_randoms)


        fig = plt.figure()
        plt.plot(hist_targets, label="targets (" + str( len(self.keep_all_custom_scores_targets[score_id])) + " boxes)", c="darkblue")
        plt.plot(hist_good_prediction_TP, label="good predictions TP (" + str( len(self.keep_all_custom_scores_good_predictions_TP[score_id])) + " boxes)", c="blueviolet")
        plt.plot(hist_good_prediction_FP, label="good predictions FP (" + str( len(self.keep_all_custom_scores_good_predictions_FP[score_id])) + " boxes)", c="magenta")
        plt.plot(hist_unknown_prediction_TP, label="unknown predictions TP (" + str( len(self.keep_all_custom_scores_unknown_predictions_TP[score_id])) + " boxes)", c="red")
        plt.plot(hist_unknown_prediction_FP, label="unknown predictions FP (" + str( len(self.keep_all_custom_scores_unknown_predictions_FP[score_id])) + " boxes)", c="orange")
        plt.plot(hist_randoms, label="random (" + str( len(self.keep_all_custom_scores_randoms[score_id])) + " boxes)", c="limegreen")
        plt.plot(hist_true_randoms, label="true random (" + str( len(self.keep_all_custom_scores_true_randoms[score_id])) + " boxes)", c="darkolivegreen")
        plt.legend(loc="best")

        if score_id == 0:
            plt.title("Histogram of boxes scores Color Contrast")
        elif score_id == 1:
            plt.title("Histogram of boxes scores Edge density")

        hist_plot = wandb.Image(plt)

        return hist_plot

    def get_wandb_metrics(self, with_print=False):

        self.log_target_background()

        output = {}

        output["Number of GT boxes"] = self.nb_gt
        output["Number of known GT boxes"] = self.nb_known_gt
        output["Number of unknown GT boxes"] = self.nb_unknown_gt

        if self.cfg.flags:
            output["TP"] = self.true_positives.item()
            output["FP"] = self.false_positives.item()
            output["FN"] = self.false_negatives.item()

            output["unknown TP"] = self.unknown_true_positives.item()
            output["unknown FP"] = self.unknown_false_positives.item()
            output["unknown FN"] = self.unknown_false_negatives.item()
            output["Number of multiple match for gt boxes"] = self.nb_multiple_match_for_gt_boxes

        if self.cfg.precision:
            known_precision = self.true_positives/(self.true_positives + self.false_positives + 1e-10)
            output["known Precision"] = known_precision.item()
            unknown_precision = self.unknown_true_positives/(self.unknown_true_positives + self.unknown_false_positives + 1e-10)
            output["unknown Precision"] = unknown_precision.item()


        if self.cfg.recall:
            known_recall = self.get_known_recall()
            output["Recall"] =  known_recall
            unknown_recall = self.unknown_true_positives/(self.nb_unknown_gt + 1e-10)
            output["Unknown Recall"] = unknown_recall.item()

        if self.cfg.f1_score:
            known_f1_score = ((known_precision * known_recall)/(known_precision + known_recall + 1e-10)) * 2
            output["known f1 score"] = known_f1_score.item()
            unknown_f1_score = ((unknown_precision * unknown_recall)/(unknown_precision + unknown_recall + 1e-10)) * 2
            output["unknown f1 score"] = unknown_f1_score.item()

        if self.cfg.mAP:
            output["known mAPs"] = self.known_maps.compute()
            output["unknown mAPs"] = self.unknown_maps.compute()



        mask = np.where(self.keep_all_custom_scores_randoms[0]  > 0)
        self.keep_all_custom_scores_randoms[0] = self.keep_all_custom_scores_randoms[0][mask]
        self.keep_all_custom_area_randoms = self.keep_all_custom_area_randoms[mask]

        # Draw histogram of color contrast scores
        CC_density_hists = self.log_histograms(0, limits_high=10, density=True)
        CC_hists = self.log_histograms(0, limits_high=10, density=False)
        CC_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True)

        

        # Draw cumulative proba of color contrast scores
        CC_cumsum_density_hists = self.log_histograms(0, limits_high=10, density=True, cumsum=True)
        CC_cumsum_hists = self.log_histograms(0, limits_high=10, density=False, cumsum=True)
        CC_cumsum_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True, cumsum=True)

        wandb.log({"Histogram of boxes scores Color Contrast": [CC_hists, CC_normalize_hists, CC_density_hists, CC_cumsum_hists, CC_cumsum_normalize_hists, CC_cumsum_density_hists]})

        # Draw histogram of edge density scores
        ED_density_hists = self.log_histograms(1, limits_high=1000000, density=True)
        ED_hists = self.log_histograms(1, limits_high=1000000, density=False)
        ED_normalize_hists = self.log_histograms(1, limits_high=1000000, density=False, normalize=True)

        # Draw histogram cumsum of edge density scores ZOOM
        ED_cumsum_hists = self.log_histograms(1, limits_high=1000000, density=False, cumsum=True)
        ED_cumsum_normalize_hists = self.log_histograms(1, limits_high=1000000, density=False, normalize=True, cumsum=True)
        ED_cumsum_density_hists = self.log_histograms(1, limits_high=1000000, density=True, normalize=True, cumsum=True)

        wandb.log({"Histogramms of boxes scores Edge Density": [ED_hists, ED_normalize_hists, ED_density_hists, ED_cumsum_hists, ED_cumsum_normalize_hists, ED_cumsum_density_hists]})

        # Draw histogram of edge density scores ZOOM
        ED_zoom_density_hists = self.log_histograms(1, limits_high=5000, density=True)
        ED_zoom_hists = self.log_histograms(1, limits_high=5000, density=False)
        ED_zoom_normalize_hists = self.log_histograms(1, limits_high=5000, density=False, normalize=True)

        wandb.log({"Histogram of boxes scores Edge Density ZOOM": [ED_zoom_hists, ED_zoom_normalize_hists, ED_zoom_density_hists]})

        #print(type(self.keep_all_custom_area_targets))
        #print(self.keep_all_custom_area_targets)

        plt.clf()
        plt.close('all')

        """
        hist_color_contrast_targets = plt.hist(self.keep_all_custom_scores_targets[0], bins=1000)
        plt.show()
        hist_color_contrast_targets = plt.hist(self.keep_all_custom_scores_good_predictions_FP[0], bins=1000)
        plt.show()
        """
        """
        hist_boxes_size_targets = plt.hist(self.keep_all_custom_area_targets)
        mask_small = self.keep_all_custom_area_targets < 32 * 32
        mask_big = self.keep_all_custom_area_targets > 96 * 96
        print("nb_small : ",len(self.keep_all_custom_area_targets[mask_small]))
        print("nb_medium: ",len(self.keep_all_custom_area_targets[~mask_small & ~mask_big]))
        print("nb_large : ",len(self.keep_all_custom_area_targets[mask_big]))
        plt.show()
        hist_boxes_size_prediction_bad = plt.hist(self.keep_all_custom_area_prediction_bad)
        mask_small = self.keep_all_custom_area_prediction_bad < 32 * 32
        mask_big = self.keep_all_custom_area_prediction_bad > 96 * 96
        print("nb_small : ",len(self.keep_all_custom_area_prediction_bad[mask_small]))
        print("nb_medium: ",len(self.keep_all_custom_area_prediction_bad[~mask_small & ~mask_big]))
        print("nb_large : ",len(self.keep_all_custom_area_prediction_bad[mask_big]))
        plt.show()
        hist_boxes_size_predictions_good = plt.hist(self.keep_all_custom_area_prediction_good)
        mask_small = self.keep_all_custom_area_prediction_good < 32 * 32
        mask_big = self.keep_all_custom_area_prediction_good > 96 * 96
        print("nb_small : ",len(self.keep_all_custom_area_prediction_good[mask_small]))
        print("nb_medium: ",len(self.keep_all_custom_area_prediction_good[~mask_small & ~mask_big]))
        print("nb_large : ",len(self.keep_all_custom_area_prediction_good[mask_big]))
        plt.show()
        """
            self.keep_all_custom_scores_good_predictions_FP[0] = np.append(self.keep_all_custom_scores_good_predictions_FP[0], prediction["custom_scores"]["color_contrast"][~prediction["tags"]].cpu())
            self.keep_all_custom_scores_good_predictions_FP[1] = np.append(self.keep_all_custom_scores_good_predictions_FP[1], prediction["custom_scores"]["edge_density"][~prediction["tags"]].cpu())

            area_boxes =  (prediction["boxes"][:, 2].int() - prediction["boxes"][:, 0].int()) * (prediction["boxes"][:, 3].int() - prediction["boxes"][:, 1].int())
            self.keep_all_custom_area_prediction_bad = np.append(self.keep_all_custom_area_prediction_bad, area_boxes[~prediction["tags"]].cpu().numpy())
            self.keep_all_custom_area_prediction_good = np.append(self.keep_all_custom_area_prediction_good, area_boxes[prediction["tags"]].cpu().numpy())

        if unknown_predictions != None:
            # Select only unknown predictions 
            for prediction in unknown_predictions:
                self.keep_all_custom_scores_unknown_predictions_TP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_TP[0], prediction["custom_scores"]["color_contrast"][prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_TP[1] = np.append(self.keep_all_custom_scores_unknown_predictions_TP[1], prediction["custom_scores"]["edge_density"][prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_FP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_FP[0], prediction["custom_scores"]["color_contrast"][~prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_FP[1] = np.append(self.keep_all_custom_scores_unknown_predictions_FP[1], prediction["custom_scores"]["edge_density"][~prediction["tags"]].cpu())
            
