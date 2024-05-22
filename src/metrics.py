from torchvision.ops import box_area, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import BinaryCalibrationError
import torch
import wandb
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pprint import pprint


class MetricModule():

    def __init__(self, cfg_scores, cfg, nb_classes_model=None, nb_classes_dataset=None, known_classes=None, class_metric=True):

        self.nb_classes_model = nb_classes_model
        self.nb_classes_dataset = nb_classes_dataset
        self.cfg = cfg
        self.threshold_score_minimum = cfg_scores.threshold_score_minimum 
        self.score_threshold = cfg_scores.threshold_score
        self.iou_threshold = cfg_scores.iou_threshold
        self.known_classes = known_classes

        self.is_metrics_on_raw_predictions = cfg.metrics_on_raw_predictions
        self.is_metrics_on_proposals_predictions = cfg.metrics_on_proposals_predictions

        self.calibration_bins = cfg.calibration_nb_bins
        if self.calibration_bins > 0:
            self.calibration_metric_classif_pred = BinaryCalibrationError(n_bins=self.calibration_bins)
            self.calibration_metric_iou_gt = BinaryCalibrationError(n_bins=self.calibration_bins)
            self.calibration_metric_centerness_gt = BinaryCalibrationError(n_bins=self.calibration_bins)

        if self.cfg.mAP:
            self.known_maps = MeanAveragePrecision(box_format="xyxy", class_metrics=class_metric, extended_summary=False)
            self.current_known_map = None
            self.current_unknown_map = None
            self.unknown_maps = MeanAveragePrecision(box_format="xyxy", class_metrics=False, extended_summary=False)
        self.reset()

    def reset(self):

        self.true_positives = torch.tensor(0)
        self.false_positives = torch.tensor(0)
        self.false_negatives = torch.tensor(0)
        self.unknown_true_positives = torch.tensor(0)
        self.unknown_false_positives = torch.tensor(0)
        self.unknown_false_negatives = torch.tensor(0)

        self.raw_true_positives = torch.tensor(0)
        self.raw_false_positives = torch.tensor(0)
        self.raw_false_negatives = torch.tensor(0)
        self.raw_false_negatives_known = torch.tensor(0)
        self.raw_false_negatives_unknown = torch.tensor(0)
        self.raw_nb_gt_known = torch.tensor(0)
        self.raw_nb_gt_unknown = torch.tensor(0)


        self.proposals_true_positives = torch.tensor(0)
        self.proposals_false_positives = torch.tensor(0)
        self.proposals_false_negatives = torch.tensor(0)
        self.proposals_false_negatives_known = torch.tensor(0)
        self.proposals_false_negatives_unknown = torch.tensor(0)
        self.proposals_nb_gt_known = torch.tensor(0)
        self.proposals_nb_gt_unknown = torch.tensor(0)

        self.nb_images = torch.tensor(0)

        self.nb_gt_all = torch.tensor(0)
        self.nb_gt = torch.tensor(0)
        self.nb_known_gt = torch.tensor(0)
        self.nb_unknown_gt = torch.tensor(0)
        self.nb_background_gt = torch.tensor(0)

        self.nb_multiple_match_for_gt_boxes = 0

        self.keep_all_custom_area_targets = np.array([], dtype=int)
        self.keep_all_custom_area_prediction_bad = np.array([], dtype=int)
        self.keep_all_custom_area_prediction_good = np.array([], dtype=int)
        self.keep_all_custom_area_randoms = np.array([], dtype=int)
        self.keep_all_custom_area_true_randoms = np.array([], dtype=int)

        # scores ares : 1: Color contrst, 2: edge_density, 3: pourcent drivable , 4: luminance, 5: str
        self.keep_all_custom_scores_targets =  [np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float) ]
        self.keep_all_custom_scores_good_predictions_TP =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_good_predictions_FP = [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_unknown_predictions_TP =  [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_unknown_predictions_FP = [np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_randoms =  [np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)]
        self.keep_all_custom_scores_true_randoms =  [np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)]

        if self.cfg.mAP:
            self.known_maps.reset()
            self.unknown_maps.reset()

        if self.calibration_bins > 0:
            self.calibration_metric_classif_pred.reset()
            self.calibration_metric_iou_gt.reset()
            self.calibration_metric_centerness_gt.reset()



        self.intersection_unknown_FP_TP_on_CC = 0
        self.intersection_random_and_targets_on_ED = 0

        self.open_set_errors = 0



    def get_open_set_errors(self):
        return self.open_set_errors

    # PRECISION

    def get_known_precision(self):
        known_precision = self.true_positives/(self.true_positives + self.false_positives + 1e-10)
        return known_precision.item()

    def get_unknown_precision(self):
        unknown_precision = self.unknown_true_positives/(self.unknown_true_positives + self.unknown_false_positives + 1e-10)
        return unknown_precision.item()

    # RECALL

    def get_known_recall(self):
        known_recall = self.true_positives/(self.true_positives + self.false_negatives + 1e-10)
        return known_recall.item()

    def get_unknown_recall(self):
        unknown_recall = self.unknown_true_positives/(self.unknown_true_positives + self.unknown_false_negatives + 1e-10)
        return unknown_recall.item()

    def get_raw_recall(self):
        raw_false_negatives = self.raw_false_negatives_known + self.raw_false_negatives_unknown
        raw_recall = self.raw_true_positives/(self.raw_true_positives + raw_false_negatives + 1e-10)
        return raw_recall.item()


    # COVERAGE

    def get_pourcent_coverage(self):
        return (((self.nb_gt - (self.false_negatives + self.unknown_false_negatives)) / self.nb_gt) * 100).item()

    def get_pourcent_coverage_known(self):
        return (((self.nb_known_gt - self.false_negatives) / self.nb_known_gt) * 100).item()

    def get_pourcent_coverage_unknown(self):
        return (((self.nb_unknown_gt - self.unknown_false_negatives) / self.nb_unknown_gt) * 100).item()

    def get_raw_pourcent_coverage(self):
        raw_nb_gt = self.raw_nb_gt_known + self.raw_nb_gt_unknown
        raw_false_negatives = self.raw_false_negatives_known + self.raw_false_negatives_unknown
        return (((raw_nb_gt - raw_false_negatives) / raw_nb_gt) * 100).item()

    def get_raw_pourcent_coverage_known(self):
        return (((self.raw_nb_gt_known - self.raw_false_negatives_known) / self.raw_nb_gt_known) * 100).item()

    def get_raw_pourcent_coverage_unknown(self):
        return (((self.raw_nb_gt_unknown - self.raw_false_negatives_unknown) / self.raw_nb_gt_unknown) * 100).item()


    def get_proposals_pourcent_coverage(self):
        proposals_nb_gt = self.proposals_nb_gt_known + self.proposals_nb_gt_unknown
        proposals_false_negatives = self.proposals_false_negatives_known + self.proposals_false_negatives_unknown
        return (((proposals_nb_gt - proposals_false_negatives) / proposals_nb_gt) * 100).item()

    def get_proposals_pourcent_coverage_known(self):
        return (((self.proposals_nb_gt_known - self.proposals_false_negatives_known) / self.proposals_nb_gt_known) * 100).item()

    def get_proposals_pourcent_coverage_unknown(self):
        return (((self.proposals_nb_gt_unknown - self.proposals_false_negatives_unknown) / self.proposals_nb_gt_unknown) * 100).item()






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



    def pca_on_targets_and_background(self):

        area_targets = self.keep_all_custom_area_targets
        cc_targets = self.keep_all_custom_scores_targets[0]
        edge_targets = self.keep_all_custom_scores_targets[1]
        std_targets = self.keep_all_custom_scores_targets[2]
        luminance_targets = self.keep_all_custom_scores_targets[3]
        #drivable_pourcent = self.keep_all_custom_scores_targets[4]

        area_randoms = self.keep_all_custom_area_randoms
        cc_randoms = self.keep_all_custom_scores_randoms[0]
        edge_randoms = self.keep_all_custom_scores_randoms[1]
        std_randoms = self.keep_all_custom_scores_randoms[2]
        luminance_randoms = self.keep_all_custom_scores_randoms[3]

        targets_data = np.column_stack((area_targets, cc_targets, edge_targets, std_targets, luminance_targets))
        randoms_data = np.column_stack((area_randoms, cc_randoms, edge_randoms, std_randoms, luminance_randoms))
        feature_names = ["Area", "Color Contrast", "Edge", "std", "luminance"]

        """
        targets_data = np.column_stack((area_targets, cc_targets, std_targets, luminance_targets))
        randoms_data = np.column_stack((area_randoms, cc_randoms, std_randoms, luminance_randoms))
        feature_names = ["Area", "Color Contrast", "std", "luminance"]
        """

        targets_labels = np.ones((targets_data.shape[0], 1))  
        randoms_labels = 2 * np.ones((randoms_data.shape[0], 1))

        data = np.vstack((np.hstack((targets_data, targets_labels)), np.hstack((randoms_data, randoms_labels))))


        # Standarsize Data
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data[:, :-1])

        
        #TSNE

        model = TSNE(n_components=2, random_state=0)
        res = model.fit_transform(data_standardized) 

        targets_points = res[data[:, -1] == 1]
        randoms_points = res[data[:, -1] == 2]

        plt.figure()
        plt.scatter(targets_points[:, 0], targets_points[:, 1], c='b', label='Targets', alpha=0.5)
        plt.scatter(randoms_points[:, 0], randoms_points[:, 1], c='g', label='Background', alpha=0.5)



        model = TSNE(perplexity=10)
        res = model.fit_transform(data_standardized) 

        targets_points = res[data[:, -1] == 1]
        randoms_points = res[data[:, -1] == 2]

        plt.figure()
        plt.scatter(targets_points[:, 0], targets_points[:, 1], c='b', label='Targets', alpha=0.5)
        plt.scatter(randoms_points[:, 0], randoms_points[:, 1], c='g', label='Background', alpha=0.5)
        plt.title("perplexity 10")



        model = TSNE(perplexity=40)
        res = model.fit_transform(data_standardized) 

        targets_points = res[data[:, -1] == 1]
        randoms_points = res[data[:, -1] == 2]

        plt.figure()
        plt.scatter(targets_points[:, 0], targets_points[:, 1], c='b', label='Targets', alpha=0.5)
        plt.scatter(randoms_points[:, 0], randoms_points[:, 1], c='g', label='Background', alpha=0.5)
        plt.title("perplexity 40")

        pca = PCA(n_components=2)  

        principal_components = pca.fit_transform(data_standardized)
        explained_variance = pca.explained_variance_ratio_

        covariance = pca.get_covariance()

        plt.figure()
        print(covariance)
        plt.imshow(covariance)


        targets_points = principal_components[data[:, -1] == 1]
        randoms_points = principal_components[data[:, -1] == 2]


        plt.figure()
        plt.scatter(targets_points[:, 0], targets_points[:, 1], c='b', label='Targets', alpha=0.5)
        plt.scatter(randoms_points[:, 0], randoms_points[:, 1], c='g', label='Background', alpha=0.5)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA 2D Visualization\nExplained Variance: PC1 = {explained_variance[0]:.2%}, PC2 = {explained_variance[1]:.2%}')
        plt.grid(True)

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Plot the circle of correlations
        for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
            plt.arrow(0, 0, x, y, color='r', alpha=0.9, width=0.003)
            plt.text(x, y, feature_names[i], color='r')

        plt.figure()

        # Principal components correlation coefficients
        loadings = pca.components_

        # Number of features before PCA
        n_features = pca.n_features_in_

        # PC names
        pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]

        # Match PC names to loadings
        pc_loadings = dict(zip(pc_list, loadings))


        # Get the loadings of x and y axes
        xs = loadings[0]
        ys = loadings[1]

        # Plot the loadings on a scatterplot
        for i, varnames in enumerate(feature_names):
            plt.scatter(xs[i], ys[i], s=200)
            plt.arrow(
                0, 0, # coordinates of arrow base
                xs[i], # length of the arrow along x
                ys[i], # length of the arrow along y
                color='r',
                head_width=0.01
                )
            plt.text(xs[i], ys[i], varnames)        

        # Define the axes
        xticks = np.linspace(-0.8, 0.8, num=5)
        yticks = np.linspace(-0.8, 0.8, num=5)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        # Show plot
        plt.title('2D Loading plot')
        plt.show()


    def log_target_background(self):

        if len(self.keep_all_custom_area_targets) == 0 or len(self.keep_all_custom_area_randoms) == 0:
            return

        plt.clf()
        plt.close('all')

        nb_interval = 100

        fig, axs = plt.subplots(3, 3)

        target_heatmap_cc_edge = np.zeros((nb_interval , nb_interval ), dtype=int)
        target_heatmap_area_edge = np.zeros((nb_interval , nb_interval ), dtype=int)
        target_heatmap_area_cc = np.zeros((nb_interval , nb_interval ), dtype=int)

        def get_interval(array, nb_interval):

            value_by_interval = len(array) // nb_interval

            sorted_array = np.sort(array.copy())
            
            intervals = [ sorted_array[i * value_by_interval] for i in range(1, nb_interval)]

            return np.round(intervals, 1) 

        cc_intervals = get_interval(self.keep_all_custom_scores_targets[0], nb_interval)
        edge_intervals = get_interval(self.keep_all_custom_scores_targets[1], nb_interval)
        area_intervals = get_interval(self.keep_all_custom_area_targets, nb_interval)



        for i, area in enumerate(self.keep_all_custom_area_targets):

            cc = self.keep_all_custom_scores_targets[0][i]
            edge = self.keep_all_custom_scores_targets[1][i]

            if cc == -1 or edge == -1:
                continue


            cc_index = np.searchsorted(cc_intervals, cc) 
            edge_index = np.searchsorted(edge_intervals, edge) 
            area_index = np.searchsorted(area_intervals, area) 

            #print(cc_index, edge_index, area_index)

            target_heatmap_cc_edge[cc_index, edge_index] += 1
            target_heatmap_area_edge[area_index, edge_index] += 1
            target_heatmap_area_cc[area_index, cc_index] += 1

        label_each = 8

        # Plot the first heatmap
        axs[0][0].imshow(target_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[0][0].set_title(" cc over edge")
        axs[0][0].set_xlabel("Color contrast")
        axs[0][0].set_ylabel("Edge")
        axs[0][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])
        axs[0][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[0][1].imshow(target_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[0][1].set_title(" area over edge")
        axs[0][1].set_xlabel("Area")
        axs[0][1].set_ylabel("Edge")
        axs[0][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[0][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[0][2].imshow(target_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[0][2].set_title(" area over cc")
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

            #print(cc_index, edge_index, area_index)

            random_heatmap_cc_edge[cc_index, edge_index] += 1
            random_heatmap_area_edge[area_index, edge_index] += 1
            random_heatmap_area_cc[area_index, cc_index] += 1


        # Plot the first heatmap
        axs[1][0].imshow(random_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[1][0].set_title(" cc over edge")
        axs[1][0].set_xlabel("Color contrast")
        axs[1][0].set_ylabel("Edge")
        axs[1][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=cc_intervals[0::label_each])
        axs[1][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[1][1].imshow(random_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[1][1].set_title(" area over edge")
        axs[1][1].set_xlabel("Area")
        axs[1][1].set_ylabel("Edge")
        axs[1][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=area_intervals[0::label_each])
        axs[1][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=edge_intervals[0::label_each])

        axs[1][2].imshow(random_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[1][2].set_title(" area over cc")
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

            #print(cc_index, edge_index, area_index)

            random_heatmap_cc_edge[cc_index, edge_index] += 1
            random_heatmap_area_edge[area_index, edge_index] += 1
            random_heatmap_area_cc[area_index, cc_index] += 1

        """
        print(cc_intervals, random_cc_intervals)
        print(edge_intervals, random_edge_intervals)
        print(area_intervals, random_area_intervals)
        """

        # Plot the first heatmap
        axs[2][0].imshow(random_heatmap_cc_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[2][0].set_title(" cc over edge")
        axs[2][0].set_xlabel("Color contrast")
        axs[2][0].set_ylabel("Edge")
        axs[2][0].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_cc_intervals[0::label_each])
        axs[2][0].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_edge_intervals[0::label_each])

        axs[2][1].imshow(random_heatmap_area_edge, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[2][1].set_title(" area over edge")
        axs[2][1].set_xlabel("Area")
        axs[2][1].set_ylabel("Edge")
        axs[2][1].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_area_intervals[0::label_each])
        axs[2][1].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_edge_intervals[0::label_each])

        axs[2][2].imshow(random_heatmap_area_cc, cmap='plasma', interpolation='nearest', aspect='auto')
        #axs[2][2].set_title(" area over cc")
        axs[2][2].set_xlabel("Area")
        axs[2][2].set_ylabel("Color contrast")
        axs[2][2].set_xticks(np.arange(nb_interval-1)[0::label_each], labels=random_area_intervals[0::label_each])
        axs[2][2].set_yticks(np.arange(nb_interval-1)[0::label_each], labels=random_cc_intervals[0::label_each])



        self.pca_on_targets_and_background()
        plt.show()
#        exit()


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

    def log_wandb_histograms(self):

        #------------------ drivable pourcent hist ------------

        # Draw histogram of color contrast scores
        DP_density_hists = self.log_histograms(0, limits_high=10, density=True)
        DP_hists = self.log_histograms(0, limits_high=10, density=False)
        DP_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True)

        
        # Draw cumulative proba of color contrast scores
        DP_cumsum_density_hists = self.log_histograms(0, limits_high=10, density=True, cumsum=True)
        DP_cumsum_hists = self.log_histograms(0, limits_high=10, density=False, cumsum=True)
        DP_cumsum_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True, cumsum=True)

        wandb.log({"Histogram of boxes scores Drivable Pourcent": [DP_hists, DP_normalize_hists, DP_density_hists, DP_cumsum_hists, DP_cumsum_normalize_hists, DP_cumsum_density_hists]})

        #------------------ Color contrast hist ------------

        """
        # Draw histogram of color contrast scores
        CC_density_hists = self.log_histograms(0, limits_high=10, density=True)
        CC_hists = self.log_histograms(0, limits_high=10, density=False)
        CC_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True)

        

        # Draw cumulative proba of color contrast scores
        CC_cumsum_density_hists = self.log_histograms(0, limits_high=10, density=True, cumsum=True)
        CC_cumsum_hists = self.log_histograms(0, limits_high=10, density=False, cumsum=True)
        CC_cumsum_normalize_hists = self.log_histograms(0, limits_high=10, density=False, normalize=True, cumsum=True)

        wandb.log({"Histogram of boxes scores Color Contrast": [CC_hists, CC_normalize_hists, CC_density_hists, CC_cumsum_hists, CC_cumsum_normalize_hists, CC_cumsum_density_hists]})
        """

        #------------------ Edge density hist ------------

        """
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
        """
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

        """
        fig = plt.figure()
        plt.plot(hist_boxes_size_targets, label="targets (" + str( len(self.keep_all_custom_area_targets)) + " boxes)", c="darkblue")
        plt.plot(hist_boxes_size_prediction_good, label="good predictions (" + str( len(self.keep_all_custom_area_good_predictions_good)) + " boxes)", c="blueviolet")
        plt.plot(hist_boxes_size_prediction_bad, label="bad predictions  (" + str( len(self.keep_all_custom_area_good_predictions_bad)) + " boxes)", c="magenta")
        plt.legend(loc="best")
        """


        # Draw the scatter plot and marginals.
        """
        self.scatter_hist(self.keep_all_custom_scores_targets[0], self.keep_all_custom_area_targets, label="targets boxes area over score (" + str( len(self.keep_all_custom_scores_targets[0])) + " boxes)")
        target_area_over_score = wandb.Image(plt)

        self.scatter_hist(self.keep_all_custom_scores_good_predictions_TP[0], self.keep_all_custom_area_prediction_good,  label="good predictions boxes area over score (" + str( len(self.keep_all_custom_scores_good_predictions_TP[0])) + " boxes)")
        prediction_good_area_over_score = wandb.Image(plt)

        self.scatter_hist(self.keep_all_custom_scores_good_predictions_FP[0], self.keep_all_custom_area_prediction_bad,  label="bad predictions boxes area over score (" + str( len(self.keep_all_custom_scores_good_predictions_FP[0])) + " boxes)")
        prediction_bad_area_over_score = wandb.Image(plt)

        self.scatter_hist(self.keep_all_custom_scores_randoms[0], self.keep_all_custom_area_randoms,  label="random predictions boxes area over score (" + str( len(self.keep_all_custom_scores_randoms[0])) + " boxes)")
        randoms_area_over_score = wandb.Image(plt)

        self.scatter_hist(self.keep_all_custom_scores_true_randoms[0], self.keep_all_custom_area_true_randoms,  label="true random boxes area over score (" + str( len(self.keep_all_custom_scores_true_randoms[0])) + " boxes)")
        true_custom_area_over_score = wandb.Image(plt)
        
        wandb.log({"boxes scores Color Contrast over area": [target_area_over_score, prediction_bad_area_over_score, prediction_good_area_over_score, randoms_area_over_score, true_custom_area_over_score]})
        """



    def get_wandb_metrics(self, with_print=False):

        self.log_target_background()

        output = {}

        output["Number of GT boxes"] = self.nb_gt.item()
        output["Number of known GT boxes"] = self.nb_known_gt.item()
        output["Number of unknown GT boxes"] = self.nb_unknown_gt.item()
        output["Number of background GT boxes"] = self.nb_background_gt.item()

        if self.cfg.flags:
            output["TP"] = self.true_positives.item()
            output["FP"] = self.false_positives.item()
            output["FN"] = self.false_negatives.item()

            output["unknown TP"] = self.unknown_true_positives.item()
            output["unknown FP"] = self.unknown_false_positives.item()
            output["unknown FN"] = self.unknown_false_negatives.item()
            output["Number of multiple match for gt boxes"] = self.nb_multiple_match_for_gt_boxes

        if self.cfg.precision:
            known_precision = self.get_known_precision()
            output["known Precision"] = known_precision
            unknown_precision = self.get_unknown_precision()
            output["unknown Precision"] = unknown_precision


        if self.cfg.recall:
            known_recall = self.get_known_recall()
            output["Recall"] =  known_recall
            unknown_recall = self.get_unknown_recall()
            output["Unknown Recall"] = unknown_recall

        if self.cfg.coverage:
            output["Global Coverage"] = self.get_pourcent_coverage()
            output["Known Coverage"] = self.get_pourcent_coverage_known()
            output["Unknown Coverage"] = self.get_pourcent_coverage_unknown()

        if self.cfg.f1_score:
            known_f1_score = ((known_precision * known_recall)/(known_precision + known_recall + 1e-10)) * 2
            output["known f1 score"] = known_f1_score.item()
            unknown_f1_score = ((unknown_precision * unknown_recall)/(unknown_precision + unknown_recall + 1e-10)) * 2
            output["unknown f1 score"] = unknown_f1_score.item()

        if self.cfg.mAP:
            output["known mAPs"] = self.known_maps.compute()
            output["unknown mAPs"] = self.unknown_maps.compute()
            self.current_known_map = output["known mAPs"]
            self.current_unknown_map = output["unknown mAPs"] 


        if self.is_metrics_on_proposals_predictions: 

            output["Proposals coverage"] = self.get_proposals_pourcent_coverage()
            output["Proposals coverage unknown"] = self.get_proposals_pourcent_coverage_unknown()
            output["Proposals coverage known"] = self.get_proposals_pourcent_coverage_known()

        if self.is_metrics_on_raw_predictions: 
            #self.reliability_plots()

            output["Number of raw prediction boxes"] = (self.raw_false_positives + self.raw_true_positives).int().item()
            if self.nb_images.item() != 0:
                output["Number of raw prediction boxes per imgs"] = output["Number of raw prediction boxes"] // self.nb_images.item()
            output["Raw Recall"] = self.get_raw_recall()
            output["Raw coverage"] = self.get_raw_pourcent_coverage()
            output["Raw coverage unknown"] = self.get_raw_pourcent_coverage_unknown()
            output["Raw coverage known"] = self.get_raw_pourcent_coverage_known()



        mask = np.where(self.keep_all_custom_scores_randoms[0]  > 0)
        self.keep_all_custom_scores_randoms[0] = self.keep_all_custom_scores_randoms[0][mask]
        self.keep_all_custom_area_randoms = self.keep_all_custom_area_randoms[mask]

        #self.log_wandb_histograms()

        if with_print:
            if self.cfg.mAP:
                self.print_metrics(known_map=output["known mAPs"], unknown_map=output["unknown mAPs"])
            else:
                self.print_metrics()


        return output


    def print_metrics(self, known_map=None, unknown_map=None):

        print("Number of GT boxes: ", self.nb_gt, " known: ", self.nb_known_gt, " unknown ", self.nb_unknown_gt, " background ", self.nb_background_gt)

        if self.cfg.flags:
            print()
            print("TP (correct): ", self.true_positives.item())
            print("FP (louper): ", self.false_positives.item())
            print("FN (should have detect): ", self.false_negatives.item())
            print()
            print("unknown TP (correct): ",             self.unknown_true_positives.item())
            print("unknown FP (louper): ",              self.unknown_false_positives.item())
            print("unknown FN (should have detect): ",  self.unknown_false_negatives.item())
            print()
            print(" Number of multiple match for gt boxes : ", self.nb_multiple_match_for_gt_boxes )

        if self.cfg.precision:
            print()
            known_precision = self.get_known_precision()
            print("known Precision : ", known_precision)
            unknown_precision = self.get_unknown_precision()
            print("unknown Precision : ", unknown_precision)


        if self.cfg.recall:
            print()
            known_recall = self.get_known_recall()
            print("Recall : ", known_recall)
            unknown_recall = self.get_unknown_recall()
            print("Unknown Recall : ", unknown_recall)#, "                                         nb UTP :", self.unknown_true_positives, "nb UFN:", self.unknown_false_negatives, " nb UGT :", self.nb_unknown_gt)

        if self.cfg.coverage:
            print()
            print("Global Coverage ", self.get_pourcent_coverage())
            print("Known Coverage  ", self.get_pourcent_coverage_known())
            print("Unknown Coverage", self.get_pourcent_coverage_unknown())

        if self.is_metrics_on_raw_predictions: 
            print(f"Raw Recall : {self.get_raw_recall():.2f}")
            print("Raw coverage:", self.get_raw_pourcent_coverage())
            print("Raw coverage known:", self.get_raw_pourcent_coverage_known())
            print("Raw coverage unknown:", self.get_raw_pourcent_coverage_unknown())

        if self.is_metrics_on_proposals_predictions: 
            print("Proposals coverage:", self.get_proposals_pourcent_coverage())
            print("Proposals coverage known:", self.get_proposals_pourcent_coverage_known())
            print("Proposals coverage unknown:", self.get_proposals_pourcent_coverage_unknown())

        if self.cfg.f1_score:
            print()
            known_f1_score = ((known_precision * known_recall)/(known_precision + known_recall + 1e-10)) * 2
            print("known f1 score : ", known_f1_score.item())
            unknown_f1_score = ((unknown_precision * unknown_recall)/(unknown_precision + unknown_recall + 1e-10)) * 2
            print("unknown f1 score : ", unknown_f1_score.item())

        if self.cfg.mAP:
            print()
            if known_map == None:
                self.current_known_map = self.known_maps.compute() 
                print(" known mAPs : ", self.current_known_map)
            else:
                print(" known mAPs : ", known_map)
            print()
            if unknown_map == None:
                self.current_unknown_map = self.unknown_maps.compute() 
                print(" unknown mAPs : ", self.current_unknown_map)
            else:
                print(" unknown mAPs : ", unknown_map)

        
        print("Open set erors", self.open_set_errors / self.nb_unknown_gt, self.open_set_errors)


        print()


    def update_randoms(self, customs_scores_concatenate_randoms, customs_scores_concatenate_true_randoms): 

        for i, customs_scores_random in enumerate(customs_scores_concatenate_randoms["edge_density_scores"]):
            self.keep_all_custom_scores_randoms[1] = np.append(self.keep_all_custom_scores_randoms[1], customs_scores_random)
        for i, customs_scores_random in enumerate(customs_scores_concatenate_randoms["std"]):
            self.keep_all_custom_scores_randoms[2] = np.append(self.keep_all_custom_scores_randoms[2], customs_scores_random)
        for i, customs_scores_random in enumerate(customs_scores_concatenate_randoms["luminance"]):
            self.keep_all_custom_scores_randoms[3] = np.append(self.keep_all_custom_scores_randoms[3], customs_scores_random)
        for i, customs_scores_random in enumerate(customs_scores_concatenate_randoms["color_contrast_scores"]):
            self.keep_all_custom_scores_randoms[0] = np.append(self.keep_all_custom_scores_randoms[0], customs_scores_random)

            if len(customs_scores_concatenate_randoms["boxes"][i]["boxes"]) == 0:
                self.keep_all_custom_area_randoms = np.append(self.keep_all_custom_area_randoms, np.array([], dtype=int))
            else:
                boxes = customs_scores_concatenate_randoms["boxes"][i]["boxes"].int()
                area_boxes =  (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                self.keep_all_custom_area_randoms = np.append(self.keep_all_custom_area_randoms, area_boxes.cpu().numpy())
         
        for i, customs_scores_true_random in enumerate(customs_scores_concatenate_true_randoms["edge_density_scores"]):
            self.keep_all_custom_scores_true_randoms[1] = np.append(self.keep_all_custom_scores_true_randoms[1], customs_scores_true_random)
        for i, customs_scores_true_random in enumerate(customs_scores_concatenate_true_randoms["std"]):
            self.keep_all_custom_scores_true_randoms[2] = np.append(self.keep_all_custom_scores_true_randoms[2], customs_scores_true_random)
        for i, customs_scores_true_random in enumerate(customs_scores_concatenate_true_randoms["color_contrast_scores"]):
            self.keep_all_custom_scores_true_randoms[0] = np.append(self.keep_all_custom_scores_true_randoms[0], customs_scores_true_random)

            if len(customs_scores_concatenate_true_randoms["boxes"][i]["boxes"]) == 0:
                self.keep_all_custom_area_true_randoms = np.append(self.keep_all_custom_area_true_randoms, np.array([], dtype=int))
            else:
                boxes = customs_scores_concatenate_true_randoms["boxes"][i]["boxes"].int()
                area_boxes =  (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                self.keep_all_custom_area_true_randoms = np.append(self.keep_all_custom_area_true_randoms, area_boxes.cpu().numpy())

    def compute_reliability(self, calibration_metric, bins):

        accuracies = torch.cat(calibration_metric.accuracies).cpu().numpy()
        confidences = torch.cat(calibration_metric.confidences).cpu().numpy()

        # Find the indices of the bins to which each value in confidences belongs
        bin_indices = np.digitize(confidences, bins)

        # Initialize arrays to store mean accuracies and number of samples in each bin
        mean_accuracies = np.zeros(self.calibration_bins)
        num_samples_in_bin = np.zeros(self.calibration_bins)
        
        # Iterate over each bin
        for i in range(self.calibration_bins):
            # Find indices of samples in current bin
            bin_indices_for_bin_i = np.where(bin_indices == i + 1)
            accuracies_for_bin_i = accuracies[bin_indices_for_bin_i].astype(bool)

            if len(accuracies_for_bin_i) > 0:
                # Calculate mean accuracy for samples in current bin
                mean_accuracy_for_bin_i = np.mean(accuracies_for_bin_i)
                # Store mean accuracy and number of samples in current bin
                mean_accuracies[i] = mean_accuracy_for_bin_i
            num_samples_in_bin[i] = len(bin_indices_for_bin_i[0])

        return mean_accuracies, num_samples_in_bin
        

    def reliability_plots(self):

        n_bins = self.calibration_bins

        # Calculate bin edges
        bins = np.linspace(0, 1, n_bins + 1)

        classif_accuracies, classif_num_samples = self.compute_reliability(self.calibration_metric_classif_pred, bins)
        iou_gt_accuracies, iou_gt_num_samples = self.compute_reliability(self.calibration_metric_iou_gt, bins)
        centerness_gt_accuracies, centerness_gt_num_samples = self.compute_reliability(self.calibration_metric_centerness_gt, bins)
        
        # Plot reliability diagram
        fig, axs = plt.subplots(3)

        iou_bar = axs[0].bar(bins[:-1], iou_gt_accuracies, width=1/n_bins, align='edge', alpha=0.3, label='iou_gt')
        centerness_bar = axs[0].bar(bins[:-1], centerness_gt_accuracies, width=1/n_bins, align='edge', alpha=0.3, label='centerness_gt', linewidth=0.05)

        axs[0].set_xlabel('Confidence')
        axs[0].set_xlim(0, 1)
        axs[0].set_ylabel('Mean Accuracy')
        axs[0].set_ylim(0, 1)
        axs[0].legend([iou_bar, centerness_bar], ['GT iou', 'GT centerness'])

        classif_bar = axs[1].bar(bins[:-1], classif_accuracies, width=1/n_bins, align='edge', alpha=0.3, label='classif_pred')

        axs[1].set_xlabel('Confidence')
        axs[1].set_xlim(0, 1)
        axs[1].set_ylabel('Mean Accuracy')
        axs[1].set_ylim(0, 1)
        axs[1].legend([classif_bar], ['Score classif'])

        axs[0].plot(bins[:-1], bins[:-1], marker='o', linestyle='-')
        axs[1].plot(bins[:-1], bins[:-1], marker='o', linestyle='-')
        axs[0].set_title("Reliability diagram Ground Truth")
        axs[1].set_title(f"Reliability diagram predicted scores\nRecall: {self.get_raw_recall():.2f} percent coverage: {self.get_raw_pourcent_coverage():.2f} K:{self.get_raw_pourcent_coverage_known():.2f} U:{self.get_raw_pourcent_coverage_unknown():.2f} for {len(self.calibration_metric_classif_pred.accuracies) * len(self.calibration_metric_classif_pred.accuracies[0])}")
        print(f"Raw Recall : {self.get_raw_recall():.2f}")
        print("Raw coverage:", self.get_raw_pourcent_coverage())
        print("Raw coverage known:", self.get_raw_pourcent_coverage_known())
        print("Raw coverage unknown:", self.get_raw_pourcent_coverage_unknown())


        max_sample = classif_num_samples.sum()
        axs[2].bar(bins[:-1], classif_num_samples/max_sample, width=1/n_bins, align='edge', alpha=0.3)
        axs[2].bar(bins[:-1], iou_gt_num_samples/max_sample, width=1/n_bins, align='edge', alpha=0.3)
        axs[2].bar(bins[:-1], centerness_gt_num_samples/max_sample, width=1/n_bins, align='edge', alpha=0.3)
        axs[2].set_ylim(0, 0.25)
        axs[2].set_xlim(0, 1)
        axs[2].legend(['classif', 'iou', 'centerness'])
        axs[2].set_title("Pourcentage of sample in each bin") 

        fig.suptitle(self.plt_name)

        plt.tight_layout()
        #plt.show()
        plt.savefig("img_output/" + 'reliabilty_diagram_' + self.plt_name + '.png')

    def reliability_plot(self, confidences, accuracies, n_bins):
        # Calculate bin edges
        bins = np.linspace(0, 1, n_bins + 1)
        
        # Find the indices of the bins to which each value in confidences belongs
        bin_indices = np.digitize(confidences, bins)
        
        # Initialize arrays to store mean accuracies and number of samples in each bin
        mean_accuracies = np.zeros(n_bins)
        num_samples_in_bin = np.zeros(n_bins)
        
        # Iterate over each bin
        for i in range(n_bins):
            # Find indices of samples in current bin
            bin_indices_for_bin_i = np.where(bin_indices == i + 1)
            accuracies_for_bin_i = accuracies[bin_indices_for_bin_i].astype(bool)

            if len(accuracies_for_bin_i) > 0:
                # Calculate mean accuracy for samples in current bin
                mean_accuracy_for_bin_i = np.mean(accuracies_for_bin_i)
                # Store mean accuracy and number of samples in current bin
                mean_accuracies[i] = mean_accuracy_for_bin_i
            num_samples_in_bin[i] = len(bin_indices_for_bin_i[0])
        
        # Plot reliability diagram
        fig, axs = plt.subplots(1)

        axs.bar(bins[:-1], mean_accuracies, width=1/n_bins, align='edge')
        axs.set_xlabel('Confidence')
        axs.set_xlim(0, 1)
        axs.set_ylabel('Mean Accuracy')
        axs.set_ylim(0, 1)
        axs.legend('Reliability Plot')
        plt.plot(bins[:-1], bins[:-1], marker='o', linestyle='-')
        plt.show()


    def update_proposals(self, known_targets, unknown_targets, proposals, targets):

        if not self.is_metrics_on_proposals_predictions: 
            return

        if not self.is_metrics_on_raw_predictions: #already update on raw of is on
            self.nb_images += len(proposals)

        for batch_index in range(len(proposals)):
            #print("prediction :", predictions[batch_index]["tags_raw"])
            self.proposals_true_positives += proposals[batch_index]["tags_proposals"].sum().detach().cpu()
            self.proposals_false_positives += (proposals[batch_index]["tags_proposals"] == False).sum().detach().cpu()
            self.proposals_false_negatives_known += (known_targets[batch_index]["tags_proposals"] == False).sum().detach().cpu()
            self.proposals_false_negatives_unknown += (unknown_targets[batch_index]["tags_proposals"] == False).sum().detach().cpu()
            self.proposals_nb_gt_known += len(known_targets[batch_index]["labels"])
            self.proposals_nb_gt_unknown += len(unknown_targets[batch_index]["labels"])

            self.proposals_false_negatives += (targets[batch_index]["tags_proposals"] == False).sum().detach().cpu()

    def update_raw(self, known_targets, unknown_targets, predictions, targets):

        if not self.is_metrics_on_raw_predictions: 
            return

        self.nb_images += len(predictions)

        for batch_index in range(len(predictions)):
            #print("prediction :", predictions[batch_index]["tags_raw"])
            self.raw_true_positives += predictions[batch_index]["tags_raw"].sum().detach().cpu()
            self.raw_false_positives += (predictions[batch_index]["tags_raw"] == False).sum().detach().cpu()
            self.raw_false_negatives_known += (known_targets[batch_index]["tags_raw"] == False).sum().detach().cpu()
            self.raw_false_negatives_unknown += (unknown_targets[batch_index]["tags_raw"] == False).sum().detach().cpu()
            self.raw_nb_gt_known += len(known_targets[batch_index]["labels"])
            self.raw_nb_gt_unknown += len(unknown_targets[batch_index]["labels"])

            self.raw_false_negatives += (targets[batch_index]["tags_raw"] == False).sum().detach().cpu()



        """
        if self.calibration_bins > 0:

            predictions_scores = []
            predictions_iou = []
            predictions_centerness= []
            predictions_tags = []
            centerness = False
            for prediction in predictions:
                predictions_scores.append(prediction["scores"])
                predictions_tags.append(prediction["tags_raw"])
                predictions_iou.append(prediction["IoU_tags_raw"])
                if "centerness" in prediction:
                    centerness = True
                    predictions_centerness.append(prediction["centerness"])

            predictions_scores = torch.cat(predictions_scores, dim=0)
            predictions_tags = torch.cat(predictions_tags, dim=0)
            predictions_iou = torch.cat(predictions_iou, dim=0)
            if centerness:
                predictions_centerness = torch.cat(predictions_centerness, dim=0)

            self.calibration_metric_classif_pred.update(preds=predictions_scores, target=predictions_tags)
            self.calibration_metric_iou_gt.update(preds=predictions_iou, target=predictions_tags)
            if centerness:
                self.calibration_metric_centerness_gt.update(preds=predictions_centerness, target=predictions_tags)
        """


    #def update(self, known_predictions, known_targets, unknown_predictions, unknown_targets, targets):
    def update(self, known_targets, unknown_targets, background_targets, known_predictions, unknown_predictions, targets):

        # Update flags metrics
        for batch_index in range(len(targets)):

            if unknown_predictions != None:
                #print("error device ", unknown_predictions, batch_index, self.unknown_true_positives)
                self.unknown_true_positives += unknown_predictions[batch_index]["tags"].sum().detach().cpu()
                self.unknown_false_positives += (unknown_predictions[batch_index]["tags"] == False).sum().detach().cpu()
                self.unknown_false_negatives += (unknown_targets[batch_index]["tags"] == False).sum().detach().cpu() if "tags" in unknown_targets[batch_index] else len(unknown_targets[batch_index])
                self.nb_unknown_gt += len(unknown_targets[batch_index]["tags"]) if "tags" in unknown_targets[batch_index] else 0

            self.true_positives += known_predictions[batch_index]["tags"].sum().detach().cpu()
            self.false_positives += (known_predictions[batch_index]["tags"] == False).sum().detach().cpu()
            self.false_negatives += (known_targets[batch_index]["tags"] == False).sum().detach().cpu()
            self.nb_known_gt += len(known_targets[batch_index]["tags"])

            self.nb_gt += len(known_targets[batch_index]["tags"]) + len(unknown_targets[batch_index]["tags"]) + len(background_targets[batch_index]["labels"])

            if "tags_KP_with_UT" in known_predictions[batch_index]:
                self.open_set_errors += (known_predictions[batch_index]["tags_KP_with_UT"] == True).sum().detach().cpu()

        # Update mAP
        if self.cfg.mAP:

            # set the same unknown labels for targets and predictions (put all labels to 255)
            unknown_targets_map = []
            for unknown_target in unknown_targets:
                unknown_target_map = copy.deepcopy(unknown_target)
                unknown_target_map["labels"].fill_(255)
                unknown_targets_map.append(unknown_target_map)

            unknown_predictions_map = []
            for unknown_pred in unknown_predictions:
                unknown_pred_map = copy.deepcopy(unknown_pred)
                unknown_pred_map["labels"].fill_(255)
                unknown_predictions_map.append(unknown_pred_map)


            self.known_maps.update(preds=known_predictions, target=known_targets)
            self.unknown_maps.update(preds=unknown_predictions_map, target=unknown_targets_map)

    #---------------- Keep scores -----------------

    def save_scores(self, targets, known_predictions, unknown_predictions):

        # Update color contrast keeping scores
        for target in targets:
            self.keep_all_custom_scores_targets[0] = np.append(self.keep_all_custom_scores_targets[0], target["custom_scores"]["color_contrast"].cpu())
            self.keep_all_custom_scores_targets[1] = np.append(self.keep_all_custom_scores_targets[1], target["custom_scores"]["edge_density"].cpu())
            if "score_drivable_pourcent" in target:
                self.keep_all_custom_scores_targets[0] = np.append(self.keep_all_custom_scores_targets[0], target["score_drivable_pourcent"].cpu())
            self.keep_all_custom_scores_targets[3] = np.append(self.keep_all_custom_scores_targets[3], target["custom_scores"]["std"].cpu())
            self.keep_all_custom_scores_targets[4] = np.append(self.keep_all_custom_scores_targets[4], target["custom_scores"]["luminance"].cpu())

            if len(target["boxes"]) == 0:
                self.keep_all_custom_area_targets = np.append(self.keep_all_custom_area_targets, np.array([], dtype=int))
            else:
                area_boxes =  (target["boxes"][:, 2].int() - target["boxes"][:, 0].int()) * (target["boxes"][:, 3].int() - target["boxes"][:, 1].int())
                self.keep_all_custom_area_targets = np.append(self.keep_all_custom_area_targets, area_boxes.cpu().numpy())

        # Select only known # Good predictions
        for prediction in known_predictions:

            if len(prediction["boxes"]) == 0:
                continue

            self.keep_all_custom_scores_good_predictions_TP[0] = np.append(self.keep_all_custom_scores_good_predictions_TP[0], prediction["custom_scores"]["color_contrast"][prediction["tags"]].cpu())
            if "score_drivable_pourcent" in prediction:
                self.keep_all_custom_scores_good_predictions_TP[0] = np.append(self.keep_all_custom_scores_good_predictions_TP[0], prediction["score_drivable_pourcent"][prediction["tags"]].cpu())
            self.keep_all_custom_scores_good_predictions_TP[1] = np.append(self.keep_all_custom_scores_good_predictions_TP[1], prediction["custom_scores"]["edge_density"][prediction["tags"]].cpu())


            self.keep_all_custom_scores_good_predictions_FP[0] = np.append(self.keep_all_custom_scores_good_predictions_FP[0], prediction["custom_scores"]["color_contrast"][~prediction["tags"]].cpu())
            if "score_drivable_pourcent" in prediction:
                self.keep_all_custom_scores_good_predictions_FP[0] = np.append(self.keep_all_custom_scores_good_predictions_FP[0], prediction["score_drivable_pourcent"][~prediction["tags"]].cpu())
            self.keep_all_custom_scores_good_predictions_FP[1] = np.append(self.keep_all_custom_scores_good_predictions_FP[1], prediction["custom_scores"]["edge_density"][~prediction["tags"]].cpu())

            area_boxes =  (prediction["boxes"][:, 2].int() - prediction["boxes"][:, 0].int()) * (prediction["boxes"][:, 3].int() - prediction["boxes"][:, 1].int())
            self.keep_all_custom_area_prediction_bad = np.append(self.keep_all_custom_area_prediction_bad, area_boxes[~prediction["tags"]].cpu().numpy())
            self.keep_all_custom_area_prediction_good = np.append(self.keep_all_custom_area_prediction_good, area_boxes[prediction["tags"]].cpu().numpy())

        if unknown_predictions != None:
            # Select only unknown predictions 
            for prediction in unknown_predictions:
                self.keep_all_custom_scores_unknown_predictions_TP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_TP[0], prediction["custom_scores"]["color_contrast"][prediction["tags"]].cpu())
                if "score_drivable_pourcent" in prediction:
                    self.keep_all_custom_scores_unknown_predictions_TP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_TP[0], prediction["score_drivable_pourcent"][prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_TP[1] = np.append(self.keep_all_custom_scores_unknown_predictions_TP[1], prediction["custom_scores"]["edge_density"][prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_FP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_FP[0], prediction["custom_scores"]["color_contrast"][~prediction["tags"]].cpu())
                if "score_drivable_pourcent" in prediction:
                    self.keep_all_custom_scores_unknown_predictions_FP[0] = np.append(self.keep_all_custom_scores_unknown_predictions_FP[0], prediction["score_drivable_pourcent"][~prediction["tags"]].cpu())
                self.keep_all_custom_scores_unknown_predictions_FP[1] = np.append(self.keep_all_custom_scores_unknown_predictions_FP[1], prediction["custom_scores"]["edge_density"][~prediction["tags"]].cpu())
            
