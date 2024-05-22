import numpy as np 
import torch
import wandb

import matplotlib.pyplot as plt
import pdb

from rl_environment import RL_Environment


class RL_Model():

    def __init__(self, nb_states, nb_actions, save_path):

        nb_component = 3
        self.cumulative_scores_components_true = np.zeros((nb_states, nb_component), dtype=float)
        self.cumulative_scores_components_false = np.zeros((nb_states, nb_component), dtype=float)
        self.cumulative_scores_nb_tags_true = np.zeros((nb_states, 2), dtype=int)

        self.cumulative_scores = np.zeros((nb_states, nb_actions), dtype=float)
        self.cumulative_states_seen = np.zeros(nb_states, dtype=int)

        self.tp_cumulative_states_error_known = np.zeros(nb_states, dtype=int)
        self.fp_cumulative_states_error_known = np.zeros(nb_states, dtype=int)
        self.tp_cumulative_states_error_unknown = np.zeros(nb_states, dtype=int)
        self.fp_cumulative_states_error_unknown = np.zeros(nb_states, dtype=int)
        self.tn_cumulative_states_error_background= np.zeros(nb_states, dtype=int)

        self.save_path = save_path

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.actions_names = ["Known", "Unknown", "Background"]

        self.is_cumulative_scores_calculated = False
        """
        self.load(800)
        self.plot(RL_Environment(None, 3))
        exit()
        """

    def calculate_cumulative_scores_with_components(self):

        tags_true = self.cumulative_scores_nb_tags_true[:, 0]
        tags_false = self.cumulative_scores_nb_tags_true[:, 1] - self.cumulative_scores_nb_tags_true[:, 0]

        iou_true = self.cumulative_scores_components_true[:, 0]
        edge_true = self.cumulative_scores_components_true[:, 1]
        semantic_segmentation_score_true = self.cumulative_scores_components_true[:, 2]
        iou_false = self.cumulative_scores_components_false[:, 0]
        edge_false = self.cumulative_scores_components_false[:, 1]
        semantic_segmentation_score_false = self.cumulative_scores_components_false[:, 2]

        #print("tags true", tags_true, "\niou true ", iou_true, "\nedge true ", edge_true, "\n SSGT ", semantic_segmentation_score_true)
        #print("tags false", tags_false, "\niou false", iou_false, "\nedge false", edge_false, "\n SSGT ", semantic_segmentation_score_false)

        self.cumulative_scores[:, 0] =  1 * iou_true    +  1 * semantic_segmentation_score_true        + 0 * edge_true  +  1 * tags_true +  0 * iou_false   + -1 * (1 - semantic_segmentation_score_false) + 0 * edge_false + -1 * tags_false

        self.cumulative_scores[:, 1] =  0.1 * iou_true  +  1 * semantic_segmentation_score_true        + 0 * edge_true  +  0 * tags_true +  0 * iou_false   +  1 * semantic_segmentation_score_false       + 0 * edge_false +  0 * tags_false

        self.cumulative_scores[:, 2] =  0 * iou_true    + -1 * semantic_segmentation_score_true        + 0 * edge_true  + -1 * tags_true +  0 * iou_false   +  1 * (1 - semantic_segmentation_score_false) + 0 * edge_false +  1 * tags_false


    def forward(self, states):

        if not self.is_cumulative_scores_calculated :
            self.calculate_cumulative_scores_with_components()
            self.is_cumulative_scores_calculated = True
            print("First forward RL ! Recalculating scores ")


        actions = []
        for state in states:
            actions.append(np.argmax(self.cumulative_scores[state] / self.cumulative_states_seen[state, np.newaxis], axis=1))

        return actions

    def sort_predictions(self, predictions, states):

        actions = self.forward(states)

        # Select only known
        known_predictions = []
        unknown_predictions = []
        background_predictions = []

        for i, prediction in enumerate(predictions):
            unknown_prediction = {}
            known_prediction = {}
            background_prediction = {}

            #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
            known_mask = actions[i] == 0
            unknown_mask = actions[i] == 1
            background_mask = actions[i] == 2

            for key, value in prediction.items():

                if key == "custom_scores":
                    known_new_custom_scores = {}
                    unknown_new_custom_scores = {}
                    background_new_custom_scores = {}
                    for key2, value2 in prediction["custom_scores"].items():
                        known_new_custom_scores[key2] = value2[known_mask]
                        unknown_new_custom_scores[key2] = value2[unknown_mask]
                        background_new_custom_scores[key2] = value2[background_mask]
                    known_prediction[key] = known_new_custom_scores
                    unknown_prediction[key] = unknown_new_custom_scores
                    background_prediction[key] = background_new_custom_scores
                    continue

                if not torch.is_tensor(value):
                    continue

                known_prediction[key] = value[known_mask]
                unknown_prediction[key] = value[unknown_mask]
                background_prediction[key] = value[background_mask]

            known_predictions.append(known_prediction)
            unknown_predictions.append(unknown_prediction)
            background_predictions.append(background_prediction)


        return known_predictions, unknown_predictions, background_predictions

    def learn(self, states, components):

        for i, state in enumerate(states):

            for j, s in enumerate(state):
                self.cumulative_states_seen[s] += 1
                self.cumulative_scores_components_true[s] += components[i][j, 1:] * (components[i][j, 0])
                self.cumulative_scores_components_false[s] += components[i][j, 1:] * (components[i][j, 0] * -1 + 1)
                self.cumulative_scores_nb_tags_true[s, 0] += components[i][j, 0].astype(int)
                self.cumulative_scores_nb_tags_true[s, 1] += 1


    def save(self, idx, txt=""):


        np.savez(self.save_path + str(idx) + txt, cumulative_scores=self.cumulative_scores, cumulative_states_seen=self.cumulative_states_seen)
        #rl_environment = RL_Environment(None, 3)
        #self.plot(rl_environment)

    def load(self, idx):

        data = np.load(self.save_path + str(idx) + ".npz")
        self.cumulative_scores = data["cumulative_scores"]
        self.cumulative_states_seen = data["cumulative_states_seen"]
        self.plot(RL_Environment(None, 3))

    def add_unknown_error_states(self, tp_states, fp_states):
        self.tp_cumulative_states_error_unknown[tp_states] += 1
        self.fp_cumulative_states_error_unknown[fp_states] += 1

    def add_known_error_states(self, tp_states, fp_states):
        self.tp_cumulative_states_error_known[tp_states] += 1
        self.fp_cumulative_states_error_known[fp_states] += 1

    def add_background_error_states(self, tn_states):
        self.tn_cumulative_states_error_background[tn_states] += 1


# ----------------------- Plot ----------------------------------------------------------


    def plot(self, rl_env, save_wandb=True, plot=False):

        wandb_images = []
        wandb_images.append(self.plot_states(rl_env))
        wandb_images.append(self.plot_nb_states_seens(rl_env))
        #self.plot_actions()
        plots = self.plot_errors(rl_env)
        wandb_images.append(plots[0])
        wandb_images.append(plots[1])
        wandb_images.append(plots[2])
        

        wandb.log({"RL_Plots/": wandb_images})

        if plot:
            plt.show()

    def plot_heatmap_on_ax(ax, heatmap, rl_env, title, add_text=False, text_array=None, font_size=6):
        # Plot the first heatmap
        ax.imshow(heatmap, cmap='plasma', interpolation='nearest', aspect='auto')
        ax.set_title(title)

        # Loop over data dimensions and create text annotations.
        if add_text:
            for i in range(len(rl_env.color_contrast_intervals_labels)):
                for j in range(len(rl_env.prediction_intervals_labels)):
                    text = ax.text(j, i, text_array[i, j], ha="center", va="center", color="black", size=font_size, fontdict=None)

        ax.set_xticks(np.arange(len(rl_env.prediction_intervals_labels)), labels=rl_env.prediction_intervals_labels)
        ax.set_yticks(np.arange(len(rl_env.color_contrast_intervals_labels)), labels=rl_env.color_contrast_intervals_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Prediction upper bound")
        ax.set_ylabel("Color Contrast upper bound")

    def plot_errors(self, rl_env):

        title = "Number of known error boxes seen for each states (" + str(self.fp_cumulative_states_error_known.sum()) + " boxes seen), more yellow is more FP, label is TP / FP" 
        text_array = np.vstack((self.tp_cumulative_states_error_known, self.fp_cumulative_states_error_known)).T
        known_errors_plot = self.plot_array(rl_env, self.fp_cumulative_states_error_known, title, text_array=text_array)

        title = "Number of unknown error boxes seen for each states (" + str(self.fp_cumulative_states_error_unknown.sum()) + " boxes seen), more yellow is more FP, label is TP / FP"
        text_array = np.vstack((self.tp_cumulative_states_error_unknown, self.fp_cumulative_states_error_unknown)).T
        unknown_errors_plot = self.plot_array(rl_env, self.fp_cumulative_states_error_unknown, title, text_array=text_array)

        title = "Number of background error boxes seen for each states (" + str(self.tn_cumulative_states_error_background.sum()) + " boxes seen), more yellow is more misses targets as background"
        background_errors_plot = self.plot_array(rl_env, self.tn_cumulative_states_error_background, title)


        return (known_errors_plot, unknown_errors_plot, background_errors_plot)

    def plot_nb_states_seens(self, rl_env):

        title = "Number of boxes seen for each states (" + str(self.cumulative_states_seen.sum()) + " boxes seen)"
        return self.plot_array(rl_env, self.cumulative_states_seen, title)
            
    def plot_array(self, rl_env, array, title, text_size=5, text_array=None):

        big_boxes = np.copy(array[np.arange(360) % 30 < 10])
        medium_boxes = np.copy(array[(np.arange(360) % 30 > 9) & (np.arange(360) % 30 < 20)])
        small_boxes = np.copy(array[np.arange(360) % 30 > 19])

        big_boxes = np.reshape(big_boxes, (12, 10))
        medium_boxes = np.reshape(medium_boxes, (12, 10))
        small_boxes = np.reshape(small_boxes, (12, 10))

        # Create three subplots for the heatmaps
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)

        if text_array is None:
            RL_Model.plot_heatmap_on_ax(axs[0], big_boxes, rl_env, 'Big boxes', add_text=True, text_array=big_boxes)
            RL_Model.plot_heatmap_on_ax(axs[1], medium_boxes, rl_env, 'Medium boxes', add_text=True, text_array=medium_boxes)
            RL_Model.plot_heatmap_on_ax(axs[2], small_boxes, rl_env, 'small boxes', add_text=True, text_array=small_boxes)

        else:
            text_array = self.create_texts_array(text_array)
            RL_Model.plot_heatmap_on_ax(axs[0], big_boxes, rl_env, 'Big boxes', add_text=True, text_array=text_array[0], font_size=text_size)
            RL_Model.plot_heatmap_on_ax(axs[1], medium_boxes, rl_env, 'Medium boxes', add_text=True, text_array=text_array[1], font_size=text_size)
            RL_Model.plot_heatmap_on_ax(axs[2], small_boxes, rl_env, 'small boxes', add_text=True, text_array=text_array[2], font_size=text_size)

        # Show the plot
        plt.tight_layout()

        return wandb.Image(plt)


    def create_texts_array(self, array):

        big_boxes_masks = np.arange(360) % 30 < 10
        medium_boxes_masks = (np.arange(360) % 30 > 9) & (np.arange(360) % 30 < 20)
        small_boxes_masks = np.arange(360) % 30 > 19

        def join_txt(text): return np.asarray(" / ".join(text),dtype=object)

        array = np.round(np.copy(array), 1)

        big_boxes_text = array[big_boxes_masks]
        big_boxes_text = np.apply_along_axis(join_txt, 1, big_boxes_text.astype(str))
        big_boxes_text = np.reshape(big_boxes_text, (12, 10))

        medium_boxes_text = array[medium_boxes_masks]
        medium_boxes_text = np.apply_along_axis(join_txt, 1, medium_boxes_text.astype(str))
        medium_boxes_text = np.reshape(medium_boxes_text, (12, 10))

        small_boxes_text = array[small_boxes_masks]
        small_boxes_text = np.apply_along_axis(join_txt, 1, small_boxes_text.astype(str))
        small_boxes_text = np.reshape(small_boxes_text, (12, 10))

        return (big_boxes_text, medium_boxes_text, small_boxes_text)

    def plot_states(self, rl_env):

        array = np.argmax(self.cumulative_scores, axis=1)
        title = "States were we select known (yellow) or not (blue)"
        return self.plot_array(rl_env, array, title, text_array=self.cumulative_scores / self.cumulative_states_seen[:, np.newaxis])

    def plot_actions(self):

        fig, ax = plt.subplots()

        color = np.copy(self.cumulative_scores)


        color[np.arange(color.shape[0]), np.argmax(color, axis=1)] = 3000

        im = ax.imshow(color, aspect='auto')

        #ax.set_yticks(np.arange(len(states_description)), labels=states_description)
        ax.set_xticks(np.arange(len(self.actions_names)), labels=self.actions_names)

        fig.tight_layout()

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
        # Loop over data dimensions and create text annotations.
        for i in range(self.nb_states):
            for j in range(self.nb_actions):
                text = ax.text(j, i, round(self.cumulative_scores[i, j], 2),
                               ha="center", va="center", color="b")

        fig.tight_layout()


