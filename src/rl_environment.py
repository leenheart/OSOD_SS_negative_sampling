import numpy as np
import torch
import matplotlib.pyplot as plt

class RL_Environment():

    def __init__(self, cfg_rl_environment, nb_actions):

        self.nb_actions = nb_actions
        self.predictions_intervals = np.array([i/10 for i in range(1, 10)])
        self.boxes_areas_intervals = np.array([32*32, 96*96])
        self.color_contrast_intervals = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 25, 50])

        self.states_descriptions = []
        for _ in range(3 * 10 * 12):
            self.states_descriptions.append("")


        self.prediction_intervals_labels = np.concatenate([self.predictions_intervals.astype(str), np.array(["+inf"])])
        self.boxe_intervals_labels = np.concatenate([self.boxes_areas_intervals.astype(str), np.array(["+inf"])])
        self.color_contrast_intervals_labels = np.concatenate([self.color_contrast_intervals.astype(str), np.array(["+inf"])])

        for x, prediction_inter in enumerate(self.prediction_intervals_labels):
            for y, boxe_inter in enumerate(self.boxe_intervals_labels):
                for z, color_contrast_inter in enumerate(self.color_contrast_intervals_labels):
                    label = " boxe area < " + boxe_inter + ", pred score < " + prediction_inter + ", CC < " + color_contrast_inter
                    self.states_descriptions[x + y * 10 + z * 10 * 3] = label 


        self.nb_predictions_intervals = len(self.predictions_intervals) + 1
        self.nb_boxes_areas_intervals = len(self.boxes_areas_intervals) + 1
        self.nb_color_contrast_intervals = len(self.color_contrast_intervals) + 1

    def get_indexes_from_interval(values, interval):

        values = torch.nan_to_num(values)
        indexes = []

        for value in values:

            if value >= interval[-1]:
                indexes.append(len(interval))
                continue

            for i, borne_sup in enumerate(interval):

                if value < borne_sup:
                    indexes.append(i)
                    break



        return np.array(indexes)

    def get_states(self, predictions, mask=...):

        prediction_index = RL_Environment.get_indexes_from_interval(predictions["scores"][mask], self.predictions_intervals)
        boxe_area_index =  RL_Environment.get_indexes_from_interval(predictions["boxes_area"][mask], self.boxes_areas_intervals)
        color_contrast_index = RL_Environment.get_indexes_from_interval(predictions["custom_scores"]["color_contrast"][mask], self.color_contrast_intervals) 

        return np.array(prediction_index + boxe_area_index * self.nb_predictions_intervals + color_contrast_index * self.nb_predictions_intervals * self.nb_boxes_areas_intervals, dtype=int)

    def get_states_batch(self, predictions_batch):

        states = []
        for predictions in predictions_batch: 

            states.append(self.get_states(predictions))

        return states


    def get_components(self, predictions):

        if len(predictions) == 0:
            return np.array([])

        components = []
        for prediction in predictions: 

            tags = prediction["tags"].cpu()
            #edge = torch.clamp(prediction["custom_scores"]["edge_density"].cpu(), max = 1000) / 1000
            edge = prediction["custom_scores"]["edge_density"].cpu()
            iou = prediction["IoU"].cpu()
            semantic_segmentation_gt = prediction["score_semantic_segmentation"].cpu()

            component = np.stack((tags, iou, edge, semantic_segmentation_gt), axis=1)

            components.append(component)

        return components

    """
    def test_rewards_components_coherents(self, rewards, components):

        components_true, components_false, nb_tags = components

        tags_true = nb_tags[:, 0]
        tags_false = nb_tags[:, 1] - nb_tags[:, 0]

        iou_true = components_true[:, 0]
        edge_true = components_true[:, 1]
        iou_false = components_false[:, 0]
        edge_false = components_false[:, 1]

        np.set_printoptions(precision=10)

        print("tags true", tags_true, "\niou true ", iou_true, "\nedge true ", edge_true)
        print("tags false", tags_false, "\niou false", iou_false, "\nedge false", edge_false)
        components_reward =   1 * iou_true  +    0  * edge_true +    0 * tags_true + 0 * iou_false +   0 * edge_false + -1 * tags_false
        is_compare_good = np.isclose(rewards[:, 0], components_reward)
        if not np.all(is_compare_good):
            print(" rewards ", rewards[:, 0])
            print(" components rewars ", components_reward) 
            print("ERROR in components, not equal to cumulate reward !, for known action")
            print("compare : ", rewards[:, 0][~is_compare_good], components_reward[~is_compare_good])

            print("tags true", tags_true[~is_compare_good], "\niou true ", iou_true[~is_compare_good], "\nedge true ", edge_true[~is_compare_good])
            print("tags false", tags_false[~is_compare_good], "\niou false", iou_false[~is_compare_good], "\nedge false", edge_false[~is_compare_good])
            exit()


        components_reward =   0.1 * iou_true  +    0.1  * edge_true +    0 * tags_true + 0 * iou_false +   1 * edge_false + 0 * tags_false
        is_compare_good = np.isclose(rewards[:, 1], components_reward) 
        if not np.all(is_compare_good):
            print(" rewards ", rewards[:, 1])
            print(" components rewars ", components_reward) 
            print("ERROR in components, not equal to cumulate reward !, for unknown action")
            print(rewards[:, 1][~is_compare_good], components_reward[~is_compare_good])
            exit()

        components_reward =  0 * iou_true  +    -1  * edge_true +    -1 * tags_true + 0 * iou_false +   -1 * edge_false + 1 * tags_false
        is_compare_good = np.isclose(rewards[:, 2], components_reward)
        if not np.all(is_compare_good):
            print(" rewards ", rewards[:, 2])
            print(" components rewars ", components_reward) 
            print("ERROR in components, not equal to cumulate reward !, for background action")
            print(rewards[:, 2][~is_compare_good], components_reward[~is_compare_good])
            exit()
    """




