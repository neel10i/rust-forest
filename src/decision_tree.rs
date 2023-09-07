use std::collections::HashMap;

struct DecisionTree {
    left: Option<Box<DecisionTree>>,
    right: Option<Box<DecisionTree>>,
    feature_index: usize,
    threshold: f64,
    class: Option<usize>,
}

impl DecisionTree {
    fn new() -> DecisionTree {
        DecisionTree {
            left: None,
            right: None,
            feature_index: 0,
            threshold: 0.0,
            class: None,
        }
    }

    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<usize>, max_depth: usize) {

        if max_depth == 0 || labels.iter().all(|&label| label == labels[0]) {
            self.class = Some(labels[0]);
            return;
        }

        let (best_feature, best_threshold) = find_best_split(data, labels);

        let (left_data, left_labels, right_data, right_labels) =
            split_data(data, labels, best_feature, best_threshold);

        if !left_data.is_empty() {
            self.left = Some(Box::new(DecisionTree::new()));
            self.left.as_mut().unwrap().train(&left_data, &left_labels, max_depth - 1);
        }
        if !right_data.is_empty() {
            self.right = Some(Box::new(DecisionTree::new()));
            self.right.as_mut().unwrap().train(&right_data, &right_labels, max_depth - 1);
        }
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<usize> {
        let mut predictions = Vec::new();
        for instance in data.iter() {
            predictions.push(self.predict_instance(instance));
        }
        predictions
    }

    fn predict_instance(&self, instance: &Vec<f64>) -> usize {
        match &self.class {
            Some(class) => *class,
            None => {
                if instance[self.feature_index] <= self.threshold {
                    self.left.as_ref().unwrap().predict_instance(instance)
                } else {
                    self.right.as_ref().unwrap().predict_instance(instance)
                }
            }
        }
    }
}

fn split_data(
    data: &Vec<Vec<f64>>,
    labels: &Vec<usize>,
    feature_index: usize,
    threshold: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>) {
    let mut left_data = Vec::new();
    let mut left_labels = Vec::new();
    let mut right_data = Vec::new();
    let mut right_labels = Vec::new();

    for (i, instance) in data.iter().enumerate() {
        if instance[feature_index] <= threshold {
            left_data.push(instance.clone());
            left_labels.push(labels[i]);
        } else {
            right_data.push(instance.clone());
            right_labels.push(labels[i]);
        }
    }

    (left_data, left_labels, right_data, right_labels)
}

fn calculate_gini_impurity(labels: &Vec<usize>) -> f64 {
    let mut class_counts = HashMap::new();

    for &label in labels.iter() {
        let count = class_counts.entry(label).or_insert(0);
        *count += 1;
    }

    let total_samples = labels.len() as f64;
    let mut gini = 1.0;

    for &count in class_counts.values() {
        let probability = count as f64 / total_samples;
        gini -= probability * probability;
    }

    gini
}

fn find_best_split(data: &Vec<Vec<f64>>, labels: &Vec<usize>) -> (usize, f64) {
    let mut best_gini = 1.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for feature in 0..data[0].len() {
        let unique_values: Vec<f64> = data.iter().map(|instance| instance[feature]).collect();
        let unique_values = unique_values.iter().cloned().collect::<Vec<_>>();
        let unique_values = unique_values.into_iter().collect::<HashSet<_>>();
        let sorted_values: Vec<f64> = unique_values.into_iter().collect();

        for i in 0..sorted_values.len() - 1 {
            let threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0;
            let (left_labels, right_labels): (Vec<_>, Vec<_>) =
                labels.iter().partition(|&&label| data[label][feature] <= threshold);
            let gini_left = calculate_gini_impurity(&left_labels);
            let gini_right = calculate_gini_impurity(&right_labels);
            let weighted_gini = (gini_left * left_labels.len() as f64
                + gini_right * right_labels.len() as f64)
                / labels.len() as f64;

            if weighted_gini < best_gini {
                best_gini = weighted_gini;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    (best_feature, best_threshold)
}

