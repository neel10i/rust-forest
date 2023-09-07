use std::collections::HashMap;

pub struct DecisionTree {
    feature: usize,
    threshold: f64,
    label: Option<usize>,
    left: Option<Box<DecisionTree>>,
    right: Option<Box<DecisionTree>>,
}

impl DecisionTree {
    pub fn new() -> DecisionTree {
        DecisionTree {
            feature: 0,
            threshold: 0.0,
            label: None,
            left: None,
            right: None,
        }
    }

    pub fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<usize>, max_depth: usize) {
        if max_depth == 0 || all_same(labels) {
            self.label = Some(most_common(labels));
            return;
        }

        let (best_feature, best_threshold) = find_best_split(data, labels);
        if best_feature == 0 && best_threshold == 0.0 {
            self.label = Some(most_common(labels));
            return;
        }

        self.feature = best_feature;
        self.threshold = best_threshold;

        let (left_data, left_labels, right_data, right_labels) = split_data(data, labels, best_feature, best_threshold);

        self.left = Some(Box::new(DecisionTree::new()));
        self.right = Some(Box::new(DecisionTree::new()));

        self.left.as_mut().unwrap().train(&left_data, &left_labels, max_depth - 1);
        self.right.as_mut().unwrap().train(&right_data, &right_labels, max_depth - 1);
    }

    pub fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<usize> {
        data.iter().map(|instance| self.predict_instance(instance)).collect()
    }

    pub fn predict_instance(&self, instance: &Vec<f64>) -> usize {
        if let Some(label) = self.label {
            return label;
        }
        if instance[self.feature] <= self.threshold {
            return self.left.as_ref().unwrap().predict_instance(instance);
        } else {
            return self.right.as_ref().unwrap().predict_instance(instance);
        }
    }
}

fn all_same(labels: &Vec<usize>) -> bool {
    let first = labels[0];
    for &label in labels {
        if label != first {
            return false;
        }
    }
    true
}

fn most_common(labels: &Vec<usize>) -> usize {
    let mut label_count = HashMap::new();
    for &label in labels {
        *label_count.entry(label).or_insert(0) += 1;
    }
    *label_count.iter().max_by_key(|&(_, &count)| count).unwrap().0
}

fn split_data(data: &Vec<Vec<f64>>, labels: &Vec<usize>, feature: usize, threshold: f64) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>) {
    let mut left_data = Vec::new();
    let mut left_labels = Vec::new();
    let mut right_data = Vec::new();
    let mut right_labels = Vec::new();

    for (index, instance) in data.iter().enumerate() {
        if instance[feature] <= threshold {
            left_data.push(instance.clone());
            left_labels.push(labels[index]);
        } else {
            right_data.push(instance.clone());
            right_labels.push(labels[index]);
        }
    }

    (left_data, left_labels, right_data, right_labels)
}

fn calculate_gini_impurity(labels: &Vec<usize>) -> f64 {
    let mut label_count = HashMap::new();
    for &label in labels {
        *label_count.entry(label).or_insert(0) += 1;
    }

    let total_samples = labels.len() as f64;
    let mut impurity = 1.0;

    for count in label_count.values() {
        let prob = *count as f64 / total_samples;
        impurity -= prob * prob;
    }

    impurity
}

fn find_best_split(data: &Vec<Vec<f64>>, labels: &Vec<usize>) -> (usize, f64) {
    let mut best_gini = 1.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for feature in 0..data[0].len() {
        let mut unique_values: Vec<f64> = data.iter().map(|instance| instance[feature]).collect();
        unique_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup();

        for i in 0..unique_values.len() - 1 {
            let threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
            let (left_labels, right_labels): (Vec<_>, Vec<_>) =
                labels.iter().enumerate().partition(|&(_, &label)| data[label][feature] <= threshold);
            let gini_left = calculate_gini_impurity(&left_labels.iter().map(|(_, &label)| label).collect());
            let gini_right = calculate_gini_impurity(&right_labels.iter().map(|(_, &label)| label).collect());
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

    println!("Best split: Feature={}, Threshold={}, Gini={}", best_feature, best_threshold, best_gini);
    (best_feature, best_threshold)
}
