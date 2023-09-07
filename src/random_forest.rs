use std::collections::HashMap;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(n_trees: usize) -> RandomForest {
        let trees = (0..n_trees).map(|_| DecisionTree::new()).collect();
        RandomForest { trees }
    }

    pub fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<usize>, max_depth: usize) {
        for tree in &mut self.trees {
            tree.train(data, labels, max_depth);
        }
    }

    pub fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<usize> {
        let mut predictions = Vec::new();
        for instance in data.iter() {
            let mut class_counts = HashMap::new();
            for tree in &self.trees {
                let class = tree.predict_instance(instance);
                let count = class_counts.entry(class).or_insert(0);
                *count += 1;
            }
            let majority_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .unwrap()
                .0;
            predictions.push(majority_class);
        }
        predictions
    }
}

pub fn create_random_forest(n_trees: usize) -> RandomForest {
    RandomForest::new(n_trees)
}
