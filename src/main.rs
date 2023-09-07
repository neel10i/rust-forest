mod decision_tree; 
mod random_forest; 

use decision_tree::create_decision_tree;
use random_forest::create_random_forest;

fn main() {
    let decision_tree = create_decision_tree();

    let n_trees = 10;
    let mut random_forest = create_random_forest(n_trees);

    let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let labels = vec![0, 1, 0];

    let max_depth = 5; 
    random_forest.train(&data, &labels, max_depth);

    let test_data = vec![vec![4.0, 5.0], vec![1.0, 1.0]];
    let predictions = random_forest.predict(&test_data);

    println!("Predictions: {:?}", predictions);
}
