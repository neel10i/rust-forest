mod decision_tree; 
mod random_forest; 

use crate::decision_tree::DecisionTree;

use random_forest::create_random_forest;

fn main() {
    let decision_tree = DecisionTree::new();

    let n_trees = 10;
    let mut random_forest = create_random_forest(n_trees);

    let data = vec![
    vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0], vec![1.5, 2.5], vec![2.5, 3.5],
    vec![5.0, 6.0], vec![6.0, 7.0], vec![7.0, 8.0], vec![5.5, 6.5], vec![6.5, 7.5],
    vec![10.0, 10.0], vec![11.0, 11.0], vec![12.0, 12.0], vec![10.5, 10.5], vec![11.5, 11.5]
    ];
    
    let labels = vec![
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2
    ];

    let max_depth = 5; 
    random_forest.train(&data, &labels, max_depth);

    let test_data = vec![vec![4.0, 5.0], vec![1.0, 1.0]];
    let predictions = random_forest.predict(&test_data);

    println!("Predictions: {:?}", predictions);
}

